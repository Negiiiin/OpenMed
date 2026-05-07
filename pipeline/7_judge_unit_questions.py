#!/usr/bin/env python3
# vLLM env vars must be set before any import.
import os
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

"""
Step 7: Run a model under test on the step-6 unit-question rubric and score
each unit with three independent LLM judges.

For each input row (one MCQ with its unit-question rubric from step 6) the
script:
  1. Runs the model-under-test N times (num_runs) to obtain diverse responses.
  2. Scores each response against the rubric using three judges:
       Perception judge -- VLM (sees the image); scores observation units.
       Knowledge judge  -- text LLM; scores knowledge units.
       Reasoning judge  -- text LLM; scores inference units.
     Each judge returns three numeric axes per unit:
       presence    in {0, 1, 2}   (0=absent, 1=partial, 2=clearly asserted)
       correctness in {-1, 0, 1}  (-1=wrong, 0=N/A, 1=correct)
       consistency in {-1, 0, 1}  (-1=contradiction, 0=N/A, 1=consistent)
  3. Tags each unit with a deterministic failure mode:
       omission / factual_error / internal_inconsistency / chain_break /
       option_elimination / judge_error / ok

Inference backends:
    vllm      -- local GPU model loaded via vLLM (default)
    openai    -- any OpenAI-compatible chat-completions endpoint (no GPU)
    anthropic -- Anthropic messages API (no GPU)

Outputs:
    <output_path>         JSONL; one line per (row, run) with judge scores.
    <output_summary_path> JSON; corpus-level metrics and failure histogram.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from prompts import (
    PKR_KNOWLEDGE_JUDGE_PROMPT,
    PKR_PERCEPTION_VLM_JUDGE_PROMPT,
    PKR_REASONING_JUDGE_PROMPT,
)


# ==========================================
#  MODEL-UNDER-TEST SYSTEM PROMPTS
# ==========================================

_MCQ_SYSTEM_PROMPT = (
    "You are a medical vision-language model. "
    "Answer the multiple-choice question using the image.\n\n"
    "Respond using the following format exactly:\n"
    "<think>\n{your reasoning here}\n</think>\n"
    "<answer>X</answer>\n\n"
    "1) Return EXACTLY one of A, B, C, D, or E as the final answer letter.\n"
    "2) Put that letter inside <answer>...</answer>.\n"
    "3) Put concise but complete clinical reasoning in <think>.\n"
    "4) Do NOT include any extra content after </answer>."
)

_OPEN_VQA_SYSTEM_PROMPT = (
    "You are a medical vision-language model. "
    "Answer the clinical question using the image.\n\n"
    "Respond using the following format exactly:\n"
    "<think>\n{your reasoning here}\n</think>\n"
    "<answer>...</answer>\n\n"
    "Put complete clinical reasoning in <think> and the final answer in <answer>."
)


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    output_summary_path: str

    # Inference backend: "vllm" | "openai" | "anthropic"
    inference_backend: str = "vllm"

    # vLLM settings (backend=vllm)
    model_id: str = "OctoMed/OctoMed-7B"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 32
    min_pixels: int = 262144
    max_pixels: int = 262144

    # OpenAI-compatible inference settings (backend=openai)
    inference_api_key_env: str = "OPENAI_API_KEY"
    inference_base_url: Optional[str] = None
    inference_request_timeout: float = 120.0

    # Anthropic inference settings (backend=anthropic)
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"

    # Sampling
    temperature: float = 1.2
    top_p: float = 0.95
    top_k: int = -1
    repetition_penalty: float = 1.05
    max_new_tokens: int = 4096
    stop_on_answer_tag: bool = True
    num_runs: int = 5
    run_seed_base: int = 42
    enable_thinking: bool = False
    # "auto" | "mcq" | "open_vqa"
    prompt_mode: str = "auto"

    # Judges (OpenAI-compatible; perception judge must be a VLM)
    perception_judge_model: str = "gpt-4o"
    knowledge_judge_model: str = "gpt-4o"
    reasoning_judge_model: str = "gpt-4o"
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_base_url: Optional[str] = None
    judge_reasoning_effort: Optional[str] = "low"
    judge_max_tokens: int = 4096
    judge_temperature: float = 0.0
    save_evidence: bool = True

    # Row slicing
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    max_rows: Optional[int] = None

    # IO
    save_every_n: int = 5
    verbose_errors: bool = True
    delay: float = 0.0


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        path = os.path.join(_SCRIPT_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _opt(k: str) -> Optional[str]:
        v = raw.get(k)
        return str(v).strip() if v is not None and str(v).strip() else None

    def _jm(k: str) -> str:
        return str(raw.get(k) or raw.get("judge_model") or "gpt-4o").strip()

    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        output_summary_path=raw["output_summary_path"],
        inference_backend=str(raw.get("inference_backend", "vllm")).strip().lower(),
        model_id=str(raw.get("model_id", "OctoMed/OctoMed-7B")).strip(),
        max_model_len=int(raw.get("max_model_len", 16384)),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9)),
        tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
        max_num_seqs=int(raw.get("max_num_seqs", 32)),
        min_pixels=int(raw.get("min_pixels", 262144)),
        max_pixels=int(raw.get("max_pixels", 262144)),
        inference_api_key_env=str(raw.get("inference_api_key_env", "OPENAI_API_KEY")).strip(),
        inference_base_url=_opt("inference_base_url"),
        inference_request_timeout=float(raw.get("inference_request_timeout", 120.0)),
        anthropic_api_key_env=str(raw.get("anthropic_api_key_env", "ANTHROPIC_API_KEY")).strip(),
        temperature=float(raw.get("temperature", 1.2)),
        top_p=float(raw.get("top_p", 0.95)),
        top_k=int(raw.get("top_k", -1)),
        repetition_penalty=float(raw.get("repetition_penalty", 1.05)),
        max_new_tokens=int(raw.get("max_new_tokens", 4096)),
        stop_on_answer_tag=bool(raw.get("stop_on_answer_tag", True)),
        num_runs=max(1, int(raw.get("num_runs", 5))),
        run_seed_base=int(raw.get("run_seed_base", 42)),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        prompt_mode=str(raw.get("prompt_mode", "auto")).strip().lower(),
        perception_judge_model=_jm("perception_judge_model"),
        knowledge_judge_model=_jm("knowledge_judge_model"),
        reasoning_judge_model=_jm("reasoning_judge_model"),
        judge_api_key_env=str(raw.get("judge_api_key_env", "OPENAI_API_KEY")).strip(),
        judge_base_url=_opt("judge_base_url"),
        judge_reasoning_effort=_opt("judge_reasoning_effort"),
        judge_max_tokens=int(raw.get("judge_max_tokens", 4096)),
        judge_temperature=float(raw.get("judge_temperature", 0.0)),
        save_evidence=bool(raw.get("save_evidence", True)),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        max_rows=raw.get("max_rows"),
        save_every_n=max(1, int(raw.get("save_every_n", 5))),
        verbose_errors=bool(raw.get("verbose_errors", True)),
        delay=float(raw.get("delay", 0.0)),
    )


# ==========================================
#  IO HELPERS
# ==========================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def normalize_letter(x: Any) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return ""
    m = re.search(r"\b([A-E])\b", s)
    return m.group(1) if m else (s[:1] if s[:1] in "ABCDE" else "")


_ANSWER_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)


def extract_answer_letter(text: str) -> str:
    if not text:
        return ""
    for m in reversed(list(_ANSWER_RE.finditer(text))):
        inner = (m.group(1) or "").strip().upper()
        hit = re.search(r"\b([A-E])\b", inner)
        if hit:
            return hit.group(1)
        if inner[:1] in "ABCDE":
            return inner[:1]
    return ""


def choices_to_list(choices: Any) -> List[str]:
    if isinstance(choices, dict):
        return [str(v).strip() for _, v in sorted(choices.items()) if str(v).strip()]
    if isinstance(choices, list):
        return [str(c).strip() for c in choices if str(c).strip()]
    return []


def format_question_block(question: str, choice_list: List[str]) -> str:
    q = str(question or "").strip()
    if not choice_list:
        return q
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choice_list))
    return f"{q}\n\nOptions:\n{opts}"


def resolve_image(raw: str, input_jsonl: str) -> str:
    p = str(raw or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p if os.path.exists(p) else ""
    here = os.path.dirname(os.path.abspath(input_jsonl))
    c = os.path.join(here, p)
    return c if os.path.exists(c) else ""


def row_image_path(row: Dict[str, Any]) -> str:
    for k in ("subfig_path", "image", "full_fig_path", "image_path"):
        v = row.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def row_ref_letter(row: Dict[str, Any]) -> str:
    for k in ("answer", "answer_letter", "ground_truth", "correct_answer"):
        v = row.get(k)
        if v and str(v).strip():
            n = normalize_letter(v)
            if n:
                return n
    return ""


def encode_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


# ==========================================
#  RUBRIC: unit_questions -> per-layer payloads
# ==========================================

_AXIS_TO_LAYER = {
    "observation": "perception", "perception": "perception",
    "knowledge": "knowledge",
    "inference": "reasoning",    "reasoning": "reasoning",
}


def _presence_fallback(topic: str, claim: str, layer: str) -> str:
    anchor = (topic or claim).rstrip(".")
    if not anchor:
        return ""
    stem = ("Does the response make an inferential link about"
            if layer == "reasoning" else "Does the response discuss")
    return f"{stem} {anchor}?"


def _correctness_fallback(claim: str, layer: str) -> str:
    c = (claim or "").rstrip(".")
    if not c:
        return ""
    verb = "correctly conclude" if layer == "reasoning" else "correctly state"
    return f"Does the response {verb} that {c}?"


def _unit_to_payload(u: Dict[str, Any], *, layer: str, uid: str) -> Dict[str, Any]:
    topic = str(u.get("topic") or "").strip()
    claim = str(u.get("claim") or u.get("claim_text") or "").strip()
    pq    = str(u.get("presence_question") or "").strip()    or _presence_fallback(topic, claim, layer)
    cq    = (str(u.get("correctness_question") or "").strip()
             or str(u.get("question") or "").strip()
             or _correctness_fallback(claim, layer))
    common = dict(topic=topic, claim_text=claim,
                  presence_question=pq, correctness_question=cq,
                  _layer=layer, _uid=uid,
                  _importance=str(u.get("importance") or "").lower())
    if layer == "perception":
        return {"claim_id": uid, **common}
    if layer == "knowledge":
        return {"knowledge_id": uid, **common}
    return {"reasoning_id": uid,
            "relation": u.get("relation"),
            "conclusion": u.get("conclusion"),
            "premise_perception_ids": list(u.get("premise_perception_ids") or []),
            "premise_knowledge_ids":  list(u.get("premise_knowledge_ids") or []),
            **common}


def group_units(row: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict]] = {"perception": [], "knowledge": [], "reasoning": []}
    for u in (row.get("unit_questions") or []):
        if not isinstance(u, dict):
            continue
        uid   = str(u.get("unit_id") or "").strip()
        layer = _AXIS_TO_LAYER.get(str(u.get("axis") or "").lower(), "reasoning")
        if uid:
            out[layer].append(_unit_to_payload(u, layer=layer, uid=uid))
    return out


# ==========================================
#  INFERENCE BACKENDS
# ==========================================

def build_vllm(cfg: Config) -> Tuple[Any, Any]:
    from vllm import LLM
    from transformers import AutoProcessor

    hf_overrides: Optional[Dict[str, Any]] = None
    if "OctoMed" in cfg.model_id or "Qwen2.5-VL" in cfg.model_id:
        hf_overrides = {"rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]}}
    llm = LLM(
        model=cfg.model_id, trust_remote_code=True, dtype="bfloat16",
        max_model_len=cfg.max_model_len, tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs, limit_mm_per_prompt={"image": 1},
        hf_overrides=hf_overrides,
    )
    proc = AutoProcessor.from_pretrained(
        cfg.model_id, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, trust_remote_code=True,
    )
    return llm, proc


def _pick_system_prompt(cfg: Config, question_text: str, ref_letter: str) -> str:
    if cfg.prompt_mode == "mcq":
        return _MCQ_SYSTEM_PROMPT
    if cfg.prompt_mode == "open_vqa":
        return _OPEN_VQA_SYSTEM_PROMPT
    if ref_letter or re.search(r"^\s*[A-E][\.\)]\s+", question_text or "", re.MULTILINE):
        return _MCQ_SYSTEM_PROMPT
    return _OPEN_VQA_SYSTEM_PROMPT


def _trim_at_answer_close(text: str) -> str:
    idx = (text or "").lower().find("</answer>")
    return text[:idx + len("</answer>")].strip() if idx != -1 else (text or "").strip()


def run_vllm(llm: Any, proc: Any, cfg: Config,
             *, image_path: str, question_text: str, system_prompt: str, seed: int) -> List[str]:
    from vllm import SamplingParams

    full = f"{system_prompt}\n\nQuestion:\n{question_text}\n"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "placeholder"},
        {"type": "text", "text": full},
    ]}]
    kw: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        kw["enable_thinking"] = cfg.enable_thinking
    except TypeError:
        pass
    formatted = proc.apply_chat_template(messages, **kw)
    img = Image.open(image_path).convert("RGB")

    sp_kw: Dict[str, Any] = dict(
        n=cfg.num_runs, temperature=cfg.temperature, top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty, max_tokens=cfg.max_new_tokens, seed=seed,
    )
    if cfg.top_k > 0:
        sp_kw["top_k"] = cfg.top_k
    if cfg.stop_on_answer_tag:
        sp_kw["stop"] = ["</answer>"]
    outputs = llm.generate([{"prompt": formatted, "multi_modal_data": {"image": img}}],
                           SamplingParams(**sp_kw))
    if not outputs or not outputs[0].outputs:
        return [""]
    return [_trim_at_answer_close((o.text or "").strip()) for o in outputs[0].outputs]


def _openai_n(client: Any, cfg: Config, messages: List[Dict], n: int) -> List[str]:
    ml = (cfg.model_id or "").lower()
    kw: Dict[str, Any] = {"model": cfg.model_id, "messages": messages,
                          "n": n, "temperature": cfg.temperature}
    if ml.startswith("gpt-5") or ml.startswith("o1"):
        kw["max_completion_tokens"] = cfg.max_new_tokens
    else:
        kw["max_tokens"] = cfg.max_new_tokens
    if cfg.top_p < 1.0:
        kw["top_p"] = cfg.top_p
    resp = client.chat.completions.create(**kw)
    return [_trim_at_answer_close((c.message.content or "").strip()) for c in resp.choices]


def run_openai(client: Any, cfg: Config,
               *, image_path: str, question_text: str, system_prompt: str) -> List[str]:
    data_url = encode_image(image_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "text", "text": f"Question:\n{question_text}"},
        ]},
    ]
    if "gemini" in (cfg.model_id or "").lower():
        out: List[str] = []
        for _ in range(cfg.num_runs):
            out.extend(_openai_n(client, cfg, messages, 1))
        return out
    return _openai_n(client, cfg, messages, cfg.num_runs)


def run_anthropic(client: Any, cfg: Config,
                  *, image_path: str, question_text: str, system_prompt: str) -> List[str]:
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mt  = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
           "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
        {"type": "text", "text": f"Question:\n{question_text}"},
    ]
    results: List[str] = []
    for _ in range(cfg.num_runs):
        resp = client.messages.create(
            model=cfg.model_id, max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        results.append(_trim_at_answer_close(text))
    return results


# ==========================================
#  JUDGE CLIENT + CALL
# ==========================================

_GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_OPTION_RE   = re.compile(
    r"\b(?:option|choice)\s+[A-E]\b|\banswer\s+(?:choice|option)\b"
    r"|\bcorrect\s+answer\s+is\s+[A-E]\b", re.IGNORECASE,
)
_TAG_RE = re.compile(r"</?(?:think|answer|reasoning|thought|redacted_thinking)>", re.IGNORECASE)


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()


def _is_option_heavy(text: str) -> bool:
    return bool(_OPTION_RE.search(text or ""))


def build_judge_client(cfg: Config) -> Any:
    from openai import OpenAI

    is_gemini = any("gemini" in (m or "").lower() for m in (
        cfg.perception_judge_model, cfg.knowledge_judge_model, cfg.reasoning_judge_model,
    ))
    base_url = cfg.judge_base_url or (_GEMINI_URL if is_gemini else None)
    api_key  = os.environ.get(cfg.judge_api_key_env)
    if not api_key and is_gemini:
        for k in _GEMINI_ENVS:
            if os.environ.get(k):
                api_key  = os.environ[k]
                base_url = base_url or _GEMINI_URL
                break
    if not api_key:
        raise EnvironmentError(f"Judge API key env not set: {cfg.judge_api_key_env}")
    kw: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    return OpenAI(**kw)


def _is_reasoning_judge(model: str) -> bool:
    return bool(re.search(r"\bgpt-5\b|gpt5|\bo[1-9]\b", (model or "").lower()))


def _msg_text(msg: Any) -> str:
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts: List[str] = []
        for block in c:
            txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if txt:
                parts.append(str(txt))
        return "\n".join(parts).strip()
    return str(c or "").strip()


def _chat_json(client: Any, *, cfg: Config, model: str, messages: List[Dict]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"model": model, "messages": messages}
    if _is_reasoning_judge(model):
        kw["max_completion_tokens"] = max(2048, cfg.judge_max_tokens)
        if cfg.judge_reasoning_effort:
            kw["reasoning_effort"] = cfg.judge_reasoning_effort
    else:
        kw["max_tokens"]  = cfg.judge_max_tokens
        kw["temperature"] = cfg.judge_temperature
    try:
        resp = client.chat.completions.create(**kw)
    except Exception:
        if _is_reasoning_judge(model) and "reasoning_effort" in kw:
            kw.pop("reasoning_effort")
            resp = client.chat.completions.create(**kw)
        else:
            raise
    text = _msg_text(resp.choices[0].message).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _ax(v: Any, *, ok: Tuple[int, ...], default: int) -> int:
    try:
        x = int(v)
        return x if x in ok else default
    except (TypeError, ValueError):
        return default


# ==========================================
#  PAYLOAD BUILDERS + EMPTY SENTINELS
# ==========================================

def _pp(u: Dict) -> Dict:
    return {k: u.get(k) for k in ("claim_id", "topic", "claim_text",
                                   "presence_question", "correctness_question")}

def _pk(u: Dict) -> Dict:
    return {k: u.get(k) for k in ("knowledge_id", "topic", "claim_text",
                                   "presence_question", "correctness_question")}

def _pr(u: Dict) -> Dict:
    return {k: u.get(k) for k in ("reasoning_id", "topic", "claim_text", "relation",
                                   "conclusion", "premise_perception_ids",
                                   "premise_knowledge_ids",
                                   "presence_question", "correctness_question")}

def _ep(uid: str, reason: str) -> Dict:
    return dict(claim_id=uid, presence=0, correctness=0, consistency=0, evidence="", judge_error=reason)

def _ek(uid: str, reason: str) -> Dict:
    return dict(knowledge_id=uid, presence=0, correctness=0, consistency=0, evidence="", judge_error=reason)

def _er(uid: str, reason: str) -> Dict:
    return dict(reasoning_id=uid, presence=0, correctness=0, consistency=0,
                chain_grounding_judge=dict(perception_premise_present=None,
                                           knowledge_premise_present=None,
                                           premises_correct=None),
                evidence="", judge_error=reason)


# ==========================================
#  THREE JUDGES
# ==========================================

def judge_perception(client: Any, *, cfg: Config, image_path: str,
                     case_question: str, opts_block: str, ref_answer: str,
                     model_response: str, units: List[Dict]) -> List[Dict]:
    if not units:
        return []
    user_text = (
        f"CASE QUESTION:\n{case_question}\n\n"
        f"ANSWER OPTIONS:\n{opts_block or '(none)'}\n\n"
        f"REFERENCE_ANSWER:\n{ref_answer or '(unknown)'}\n\n"
        f"MODEL FULL RESPONSE:\n{model_response}\n\n"
        f"PERCEPTION_RUBRICS_JSON:\n{json.dumps([_pp(u) for u in units], indent=2)}\n"
    )
    try:
        img_url = encode_image(image_path)
    except Exception as exc:
        return [_ep(str(u.get("claim_id") or ""), f"image_load:{exc}") for u in units]
    messages = [
        {"role": "system", "content": PKR_PERCEPTION_VLM_JUDGE_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ]
    try:
        obj = _chat_json(client, cfg=cfg, model=cfg.perception_judge_model, messages=messages)
    except Exception as exc:
        return [_ep(str(u.get("claim_id") or ""), f"call:{exc}") for u in units]
    by_id = {str(it.get("claim_id") or ""): it
             for it in (obj.get("items") or []) if isinstance(it, dict)}
    out = []
    for u in units:
        uid = str(u.get("claim_id") or "")
        it  = by_id.get(uid)
        if it is None:
            out.append(_ep(uid, "missing_in_judge_output"))
        else:
            out.append(dict(claim_id=uid,
                            presence    =_ax(it.get("presence"),    ok=(0,1,2),   default=0),
                            correctness =_ax(it.get("correctness"), ok=(-1,0,1),  default=0),
                            consistency =_ax(it.get("consistency"), ok=(-1,0,1),  default=0),
                            evidence    =str(it.get("evidence") or "").strip()))
    return out


def judge_knowledge(client: Any, *, cfg: Config,
                    case_question: str, opts_block: str, ref_answer: str,
                    model_response: str, units: List[Dict]) -> List[Dict]:
    if not units:
        return []
    user_text = (
        f"CASE QUESTION:\n{case_question}\n\n"
        f"ANSWER OPTIONS:\n{opts_block or '(none)'}\n\n"
        f"REFERENCE_ANSWER:\n{ref_answer or '(unknown)'}\n\n"
        f"MODEL FULL RESPONSE:\n{model_response}\n\n"
        f"KNOWLEDGE_RUBRICS_JSON:\n{json.dumps([_pk(u) for u in units], indent=2)}\n"
    )
    messages = [
        {"role": "system", "content": PKR_KNOWLEDGE_JUDGE_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    try:
        obj = _chat_json(client, cfg=cfg, model=cfg.knowledge_judge_model, messages=messages)
    except Exception as exc:
        return [_ek(str(u.get("knowledge_id") or ""), f"call:{exc}") for u in units]
    by_id = {str(it.get("knowledge_id") or ""): it
             for it in (obj.get("items") or []) if isinstance(it, dict)}
    out = []
    for u in units:
        uid = str(u.get("knowledge_id") or "")
        it  = by_id.get(uid)
        if it is None:
            out.append(_ek(uid, "missing_in_judge_output"))
        else:
            out.append(dict(knowledge_id=uid,
                            presence    =_ax(it.get("presence"),    ok=(0,1,2),   default=0),
                            correctness =_ax(it.get("correctness"), ok=(-1,0,1),  default=0),
                            consistency =_ax(it.get("consistency"), ok=(-1,0,1),  default=0),
                            evidence    =str(it.get("evidence") or "").strip()))
    return out


def judge_reasoning(client: Any, *, cfg: Config,
                    case_question: str, opts_block: str, ref_answer: str,
                    model_response: str, units: List[Dict]) -> List[Dict]:
    if not units:
        return []
    user_text = (
        f"CASE QUESTION:\n{case_question}\n\n"
        f"ANSWER OPTIONS:\n{opts_block or '(none)'}\n\n"
        f"REFERENCE_ANSWER:\n{ref_answer or '(unknown)'}\n\n"
        f"MODEL FULL RESPONSE:\n{model_response}\n\n"
        f"REASONING_RUBRICS_JSON:\n{json.dumps([_pr(u) for u in units], indent=2)}\n"
    )
    messages = [
        {"role": "system", "content": PKR_REASONING_JUDGE_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    try:
        obj = _chat_json(client, cfg=cfg, model=cfg.reasoning_judge_model, messages=messages)
    except Exception as exc:
        return [_er(str(u.get("reasoning_id") or ""), f"call:{exc}") for u in units]

    def _b(x: Any) -> Optional[bool]:
        if x is None: return None
        if isinstance(x, bool): return x
        s = str(x).lower()
        if s in ("true", "yes", "1"):  return True
        if s in ("false", "no", "0"): return False
        return None

    by_id = {str(it.get("reasoning_id") or ""): it
             for it in (obj.get("items") or []) if isinstance(it, dict)}
    out = []
    for u in units:
        uid = str(u.get("reasoning_id") or "")
        it  = by_id.get(uid)
        if it is None:
            out.append(_er(uid, "missing_in_judge_output"))
        else:
            cg = it.get("chain_grounding") if isinstance(it.get("chain_grounding"), dict) else {}
            out.append(dict(
                reasoning_id=uid,
                presence    =_ax(it.get("presence"),    ok=(0,1,2),   default=0),
                correctness =_ax(it.get("correctness"), ok=(-1,0,1),  default=0),
                consistency =_ax(it.get("consistency"), ok=(-1,0,1),  default=0),
                chain_grounding_judge=dict(
                    perception_premise_present=_b(cg.get("perception_premise_present")),
                    knowledge_premise_present =_b(cg.get("knowledge_premise_present")),
                    premises_correct          =_b(cg.get("premises_correct")),
                ),
                evidence=str(it.get("evidence") or "").strip(),
            ))
    return out


# ==========================================
#  DETERMINISTIC CHAIN GROUNDING
# ==========================================

def chain_grounding(*, r_unit: Dict, p_by_id: Dict, k_by_id: Dict) -> Dict[str, Any]:
    p_ids = list(r_unit.get("premise_perception_ids") or [])
    k_ids = list(r_unit.get("premise_knowledge_ids")  or [])

    def _pk2(item: Optional[Dict]) -> Tuple[bool, bool]:
        if not item: return False, False
        return int(item.get("presence") or 0) >= 1, int(item.get("correctness") or 0) == 1

    p_st = [_pk2(p_by_id.get(i)) for i in p_ids]
    k_st = [_pk2(k_by_id.get(i)) for i in k_ids]
    pp = all(s[0] for s in p_st) if p_st else None
    pc = all(s[1] for s in p_st) if p_st else None
    kp = all(s[0] for s in k_st) if k_st else None
    kc = all(s[1] for s in k_st) if k_st else None
    grounded: Optional[bool] = None
    if p_ids or k_ids:
        grounded = all(v is not False for v in (pp, pc, kp, kc))
    return dict(reasoning_id=r_unit.get("reasoning_id"),
                premise_perception_ids=p_ids, premise_knowledge_ids=k_ids,
                perception_premises_present=pp, perception_premises_correct=pc,
                knowledge_premises_present=kp, knowledge_premises_correct=kc,
                grounded=grounded)


# ==========================================
#  FAILURE-MODE TAGGING
# ==========================================

def _tag_pk(it: Dict) -> str:
    if "judge_error" in it: return "judge_error"
    p, c, s = int(it.get("presence") or 0), int(it.get("correctness") or 0), int(it.get("consistency") or 0)
    if p == 0:  return "omission"
    if c == -1: return "factual_error"
    if s == -1: return "internal_inconsistency"
    return "ok"


def _tag_r(it: Dict, *, cg: Dict, option_heavy: bool) -> str:
    if "judge_error" in it: return "judge_error"
    p, c, s = int(it.get("presence") or 0), int(it.get("correctness") or 0), int(it.get("consistency") or 0)
    if p == 0:  return "omission"
    if c == -1: return "option_elimination" if option_heavy else "factual_error"
    if s == -1: return "internal_inconsistency"
    if cg and cg.get("grounded") is False: return "chain_break"
    return "ok"


# ==========================================
#  AGGREGATION
# ==========================================

def _mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def _layer_metrics(items: List[Dict]) -> Dict[str, Any]:
    valid   = [i for i in items if "judge_error" not in i]
    present = [i for i in valid if int(i.get("presence") or 0) >= 1]
    return {
        "presence_mean":    _mean([int(i.get("presence") or 0) / 2.0 for i in valid]),
        "correctness_rate": _mean([1.0 if int(i.get("correctness") or 0) == 1 else 0.0 for i in present]),
        "consistency_rate": _mean([1.0 if int(i.get("consistency") or 0) == 1 else 0.0 for i in present]),
        "n_items":   len(items),
        "n_present": len(present),
    }


def _fail_hist(items: List[Dict]) -> Dict[str, int]:
    h: Dict[str, int] = defaultdict(int)
    for it in items:
        h[str(it.get("failure_mode") or "ok")] += 1
    return dict(h)


def _maybe_strip_evidence(items: List[Dict], save: bool) -> List[Dict]:
    if save:
        return items
    return [{k: v for k, v in it.items() if k != "evidence"} for it in items]


# ==========================================
#  SUMMARY
# ==========================================

def write_summary(path: str, *, cfg: Config, input_path: str, output_path: str,
                  n_rows: int, n_runs: int, n_lines: int,
                  n_mc_correct: int, n_mc_grounded: int,
                  skip_no_image: int, skip_no_units: int, skip_inf_err: int,
                  judge_errors: int,
                  corpus_fail: Dict[str, int],
                  corpus_layer_fail: Dict[str, Dict[str, int]],
                  corpus_sums: Dict[str, Dict[str, List[float]]],
                  corpus_chain: List[float]) -> None:
    payload = {
        "input_path": input_path, "output_path": output_path,
        "rows_judged": n_rows, "n_total_runs": n_runs, "n_lines_written": n_lines,
        "n_mc_correct": n_mc_correct,
        "mc_accuracy": (n_mc_correct / n_runs) if n_runs else None,
        "n_mc_correct_with_grounded_reasoning": n_mc_grounded,
        "rows_skipped_no_image": skip_no_image,
        "rows_skipped_no_units": skip_no_units,
        "rows_skipped_inference_error": skip_inf_err,
        "judge_errors": judge_errors,
        "models": {
            "model_id": cfg.model_id, "inference_backend": cfg.inference_backend,
            "perception_judge": cfg.perception_judge_model,
            "knowledge_judge":  cfg.knowledge_judge_model,
            "reasoning_judge":  cfg.reasoning_judge_model,
        },
        "sampling": {
            "num_runs": cfg.num_runs, "temperature": cfg.temperature,
            "top_p": cfg.top_p, "max_new_tokens": cfg.max_new_tokens,
        },
        "corpus_failure_histogram": dict(corpus_fail),
        "corpus_layer_failure_histogram": {k: dict(v) for k, v in corpus_layer_fail.items()},
        "corpus_axis_means": {
            layer: {axis: _mean(vals) for axis, vals in axes.items()}
            for layer, axes in corpus_sums.items()
        },
        "corpus_reasoning_grounded_rate": _mean(corpus_chain),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(summary_path: str, output_path: str) -> None:
    try:
        with open(summary_path) as f:
            s = json.load(f)
    except Exception:
        return
    means = s.get("corpus_axis_means") or {}
    print("\n[step7] ===== CORPUS METRICS =====")
    for layer in ("perception", "knowledge", "reasoning"):
        m   = means.get(layer) or {}
        fmt = lambda v: f"{float(v)*100:6.2f}%" if v is not None else "   n/a"
        print(f"  {layer:11s}  presence={fmt(m.get('presence_mean'))}  "
              f"correctness={fmt(m.get('correctness_rate'))}  "
              f"consistency={fmt(m.get('consistency_rate'))}")
    gr = s.get("corpus_reasoning_grounded_rate")
    if gr is not None:
        print(f"\n  reasoning_grounded_rate = {float(gr)*100:6.2f}%")
    mc = s.get("mc_accuracy")
    if mc is not None:
        print(f"  mc_accuracy             = {float(mc)*100:6.2f}%  "
              f"({s.get('n_mc_correct')}/{s.get('n_total_runs')})")
    fh = s.get("corpus_failure_histogram") or {}
    if fh:
        print("\n  failure_mode_histogram:")
        for k in sorted(fh, key=lambda kk: -int(fh.get(kk) or 0)):
            print(f"    {k:24s} {fh[k]}")
    print(f"\n[step7] details : {output_path}")
    print(f"[step7] summary : {summary_path}")


# ==========================================
#  MAIN
# ==========================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Step 7: Judge unit questions.")
    parser.add_argument("--config",              type=str, default="cfg/7_judge_unit_questions.yaml")
    parser.add_argument("--from_row",            type=int, default=None)
    parser.add_argument("--to_row",              type=int, default=None)
    parser.add_argument("--output_path",         type=str, default=None)
    parser.add_argument("--output_summary_path", type=str, default=None)
    args = parser.parse_args()

    cfg          = load_config(args.config)
    input_path   = os.path.abspath(cfg.input_path)
    output_path  = os.path.abspath(args.output_path or cfg.output_path)
    summary_path = os.path.abspath(args.output_summary_path or cfg.output_summary_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    all_rows = read_jsonl(input_path)
    n_all    = len(all_rows)
    fr       = args.from_row if args.from_row is not None else (cfg.from_row or 0)
    to       = args.to_row   if args.to_row   is not None else (cfg.to_row   or n_all)
    rows     = all_rows[fr:to]
    if cfg.max_rows is not None:
        rows = rows[: cfg.max_rows]

    print(f"[step7] input  = {input_path}")
    print(f"[step7] output = {output_path}")
    print(f"[step7] summary= {summary_path}")
    print(f"[step7] rows={len(rows)} of {n_all}  model={cfg.model_id}  "
          f"backend={cfg.inference_backend}  num_runs={cfg.num_runs}")
    print(f"[step7] judges: P={cfg.perception_judge_model}  "
          f"K={cfg.knowledge_judge_model}  R={cfg.reasoning_judge_model}")

    llm = proc = inf_client = ant_client = None
    if cfg.inference_backend == "openai":
        from openai import OpenAI
        api_key = os.environ.get(cfg.inference_api_key_env, "")
        if not api_key:
            raise EnvironmentError(f"Inference API key env var '{cfg.inference_api_key_env}' not set.")
        kw: Dict[str, Any] = {"api_key": api_key, "timeout": cfg.inference_request_timeout}
        if cfg.inference_base_url:
            kw["base_url"] = cfg.inference_base_url
        inf_client = OpenAI(**kw)
        print("[step7] OpenAI inference client ready.")
    elif cfg.inference_backend == "anthropic":
        import anthropic
        api_key = os.environ.get(cfg.anthropic_api_key_env, "")
        if not api_key:
            raise EnvironmentError(f"Anthropic key env '{cfg.anthropic_api_key_env}' not set.")
        ant_client = anthropic.Anthropic(api_key=api_key)
        print("[step7] Anthropic client ready.")
    else:
        print("[step7] Loading vLLM model ...")
        llm, proc = build_vllm(cfg)
        print("[step7] vLLM ready.")

    judge_client = build_judge_client(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    corpus_fail: Dict[str, int] = defaultdict(int)
    corpus_layer_fail: Dict[str, Dict[str, int]] = {
        "perception": defaultdict(int), "knowledge": defaultdict(int), "reasoning": defaultdict(int),
    }
    corpus_sums: Dict[str, Dict[str, List[float]]] = {
        layer: {"presence_mean": [], "correctness_rate": [], "consistency_rate": []}
        for layer in ("perception", "knowledge", "reasoning")
    }
    corpus_chain: List[float] = []
    n_rows_done = n_runs = n_lines = n_mc_correct = n_mc_grounded = 0
    skip_no_image = skip_no_units = skip_inf_err = n_judge_err = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="step7:judge"):
            rid     = str(row.get("id") or "").strip()
            grouped = group_units(row)
            ref_p, ref_k, ref_r = grouped["perception"], grouped["knowledge"], grouped["reasoning"]
            if not (ref_p or ref_k or ref_r):
                skip_no_units += 1
                continue

            ref_letter  = row_ref_letter(row)
            choice_list = choices_to_list(row.get("choices") or row.get("options"))
            opts_block  = ("\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choice_list))
                           if choice_list else "(no explicit options)")
            stem   = str(row.get("question") or "").strip()
            full_q = format_question_block(stem, choice_list)

            img_raw  = row_image_path(row)
            img_path = resolve_image(img_raw, input_path)
            if not img_path:
                tqdm.write(f"[step7] skip id={rid!r}: image not found ({img_raw!r})", file=sys.stderr)
                skip_no_image += 1
                continue

            seed       = (cfg.run_seed_base or 0) + (n_rows_done + 1) * 1000
            sys_prompt = _pick_system_prompt(cfg, full_q, ref_letter)

            try:
                if cfg.inference_backend == "openai":
                    responses = run_openai(inf_client, cfg, image_path=img_path,
                                           question_text=full_q, system_prompt=sys_prompt)
                elif cfg.inference_backend == "anthropic":
                    responses = run_anthropic(ant_client, cfg, image_path=img_path,
                                              question_text=full_q, system_prompt=sys_prompt)
                else:
                    responses = run_vllm(llm, proc, cfg, image_path=img_path,
                                         question_text=full_q, system_prompt=sys_prompt, seed=seed)
            except Exception as exc:
                skip_inf_err += 1
                tqdm.write(f"[step7] inference error id={rid!r}: {type(exc).__name__}: {exc}",
                           file=sys.stderr)
                if cfg.verbose_errors:
                    tqdm.write(traceback.format_exc(), file=sys.stderr)
                continue

            for run_idx, raw_resp in enumerate(responses):
                model_letter = extract_answer_letter(raw_resp)
                mc_correct   = bool(ref_letter and model_letter and model_letter == ref_letter)
                resp_j       = _strip_tags(raw_resp)
                opt_heavy    = _is_option_heavy(resp_j)

                try:
                    p_items = judge_perception(judge_client, cfg=cfg,
                                               image_path=img_path, case_question=stem or full_q,
                                               opts_block=opts_block, ref_answer=ref_letter,
                                               model_response=resp_j, units=ref_p)
                except Exception as exc:
                    n_judge_err += 1
                    p_items = [_ep(str(u.get("claim_id") or ""), f"call:{exc}") for u in ref_p]

                try:
                    k_items = judge_knowledge(judge_client, cfg=cfg,
                                              case_question=stem or full_q,
                                              opts_block=opts_block, ref_answer=ref_letter,
                                              model_response=resp_j, units=ref_k)
                except Exception as exc:
                    n_judge_err += 1
                    k_items = [_ek(str(u.get("knowledge_id") or ""), f"call:{exc}") for u in ref_k]

                try:
                    r_items = judge_reasoning(judge_client, cfg=cfg,
                                              case_question=stem or full_q,
                                              opts_block=opts_block, ref_answer=ref_letter,
                                              model_response=resp_j, units=ref_r)
                except Exception as exc:
                    n_judge_err += 1
                    r_items = [_er(str(u.get("reasoning_id") or ""), f"call:{exc}") for u in ref_r]

                p_by_id = {it["claim_id"]: it for it in p_items if it.get("claim_id")}
                k_by_id = {it["knowledge_id"]: it for it in k_items if it.get("knowledge_id")}
                chain_units = [
                    chain_grounding(r_unit=ru, p_by_id=p_by_id, k_by_id=k_by_id)
                    for ru in ref_r
                ]
                chain_by_id = {c["reasoning_id"]: c for c in chain_units}

                for it in p_items: it["failure_mode"] = _tag_pk(it)
                for it in k_items: it["failure_mode"] = _tag_pk(it)
                for it in r_items:
                    cg = chain_by_id.get(it.get("reasoning_id")) or {}
                    it["failure_mode"] = _tag_r(it, cg=cg, option_heavy=opt_heavy)

                per_layer = {
                    "perception": _layer_metrics(p_items),
                    "knowledge":  _layer_metrics(k_items),
                    "reasoning":  _layer_metrics(r_items),
                }
                grounded_flags = [c.get("grounded") for c in chain_units
                                  if c.get("grounded") is not None]
                grounded_rate  = (float(sum(grounded_flags) / len(grounded_flags))
                                  if grounded_flags else None)
                fail_hist = {
                    "perception": _fail_hist(p_items),
                    "knowledge":  _fail_hist(k_items),
                    "reasoning":  _fail_hist(r_items),
                }

                n_runs += 1
                if mc_correct:
                    n_mc_correct += 1
                    if grounded_rate is not None and grounded_rate >= 1.0:
                        n_mc_grounded += 1
                for layer, items in (("perception", p_items), ("knowledge", k_items),
                                     ("reasoning", r_items)):
                    for it in items:
                        corpus_fail[it["failure_mode"]] += 1
                        corpus_layer_fail[layer][it["failure_mode"]] += 1
                    pl = per_layer[layer]
                    for ak in ("presence_mean", "correctness_rate", "consistency_rate"):
                        v = pl.get(ak)
                        if v is not None:
                            corpus_sums[layer][ak].append(float(v))
                if grounded_rate is not None:
                    corpus_chain.append(grounded_rate)

                record: Dict[str, Any] = {
                    "id": rid, "run": run_idx, "seed": seed + run_idx,
                    "model_id": cfg.model_id,
                    "judge_models": dict(perception=cfg.perception_judge_model,
                                        knowledge=cfg.knowledge_judge_model,
                                        reasoning=cfg.reasoning_judge_model),
                    "ref_letter": ref_letter,
                    "model_raw_answer": raw_resp, "model_answer_letter": model_letter,
                    "mc_correct": mc_correct, "response_uses_option_language": opt_heavy,
                    "judge_perception": _maybe_strip_evidence(p_items, cfg.save_evidence),
                    "judge_knowledge":  _maybe_strip_evidence(k_items, cfg.save_evidence),
                    "judge_reasoning":  _maybe_strip_evidence(r_items, cfg.save_evidence),
                    "chain_grounding":  chain_units,
                    "per_layer_metrics": per_layer,
                    "reasoning_grounded_rate": grounded_rate,
                    "failure_mode_histogram": fail_hist,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_lines += 1

                if n_lines % cfg.save_every_n == 0:
                    fout.flush()
                    write_summary(summary_path, cfg=cfg, input_path=input_path,
                                  output_path=output_path, n_rows=n_rows_done+1,
                                  n_runs=n_runs, n_lines=n_lines,
                                  n_mc_correct=n_mc_correct, n_mc_grounded=n_mc_grounded,
                                  skip_no_image=skip_no_image, skip_no_units=skip_no_units,
                                  skip_inf_err=skip_inf_err, judge_errors=n_judge_err,
                                  corpus_fail=corpus_fail, corpus_layer_fail=corpus_layer_fail,
                                  corpus_sums=corpus_sums, corpus_chain=corpus_chain)
                if cfg.delay > 0:
                    time.sleep(cfg.delay)

            n_rows_done += 1

    write_summary(summary_path, cfg=cfg, input_path=input_path, output_path=output_path,
                  n_rows=n_rows_done, n_runs=n_runs, n_lines=n_lines,
                  n_mc_correct=n_mc_correct, n_mc_grounded=n_mc_grounded,
                  skip_no_image=skip_no_image, skip_no_units=skip_no_units,
                  skip_inf_err=skip_inf_err, judge_errors=n_judge_err,
                  corpus_fail=corpus_fail, corpus_layer_fail=corpus_layer_fail,
                  corpus_sums=corpus_sums, corpus_chain=corpus_chain)

    print(f"[step7] done. rows={n_rows_done}, runs={n_runs}, lines={n_lines}, "
          f"skip_no_image={skip_no_image}, skip_no_units={skip_no_units}, "
          f"skip_inf_err={skip_inf_err}, judge_errors={n_judge_err}")
    print_summary(summary_path, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[step7 fatal] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
