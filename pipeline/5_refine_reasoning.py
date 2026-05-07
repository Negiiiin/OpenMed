#!/usr/bin/env python3
"""
Step 5: Refine reasoning traces produced in step 4 using a vision-language model
(e.g. GPT-4o or a reasoning model such as o1/o3).

Input:  step 4 JSONL. Each row contains a `generated_mcq` list; every MCQ object
        has a `reasoning` field written by OctoMed.

Output: same rows with each MCQ object extended / overwritten with:
    reasoning_original                 draft from step 4
    reasoning_original_predicted_letter parsed answer letter from draft
    reasoning_original_matches_gold    bool / null
    reasoning                          refined reasoning trace
    reasoning_predicted_letter         parsed answer letter from refined trace
    reasoning_matches_gold             bool / null
    reasoning_refine_model             model used for refinement
    reasoning_refine_image_path        image used during refinement
    reasoning_refine_image_field       which path field was used

Rows / MCQs that are missing reasoning, question, choices, or a resolvable image
path are skipped (no output row written).

SLURM array support:
    Set ``split_by_slurm_array: true`` in the YAML (or env var
    STEP5_SPLIT_BY_SLURM_ARRAY=1) to automatically shard rows across
    SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT.
    Merge shards afterwards: ``cat *_task*.jsonl > refined.jsonl``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from prompts import REASONING_REFINEMENT_SYSTEM_PROMPT


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_tokens: int = 4096
    delay: float = 0.0
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    max_rows: Optional[int] = None
    full_fig_path_field: str = "full_fig_path"
    subfig_path_field: str = "subfig_path"
    image_context_field: str = "image_context"
    relevant_context_field: str = "relevant_image_context"
    generated_mcq_field: str = "generated_mcq"
    save_raw_response: bool = False
    verbose_errors: bool = True
    split_by_slurm_array: bool = False


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        path = os.path.join(_SCRIPT_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _str(key: str, default: str) -> str:
        v = raw.get(key)
        return str(v).strip() if v is not None and str(v).strip() else default

    def _opt_str(key: str) -> Optional[str]:
        v = raw.get(key)
        return str(v).strip() if v is not None and str(v).strip() else None

    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        model=_str("model", "gpt-4o"),
        api_key_env=_str("api_key_env", "OPENAI_API_KEY"),
        base_url=_opt_str("base_url"),
        reasoning_effort=_opt_str("reasoning_effort"),
        temperature=float(raw["temperature"]) if raw.get("temperature") is not None else None,
        max_tokens=int(raw.get("max_tokens", 4096)),
        delay=float(raw.get("delay", 0.0)),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        max_rows=raw.get("max_rows"),
        full_fig_path_field=_str("full_fig_path_field", "full_fig_path"),
        subfig_path_field=_str("subfig_path_field", "subfig_path"),
        image_context_field=_str("image_context_field", "image_context"),
        relevant_context_field=_str("relevant_context_field", "relevant_image_context"),
        generated_mcq_field=_str("generated_mcq_field", "generated_mcq"),
        save_raw_response=bool(raw.get("save_raw_response", False)),
        verbose_errors=bool(raw.get("verbose_errors", True)),
        split_by_slurm_array=bool(raw.get("split_by_slurm_array", False)),
    )


# ==========================================
#  HELPERS
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


def _normalize_letter(text: str) -> Optional[str]:
    if not text:
        return None
    c = text.strip().upper()
    m = re.match(r"^([A-Z])(?:\b|[\).\:\-])", c)
    if m:
        return m.group(1)
    if len(c) == 1 and c.isalpha():
        return c
    return None


def extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"<answer>\s*([^<]+?)\s*</answer>",
        r"\\boxed\{([^}]+)\}",
        r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*([A-Za-z]+)",
        r"(?:^|\n)\s*answer\s*[:\-]\s*([A-Za-z]+)",
    ]
    for pat in patterns:
        for m in reversed(list(re.finditer(pat, text, re.IGNORECASE | re.DOTALL))):
            norm = _normalize_letter(m.group(1).strip())
            if norm is not None:
                return norm
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return _normalize_letter(lines[-1])
    return None


def _compare(pred: Optional[str], gold: Any) -> Optional[bool]:
    if pred is None or gold is None:
        return None
    g = _normalize_letter(str(gold))
    return pred == g if g is not None else None


def _image_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _choices_text(choices: Dict[str, str]) -> str:
    return "\n".join(
        f"{k}. {(choices.get(k) or '').strip()}" for k in sorted(choices.keys())
    )


def _mcq_block(question: str, choices: Dict[str, str]) -> str:
    return question.strip() + "\n\nOptions:\n" + _choices_text(choices)


def _resolve_image(
    row: Dict[str, Any],
    scope: str,
    cfg: Config,
    input_path: str,
) -> Tuple[str, str]:
    """Return (resolved_absolute_path, field_name) for the most appropriate image."""
    subfig  = str(row.get(cfg.subfig_path_field) or "").strip()
    fullfig = str(row.get(cfg.full_fig_path_field) or "").strip()

    if scope in {"full_figure", "full"}:
        ordered = [(cfg.full_fig_path_field, fullfig), (cfg.subfig_path_field, subfig)]
    else:
        ordered = [(cfg.subfig_path_field, subfig), (cfg.full_fig_path_field, fullfig)]

    here = os.path.dirname(os.path.abspath(input_path))
    for field, raw in ordered:
        if not raw:
            continue
        for candidate in [
            raw if os.path.isabs(raw) else None,
            os.path.join(here, raw),
            os.path.abspath(raw),
        ]:
            if candidate and os.path.exists(candidate):
                return candidate, field
    return "", ""


def _pick_context(row: Dict[str, Any], scope: str, cfg: Config) -> str:
    ctx_rel  = str(row.get(cfg.relevant_context_field) or "").strip()
    ctx_full = str(row.get(cfg.image_context_field) or "").strip()
    if scope == "full_figure":
        return ctx_full
    return ctx_rel or ctx_full


def build_client(cfg: Config) -> OpenAI:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"API key not found in environment variable '{cfg.api_key_env}'. "
            "Set it before running: export OPENAI_API_KEY=your-key"
        )
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)


def _parse_response_text(response: Any) -> str:
    choice = response.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if txt:
                parts.append(str(txt))
        return "\n".join(parts).strip()
    refusal = getattr(msg, "refusal", None)
    if refusal:
        raise ValueError(f"Model refused: {refusal}")
    raise ValueError("Empty model response.")


def call_refinement_api(
    client: OpenAI,
    *,
    image_path: str,
    context_text: str,
    question: str,
    choices: Dict[str, str],
    original_reasoning: str,
    target_answer: str,
    cfg: Config,
) -> Tuple[str, str]:
    context_text = (context_text or "").strip()
    user_parts = [
        "Image context (ground-truth source; do not cite directly):",
        context_text[:3500] if context_text else "(none)",
        "",
        "MCQ:",
        _mcq_block(question, choices),
        "",
        f"Target answer: {target_answer}",
        "",
        "Draft reasoning to refine:",
        original_reasoning.strip() or "(empty)",
    ]
    user_text = "\n".join(user_parts)

    image_uri = _image_to_data_uri(image_path)
    is_reasoning_model = re.search(r"\bo[1-9]\b|gpt-5", cfg.model, re.IGNORECASE) is not None

    api_params: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": REASONING_REFINEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
        ],
    }
    if is_reasoning_model:
        api_params["max_completion_tokens"] = cfg.max_tokens
        if cfg.reasoning_effort:
            api_params["reasoning_effort"] = cfg.reasoning_effort
    else:
        api_params["max_tokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            api_params["temperature"] = cfg.temperature

    try:
        response = client.chat.completions.create(**api_params)
    except Exception:
        if is_reasoning_model and "reasoning_effort" in api_params:
            api_params.pop("reasoning_effort")
            response = client.chat.completions.create(**api_params)
        else:
            raise

    raw = _parse_response_text(response)
    return raw.strip(), raw


# ==========================================
#  MAIN PIPELINE
# ==========================================

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="Step 5: Refine reasoning traces.")
    parser.add_argument("--config", type=str, default="cfg/5_refine_reasoning.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path  = os.path.abspath(cfg.input_path)
    output_path = os.path.abspath(cfg.output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_rows = read_jsonl(input_path)
    n_all = len(all_rows)

    from_idx = cfg.from_row if cfg.from_row is not None else 0
    to_idx   = cfg.to_row   if cfg.to_row   is not None else n_all
    rows = all_rows[from_idx:to_idx]
    if cfg.max_rows is not None:
        rows = rows[: cfg.max_rows]

    env_split = os.environ.get("STEP5_SPLIT_BY_SLURM_ARRAY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    do_split = cfg.split_by_slurm_array or env_split
    task_id_raw    = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count_raw = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if do_split and task_id_raw is not None and task_count_raw is not None:
        tid, tcount = int(task_id_raw), int(task_count_raw)
        n_work = len(rows)
        chunk  = (n_work + tcount - 1) // tcount
        start, end = tid * chunk, min((tid + 1) * chunk, n_work)
        rows = rows[start:end]
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_task{tid}{ext}"
        print(
            f"SLURM array task {tid}/{tcount}: rows [{start}:{end}) "
            f"({len(rows)} rows) -> {output_path}",
            flush=True,
        )
    elif do_split:
        print(
            "[step5] WARNING: split_by_slurm_array enabled but SLURM_ARRAY_TASK_ID "
            "/ SLURM_ARRAY_TASK_COUNT not set; processing full working set.",
            flush=True,
        )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(rows)} (from {n_all} total)", flush=True)

    client = build_client(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_rows_in = n_rows_out = 0
    n_mcqs_in = n_mcqs_ok = n_mcqs_skip = n_mcqs_err = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Step 5: refine reasoning"):
            n_rows_in += 1
            gen_list = row.get(cfg.generated_mcq_field)
            if not isinstance(gen_list, list) or not gen_list:
                continue

            out_row = dict(row)
            any_written = False

            for mcq_idx, mcq in enumerate(gen_list):
                if not isinstance(mcq, dict):
                    continue
                n_mcqs_in += 1

                question          = str(mcq.get("question") or "").strip()
                choices           = mcq.get("choices") or {}
                original_reason   = str(mcq.get("reasoning") or "").strip()
                target_answer     = str(mcq.get("answer") or "").strip().upper()
                scope             = str(mcq.get("image_scope") or "subfigure").strip()

                if not question or not isinstance(choices, dict) or not original_reason or not target_answer:
                    n_mcqs_skip += 1
                    continue

                image_path, image_field = _resolve_image(row, scope, cfg, input_path)
                if not image_path:
                    n_mcqs_skip += 1
                    continue

                context_text = _pick_context(row, scope, cfg)

                try:
                    refined, raw = call_refinement_api(
                        client,
                        image_path=image_path,
                        context_text=context_text,
                        question=question,
                        choices=choices,
                        original_reasoning=original_reason,
                        target_answer=target_answer,
                        cfg=cfg,
                    )
                except Exception as exc:
                    n_mcqs_err += 1
                    err = (
                        f"[step5 error] row={n_rows_in - 1} mcq={mcq_idx} "
                        f"{type(exc).__name__}: {exc}"
                    )
                    tqdm.write(err, file=sys.stderr)
                    if cfg.verbose_errors:
                        tqdm.write(traceback.format_exc(), file=sys.stderr)
                    continue

                orig_pred    = extract_answer_letter(original_reason)
                refined_pred = extract_answer_letter(refined)
                gold         = target_answer or None

                out_mcq = dict(mcq)
                out_mcq["reasoning_original"]                  = original_reason
                out_mcq["reasoning_original_predicted_letter"] = orig_pred
                out_mcq["reasoning_original_matches_gold"]     = _compare(orig_pred, gold)
                out_mcq["reasoning"]                           = refined
                out_mcq["reasoning_predicted_letter"]          = refined_pred
                out_mcq["reasoning_matches_gold"]              = _compare(refined_pred, gold)
                out_mcq["reasoning_refine_model"]              = cfg.model
                out_mcq["reasoning_refine_image_path"]         = image_path
                out_mcq["reasoning_refine_image_field"]        = image_field
                if cfg.save_raw_response:
                    out_mcq["reasoning_refine_raw_response"] = raw

                gen_list[mcq_idx] = out_mcq
                any_written = True
                n_mcqs_ok += 1

                if cfg.delay > 0:
                    time.sleep(cfg.delay)

            if any_written:
                out_row[cfg.generated_mcq_field] = gen_list
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_rows_out += 1

    print(
        f"Done. rows_in={n_rows_in}, rows_written={n_rows_out}, "
        f"mcqs_in={n_mcqs_in}, mcqs_refined={n_mcqs_ok}, "
        f"mcqs_skipped={n_mcqs_skip}, mcqs_errors={n_mcqs_err}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[step5 fatal] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
