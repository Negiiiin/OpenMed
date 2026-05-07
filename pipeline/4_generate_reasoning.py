#!/usr/bin/env python3
# Set vLLM env vars before any import so they are picked up at process start.
import os
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

"""
Step 4: Generate reasoning traces for each MCQ produced in step 3.

Uses OctoMed (a medical VLM) via vLLM. For each generated MCQ the script:
  - Selects the image and context based on `image_scope`
      subfigure  → subfig_path  + relevant_image_context (fallback: image_context)
      full_figure → full_fig_path + image_context
  - Builds a structured prompt (paper prompt format)
  - Runs the model and extracts the predicted answer letter
  - Keeps the first sample that predicts the correct answer (if any)

Input:  step 3 JSONL (rows with `generated_mcq` list).
Output: same rows; each MCQ object extended with:
    reasoning                 full model output string
    reasoning_predicted_letter parsed answer letter (or null)
    reasoning_matches_gold    bool / null
    reasoning_context_source  which context field was used
    reasoning_image_path      which image was used
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import yaml
from PIL import Image
from tqdm import tqdm

from prompts import REASONING_GENERATION_SYSTEM_PROMPT


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    model_id: str = "OctoMed/OctoMed-7B"
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 2048
    max_model_len: int = 16384
    context_char_limit: int = 3500
    min_pixels: int = 262144
    max_pixels: int = 262144
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    batch_size: int = 16
    num_samples: int = 1
    checkpoint_every_n_rows: int = 50
    max_rows: Optional[int] = None
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    split_by_row_range: bool = False
    full_fig_path_field: str = "full_fig_path"
    subfig_path_field: str = "subfig_path"
    image_context_field: str = "image_context"
    relevant_context_field: str = "relevant_image_context"
    generated_mcq_field: str = "generated_mcq"


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        path = os.path.join(_script_dir, path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        model_id=raw.get("model_id", "OctoMed/OctoMed-7B"),
        temperature=float(raw.get("temperature", 0.2)),
        top_p=float(raw.get("top_p", 0.95)),
        max_new_tokens=int(raw.get("max_new_tokens", 2048)),
        max_model_len=int(raw.get("max_model_len", 16384)),
        context_char_limit=int(raw.get("context_char_limit", 3500)),
        min_pixels=int(raw.get("min_pixels", 262144)),
        max_pixels=int(raw.get("max_pixels", 262144)),
        tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.90)),
        batch_size=int(raw.get("batch_size", 16)),
        num_samples=int(raw.get("num_samples", 1)),
        checkpoint_every_n_rows=int(raw.get("checkpoint_every_n_rows", 50)),
        max_rows=raw.get("max_rows"),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        split_by_row_range=bool(raw.get("split_by_row_range", False)),
        full_fig_path_field=raw.get("full_fig_path_field", "full_fig_path"),
        subfig_path_field=raw.get("subfig_path_field", "subfig_path"),
        image_context_field=raw.get("image_context_field", "image_context"),
        relevant_context_field=raw.get("relevant_context_field", "relevant_image_context"),
        generated_mcq_field=raw.get("generated_mcq_field", "generated_mcq"),
    )


# ==========================================
#  HELPERS
# ==========================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
    """Parse the model's predicted answer letter from its output."""
    if not text:
        return None
    patterns = [
        r"<answer>\s*([^<]+?)\s*</answer>",
        r"\\boxed\{([^}]+)\}",
        r"<final_answer>\s*([^<]+?)\s*</final_answer>",
        r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*([A-Za-z])(?:[\).\:\-]|\b)",
        r"(?:^|\n)\s*answer\s*[:\-]\s*([A-Za-z])(?:[\).\:\-]|\b)",
        r"(?:^|\n)\s*correct\s+answer\s*[:\-]\s*([A-Za-z])(?:[\).\:\-]|\b)",
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


def format_mcq_text(question: str, choices: Dict[str, str]) -> str:
    opts = "\n".join(
        f"{k}: {(choices.get(k) or '').strip()}" for k in sorted(choices.keys())
    )
    return (question.strip() + "\n\n" + opts).strip()


def build_prompt_text(
    context_text: str,
    question: str,
    choices: Dict[str, str],
    correct_answer: str,
    context_char_limit: int,
) -> str:
    ctx = (context_text or "").strip()[:context_char_limit]
    lines = [
        REASONING_GENERATION_SYSTEM_PROMPT.strip(),
        "",
        "Grounding context (use as factual ground-truth; do not cite it as 'caption' or 'context'):",
        ctx or "(none)",
        "",
        "Question:",
        format_mcq_text(question, choices),
        "",
        f"Correct answer: {correct_answer}",
        "",
        "Write the reasoning trace now.",
    ]
    return "\n".join(lines).strip()


# ==========================================
#  vLLM MODEL
# ==========================================

def build_vllm_model(cfg: Config):
    from vllm import LLM
    from transformers import AutoProcessor

    # OctoMed-7B is based on Qwen2.5-VL; supply a clean rope_scaling config
    # to avoid conflicts between legacy and modern keys in some vLLM versions.
    hf_overrides: Optional[Dict[str, Any]] = None
    if "OctoMed" in cfg.model_id or "Qwen2.5-VL" in cfg.model_id:
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [16, 24, 24],
            }
        }

    llm = LLM(
        model=cfg.model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        hf_overrides=hf_overrides,
    )
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        trust_remote_code=True,
    )
    return llm, processor


def make_vllm_prompt(text: str, processor: Any) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def infer_batch(
    llm: Any,
    processor: Any,
    items: List[Tuple[str, str]],
    cfg: Config,
) -> List[List[str]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=cfg.num_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )
    vllm_inputs = []
    for image_path, prompt_text in items:
        formatted = make_vllm_prompt(prompt_text, processor)
        img = Image.open(image_path).convert("RGB")
        vllm_inputs.append({"prompt": formatted, "multi_modal_data": {"image": img}})

    outputs = llm.generate(vllm_inputs, sampling_params)
    return [[c.text.strip() for c in out.outputs] for out in outputs]


# ==========================================
#  MAIN PIPELINE
# ==========================================

def run(cfg: Config) -> None:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    records = load_jsonl(cfg.input_path)

    if task_id is not None and task_count is not None:
        tid, tcount = int(task_id), int(task_count)
        out_base, out_ext = os.path.splitext(cfg.output_path)
        cfg.output_path = f"{out_base}_task{tid}{out_ext}"
        if cfg.split_by_row_range:
            total = len(records)
            chunk = (total + tcount - 1) // tcount
            start, end = tid * chunk, min((tid + 1) * chunk, total)
            records = records[start:end]
            print(f"Array task {tid}/{tcount}: rows {start}-{end} ({len(records)}) -> {cfg.output_path}")

    if cfg.from_row is not None or cfg.to_row is not None:
        from_idx = cfg.from_row if cfg.from_row is not None else 0
        to_idx = min(cfg.to_row if cfg.to_row is not None else len(records), len(records))
        records = records[from_idx:to_idx] if from_idx < to_idx else []
    elif cfg.max_rows is not None and cfg.max_rows > 0:
        records = records[: cfg.max_rows]

    if not records:
        print("No records to process.")
        return

    llm, processor = build_vllm_model(cfg)

    # Flatten all (row, mcq_index) pairs into a single inference list.
    # Shape: (row_idx, gen_idx, image_path, context_source, correct_letter, prompt_text)
    FlatItem = Tuple[int, int, str, str, str, str]
    flat: List[FlatItem] = []

    for row_idx, r in enumerate(records):
        full_fig = str(r.get(cfg.full_fig_path_field) or "").strip()
        subfig   = str(r.get(cfg.subfig_path_field) or "").strip()
        ctx_full = str(r.get(cfg.image_context_field) or "").strip()
        ctx_rel  = str(r.get(cfg.relevant_context_field) or "").strip()
        gen_list = r.get(cfg.generated_mcq_field) or []

        for gen_idx, mcq in enumerate(gen_list if isinstance(gen_list, list) else []):
            if not isinstance(mcq, dict):
                continue
            question = str(mcq.get("question") or "").strip()
            choices  = mcq.get("choices") or {}
            correct  = str(mcq.get("answer") or "").strip().upper()
            scope    = str(mcq.get("image_scope") or "subfigure").strip()

            if not question or not isinstance(choices, dict) or not correct:
                continue

            if scope == "full_figure":
                image_path   = full_fig
                context_text = ctx_full
                ctx_source   = cfg.image_context_field
            else:
                image_path   = subfig
                context_text = ctx_rel or ctx_full
                ctx_source   = cfg.relevant_context_field if ctx_rel else cfg.image_context_field

            if not image_path or not os.path.exists(image_path):
                continue

            prompt_text = build_prompt_text(
                context_text=context_text,
                question=question,
                choices=choices,
                correct_answer=correct,
                context_char_limit=cfg.context_char_limit,
            )
            flat.append((row_idx, gen_idx, image_path, ctx_source, correct, prompt_text))

    if not flat:
        print("No valid MCQs found to process.")
        return

    # Track completion for periodic checkpoints.
    row_item_counts      = [0] * len(records)
    for row_idx, *_ in flat:
        row_item_counts[row_idx] += 1
    processed_counts     = [0] * len(records)
    completed_flags      = [cnt == 0 for cnt in row_item_counts]
    completed_rows       = sum(completed_flags)
    next_checkpoint      = cfg.checkpoint_every_n_rows

    for batch_start in tqdm(range(0, len(flat), cfg.batch_size), desc="Reasoning (OctoMed)"):
        chunk = flat[batch_start : batch_start + cfg.batch_size]
        items = [(c[2], c[5]) for c in chunk]
        try:
            batch_outputs = infer_batch(llm, processor, items, cfg)
        except ValueError as e:
            print(f"Skipping batch {batch_start} due to vLLM error: {e}")
            continue

        rows_done_this_batch = False
        for (row_idx, gen_idx, image_path, ctx_source, correct, _), texts in zip(chunk, batch_outputs):
            # Prefer the first sample that predicts the correct answer.
            selected_text    = ""
            selected_pred: Optional[str] = None
            selected_matches: Optional[bool] = None

            if texts:
                for t in texts:
                    pred = extract_answer_letter(t) if t else None
                    if pred is not None and pred == correct:
                        selected_text, selected_pred, selected_matches = t, pred, True
                        break
                if not selected_text:
                    # No correct sample found — keep the first output for inspection.
                    first = texts[0]
                    selected_pred    = extract_answer_letter(first) if first else None
                    selected_matches = (selected_pred == correct) if selected_pred is not None else None

            try:
                mcq = records[row_idx][cfg.generated_mcq_field][gen_idx]
                if isinstance(mcq, dict):
                    mcq["reasoning"]                  = selected_text
                    mcq["reasoning_predicted_letter"] = selected_pred
                    mcq["reasoning_matches_gold"]     = selected_matches
                    mcq["reasoning_context_source"]   = ctx_source
                    mcq["reasoning_image_path"]       = image_path
            except Exception:
                pass

            processed_counts[row_idx] += 1
            if (not completed_flags[row_idx]
                    and processed_counts[row_idx] >= row_item_counts[row_idx]):
                completed_flags[row_idx] = True
                completed_rows += 1
                rows_done_this_batch = True

        if rows_done_this_batch and completed_rows >= next_checkpoint:
            write_jsonl(records, cfg.output_path)
            print(f"Checkpoint: {completed_rows} rows complete -> {cfg.output_path}")
            while completed_rows >= next_checkpoint:
                next_checkpoint += cfg.checkpoint_every_n_rows

    write_jsonl(records, cfg.output_path)
    print(f"Wrote {len(records)} rows -> {cfg.output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: Generate reasoning traces (OctoMed).")
    ap.add_argument("--config", type=str, default="cfg/4_generate_reasoning.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
