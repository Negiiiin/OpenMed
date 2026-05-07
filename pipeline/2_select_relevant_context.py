#!/usr/bin/env python3
"""
Step 2: Select relevant image context and assign modality.
Uses an LLM to extract from image_context all passages that refer to the given
subfigure. Sends the subfigure image (when available), sub_caption, and image_context.
Outputs relevant_image_context, primary_modality, secondary_modality.
"""

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Allow importing prompts from data_2 when run from project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import yaml
from tqdm import tqdm
from openai import OpenAI

from prompts import SELECT_RELEVANT_CONTEXT_SYSTEM_PROMPT


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    output_format: str = "jsonl"
    sub_caption_field: str = "sub_caption"
    image_context_field: str = "image_context"
    image_path_field: str = "subfig_path"
    full_fig_path_field: Optional[str] = "full_fig_path"
    image_path_prefix: Optional[str] = None
    model: str = "gpt-5-nano"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_rows: Optional[int] = None
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    split_by_row_range: bool = False


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        output_format=raw.get("output_format", "jsonl"),
        sub_caption_field=raw.get("sub_caption_field", "sub_caption"),
        image_context_field=raw.get("image_context_field", "image_context"),
        image_path_field=raw.get("image_path_field", "subfig_path"),
        full_fig_path_field=raw.get("full_fig_path_field", "full_fig_path"),
        image_path_prefix=raw.get("image_path_prefix"),
        model=raw.get("model", "gpt-5-nano"),
        api_key_env=raw.get("api_key_env", "OPENAI_API_KEY"),
        temperature=float(raw.get("temperature", 0.0)),
        max_rows=raw.get("max_rows"),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        split_by_row_range=bool(raw.get("split_by_row_range", False)),
    )


# ==========================================
#  HELPERS
# ==========================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_image_context_text(image_context: Any) -> str:
    if image_context is None:
        return ""
    if isinstance(image_context, str):
        return image_context.strip()
    if isinstance(image_context, dict):
        parts = []
        for v in image_context.values():
            if isinstance(v, list):
                parts.extend([str(t).strip() for t in v if t])
            else:
                parts.append(str(v).strip())
        return " ".join(p for p in parts if p)
    return str(image_context).strip()


def clean_references(text: str) -> str:
    """Replace XML figure xrefs like <xref ...>Figure 5</xref>B with Figure 5B."""
    if not text:
        return text

    def _repl(match: Any) -> str:
        inner = match.group(1)
        trailing = match.group(2) or ""
        return f"{inner}{trailing}"

    return re.sub(r'<xref[^>]*>([^<]+)</xref>([A-Za-z])?', _repl, text)


def parse_model_response(text: str) -> Dict[str, str]:
    """Extract JSON from model response; strip code fences if present."""
    if not text:
        return {}
    text = text.strip()
    # Remove markdown code block
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
    if text.endswith("```"):
        text = text[: text.rfind("```")].rstrip()
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            valid = out.get("valid")
            if isinstance(valid, bool):
                pass
            elif isinstance(valid, str):
                valid = valid.strip().lower() in ("true", "yes", "1")
            else:
                valid = False
            return {
                "relevant_image_context": (out.get("relevant_image_context") or "").strip(),
                "primary_modality": (out.get("primary_modality") or "").strip(),
                "secondary_modality": (out.get("secondary_modality") or "").strip(),
                "valid": valid,
            }
    except json.JSONDecodeError:
        pass
    return {}


def build_client(cfg: Config) -> OpenAI:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(f"API key not found in {cfg.api_key_env}.")
    return OpenAI(api_key=api_key)


def resolve_image_path(image_path: str, base_dir: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if base_dir:
        candidate = os.path.join(base_dir, image_path)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(image_path):
        return os.path.abspath(image_path)
    return None


def encode_image(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def call_llm(
    client: OpenAI,
    sub_caption: str,
    image_context: str,
    cfg: Config,
    image_b64: Optional[str] = None,
    full_fig_b64: Optional[str] = None,
) -> Dict[str, str]:
    user_text = f"Sub-caption: {sub_caption or '(none)'}\n\nImage context: {image_context or '(none)'}"
    if full_fig_b64 or image_b64:
        image_hint = []
        content_parts: List[Any] = [{"type": "text", "text": user_text}]
        if full_fig_b64:
            image_hint.append("First image below: full compound figure (all panels).")
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_fig_b64}", "detail": "low"}})
        if image_b64:
            image_hint.append("Next image below: the subfigure for which to select relevant context.")
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}})
        content_parts[0]["text"] = user_text + " " + " ".join(image_hint)
        user_content = content_parts
    else:
        user_content = user_text
    no_temperature = "o1" in cfg.model.lower() or "gpt-5" in cfg.model.lower()
    kwargs = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SELECT_RELEVANT_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    if not no_temperature:
        kwargs["temperature"] = cfg.temperature
    try:
        response = client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        return parse_model_response(text)
    except Exception:
        return {}


def run(cfg: Config) -> None:
    # SLURM array: optionally read task-specific input (e.g. step1 filtered_task0.jsonl)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    input_path = cfg.input_path
    if task_id is not None and task_count is not None:
        task_id = int(task_id)
        task_count = int(task_count)
        out_base, out_ext = os.path.splitext(cfg.output_path)
        cfg.output_path = f"{out_base}_task{task_id}{out_ext}"
        if cfg.split_by_row_range:
            records = load_jsonl(cfg.input_path)
            total = len(records)
            chunk = (total + task_count - 1) // task_count
            start = task_id * chunk
            end = min((task_id + 1) * chunk, total)
            records = records[start:end]
            print(f"Array task {task_id}/{task_count}: rows {start}-{end} ({len(records)} records, uniform split) -> {cfg.output_path}")
        else:
            base, ext = os.path.splitext(cfg.input_path)
            task_input = f"{base}_task{task_id}{ext}"
            if os.path.exists(task_input):
                input_path = task_input
                records = load_jsonl(input_path)
                print(f"Array task {task_id}/{task_count}: read {input_path} ({len(records)} records) -> {cfg.output_path}")
            else:
                records = load_jsonl(cfg.input_path)
                total = len(records)
                chunk = (total + task_count - 1) // task_count
                start = task_id * chunk
                end = min((task_id + 1) * chunk, total)
                records = records[start:end]
                print(f"Array task {task_id}/{task_count}: rows {start}-{end} ({len(records)} records) -> {cfg.output_path}")
    else:
        records = load_jsonl(input_path)
        if cfg.from_row is not None or cfg.to_row is not None:
            from_idx = cfg.from_row if cfg.from_row is not None else 0
            to_idx = cfg.to_row if cfg.to_row is not None else len(records)
            to_idx = min(to_idx, len(records))
            records = records[from_idx:to_idx] if from_idx < to_idx else []
        elif cfg.max_rows is not None and cfg.max_rows > 0:
            records = records[: cfg.max_rows]

    if not records:
        print("No records to process.")
        return

    client = build_client(cfg)
    sub_col = cfg.sub_caption_field
    ctx_col = cfg.image_context_field
    img_col = cfg.image_path_field
    full_fig_col = cfg.full_fig_path_field or "full_fig_path"
    base_dir = cfg.image_path_prefix or os.path.dirname(os.path.abspath(cfg.input_path))
    out_records: List[Dict[str, Any]] = []
    skipped_empty_context = 0
    skipped_insufficient = 0

    for r in tqdm(records, desc="Select relevant context"):
        sub_caption = (r.get(sub_col) or "").strip()
        image_context = extract_image_context_text(r.get(ctx_col))
        if not image_context or not image_context.strip():
            skipped_empty_context += 1
            continue
        image_b64 = None
        full_fig_b64 = None
        img_path = r.get(img_col) or r.get("image_path") or ""
        resolved = resolve_image_path(str(img_path), base_dir)
        if resolved:
            image_b64 = encode_image(resolved)
        full_fig_path = r.get(full_fig_col) or ""
        full_fig_resolved = resolve_image_path(str(full_fig_path), base_dir)
        if full_fig_resolved:
            full_fig_b64 = encode_image(full_fig_resolved)
        result = call_llm(client, sub_caption, image_context, cfg, image_b64=image_b64, full_fig_b64=full_fig_b64)
        relevant = result.get("relevant_image_context", "")
        primary = result.get("primary_modality", "")
        secondary = result.get("secondary_modality", "")
        if not relevant and image_context:
            relevant = image_context
        # Clean up XML-like figure xrefs in the final text, e.g. <xref ...>Figure 5</xref>B -> Figure 5B
        relevant = clean_references(relevant)
        valid = result.get("valid", False)
        if not valid or not relevant.strip():
            skipped_insufficient += 1
            continue
        out_records.append({
            **r,
            "relevant_image_context": relevant,
            "primary_modality": primary,
            "secondary_modality": secondary,
        })

    if skipped_empty_context or skipped_insufficient:
        print(f"Skipped {skipped_empty_context} rows (empty image context), {skipped_insufficient} rows (insufficient information).")
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_records)} rows -> {cfg.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/2_select_relevant_context.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
