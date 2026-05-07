#!/usr/bin/env python3
"""
Step 2b: Assign question categories per image-context pair.

For each subfigure, uses the sub_caption and relevant_image_context to decide
which question categories are well-supported by the context (e.g. Diagnosis,
Next-step treatment, Mechanism / pathophysiology explanation, etc.).

Input:  output of step 2 (with_relevant_context.jsonl).
Output: same rows extended with `categories` (list of matching category names).
"""

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import yaml
from tqdm import tqdm
from openai import OpenAI

from prompts import QUESTION_CATEGORY_TYPES, QUESTION_CATEGORY_SYSTEM_PROMPT


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    output_format: str = "jsonl"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    image_path_field: str = "subfig_path"
    image_path_prefix: Optional[str] = None
    use_image: bool = True
    sub_caption_field: str = "sub_caption"
    context_field: str = "relevant_image_context"
    modality_field: str = "primary_modality"
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
        model=raw.get("model", "gpt-4o-mini"),
        api_key_env=raw.get("api_key_env", "OPENAI_API_KEY"),
        temperature=float(raw.get("temperature", 0.0)),
        image_path_field=raw.get("image_path_field", "subfig_path"),
        image_path_prefix=raw.get("image_path_prefix"),
        use_image=raw.get("use_image", True),
        sub_caption_field=raw.get("sub_caption_field", "sub_caption"),
        context_field=raw.get("context_field", "relevant_image_context"),
        modality_field=raw.get("modality_field", "primary_modality"),
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
    records: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"Warning: skipped {skipped} invalid lines in {path}")
    return records


def resolve_image_path(image_path: str, base_dir: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if base_dir:
        for candidate in [
            os.path.join(base_dir, image_path),
            os.path.join(base_dir, image_path.lstrip("/")),
        ]:
            if os.path.exists(candidate):
                return candidate
    return image_path if os.path.exists(image_path) else None


def encode_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def build_client(cfg: Config) -> OpenAI:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(f"API key not found in environment variable '{cfg.api_key_env}'.")
    return OpenAI(api_key=api_key)


def parse_categories(response_text: str) -> List[str]:
    """Extract category names from the XML <question_categories> block."""
    if not response_text or not isinstance(response_text, str):
        return []
    text = response_text.strip()

    # Strip markdown fences if present
    if "```xml" in text:
        text = text.split("```xml")[1].split("```")[0].strip()
    elif "```" in text:
        for part in text.split("```"):
            if "<question_categories>" in part:
                text = part.strip()
                break

    # Extract the block
    start = text.lower().find("<question_categories>")
    if start != -1:
        end = text.lower().find("</question_categories>", start)
        if end != -1:
            text = text[start : end + len("</question_categories>")]

    valid = {s.strip() for s in QUESTION_CATEGORY_TYPES}
    pat = re.compile(r"<category\s*>(.*?)</category>", re.IGNORECASE | re.DOTALL)
    seen: set = set()
    categories: List[str] = []
    for m in pat.finditer(text):
        raw = (m.group(1) or "").strip()
        # Exact match first, then case-insensitive
        canon = raw if raw in valid else next((v for v in valid if v.lower() == raw.lower()), None)
        if canon and canon not in seen:
            seen.add(canon)
            categories.append(canon)
    return categories


def assign_categories(
    client: OpenAI,
    modality: str,
    context: str,
    image_b64: Optional[str],
    cfg: Config,
) -> List[str]:
    user_text = (
        f"Modality: {modality or '(unknown)'}\n\n"
        f"Image Context:\n{context or '(none)'}"
    )
    content: List[Any] = [{"type": "text", "text": user_text}]
    if cfg.use_image and image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        })
    messages = [
        {"role": "system", "content": QUESTION_CATEGORY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    kwargs: Dict[str, Any] = {"model": cfg.model, "messages": messages}
    no_temperature = "o1" in cfg.model.lower() or "gpt-5" in cfg.model.lower()
    if not no_temperature:
        kwargs["temperature"] = cfg.temperature
    try:
        response = client.chat.completions.create(**kwargs)
        text_out = (response.choices[0].message.content or "").strip()
        return parse_categories(text_out)
    except Exception as e:
        print(f"Error calling model: {e}")
        return []


# ==========================================
#  MAIN PIPELINE
# ==========================================

def run(cfg: Config) -> None:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    input_path = cfg.input_path

    if task_id is not None and task_count is not None:
        task_id_int = int(task_id)
        task_count_int = int(task_count)
        out_base, out_ext = os.path.splitext(cfg.output_path)
        cfg.output_path = f"{out_base}_task{task_id_int}{out_ext}"

        if cfg.split_by_row_range:
            records = load_jsonl(cfg.input_path)
            total = len(records)
            chunk = (total + task_count_int - 1) // task_count_int
            start = task_id_int * chunk
            end = min((task_id_int + 1) * chunk, total)
            records = records[start:end]
            print(
                f"Array task {task_id_int}/{task_count_int}: rows {start}-{end} "
                f"({len(records)} records, uniform split) -> {cfg.output_path}"
            )
        else:
            base, ext = os.path.splitext(cfg.input_path)
            task_input = f"{base}_task{task_id_int}{ext}"
            if os.path.exists(task_input):
                input_path = task_input
                records = load_jsonl(input_path)
                print(
                    f"Array task {task_id_int}/{task_count_int}: read {input_path} "
                    f"({len(records)} records) -> {cfg.output_path}"
                )
            else:
                records = load_jsonl(cfg.input_path)
                total = len(records)
                chunk = (total + task_count_int - 1) // task_count_int
                start = task_id_int * chunk
                end = min((task_id_int + 1) * chunk, total)
                records = records[start:end]
                print(
                    f"Array task {task_id_int}/{task_count_int}: rows {start}-{end} "
                    f"({len(records)} records) -> {cfg.output_path}"
                )
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
    base_dir = cfg.image_path_prefix or os.path.dirname(os.path.abspath(cfg.input_path))
    out_records: List[Dict[str, Any]] = []

    for r in tqdm(records, desc="Assigning question categories"):
        sub_caption = (r.get(cfg.sub_caption_field) or "").strip()
        context = (r.get(cfg.context_field) or "").strip()
        modality = (r.get(cfg.modality_field) or "").strip()
        combined_text = (sub_caption + "\n\n" + context).strip()

        if not combined_text:
            out_records.append({**r, "categories": []})
            continue

        image_b64 = None
        img_path = r.get(cfg.image_path_field) or r.get("image_path") or ""
        resolved = resolve_image_path(str(img_path), base_dir)
        if resolved:
            image_b64 = encode_image(resolved)

        categories = assign_categories(client, modality, combined_text, image_b64, cfg)
        out_records.append({**r, "categories": categories})

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} rows -> {cfg.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2b: Assign question categories.")
    parser.add_argument("--config", type=str, default="cfg/2b_assign_question_categories.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
