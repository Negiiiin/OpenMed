#!/usr/bin/env python3
"""
Step 3: Generate MCQs for each question category assigned in step 2b.

Input:  JSONL from step 2b, where each row has `categories` (list of category names),
        `relevant_image_context`, `primary_modality`, `secondary_modality`, `sub_caption`,
        and image path fields.

For each row and each category the script:
  - Decides question style (short / long) and answer format (standard / binary_*).
  - Decides image scope (subfigure or full_figure) stochastically.
  - Calls the LLM with the system prompt from the paper and a structured user message.
  - Filters out __INVALID__ responses.
  - Validates and (if needed) auto-repairs the returned JSON.

Output: same rows with an added `generated_mcq` field:
  [
    {
      "category": "...",
      "style": "short" | "long",
      "question": "...",
      "choices": {"A": "...", "B": "...", ...},
      "answer": "A",
      "image_scope": "subfigure" | "full_figure"
    },
    ...
  ]
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import yaml
from tqdm import tqdm
from openai import OpenAI

from prompts import (
    MCQ_GENERATION_SYSTEM_PROMPT,
    MCQ_SHORT_ONLY_CATEGORIES,
    MCQ_LONG_ONLY_CATEGORIES,
    MCQ_BINARY_ELIGIBLE_CATEGORIES,
    QUESTION_CATEGORY_TYPES,
)


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
    max_completion_tokens: int = 512
    reasoning_effort: Optional[str] = None
    # Fraction of questions that use the full compound figure instead of the subfigure.
    full_figure_ratio: float = 0.30
    # Fraction of eligible categories that use a binary (yes/no or true/false) format.
    binary_ratio: float = 0.20
    seed: int = 42
    max_rows: Optional[int] = None
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    split_by_row_range: bool = False
    use_image: bool = True
    image_path_field_subfigure: str = "subfig_path"
    image_path_field_full: str = "full_fig_path"
    image_path_prefix: Optional[str] = None
    vision_detail: str = "high"


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    _re = raw.get("reasoning_effort")
    if _re is not None:
        _re = str(_re).strip() or None
    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        output_format=raw.get("output_format", "jsonl"),
        model=raw.get("model", "gpt-4o-mini"),
        api_key_env=raw.get("api_key_env", "OPENAI_API_KEY"),
        max_completion_tokens=int(raw.get("max_completion_tokens", 512)),
        reasoning_effort=_re,
        full_figure_ratio=float(raw.get("full_figure_ratio", 0.30)),
        binary_ratio=float(raw.get("binary_ratio", 0.20)),
        seed=int(raw.get("seed", 42)),
        max_rows=raw.get("max_rows"),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        split_by_row_range=bool(raw.get("split_by_row_range", False)),
        use_image=bool(raw.get("use_image", True)),
        image_path_field_subfigure=str(raw.get("image_path_field_subfigure", "subfig_path")),
        image_path_field_full=str(raw.get("image_path_field_full", "full_fig_path")),
        image_path_prefix=raw.get("image_path_prefix"),
        vision_detail=str(raw.get("vision_detail", "high")),
    )


# ==========================================
#  HELPERS
# ==========================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def build_client(cfg: Config) -> OpenAI:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise EnvironmentError(f"API key not found in environment variable '{cfg.api_key_env}'.")
    return OpenAI(api_key=api_key)


def resolve_image_path(image_path: str, base_dir: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if base_dir:
        for candidate in (
            os.path.join(base_dir, image_path),
            os.path.join(base_dir, image_path.lstrip("/")),
        ):
            if os.path.exists(candidate):
                return candidate
    return image_path if os.path.exists(image_path) else None


def make_image_data_url(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def get_image_data_url(
    row: Dict[str, Any],
    cfg: Config,
    image_scope: str,
) -> Optional[str]:
    if not cfg.use_image:
        return None
    base_dir = cfg.image_path_prefix or os.path.dirname(os.path.abspath(cfg.input_path))
    field_order = (
        (cfg.image_path_field_full, cfg.image_path_field_subfigure, "full_fig_path", "subfig_path")
        if image_scope == "full_figure"
        else (cfg.image_path_field_subfigure, "subfig_path", cfg.image_path_field_full)
    )
    for key in field_order:
        raw = row.get(key)
        if not raw:
            continue
        resolved = resolve_image_path(str(raw).strip(), base_dir)
        if resolved:
            url = make_image_data_url(resolved)
            if url:
                return url
    return None


def build_user_content(
    user_text: str,
    image_data_url: Optional[str],
    vision_detail: str,
) -> Union[str, List[Dict[str, Any]]]:
    if not image_data_url:
        return user_text
    detail = (vision_detail or "auto").strip().lower()
    if detail not in ("auto", "low", "high"):
        detail = "auto"
    return [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": image_data_url, "detail": detail}},
    ]


# ==========================================
#  STYLE / FORMAT DECISION
# ==========================================

def stable_hash_0_1(seed: int, key: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from seed and key."""
    h = hashlib.md5((str(seed) + "|" + key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def resolve_styles(category: str) -> List[str]:
    """Return the list of styles to generate for this category."""
    cat = category.strip()
    if cat in MCQ_SHORT_ONLY_CATEGORIES:
        return ["short"]
    if cat in MCQ_LONG_ONLY_CATEGORIES:
        return ["long"]
    return ["short", "long"]


def decide_image_scope(cfg: Config, row: Dict[str, Any], category: str, style: str) -> str:
    cat = category.strip()
    if cat in {"Anatomy / localization", "Spatial location on image (quadrant / region)",
               "Findings / description only", "Annotation / marker interpretation"} and style == "short":
        return "subfigure"
    key = f"{row.get('subfig_path', '')}|{cat}|{style}"
    return "full_figure" if stable_hash_0_1(cfg.seed, key) < cfg.full_figure_ratio else "subfigure"


def decide_mcq_format(cfg: Config, row: Dict[str, Any], category: str, style: str) -> str:
    cat = category.strip()
    if cat == "Normal vs abnormal":
        return "binary_normal_abnormal"
    if cat not in MCQ_BINARY_ELIGIBLE_CATEGORIES:
        return "standard"
    key = f"{row.get('subfig_path', '')}|{row.get('sub_caption', '')}|{cat}|{style}"
    if stable_hash_0_1(cfg.seed, key + "|pick") >= cfg.binary_ratio:
        return "standard"
    return "binary_yesno" if stable_hash_0_1(cfg.seed, key + "|fmt") < 0.5 else "binary_truefalse"


# ==========================================
#  MCQ GENERATION
# ==========================================

def build_user_prompt(
    sub_caption: str,
    context_text: str,
    primary_modality: str,
    secondary_modality: str,
    category: str,
    style: str,
    mcq_format: str,
    image_scope: str,
    has_image: bool = False,
) -> str:
    image_note = (
        "The image is attached to this message. Use it to verify what is visually present "
        "and to check stem privacy — do not describe what you see in the stem."
        if has_image
        else "No image pixels are attached; rely on the provided context."
    )
    format_instructions = {
        "standard": "Use standard multiple choice with 4-5 options (A, B, C, D, or A-E).",
        "binary_yesno": "Use exactly A = Yes and B = No.",
        "binary_truefalse": "Use exactly A = True and B = False.",
        "binary_normal_abnormal": "Use exactly A = Normal and B = Abnormal.",
    }.get(mcq_format, "Use standard multiple choice.")

    lines = [
        image_note,
        "",
        f"Modality: {primary_modality or '(unknown)'}",
        f"Secondary modality: {secondary_modality or '(none)'}",
        "",
        f"Sub-caption: {sub_caption or '(none)'}",
        "",
        f"Context: {context_text or '(none)'}",
        "",
        f"Target MCQ category: {category}",
        f"Image scope: {image_scope}",
        f"Question style: {style}",
        f"Answer format: {mcq_format}. {format_instructions}",
        "",
        "Generate exactly one MCQ JSON object as specified in the system prompt.",
        "If this example does not qualify, return the __INVALID__ JSON.",
    ]
    return "\n".join(lines)


def parse_mcq_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                t = part
                break
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def repair_json_via_model(
    client: OpenAI,
    cfg: Config,
    raw_text: str,
    category: str,
    mcq_format: str,
) -> Optional[Dict[str, Any]]:
    system = (
        "You are a JSON repair tool. Return ONLY a valid JSON object with keys: "
        "question, choices, answer, image_scope. "
        f"mcq_format is '{mcq_format}'. "
        "For binary_yesno: choices={{'A':'Yes','B':'No'}}. "
        "For binary_truefalse: choices={{'A':'True','B':'False'}}. "
        "For binary_normal_abnormal: choices={{'A':'Normal','B':'Abnormal'}}. "
        "For standard: produce 4-5 options. No markdown, no explanation."
    )
    user = (
        f"The following text was supposed to be a JSON MCQ for category: {category}.\n"
        "Extract or rewrite it as the required JSON.\n\nTEXT:\n" + (raw_text or "")
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=min(256, cfg.max_completion_tokens),
        )
        return parse_mcq_json((resp.choices[0].message.content or "").strip())
    except Exception:
        return None


_SOURCE_REF_PATTERNS = [
    "sub-caption", "subcaption", "caption", "image context", "image_context",
    "relevant image context", "relevant_image_context", "provided text",
    "provided context", "given text", "given context", "as described", "according to",
]


def has_source_reference(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _SOURCE_REF_PATTERNS)


def mcq_violates_source_privacy(mcq: Dict[str, Any]) -> bool:
    if has_source_reference(str(mcq.get("question") or "")):
        return True
    for v in (mcq.get("choices") or {}).values():
        if has_source_reference(str(v or "")):
            return True
    return False


def rewrite_source_references(
    client: OpenAI,
    cfg: Config,
    mcq: Dict[str, Any],
    category: str,
    style: str,
    image_scope: str,
    mcq_format: str,
    image_data_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    system = (
        "You are rewriting an MCQ to remove references to hidden source text. "
        "Return ONLY valid JSON: {question, choices, answer, image_scope}. "
        f"mcq_format is '{mcq_format}'. Preserve the answer choices format exactly. "
        "Do not mention sub-caption, caption, image_context, provided text, or context."
    )
    user_text = json.dumps(
        {"category": category, "style": style, "image_scope": image_scope, "mcq": mcq},
        ensure_ascii=False,
    )
    user_content = build_user_content(user_text, image_data_url, cfg.vision_detail)
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=min(512, cfg.max_completion_tokens),
        )
        out = parse_mcq_json((resp.choices[0].message.content or "").strip())
        if not isinstance(out, dict):
            return None
        if str(out.get("image_scope") or "").strip() not in {"subfigure", "full_figure"}:
            out["image_scope"] = image_scope
        return out
    except Exception:
        return None


def canonicalize_binary_choices(mcq: Dict[str, Any], mcq_format: str) -> Dict[str, Any]:
    """Enforce exact choice labels for binary formats; remap answer letter if needed."""
    canonical_map = {
        "binary_yesno":           {"A": "Yes", "B": "No"},
        "binary_truefalse":       {"A": "True", "B": "False"},
        "binary_normal_abnormal": {"A": "Normal", "B": "Abnormal"},
    }
    if mcq_format not in canonical_map:
        return mcq
    target = canonical_map[mcq_format]
    ch = mcq.get("choices") or {}
    old_ans = str(mcq.get("answer") or "").strip().upper()
    # Find which value in the model's choices matches the correct canonical answer
    if old_ans in ch:
        correct_val = str(ch[old_ans]).strip().lower()
        new_ans = next(
            (k for k, v in target.items() if v.lower() == correct_val),
            old_ans,
        )
    else:
        new_ans = old_ans
    return {**mcq, "choices": target, "answer": new_ans}


def generate_mcq(
    client: OpenAI,
    cfg: Config,
    row: Dict[str, Any],
    sub_caption: str,
    context_text: str,
    primary_modality: str,
    secondary_modality: str,
    category: str,
    style: str,
    mcq_format: str,
    image_scope: str,
    image_data_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    user_text = build_user_prompt(
        sub_caption=sub_caption,
        context_text=context_text,
        primary_modality=primary_modality,
        secondary_modality=secondary_modality,
        category=category,
        style=style,
        mcq_format=mcq_format,
        image_scope=image_scope,
        has_image=bool(image_data_url),
    )
    user_content = build_user_content(user_text, image_data_url, cfg.vision_detail)
    api_kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": MCQ_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_completion_tokens": cfg.max_completion_tokens,
    }
    effort = (cfg.reasoning_effort or "").strip()
    if effort:
        api_kwargs["reasoning_effort"] = effort

    try:
        resp = client.chat.completions.create(**api_kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"API error for category '{category}': {e}")
        return None

    mcq = parse_mcq_json(raw)
    if not isinstance(mcq, dict):
        mcq = repair_json_via_model(client, cfg, raw, category, mcq_format)
        if not isinstance(mcq, dict):
            print(f"Warning: unparseable response for category '{category}'")
            return None

    question = str(mcq.get("question") or "")
    choices = mcq.get("choices")
    answer = str(mcq.get("answer") or "")
    img_scope_out = str(mcq.get("image_scope") or "").strip()

    if not isinstance(choices, dict) or not question or not answer:
        print(f"Warning: incomplete MCQ JSON for category '{category}'")
        return None

    if question.strip().upper() == "__INVALID__":
        return None

    if img_scope_out not in {"subfigure", "full_figure"}:
        img_scope_out = image_scope

    # Post-process: remove source references from the stem
    if mcq_violates_source_privacy(mcq):
        rewritten = rewrite_source_references(
            client, cfg, mcq, category, style, image_scope, mcq_format, image_data_url
        )
        if isinstance(rewritten, dict) and not mcq_violates_source_privacy(rewritten):
            mcq = rewritten
            question = str(mcq.get("question") or question)
            choices = mcq.get("choices") or choices
            answer = str(mcq.get("answer") or answer)
            img_scope_out = str(mcq.get("image_scope") or img_scope_out)

    # Enforce canonical choices for binary formats
    mcq = canonicalize_binary_choices(
        {"question": question, "choices": choices, "answer": answer, "image_scope": img_scope_out},
        mcq_format,
    )

    return {
        "category": category,
        "style": style,
        "question": str(mcq.get("question") or ""),
        "choices": mcq.get("choices"),
        "answer": str(mcq.get("answer") or "").strip().upper(),
        "image_scope": str(mcq.get("image_scope") or img_scope_out).strip(),
    }


# ==========================================
#  MAIN PIPELINE
# ==========================================

def run(cfg: Config) -> None:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if task_id is not None and task_count is not None:
        task_id_int = int(task_id)
        task_count_int = int(task_count)
        out_base, out_ext = os.path.splitext(cfg.output_path)
        cfg.output_path = f"{out_base}_task{task_id_int}{out_ext}"
        records_all = load_jsonl(cfg.input_path)
        if cfg.split_by_row_range:
            total = len(records_all)
            chunk = (total + task_count_int - 1) // task_count_int
            start = task_id_int * chunk
            end = min((task_id_int + 1) * chunk, total)
            records = records_all[start:end]
            print(
                f"Array task {task_id_int}/{task_count_int}: rows {start}-{end} "
                f"({len(records)} records) -> {cfg.output_path}"
            )
        else:
            records = records_all
    else:
        records = load_jsonl(cfg.input_path)
        if cfg.from_row is not None or cfg.to_row is not None:
            from_idx = cfg.from_row if cfg.from_row is not None else 0
            to_idx = min(cfg.to_row if cfg.to_row is not None else len(records), len(records))
            records = records[from_idx:to_idx] if from_idx < to_idx else []
        elif cfg.max_rows is not None and cfg.max_rows > 0:
            records = records[: cfg.max_rows]

    if not records:
        print("No records to process.")
        return

    client = build_client(cfg)
    out_records: List[Dict[str, Any]] = []

    for r in tqdm(records, desc="Generating MCQs"):
        sub_caption = (r.get("sub_caption") or "").strip()
        relevant_context = (r.get("relevant_image_context") or "").strip()
        image_context = (r.get("image_context") or "").strip()
        primary_modality = (r.get("primary_modality") or "").strip()
        secondary_modality = (r.get("secondary_modality") or "").strip()

        # Read categories from step 2b output
        categories: List[str] = [
            c for c in (r.get("categories") or []) if isinstance(c, str) and c.strip()
        ]

        generated: List[Dict[str, Any]] = []

        for category in categories:
            for style in resolve_styles(category):
                image_scope = decide_image_scope(cfg, r, category, style)
                # Full-figure questions use the broader image_context; subfigure uses the relevant extract.
                context_text = (
                    image_context if image_scope == "full_figure"
                    else (relevant_context or image_context)
                )
                mcq_format = decide_mcq_format(cfg, r, category, style)
                image_data_url = get_image_data_url(r, cfg, image_scope)

                mcq = generate_mcq(
                    client=client,
                    cfg=cfg,
                    row=r,
                    sub_caption=sub_caption,
                    context_text=context_text,
                    primary_modality=primary_modality,
                    secondary_modality=secondary_modality,
                    category=category,
                    style=style,
                    mcq_format=mcq_format,
                    image_scope=image_scope,
                    image_data_url=image_data_url,
                )
                if mcq is not None:
                    generated.append(mcq)

        out_records.append({**r, "generated_mcq": generated})

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} rows -> {cfg.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Generate MCQs.")
    parser.add_argument("--config", type=str, default="cfg/3_generate_mcq.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
