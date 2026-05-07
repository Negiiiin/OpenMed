#!/usr/bin/env python3
"""
Stage 1: Data Filtering Pipeline
Applies image quality and text quality filters to the dataset.
"""

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import yaml
from PIL import Image  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from tqdm import tqdm  # type: ignore


# ==========================================
#  CONFIGURATION CLASSES
# ==========================================

@dataclass
class FilteringConfig:
    """Configuration for the dataset filtering pipeline."""

    input_path: str
    output_path: str
    columns: Dict[str, str] = field(
        default_factory=lambda: {
            "image_path": "image_path",
            "sub_caption": "sub_caption",
            "summary": "summary",
            "image_context": "image_context",
        }
    )
    context_source: str = "summary"
    extract_all_image_context: bool = True  # If False, extract only for the related image figure_id
    image_quality: Dict[str, Any] = field(default_factory=dict)
    context_quality: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "jsonl"
    save_intermediate: bool = False
    intermediate_dir: Optional[str] = None
    keywords: Optional[List[str]] = None
    max_rows: Optional[int] = None
    from_row: Optional[int] = None  # Start row index (0-based)
    to_row: Optional[int] = None  # End row index (exclusive, 0-based)


# ==========================================
#  HELPER FUNCTIONS (Config & IO)
# ==========================================

def load_filtering_config(raw: Dict[str, Any]) -> FilteringConfig:
    """Parse the filtering section from YAML into a FilteringConfig object."""

    def _normalize_keywords(value: Optional[Any]) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            return cleaned or None
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return None

    columns = raw.get("columns") or {}
    filter_cfg = FilteringConfig(
        input_path=raw.get("input_path", ""),
        output_path=raw.get("output_path", ""),
        columns={
            "image_path": columns.get("image_path", "image_path"),
            "sub_caption": columns.get("sub_caption", "sub_caption"),
            "summary": columns.get("summary", "summary"),
            "image_context": columns.get("image_context", "image_context"),
        },
        context_source=raw.get("context_source", "summary"),
        extract_all_image_context=raw.get("extract_all_image_context", True),
        image_quality=raw.get("image_quality", {}),
        context_quality=raw.get("context_quality", {}),
        output_format=raw.get("output_format", "jsonl").lower(),
        save_intermediate=raw.get("save_intermediate", False),
        intermediate_dir=raw.get("intermediate_dir"),
        keywords=_normalize_keywords(raw.get("keywords")),
        max_rows=raw.get("max_rows"),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
    )
    if filter_cfg.max_rows is not None and filter_cfg.max_rows <= 0:
        filter_cfg.max_rows = None
    if filter_cfg.from_row is not None and filter_cfg.from_row < 0:
        filter_cfg.from_row = None
    if filter_cfg.to_row is not None and filter_cfg.to_row < 0:
        filter_cfg.to_row = None
    return filter_cfg


def load_data_file(file_path: str) -> pd.DataFrame:
    """Load data from either JSONL or CSV into a DataFrame."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {exc}")
        return pd.DataFrame(data)

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    # Auto-detect based on first line
    with open(file_path, "rb") as f:
        head = f.read(1024)
        first_line = head.decode("utf-8", errors="ignore").split("\n")[0]
        if first_line.strip().startswith("{"):
            data = []
            with open(file_path, "r", encoding="utf-8") as txt_f:
                for line_num, line in enumerate(txt_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {exc}")
            return pd.DataFrame(data)
    return pd.read_csv(file_path)


def ensure_directory(path: str) -> None:
    """Create parent directory for a file path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def resolve_image_path(image_path: str, base_dir: Optional[str]) -> Optional[str]:
    """Resolve absolute image path given a base directory."""
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


# ==========================================
#  STAGE 1: IMAGE QUALITY FILTER
# ==========================================

def check_basic_quality(image_path: str, min_size: int, max_aspect: float) -> Tuple[bool, Optional[str]]:
    """Basic check for shortest side and aspect ratio."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if min(w, h) < min_size:
                return False, f"Too small ({min(w, h)}px < {min_size}px)"
            ratio = max(w / h, h / w) if h else float("inf")
            if ratio > max_aspect:
                return False, f"Extreme aspect ratio ({ratio:.2f} > {max_aspect})"
    except Exception as exc:
        return False, f"Failed to read image: {exc}"
    return True, None


def white_ratio(image_path: str, threshold: int, border_percent: float) -> float:
    """Compute ratio of near-white pixels along the border."""
    img = cv2.imread(image_path)
    if img is None:
        return 1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border = max(5, int(min(h, w) * border_percent))
    top = gray[:border, :]
    bottom = gray[h - border :, :]
    left = gray[:, :border]
    right = gray[:, w - border :]
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
    return float(np.mean(border_pixels >= threshold)) if len(border_pixels) else 0.0


def check_resolution_quality(image_path: str, min_sharpness: float = 50.0, min_effective_resolution: Optional[int] = None) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """
    Check the actual resolution quality of an image beyond just dimensions.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Failed to read image", {}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Check sharpness using Laplacian variance (detects blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var >= min_sharpness
        
        # 2. Estimate effective resolution by analyzing high-frequency content
        gray_float = gray.astype(np.float32)
        grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        detail_density = avg_gradient / 255.0
        total_pixels = w * h
        estimated_effective_pixels = total_pixels * detail_density
        
        if min_effective_resolution is None:
            min_effective_pixels = (min(w, h) * 0.1) ** 2
        else:
            min_effective_pixels = min_effective_resolution ** 2
        
        has_sufficient_detail = estimated_effective_pixels >= min_effective_pixels
        
        # 3. Check for compression artifacts
        block_size = 8
        if h >= block_size and w >= block_size:
            block_variances = []
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    block_variances.append(np.var(block))
            avg_block_variance = np.mean(block_variances) if block_variances else 0
            has_compression_artifacts = avg_block_variance < 20 and len(block_variances) > 10
        else:
            has_compression_artifacts = False
        
        metrics = {
            "laplacian_variance": float(laplacian_var),
            "avg_gradient": float(avg_gradient),
            "detail_density": float(detail_density),
            "estimated_effective_pixels": float(estimated_effective_pixels),
            "total_pixels": total_pixels,
            "width": w,
            "height": h
        }
        
        reasons = []
        if not is_sharp:
            reasons.append(f"Too blurry (sharpness={laplacian_var:.1f} < {min_sharpness})")
        if not has_sufficient_detail:
            reasons.append(f"Low effective resolution (estimated={estimated_effective_pixels:.0f} pixels < {min_effective_pixels:.0f})")
        if has_compression_artifacts:
            reasons.append("Heavy compression artifacts detected")
        
        if reasons:
            return False, "; ".join(reasons), metrics
        return True, None, metrics
        
    except Exception as exc:
        return False, f"Quality check failed: {exc}", {}


def stage1_image_quality_filter(df: pd.DataFrame, cfg: FilteringConfig, base_dir: Optional[str]) -> pd.DataFrame:
    """Remove rows whose images fail basic quality checks."""
    quality_cfg = cfg.image_quality or {}
    if not quality_cfg.get("enabled", True):
        print("Stage 1 (image quality): disabled, skipping.")
        return df

    print(f"\n--- Stage 1: Image Quality Filtering ---")
    image_col = cfg.columns["image_path"]
    if image_col not in df.columns:
        raise KeyError(f"Image column '{image_col}' not present for filtering.")
    min_size = quality_cfg.get("min_size", 256)
    max_aspect = quality_cfg.get("max_aspect", 4.0)
    white_threshold = quality_cfg.get("white_threshold", 235)
    max_white = quality_cfg.get("max_white_ratio", 0.4)
    border_percent = quality_cfg.get("border_percent", 0.15)
    
    check_resolution = quality_cfg.get("check_resolution_quality", True)
    min_sharpness = quality_cfg.get("min_sharpness", 50.0)
    min_effective_resolution = quality_cfg.get("min_effective_resolution")
    
    if check_resolution:
        print(f"  Resolution quality check: enabled (min_sharpness={min_sharpness}, min_effective_resolution={min_effective_resolution})")
    else:
        print(f"  Resolution quality check: disabled")
    
    keep_mask = []
    filtered_count = 0
    resolution_filtered_count = 0
    filter_reasons = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Checking image quality"):
        resolved = resolve_image_path(str(row[image_col]), base_dir)
        if not resolved:
            keep_mask.append(False)
            filtered_count += 1
            filter_reasons["Path not found"] = filter_reasons.get("Path not found", 0) + 1
            continue
        good, reason = check_basic_quality(resolved, min_size, max_aspect)
        if not good:
            keep_mask.append(False)
            filtered_count += 1
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue
        white = white_ratio(resolved, white_threshold, border_percent)
        if white > max_white:
            keep_mask.append(False)
            filtered_count += 1
            filter_reasons["Too much white border"] = filter_reasons.get("Too much white border", 0) + 1
            continue
        
        if check_resolution:
            res_valid, res_reason, res_metrics = check_resolution_quality(
                resolved, min_sharpness, min_effective_resolution
            )
            if not res_valid:
                keep_mask.append(False)
                filtered_count += 1
                resolution_filtered_count += 1
                filter_reasons[f"Resolution: {res_reason}"] = filter_reasons.get(f"Resolution: {res_reason}", 0) + 1
                continue
        
        keep_mask.append(True)
    
    filtered = df[keep_mask].reset_index(drop=True)
    print(f"  ✓ Removed {filtered_count} rows, {len(filtered)} rows remaining.")
    if check_resolution and resolution_filtered_count > 0:
        print(f"    - Filtered by resolution quality: {resolution_filtered_count} rows")
    if filter_reasons:
        print(f"    - Breakdown: {dict(filter_reasons)}")
    return filtered


# ==========================================
#  STAGE 2: CONTEXT QUALITY FILTER
# ==========================================

def extract_figure_id_from_path(path: str, available_keys: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract figure ID from image path.
    Looks for patterns like 'DDDT-13-4161-g0003' in the path.
    
    Args:
        path: Image path string
        available_keys: Optional list of available figure ID keys to match against.
                        If provided, will try to find exact matches first.
    
    Returns:
        Extracted figure ID or None
    """
    if not path:
        return None
    
    # If we have available keys, try to find an exact match first
    if available_keys:
        for key in available_keys:
            if key in path:
                return key
    
    # Try to match figure ID pattern (e.g., DDDT-13-4161-g0003)
    # Pattern: letters/numbers-dash-numbers-dash-numbers-dash-g-numbers
    # More specific: starts with letters, then dash-separated numbers/letters, ends with -g<digits>
    match = re.search(r'([A-Z]{2,}(?:-\d+)+-g\d+)', path)
    if match:
        return match.group(1)
    
    # Fallback: try the more general pattern
    match = re.search(r'([A-Z0-9]+(?:-[A-Z0-9]+)*-(?:g|G)\d+)', path)
    if match:
        return match.group(1)
    return None


def extract_image_context_text(image_context: Any, figure_id: Optional[str] = None, extract_all: bool = True) -> str:
    """
    Extract text from image_context field.
    Handles both string and dict formats (dict maps figure IDs to lists of text).
    
    Args:
        image_context: The image_context field (string or dict)
        figure_id: Optional figure ID to extract text for a specific figure only.
                   If None and extract_all=False, will try to extract from available context.
        extract_all: If True (default), extract text from all figure IDs.
                     If False and figure_id is provided, only extract for that figure.
    
    Returns:
        Extracted text as a string
    """
    if image_context is None:
        return ""
    if isinstance(image_context, str):
        return image_context.strip()
    if isinstance(image_context, dict):
        texts = []
        if extract_all:
            # Extract all text from all figure IDs and join them
            for fig_id, text_list in image_context.items():
                if isinstance(text_list, list):
                    texts.extend([str(t).strip() for t in text_list if t])
                else:
                    texts.append(str(text_list).strip())
        else:
            # Extract only for the specific figure_id
            if figure_id and figure_id in image_context:
                text_list = image_context[figure_id]
                if isinstance(text_list, list):
                    texts.extend([str(t).strip() for t in text_list if t])
                else:
                    texts.append(str(text_list).strip())
            elif figure_id:
                # Try to find a partial match (in case the figure_id format differs slightly)
                for key in image_context.keys():
                    if figure_id in key or key in figure_id:
                        text_list = image_context[key]
                        if isinstance(text_list, list):
                            texts.extend([str(t).strip() for t in text_list if t])
                        else:
                            texts.append(str(text_list).strip())
                        break
        return " ".join([t for t in texts if t])
    return str(image_context).strip()


def load_qwen_model(model_path: str, device: str, use_vllm: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if use_vllm:
        from vllm import LLM  # type: ignore
        llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
        return llm, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def generate_with_qwen(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float, use_vllm: bool) -> str:
    """Generate text for a single prompt (legacy function for non-batched calls)."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if use_vllm:
        from vllm import SamplingParams  # type: ignore
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature if temperature > 0 else None, top_p=top_p, repetition_penalty=repetition_penalty)
        outputs = model.generate([prompt_text], sampling_params)
        return outputs[0].outputs[0].text.strip()

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(output[:, input_ids.shape[-1] :], skip_special_tokens=True)[0].strip()

def generate_batch_with_qwen(model, tokenizer, system_prompt: str, user_prompts: List[str], max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float, use_vllm: bool) -> List[str]:
    """Generate text for a batch of prompts. Optimized for vLLM batch processing."""
    if use_vllm:
        from vllm import SamplingParams  # type: ignore
        prompt_texts = []
        for user_prompt in user_prompts:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prompt_text)
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, 
            temperature=temperature if temperature > 0 else None, 
            top_p=top_p, 
            repetition_penalty=repetition_penalty
        )
        outputs = model.generate(prompt_texts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]
    
    results = []
    for user_prompt in user_prompts:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty, pad_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.batch_decode(output[:, input_ids.shape[-1] :], skip_special_tokens=True)[0].strip()
        results.append(result)
    return results

def stage2_context_quality_filter(df: pd.DataFrame, cfg: FilteringConfig, reject_path: Optional[str] = None) -> pd.DataFrame:
    ctx_cfg = cfg.context_quality or {}
    if not ctx_cfg.get("enabled", True):
        print("Stage 2 (context quality): disabled, skipping.")
        return df

    print(f"\n--- Stage 2: Context Quality Filtering ---")
    min_length = ctx_cfg.get("min_length", 50)
    model_path = ctx_cfg.get("model_path")
    if not model_path:
        return df

    model, tokenizer = load_qwen_model(model_path, ctx_cfg.get("device", "cuda"), use_vllm=ctx_cfg.get("use_vllm", False))
    
    sub_caption_col = cfg.columns["sub_caption"]
    summary_col = cfg.columns["summary"]
    image_context_col = cfg.columns["image_context"]
    
    # Pre-filter length
    def combine(row): 
        sub_cap = str(row.get(sub_caption_col, ""))
        if cfg.context_source == "summary":
            ctx = str(row.get(summary_col, ""))
        else:
            image_context_raw = row.get(image_context_col)
            # Extract figure ID from path if we want only the related image
            figure_id = None
            if not cfg.extract_all_image_context:
                path = None
                for path_col in ["full_fig_path", "subfig_path", cfg.columns.get("image_path", "image_path")]:
                    if path_col in row:
                        path = str(row.get(path_col, ""))
                        if path:
                            break
                if path:
                    # Get available keys from image_context if it's a dict
                    available_keys = None
                    if isinstance(image_context_raw, dict):
                        available_keys = list(image_context_raw.keys())
                    figure_id = extract_figure_id_from_path(path, available_keys=available_keys)
            ctx = extract_image_context_text(
                image_context_raw,
                figure_id=figure_id,
                extract_all=cfg.extract_all_image_context
            )
        return " ".join([sub_cap, ctx])
    df = df[df.apply(combine, axis=1).str.len() >= min_length].reset_index(drop=True)

    keep_mask = []
    rows = df.to_dict("records")
    batch_size = ctx_cfg.get("batch_size", 8)
    user_template = ctx_cfg.get("user_prompt_template", "Sub-caption: {{sub_caption}}\nContext: {{image_context}}\nAnswer YES or NO:")
    
    max_new_tokens = ctx_cfg.get("max_new_tokens", 512)
    temperature = ctx_cfg.get("temperature", 0.1)
    top_p = ctx_cfg.get("top_p", 0.9)
    repetition_penalty = ctx_cfg.get("repetition_penalty", 1.0)
    use_vllm = ctx_cfg.get("use_vllm", False)
    # Safety: cap context length to avoid exceeding model max context (tokens)
    # This is a character-level cap; effective token length will be lower.
    max_context_chars = ctx_cfg.get("max_context_chars", 8000)
    
    system_prompt = ctx_cfg.get("system_prompt", "")
    
    print(f"  Evaluating {len(rows)} rows with Qwen model...")
    print(f"  Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    print(f"  Using vLLM: {use_vllm}")
    if use_vllm:
        print(f"  vLLM batch processing enabled - processing in batches of {batch_size}")
    
    for start in tqdm(range(0, len(rows), batch_size), desc="  Evaluating context"):
        batch = rows[start : start + batch_size]
        batch_indices = list(range(start, min(start + batch_size, len(rows))))
        prompts = []
        for r in batch:
            sub_caption_text = str(r.get(sub_caption_col, ""))
            if cfg.context_source == "summary":
                context_text = str(r.get(summary_col, ""))
            else:
                image_context_raw = r.get(image_context_col)
                # Extract figure ID from path if we want only the related image
                figure_id = None
                if not cfg.extract_all_image_context:
                    path = None
                    for path_col in ["full_fig_path", "subfig_path", cfg.columns.get("image_path", "image_path")]:
                        if path_col in r:
                            path = str(r.get(path_col, ""))
                            if path:
                                break
                    if path:
                        # Get available keys from image_context if it's a dict
                        available_keys = None
                        if isinstance(image_context_raw, dict):
                            available_keys = list(image_context_raw.keys())
                        figure_id = extract_figure_id_from_path(path, available_keys=available_keys)
                context_text = extract_image_context_text(
                    image_context_raw,
                    figure_id=figure_id,
                    extract_all=cfg.extract_all_image_context
                )

            # Truncate excessively long context to keep prompt within model limits
            if len(context_text) > max_context_chars:
                context_text = context_text[:max_context_chars]
            prompt = user_template.replace("{{sub_caption}}", sub_caption_text)
            prompt = prompt.replace("{{image_context}}", context_text)
            prompts.append(prompt)
        
        if use_vllm:
            responses = generate_batch_with_qwen(
                model, tokenizer, system_prompt, prompts, 
                max_new_tokens, temperature, top_p, repetition_penalty, use_vllm
            )
        else:
            responses = []
            for prompt in prompts:
                resp = generate_with_qwen(
                    model, tokenizer, system_prompt, prompt, 
                    max_new_tokens, temperature, top_p, repetition_penalty, use_vllm
                )
                responses.append(resp)
        
        batch_keep = ["yes" in r.lower() or "sufficient" in r.lower() for r in responses]
        keep_mask.extend(batch_keep)

        # write rejects for this batch immediately
        if reject_path:
            rejected_batch = []
            for i, kept in enumerate(batch_keep):
                if not kept:
                    rec = dict(batch[i])
                    # extract only the related subfigure's context (not the full dict)
                    image_context_raw = rec.get(image_context_col)
                    if isinstance(image_context_raw, dict):
                        path = None
                        for path_col in ["full_fig_path", "subfig_path", cfg.columns.get("image_path", "image_path")]:
                            if path_col in rec:
                                path = str(rec.get(path_col, ""))
                                if path:
                                    break
                        figure_id = extract_figure_id_from_path(path, available_keys=list(image_context_raw.keys())) if path else None
                        rec[image_context_col] = extract_image_context_text(
                            image_context_raw, figure_id=figure_id, extract_all=False
                        )
                    rec["_filter_stage"] = "stage2_text_quality"
                    rec["_qwen_response"] = responses[i]
                    rejected_batch.append(rec)
            if rejected_batch:
                ensure_directory(reject_path)
                with open(reject_path, "a", encoding="utf-8") as rf:
                    for rec in rejected_batch:
                        rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rf.flush()

    filtered = df[keep_mask].reset_index(drop=True)
    print(f"  ✓ Removed {len(df) - len(filtered)} rows, {len(filtered)} rows remaining.")
    return filtered


# ==========================================
#  MAIN PIPELINE
# ==========================================

def run_filtering_pipeline(cfg: FilteringConfig) -> pd.DataFrame:
    if not cfg.input_path:
        raise ValueError("input_path must be set")
    print(f"Running filtering pipeline on {cfg.input_path}")
    df = load_data_file(cfg.input_path)

    # Optional: slice raw input first (used by SLURM array jobs)
    if cfg.from_row is not None or cfg.to_row is not None:
        from_idx = cfg.from_row if cfg.from_row is not None else 0
        to_idx = cfg.to_row if cfg.to_row is not None else len(df)
        to_idx = min(to_idx, len(df))
        if from_idx < to_idx:
            df = df.iloc[from_idx:to_idx].reset_index(drop=True)
        else:
            df = df.iloc[0:0].reset_index(drop=True)
        cfg.from_row = None
        cfg.to_row = None

    # Step 1: Keyword filtering (first)
    if cfg.keywords:
        print(f"\n--- Keyword Filtering ---")
        initial_count = len(df)
        lowered = [kw.lower() for kw in cfg.keywords]
        text_cols = [cfg.columns["sub_caption"], cfg.columns[cfg.context_source]]
        mask = None
        for col in text_cols:
            if col in df.columns:
                # Handle image_context as dict by extracting text first
                if col == cfg.columns["image_context"] and cfg.context_source == "image_context":
                    # Extract text from image_context dict for keyword matching
                    def extract_text_for_keywords(row_val):
                        text = extract_image_context_text(row_val, extract_all=cfg.extract_all_image_context)
                        return text.lower()
                    col_mask = df[col].apply(extract_text_for_keywords).apply(lambda v: any(k in v for k in lowered))
                else:
                    col_mask = df[col].astype(str).str.lower().apply(lambda v: any(k in v for k in lowered))
                if mask is None:
                    mask = col_mask
                else:
                    mask = mask | col_mask
        
        if mask is None:
            print(f"Warning: None of the expected columns {text_cols} found in dataframe. Available columns: {list(df.columns)}")
            print(f"Skipping keyword filtering, keeping all {initial_count} rows.")
        else:
            df = df[mask].reset_index(drop=True)
            print(f"  ✓ Removed {initial_count - len(df)} rows, {len(df)} rows remaining after keyword filtering.")
    
    # Step 2: Row selection (second)
    if cfg.from_row is not None or cfg.to_row is not None:
        print(f"\n--- Row Selection (Range) ---")
        initial_count = len(df)
        from_idx = cfg.from_row if cfg.from_row is not None else 0
        to_idx = cfg.to_row if cfg.to_row is not None else len(df)
        
        # Validate range
        if from_idx >= len(df):
            print(f"Warning: from_row ({from_idx}) >= total rows ({len(df)}). No rows selected.")
            df = df.iloc[0:0].reset_index(drop=True)  # Empty dataframe
        else:
            to_idx = min(to_idx, len(df))
            if from_idx >= to_idx:
                print(f"Warning: from_row ({from_idx}) >= to_row ({to_idx}). No rows selected.")
                df = df.iloc[0:0].reset_index(drop=True)  # Empty dataframe
            else:
                df = df.iloc[from_idx:to_idx].reset_index(drop=True)
                print(f"Selected rows {from_idx} to {to_idx-1} ({len(df)} rows from {initial_count} available).")
    elif cfg.max_rows:
        print(f"\n--- Row Selection ---")
        initial_count = len(df)
        df = df.head(cfg.max_rows).reset_index(drop=True)
        print(f"  ✓ Selected {len(df)} rows (from {initial_count} available).")
    
    # Step 3: Quality checks (third)
    base_dir = os.path.dirname(os.path.abspath(cfg.input_path))

    def _save_rejects(rejected_df: pd.DataFrame, stage: str) -> None:
        """Optionally save rejected rows alongside the output file."""
        if rejected_df.empty:
            return
        base, ext = os.path.splitext(cfg.output_path)
        reject_path = f"{base}_rejected_{stage}{ext}"
        ensure_directory(reject_path)
        rejected_df["_filter_stage"] = stage
        if cfg.output_format == "jsonl":
            with open(reject_path, "a", encoding="utf-8") as f:
                for _, row in rejected_df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        else:
            rejected_df.to_csv(reject_path, mode="a", index=False, header=not os.path.exists(reject_path))
        print(f"  → Saved {len(rejected_df)} rejected rows to {reject_path}")

    df_before = df.copy()
    df = stage1_image_quality_filter(df, cfg, base_dir)
    _save_rejects(df_before[~df_before.index.isin(df.index)], "stage1_image_quality")

    if not df.empty:
        base, ext = os.path.splitext(cfg.output_path)
        stage2_reject_path = f"{base}_rejected_stage2_text_quality{ext}"
        df = stage2_context_quality_filter(df, cfg, reject_path=stage2_reject_path)
    
    # Extract and save only the related image context before saving
    if not df.empty and cfg.columns.get("image_context") in df.columns:
        print(f"\n--- Extracting Related Image Context ---")
        image_context_col = cfg.columns["image_context"]
        
        def extract_related_context(row):
            """Extract only the related image's context from image_context dict."""
            image_context_raw = row.get(image_context_col)
            if not isinstance(image_context_raw, dict):
                # If it's already a string or None, keep it as is
                return image_context_raw
            
            # Extract figure ID from path
            path = None
            for path_col in ["full_fig_path", "subfig_path", cfg.columns.get("image_path", "image_path")]:
                if path_col in row:
                    path = str(row.get(path_col, ""))
                    if path:
                        break
            
            if path:
                available_keys = list(image_context_raw.keys())
                figure_id = extract_figure_id_from_path(path, available_keys=available_keys)
                if figure_id:
                    # Extract only for this specific figure
                    return extract_image_context_text(
                        image_context_raw,
                        figure_id=figure_id,
                        extract_all=False  # Only extract for the related image
                    )
            
            # Fallback: if we can't find the figure ID, return empty string
            return ""
        
        # Apply extraction to each row
        df[image_context_col] = df.apply(extract_related_context, axis=1)
        print(f"  ✓ Extracted related image context for {len(df)} rows")
    
    # Save output
    ensure_directory(cfg.output_path)
    if cfg.output_format == "jsonl":
        with open(cfg.output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    else:
        df.to_csv(cfg.output_path, index=False)
    
    return df


def load_config(path: str) -> FilteringConfig:
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return load_filtering_config(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/1_filter.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # SLURM array job: process a slice of rows and write to task-specific output
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if task_id is not None and task_count is not None:
        task_id = int(task_id)
        task_count = int(task_count)
        total = len(load_data_file(cfg.input_path))
        chunk = (total + task_count - 1) // task_count
        cfg.from_row = task_id * chunk
        cfg.to_row = min((task_id + 1) * chunk, total)
        base, ext = os.path.splitext(cfg.output_path)
        cfg.output_path = f"{base}_task{task_id}{ext}"
        print(f"Array task {task_id}/{task_count}: rows {cfg.from_row}-{cfg.to_row} -> {cfg.output_path}")
    df = run_filtering_pipeline(cfg)
    print("\n--- Done. Head of output: ---")
    print(df.head())


if __name__ == "__main__":
    main()
