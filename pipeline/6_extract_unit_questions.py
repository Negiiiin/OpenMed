#!/usr/bin/env python3
"""
Step 6: Extract a unit-question rubric from each refined reasoning trace.

Each reasoning trace is decomposed into N yes/no UNIT QUESTIONS along three
axes:

    observation  -- visible image findings the trace describes
    knowledge    -- general medical facts the trace relies on
    inference    -- case-specific bridges from findings + facts to the conclusion

Every unit carries two probes that the downstream judge (step 7) scores
independently:

    presence_question    -- LENIENT: did the model mention this topic at all?
    correctness_question -- STRICT:  did the model get the details right?

An anti-hallucination guard rejects any unit whose ``source_quote`` cannot be
found verbatim (modulo whitespace) in the source reasoning trace.

Input:  step 5 JSONL -- rows with ``generated_mcq`` list; each MCQ object has
        a ``reasoning`` field.
Output: flat JSONL -- one row per MCQ (carries forward image paths, question,
        choices, answer, category, modality) with a ``unit_questions`` list.

SLURM array support:
    Pass ``--from_row`` / ``--to_row`` and ``--output_path`` on the CLI to
    process a disjoint row slice per task. Merge shards afterward.
"""

from __future__ import annotations

import argparse
import json
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

from prompts import UNIT_QUESTIONS_EXTRACT_PROMPT


# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    input_path: str
    output_path: str
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = "medium"
    temperature: Optional[float] = 0.0
    # Visible-token budget. For reasoning models (o1/o3/gpt-5*) this becomes
    # max_completion_tokens and must cover hidden reasoning tokens -- keep it
    # large. The call retries at 2x and 4x on empty / truncated responses.
    max_completion_tokens: int = 16384
    request_timeout: float = 180.0
    # Per-axis caps (communicated to the LLM via the prompt template).
    max_observation: int = 4
    max_knowledge: int = 3
    max_inference: int = 3
    min_total: int = 3
    max_total: int = 8
    # Which field inside each MCQ object holds the reasoning trace.
    reasoning_field: str = "reasoning"
    generated_mcq_field: str = "generated_mcq"
    delay: float = 0.0
    from_row: Optional[int] = None
    to_row: Optional[int] = None
    max_rows: Optional[int] = None
    save_every_n: int = 10
    verbose_errors: bool = True


def load_config(path: str) -> Config:
    if not os.path.isabs(path):
        path = os.path.join(_SCRIPT_DIR, path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _opt_str(key: str) -> Optional[str]:
        v = raw.get(key)
        return str(v).strip() if v is not None and str(v).strip() else None

    return Config(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        model=str(raw.get("model", "gpt-4o-mini")).strip(),
        api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")).strip(),
        base_url=_opt_str("base_url"),
        reasoning_effort=_opt_str("reasoning_effort"),
        temperature=float(raw["temperature"]) if raw.get("temperature") is not None else None,
        max_completion_tokens=max(1024, int(raw.get("max_completion_tokens", 16384))),
        request_timeout=float(raw.get("request_timeout", 180.0)),
        max_observation=max(0, int(raw.get("max_observation", 4))),
        max_knowledge=max(0, int(raw.get("max_knowledge", 3))),
        max_inference=max(0, int(raw.get("max_inference", 3))),
        min_total=max(0, int(raw.get("min_total", 3))),
        max_total=max(1, int(raw.get("max_total", 8))),
        reasoning_field=str(raw.get("reasoning_field", "reasoning")).strip(),
        generated_mcq_field=str(raw.get("generated_mcq_field", "generated_mcq")).strip(),
        delay=float(raw.get("delay", 0.0)),
        from_row=raw.get("from_row"),
        to_row=raw.get("to_row"),
        max_rows=raw.get("max_rows"),
        save_every_n=max(1, int(raw.get("save_every_n", 10))),
        verbose_errors=bool(raw.get("verbose_errors", True)),
    )


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
    if cfg.request_timeout > 0:
        kwargs["timeout"] = cfg.request_timeout
    return OpenAI(**kwargs)


# ==========================================
#  IO
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


# ==========================================
#  REASONING EXTRACTION
# ==========================================

_THINK_RE = re.compile(
    r"<\s*(?:think|reasoning|thought)\s*>(.*?)<\s*/\s*(?:think|reasoning|thought)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_RE = re.compile(r"<\s*answer\s*>.*?<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE    = re.compile(r"</?\s*(?:think|reasoning|thought|answer)\s*>", re.IGNORECASE)


def extract_think_block(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    blocks = [m.strip() for m in _THINK_RE.findall(text) if str(m).strip()]
    if blocks:
        return "\n\n".join(blocks).strip()
    text = _ANSWER_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return text.strip()


def format_mcq_context(mcq: Dict[str, Any]) -> str:
    parts: List[str] = []
    q = str(mcq.get("question") or "").strip()
    if q:
        parts.append(q)
    choices = mcq.get("choices") or {}
    if isinstance(choices, dict):
        lines = [f"{k}. {v}" for k, v in sorted(choices.items()) if str(v).strip()]
        if lines:
            parts.append("Options:\n" + "\n".join(lines))
    answer = str(mcq.get("answer") or "").strip()
    if answer:
        parts.append(f"Reference answer: {answer}")
    return "\n\n".join(parts) if parts else "(unavailable)"


# Fields carried forward from the MCQ object and parent row into flat output.
_MCQ_FIELDS = ("category", "style", "question", "choices", "answer", "image_scope",
               "reasoning_image_path", "reasoning_context_source")
_ROW_FIELDS = ("id", "subfig_path", "full_fig_path",
               "modality", "primary_modality",
               "image_context", "relevant_image_context")


def build_flat_row(row: Dict[str, Any], mcq: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _ROW_FIELDS:
        if k in row:
            out[k] = row[k]
    for k in _MCQ_FIELDS:
        if k in mcq:
            out[k] = mcq[k]
    return out


# ==========================================
#  OPENAI CALL + RETRY
# ==========================================

def _is_reasoning_model(model: str) -> bool:
    return bool(re.search(r"\bo[1-9]\b|gpt-5", model, re.IGNORECASE))


def _assistant_text(message: Any) -> str:
    c = getattr(message, "content", None)
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts: List[str] = []
        for block in c:
            txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if txt:
                parts.append(str(txt))
        return "\n".join(parts).strip()
    return str(c).strip() if c else ""


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in response")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def _step_effort_down(effort: Optional[str]) -> Optional[str]:
    chain = ["high", "medium", "low", "minimal"]
    cur = (effort or "").strip().lower()
    if cur not in chain:
        return "minimal"
    return chain[min(chain.index(cur) + 1, len(chain) - 1)]


def _build_prompt(cot: str, case_question: str, cfg: Config) -> str:
    return (
        UNIT_QUESTIONS_EXTRACT_PROMPT
        .replace("{min_total}",       str(cfg.min_total))
        .replace("{max_total}",       str(cfg.max_total))
        .replace("{max_observation}", str(cfg.max_observation))
        .replace("{max_knowledge}",   str(cfg.max_knowledge))
        .replace("{max_inference}",   str(cfg.max_inference))
        .replace("{case_question}",   case_question)
        .replace("{cot}",             cot)
    )


def call_extract(
    client: OpenAI,
    *,
    cot: str,
    case_question: str,
    cfg: Config,
) -> Dict[str, Any]:
    """Call the LLM with a retry chain that handles empty / truncated responses.

    Reasoning models can exhaust their visible-token budget on hidden chain-of-
    thought. Each retry doubles the budget; on the final attempt the reasoning
    effort is also stepped down so more tokens go to visible JSON output.
    """
    prompt    = _build_prompt(cot, case_question, cfg)
    is_reason = _is_reasoning_model(cfg.model)
    base      = cfg.max_completion_tokens if is_reason else min(cfg.max_completion_tokens, 4096)
    effort0   = cfg.reasoning_effort if is_reason else None

    attempts: List[Tuple[int, Optional[str]]] = (
        [(base, effort0), (base * 2, effort0), (base * 4, _step_effort_down(effort0))]
        if is_reason else
        [(base, None), (min(base * 2, 16384), None)]
    )

    last_finish: Optional[str] = None
    last_text   = ""
    last_exc: Optional[BaseException] = None

    for idx, (max_toks, effort) in enumerate(attempts):
        params: Dict[str, Any] = {
            "model":    cfg.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_reason:
            params["max_completion_tokens"] = max_toks
            if effort:
                params["reasoning_effort"] = effort
        else:
            params["max_tokens"] = max_toks
            if cfg.temperature is not None:
                params["temperature"] = cfg.temperature

        try:
            resp = client.chat.completions.create(**params)
        except TypeError:
            if "reasoning_effort" in params:
                params.pop("reasoning_effort")
                resp = client.chat.completions.create(**params)
            else:
                raise
        except Exception as exc:
            last_exc = exc
            if idx == len(attempts) - 1:
                raise
            continue

        choice = resp.choices[0] if resp.choices else None
        if choice is None:
            continue
        last_finish = getattr(choice, "finish_reason", None)
        last_text   = _assistant_text(choice.message)

        if last_text:
            try:
                return _parse_json_object(last_text)
            except ValueError as exc:
                last_exc = exc
                if idx < len(attempts) - 1:
                    continue
                raise

        last_exc = ValueError("empty response")

    snippet = (last_text or "").strip()[:200]
    raise ValueError(
        f"empty/unparseable response after {len(attempts)} attempts; "
        f"finish_reason={last_finish!r}; snippet={snippet!r}; last_error={last_exc}"
    )


# ==========================================
#  UNIT VALIDATION + CAPPING
# ==========================================

_AXES        = ("observation", "knowledge", "inference")
_IMPORTANCES = ("core", "supporting")
_WS_RE       = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", str(s or "")).strip().lower()


def _quote_in_cot(quote: str, cot_norm: str) -> bool:
    q = _norm(quote)
    if not q or len(q) < 3:
        return False
    if q in cot_norm:
        return True
    table = {ord(c): repl for c, repl in [
        ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
        ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-"),
    ]}
    return q.translate(table) in cot_norm.translate(table)


def _wc(s: str) -> int:
    return len([w for w in str(s or "").split() if w])


def _normalize_unit(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    axis = str(raw.get("axis") or "").strip().lower()
    if axis not in _AXES:
        return None
    topic         = str(raw.get("topic") or "").strip()
    claim         = str(raw.get("claim") or "").strip()
    presence_q    = str(raw.get("presence_question") or "").strip()
    correctness_q = str(raw.get("correctness_question") or "").strip()
    source_quote  = str(raw.get("source_quote") or "").strip()
    importance    = str(raw.get("importance") or "supporting").strip().lower()
    if importance not in _IMPORTANCES:
        importance = "supporting"

    # Back-compat: single `question` key -> treat as correctness_question.
    if not correctness_q:
        legacy = str(raw.get("question") or "").strip()
        if legacy:
            correctness_q = legacy
    if not presence_q and (topic or claim):
        anchor = topic or claim.rstrip(".")
        stem = ("Does the response make an inferential link about"
                if axis == "inference" else "Does the response discuss")
        presence_q = f"{stem} {anchor.rstrip('.')}?"

    if not (topic and claim and presence_q and correctness_q and source_quote):
        return None
    if not (1 <= _wc(topic) <= 15):        return None
    if not (4 <= _wc(claim) <= 60):        return None
    if not (4 <= _wc(presence_q) <= 40):   return None
    if not (6 <= _wc(correctness_q) <= 60): return None
    if not (2 <= _wc(source_quote) <= 60): return None

    return dict(axis=axis, topic=topic, claim=claim,
                presence_question=presence_q, correctness_question=correctness_q,
                source_quote=source_quote, importance=importance)


def _dedupe(units: List[Dict]) -> List[Dict]:
    seen: set = set()
    out = []
    for u in units:
        key = (u["axis"], _norm(u["claim"]))
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def _cap_axis(units: List[Dict], cfg: Config) -> List[Dict]:
    caps   = {"observation": cfg.max_observation, "knowledge": cfg.max_knowledge, "inference": cfg.max_inference}
    counts = {a: 0 for a in _AXES}
    # Core items take priority; original order preserved within each tier.
    indexed = sorted(enumerate(units), key=lambda t: (0 if t[1]["importance"] == "core" else 1, t[0]))
    kept: List[Tuple[int, Dict]] = []
    for i, u in indexed:
        if counts[u["axis"]] < caps[u["axis"]]:
            counts[u["axis"]] += 1
            kept.append((i, u))
    return [u for _, u in sorted(kept)]


def _cap_total(units: List[Dict], cfg: Config) -> List[Dict]:
    if len(units) <= cfg.max_total:
        return units
    indexed = sorted(enumerate(units), key=lambda t: (0 if t[1]["importance"] == "core" else 1, t[0]))
    return [u for _, u in sorted(indexed[: cfg.max_total])]


def normalize_units(
    obj: Dict[str, Any], *, cot: str, cfg: Config
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    raw_list = obj.get("unit_questions")
    stats = dict(total_in=0, dropped_schema=0, dropped_unsupported_quote=0,
                 dropped_dedupe=0, dropped_axis_cap=0, dropped_total_cap=0)
    if not isinstance(raw_list, list):
        return [], stats

    stats["total_in"] = len(raw_list)
    cot_norm = _norm(cot)
    candidates: List[Dict] = []
    for raw in raw_list:
        u = _normalize_unit(raw)
        if u is None:
            stats["dropped_schema"] += 1
        elif not _quote_in_cot(u["source_quote"], cot_norm):
            stats["dropped_unsupported_quote"] += 1
        else:
            candidates.append(u)

    deduped = _dedupe(candidates)
    stats["dropped_dedupe"] = len(candidates) - len(deduped)
    capped_ax = _cap_axis(deduped, cfg)
    stats["dropped_axis_cap"] = len(deduped) - len(capped_ax)
    capped_tot = _cap_total(capped_ax, cfg)
    stats["dropped_total_cap"] = len(capped_ax) - len(capped_tot)

    final = [{"unit_id": f"u{i}", **u} for i, u in enumerate(capped_tot, 1)]
    return final, stats


def axis_counts(units: List[Dict]) -> Dict[str, int]:
    out = {a: 0 for a in _AXES}
    for u in units:
        if u.get("axis") in out:
            out[u["axis"]] += 1
    return out


# ==========================================
#  MAIN
# ==========================================

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Step 6: Extract unit-question rubric.")
    parser.add_argument("--config",      type=str, default="cfg/6_extract_unit_questions.yaml")
    parser.add_argument("--from_row",    type=int, default=None)
    parser.add_argument("--to_row",      type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Override output path (useful for per-shard SLURM runs).")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    input_path  = os.path.abspath(cfg.input_path)
    output_path = os.path.abspath(args.output_path or cfg.output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    all_rows = read_jsonl(input_path)
    n_all    = len(all_rows)
    fr       = args.from_row if args.from_row is not None else (cfg.from_row or 0)
    to       = args.to_row   if args.to_row   is not None else (cfg.to_row   or n_all)
    rows     = all_rows[fr:to]
    if cfg.max_rows is not None:
        rows = rows[: cfg.max_rows]

    print(f"[step6] input={input_path}")
    print(f"[step6] output={output_path}")
    print(
        f"[step6] rows={len(rows)} of {n_all}  model={cfg.model}  "
        f"caps obs={cfg.max_observation} know={cfg.max_knowledge} "
        f"infer={cfg.max_inference}  total={cfg.min_total}-{cfg.max_total}"
    )

    client = build_client(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_rows_in = n_flat_in = n_written = n_missing = n_errors = n_units = 0
    drop_totals = dict(total_in=0, dropped_schema=0, dropped_unsupported_quote=0,
                       dropped_dedupe=0, dropped_axis_cap=0, dropped_total_cap=0)

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="step6:unit_questions"):
            n_rows_in += 1
            gen_list = row.get(cfg.generated_mcq_field)
            if not isinstance(gen_list, list):
                continue

            for mcq in gen_list:
                if not isinstance(mcq, dict):
                    continue
                n_flat_in += 1

                cot = extract_think_block(str(mcq.get(cfg.reasoning_field) or ""))
                if not cot:
                    n_missing += 1
                    continue

                try:
                    obj = call_extract(client, cot=cot,
                                       case_question=format_mcq_context(mcq), cfg=cfg)
                except Exception as exc:
                    n_errors += 1
                    tqdm.write(
                        f"[step6 error] row={n_rows_in - 1} "
                        f"id={row.get('id')!r}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if cfg.verbose_errors:
                        tqdm.write(traceback.format_exc(), file=sys.stderr)
                    continue

                units, drops = normalize_units(obj, cot=cot, cfg=cfg)
                for k in drop_totals:
                    drop_totals[k] += drops.get(k, 0)
                n_units += len(units)

                out_row = build_flat_row(row, mcq)
                out_row.update(
                    reference_cot=cot,
                    unit_questions=units,
                    unit_questions_axis_counts=axis_counts(units),
                    unit_questions_model=cfg.model,
                )
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % cfg.save_every_n == 0:
                    fout.flush()
                if cfg.delay > 0:
                    time.sleep(cfg.delay)

    avg = n_units / max(1, n_written)
    print(
        f"[step6] done. rows_in={n_rows_in} mcqs_in={n_flat_in} "
        f"written={n_written} missing_cot={n_missing} errors={n_errors} "
        f"units={n_units} (avg {avg:.1f}/row)"
    )
    print(
        f"[step6] drops: in={drop_totals['total_in']} "
        f"schema={drop_totals['dropped_schema']} "
        f"unsupported_quote={drop_totals['dropped_unsupported_quote']} "
        f"dedupe={drop_totals['dropped_dedupe']} "
        f"axis_cap={drop_totals['dropped_axis_cap']} "
        f"total_cap={drop_totals['dropped_total_cap']}"
    )
    print(f"[step6] output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[step6 fatal] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
