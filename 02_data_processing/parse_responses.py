#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 02_data_processing/parse_responses.py
=============================================================================
 Extracts the chosen answer option from every raw API response.

 Parsing strategy (in order):
   1. strict   : "Answer: Option N"
   2. fallback : last "Option N" mention in the response
   3. bare     : a lone integer on the final line
 Every extracted value is validated against the item's option range
 (1..n_options); out-of-range extractions are marked invalid.

 The per-model parse rate printed here is the Day-2/3 quality gate:
 >= 95% is required before analysis.

 Usage:
   python 02_data_processing/parse_responses.py

 Saves (in 02_data_processing/outputs/):
   parsed_responses.csv   one row per API call, with answer_idx + parse_method
   parse_log.json         machine-readable per-model/per-condition stats
   parse_report.txt       human-readable report (paste-able into the paper's
                          data-quality paragraph)
=============================================================================
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRICT_RE = re.compile(r"Answer:\s*\**\s*Option\s*(\d+)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{[^}]*?(\d+)[^}]*?\}")
ANSWER_IS_RE = re.compile(
    r"(?:answer|choice|select|choose|pick|go with)(?:\s+is)?\s*:?\s*\**\s*"
    r"(?:option\s*)?\(?(\d+)\)?", re.IGNORECASE)
FALLBACK_RE = re.compile(r"Option\s*\**\s*\(?(\d+)\)?", re.IGNORECASE)
BARE_RE = re.compile(r"^\s*\**\s*(\d{1,2})\s*\**\s*\.?\s*$")
DIGIT_RE = re.compile(r"\b(\d{1,2})\b")

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


def extract_answer(text: str, n_options: int):
    """Return (answer_idx or None, parse_method).

    Tiered, range-aware extraction: every tier only accepts values within
    1..n_options, so echoed stimulus numbers (e.g. an anchor like "74%")
    can never be mistaken for an answer.
    """
    if not text or not text.strip():
        return None, "empty"
    m = STRICT_RE.search(text)
    if m and 1 <= int(m.group(1)) <= n_options:
        return int(m.group(1)), "strict"
    for pattern, name in [(BOXED_RE, "boxed"), (ANSWER_IS_RE, "answer_is"),
                          (FALLBACK_RE, "fallback")]:
        vals = [int(x) for x in pattern.findall(text)
                if 1 <= int(x) <= n_options]
        if vals:
            return vals[-1], name  # last in-range mention = final answer
    last_line = text.strip().splitlines()[-1]
    m = BARE_RE.match(last_line)
    if m and 1 <= int(m.group(1)) <= n_options:
        return int(m.group(1)), "bare"
    if len(text.strip()) < 120:
        vals = [int(x) for x in DIGIT_RE.findall(text)
                if 1 <= int(x) <= n_options]
        if len(set(vals)) == 1:
            return vals[0], "short_digit"
    return None, "unparseable"


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_responses_dir"]
    out_dir = ROOT / cfg["paths"]["processing_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(" ANCHORED MINDS -- PARSE RESPONSES")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 70)

    files = sorted(raw_dir.glob("*__b*.jsonl"))
    if not files:
        log(f"\n[ERROR] No raw response files in {raw_dir.relative_to(ROOT)}.")
        log("        Run run_experiments.py first.")
        return 1
    log(f"\n[1/3] Found {len(files)} raw response files")

    rows, n_dupes, n_failed_calls = [], 0, 0
    seen_task_ids = set()
    for f in files:
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated line from a killed run; the call was re-run
            if rec.get("status") != "ok":
                n_failed_calls += 1
                continue
            if rec["task_id"] in seen_task_ids:
                n_dupes += 1
                continue  # keep first occurrence (rerun duplicates)
            seen_task_ids.add(rec["task_id"])
            answer_idx, method = extract_answer(rec.get("response_text", ""),
                                                rec["n_options"])
            usage = rec.get("usage") or {}
            rows.append({
                "task_id": rec["task_id"],
                "model": rec["model"],
                "model_id": rec["model_id"],
                "reasoning": rec["reasoning"],
                "pair": rec.get("pair"),
                "budget": rec["budget"],
                "item_id": rec["item_id"],
                "bias": rec["bias"],
                "condition": rec["condition"],
                "sample_idx": rec["sample_idx"],
                "n_options": rec["n_options"],
                "k": rec["k"],
                "flip_treatment": rec["flip_treatment"],
                "answer_idx": answer_idx,
                "parse_method": method,
                "parsed": answer_idx is not None,
                "latency_s": rec.get("latency_s"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details")
                                     or {}).get("reasoning_tokens"),
            })
    df = pd.DataFrame(rows)
    log(f"      Loaded {len(df):,} ok responses "
        f"({n_failed_calls} failed calls skipped, {n_dupes} duplicates dropped)")

    # -------------------------------------------------------- per-model stats
    # The gate is computed on ANALYZABLE responses only: budget-0 rows of
    # models flagged cannot_disable_thinking are excluded from scoring by
    # design, so they are reported separately and do not count against the
    # gate.
    no_disable = {m["name"] for m in cfg["models"]
                  if m.get("cannot_disable_thinking")}
    excluded_mask = df["model"].isin(no_disable) & (df["budget"] == 0)
    if excluded_mask.sum():
        exc = df[excluded_mask]
        log(f"\n      Note: {excluded_mask.sum():,} responses are in "
            f"design-excluded b0 cells of {sorted(no_disable)}")
        log(f"            (parse rate there: {exc['parsed'].mean()*100:.1f}%, "
            f"not counted toward the gate)")
    df_gate = df[~excluded_mask]

    log(f"\n[2/3] Parse rates on analyzable data (gate: >= 95% per model)")
    stats = {}
    gate_ok = True
    for model, sub in df_gate.groupby("model"):
        rate = sub["parsed"].mean() * 100
        methods = sub["parse_method"].value_counts().to_dict()
        stats[model] = {"n": len(sub), "parse_rate_pct": round(rate, 2),
                        "methods": methods}
        mark = "\u2713" if rate >= 95 else "\u2717"
        gate_ok &= rate >= 95
        log(f"      [{mark}] {model:<30} {rate:6.2f}%  (n={len(sub):,})")
    overall = df_gate["parsed"].mean() * 100
    log(f"      {'-' * 52}")
    log(f"          {'OVERALL (analyzable)':<30} {overall:6.2f}%  "
        f"(n={len(df_gate):,})")

    # show a few unparseable examples for debugging
    bad = df_gate[~df_gate["parsed"]]
    if len(bad):
        log(f"\n      {len(bad)} unparseable responses; first task_ids:")
        for tid in bad["task_id"].head(5):
            log(f"        - {tid}")

    # -------------------------------------------------------------------- save
    csv_path = out_dir / "parsed_responses.csv"
    df.to_csv(csv_path, index=False)
    log(f"\n[3/3] Saved parsed responses -> {csv_path.relative_to(ROOT)}")

    log_json = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_responses": len(df),
        "n_failed_calls_skipped": n_failed_calls,
        "n_duplicates_dropped": n_dupes,
        "overall_parse_rate_pct": round(overall, 2),
        "per_model": stats,
        "gate_95_passed": bool(gate_ok),
    }
    (out_dir / "parse_log.json").write_text(json.dumps(log_json, indent=2),
                                            encoding="utf-8")

    log("\n" + "=" * 70)
    if gate_ok:
        log(f" RESULT: PARSE GATE PASSED \u2713  (overall {overall:.2f}%)")
        log(" Next step: python 02_data_processing/compute_bias_scores.py")
    else:
        log(f" RESULT: PARSE GATE FAILED \u2717  -- inspect the unparseable "
            "responses above before proceeding.")
    log("=" * 70)

    (out_dir / "parse_report.txt").write_text("\n".join(LOG_LINES) + "\n",
                                              encoding="utf-8")
    print(f"\nReport saved -> {out_dir.relative_to(ROOT)}/parse_report.txt")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())