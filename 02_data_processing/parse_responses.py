#!/usr/bin/env python3
"""
02_data_processing/parse_responses.py

Extracts the chosen answer option from every raw API response.

Tiered, range-aware parsing (first tier that yields a value wins):
  1. strict      : "Answer: Option N"
  2. boxed       : \\boxed{N}
  3. answer_is   : "answer is N", "I choose option N", ...
  4. fallback    : any "Option N" mention
  5. bare        : a lone integer on the final line
  6. short_digit : a single distinct integer in a response under 120 chars
Tiers 2-4 take the LAST in-range mention. Every tier only accepts values in
1..n_options, so echoed stimulus numbers cannot be mistaken for answers.

The per-model parse rate (target >= 95%) is computed on analyzable cells,
i.e. excluding the budget-0 cells of models flagged cannot_disable_thinking.
A model below the target produces a warning, not a failure: its unparsed
responses are excluded from scoring (in the released data, Qwen3-Instruct at
92.8%, which often answered with a raw quantity instead of an option index).

Exit status: 0 on success (including parse-rate warnings); 1 if no raw files
are found, a file contains no readable records, or a record lacks required
fields.

Usage:
  python 02_data_processing/parse_responses.py

Saves (in 02_data_processing/outputs/):
  parsed_responses.csv   one row per API call, with answer_idx + parse_method
  parse_log.json         machine-readable per-model stats
  parse_report.txt       human-readable report
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

REQUIRED_FIELDS = ["task_id", "model", "model_id", "reasoning", "budget",
                   "item_id", "bias", "condition", "sample_idx", "n_options",
                   "k", "flip_treatment"]
PARSE_RATE_TARGET = 95.0

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

    rows, n_dupes, n_failed_calls, n_malformed = [], 0, 0, 0
    seen_task_ids = set()
    unreadable_files, incomplete = [], []
    for f in files:
        n_valid = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_malformed += 1  # e.g. a truncated line from an interrupted run
                continue
            n_valid += 1
            if rec.get("status") != "ok":
                n_failed_calls += 1
                continue
            if rec["task_id"] in seen_task_ids:
                n_dupes += 1
                continue  # keep first occurrence (rerun duplicates)
            missing = [k for k in REQUIRED_FIELDS if k not in rec]
            if missing:
                incomplete.append((f.name, rec.get("task_id"), missing))
                continue
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
        if n_valid == 0:
            unreadable_files.append(f.name)

    if unreadable_files or incomplete or not rows:
        log("\n[ERROR] Raw data could not be read:")
        for name in unreadable_files:
            log(f"        {name}: no readable records")
        for name, tid, missing in incomplete[:5]:
            log(f"        {name}: record {tid} lacks fields {missing}")
        if len(incomplete) > 5:
            log(f"        ... {len(incomplete) - 5} more incomplete records")
        if not rows:
            log("        no successful responses found")
        return 1

    df = pd.DataFrame(rows)
    log(f"      Loaded {len(df):,} ok responses "
        f"({n_failed_calls} failed calls skipped, {n_dupes} duplicates dropped)")
    if n_malformed:
        log(f"      [!] WARNING: {n_malformed} malformed JSON line(s) skipped")

    # -------------------------------------------------------- per-model stats
    # Parse rates are computed on ANALYZABLE responses only: budget-0 rows of
    # models flagged cannot_disable_thinking are excluded from scoring by
    # design, so they are reported separately.
    no_disable = {m["name"] for m in cfg["models"]
                  if m.get("cannot_disable_thinking")}
    excluded_mask = df["model"].isin(no_disable) & (df["budget"] == 0)
    if excluded_mask.sum():
        exc = df[excluded_mask]
        log(f"\n      Note: {excluded_mask.sum():,} responses are in "
            f"design-excluded b0 cells of {sorted(no_disable)}")
        log(f"            (parse rate there: {exc['parsed'].mean()*100:.1f}%, "
            f"not counted toward the parse rates below)")
    df_an = df[~excluded_mask]

    log(f"\n[2/3] Parse rates on analyzable data (target: >= "
        f"{PARSE_RATE_TARGET:.0f}% per model)")
    stats, below = {}, {}
    for model, sub in df_an.groupby("model"):
        rate = sub["parsed"].mean() * 100
        methods = sub["parse_method"].value_counts().to_dict()
        stats[model] = {"n": len(sub), "parse_rate_pct": round(rate, 2),
                        "methods": methods}
        mark = "\u2713" if rate >= PARSE_RATE_TARGET else "!"
        if rate < PARSE_RATE_TARGET:
            below[model] = rate
        log(f"      [{mark}] {model:<30} {rate:6.2f}%  (n={len(sub):,})")
    overall = df_an["parsed"].mean() * 100
    log(f"      {'-' * 52}")
    log(f"          {'OVERALL (analyzable)':<30} {overall:6.2f}%  "
        f"(n={len(df_an):,})")

    # show a few unparseable examples for inspection
    bad = df_an[~df_an["parsed"]]
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
        "parse_rate_target_pct": PARSE_RATE_TARGET,
        "models_below_target": {m: round(r, 2) for m, r in below.items()},
    }
    (out_dir / "parse_log.json").write_text(json.dumps(log_json, indent=2),
                                            encoding="utf-8")

    log("\n" + "=" * 70)
    log(f" RESULT: PARSING COMPLETE \u2713  (overall {overall:.2f}%)")
    if below:
        log(f" [!] WARNING: {len(below)} model(s) below the "
            f"{PARSE_RATE_TARGET:.0f}% parse-rate target: "
            + ", ".join(f"{m} ({r:.2f}%)" for m, r in below.items()))
        log("     Unparsed responses are excluded from scoring. In the "
            "released data this")
        log("     is the documented Qwen3-Instruct exclusion (raw quantities "
            "instead of an")
        log("     option index); for new data, inspect the task_ids above.")
    log(" Next step: python 02_data_processing/compute_bias_scores.py")
    log("=" * 70)

    (out_dir / "parse_report.txt").write_text("\n".join(LOG_LINES) + "\n",
                                              encoding="utf-8")
    print(f"\nReport saved -> {out_dir.relative_to(ROOT)}/parse_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())