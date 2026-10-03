#!/usr/bin/env python3
"""
02_data_processing/extract_usage.py

Flattens the token usage reported by the API gateway for every call into a
single CSV (no response text). Realized reasoning tokens from this file are
the dose variable for the manipulation check, RQ2, and RQ3.

reasoning_tokens is read from usage.completion_tokens_details.reasoning_tokens
(falling back to usage.reasoning_tokens); it is 0 for calls without thinking.

Usage:
  python 02_data_processing/extract_usage.py

Saves:
  data/per_call_usage.csv   one row per API call (12,350 rows):
                            model, budget, item_id, condition, sample_idx,
                            prompt_tokens, completion_tokens, reasoning_tokens
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ["model", "budget", "item_id", "condition", "sample_idx",
           "prompt_tokens", "completion_tokens", "reasoning_tokens"]


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_responses_dir"]
    out_path = ROOT / cfg["paths"]["usage_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*__b*.jsonl"))
    if not files:
        print(f"[ERROR] No raw response files in {raw_dir.relative_to(ROOT)}.")
        return 1

    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = rec.get("usage") or {}
            details = u.get("completion_tokens_details") or {}
            rows.append({
                "model": rec["model"],
                "budget": rec["budget"],
                "item_id": rec["item_id"],
                "condition": rec["condition"],
                "sample_idx": rec["sample_idx"],
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens",
                                                u.get("reasoning_tokens")),
            })

    df = (pd.DataFrame(rows, columns=COLUMNS)
            .sort_values(["model", "budget", "item_id", "condition",
                          "sample_idx"])
            .reset_index(drop=True))
    for c in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[c] = df[c].astype("Int64")
    df.to_csv(out_path, index=False)

    print(f"Read {len(files)} raw files -> {len(df):,} calls "
          f"(reasoning_tokens present for "
          f"{df['reasoning_tokens'].notna().sum():,})")
    print(f"Saved -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
