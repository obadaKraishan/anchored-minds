#!/usr/bin/env python3
"""
02_data_processing/purge_truncated.py

Data-collection utility (not part of the analysis pipeline). Removes
unparseable responses from the raw JSONL checkpoints so that
run_experiments.py re-collects exactly those calls. It was used during
collection to re-collect responses whose visible answer had been truncated
by long deliberation. Budget-0 responses of models flagged
cannot_disable_thinking are kept, since those cells are excluded from
analysis by design.

WARNING: modifies files in 01_data_collection/outputs/raw_responses/.
Running it on the released data is unnecessary.

Workflow:
  1. python 02_data_processing/parse_responses.py
  2. python 02_data_processing/purge_truncated.py
  3. python 01_data_collection/run_experiments.py      (fills the gaps)
  4. re-run parse -> scores -> validate

Idempotent; prints exactly what it removes.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_responses_dir"]
    parsed_path = ROOT / cfg["paths"]["processing_dir"] / "parsed_responses.csv"
    if not parsed_path.exists():
        print("[ERROR] parsed_responses.csv not found. Run parse_responses.py "
              "first (the purge decision is based on the CURRENT parser).")
        return 1

    df = pd.read_csv(parsed_path)
    no_disable = {m["name"] for m in cfg["models"]
                  if m.get("cannot_disable_thinking")}

    bad = df[~df["parsed"]].copy()
    keep_excluded = bad["model"].isin(no_disable) & (bad["budget"] == 0)
    purge = bad[~keep_excluded]
    purge_ids = set(purge["task_id"])

    print("=" * 70)
    print(" ANCHORED MINDS -- PURGE TRUNCATED/UNPARSEABLE RESPONSES")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print(f" Unparseable responses           : {len(bad):,}")
    print(f"   in excluded b0 cells (kept)   : {keep_excluded.sum():,}  "
          f"(models: {sorted(no_disable) if no_disable else 'none flagged'})")
    print(f"   to purge for re-collection    : {len(purge_ids):,}")
    if len(purge):
        print("\n   purge breakdown by model x budget:")
        for (m, b), n in purge.groupby(["model", "budget"]).size().items():
            print(f"     {m:<28} b{b:<6} {n:>5}")

    if not purge_ids:
        print("\n Nothing to purge \u2713")
        return 0

    total_removed = 0
    for f in sorted(raw_dir.glob("*__b*.jsonl")):
        lines = f.read_text().splitlines()
        kept = []
        removed = 0
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # drop truncated junk lines too
            if rec.get("task_id") in purge_ids:
                removed += 1
            else:
                kept.append(line)
        if removed:
            f.write_text("\n".join(kept) + ("\n" if kept else ""),
                         encoding="utf-8")
            print(f"   {f.name:<44} -{removed}")
            total_removed += removed

    print(f"\n Removed {total_removed:,} responses from checkpoints.")
    print(" Next: python 01_data_collection/run_experiments.py")
    print("       (re-collects exactly these calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
