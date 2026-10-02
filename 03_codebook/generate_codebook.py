#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 03_codebook/generate_codebook.py
=============================================================================
 Auto-generates the variable codebook for master_scores.csv: every column's
 name, type, observed range/values, and a curated description of its
 derivation. Ships in the repo (codebook.md) and feeds the paper's Method.

 Usage:
   python 03_codebook/generate_codebook.py

 Saves (in 03_codebook/outputs/): codebook.csv, codebook.md
=============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DESCRIPTIONS = {
    "model": "Model panel name from config.yaml (7 models, 4 families).",
    "reasoning": "True if the model supports a controllable thinking budget.",
    "pair": "Model family grouping a reasoning model with its non-reasoning "
            "sibling (claude / openai / deepseek / qwen).",
    "budget": "Reasoning-token ceiling for the cell (0 = extended thinking "
              "disabled; models flagged cannot_disable_thinking contribute "
              "budgets > 0 only).",
    "item_id": "Frozen battery item identifier (e.g. ANCH-003).",
    "bias": "Cognitive bias targeted by the item (Malberg et al. 2024 "
            "benchmark label).",
    "n_options": "Number of ordinal answer options on the item's scale "
                 "(7 or 11).",
    "k": "Direction parameter from the benchmark's metric_params: +1/-1 so "
         "that positive bias_score always means a shift in the "
         "bias-predicted direction.",
    "mean_control": "Mean chosen option index across samples, control "
                    "condition (no bias manipulation).",
    "mean_treatment": "Mean chosen option index across samples, treatment "
                      "condition (bias manipulation present; scale reversed "
                      "first when flip_treatment is set).",
    "mean_treatment_verbalized": "Mean chosen option index, verbalized "
                                 "treatment (RQ4; Anchoring items only).",
    "std_control": "SD of chosen option index, control condition.",
    "std_treatment": "SD of chosen option index, treatment condition.",
    "std_treatment_verbalized": "SD, verbalized treatment condition.",
    "count_control": "Parsed samples in the control condition (target 10).",
    "count_treatment": "Parsed samples in the treatment condition "
                       "(target 10).",
    "count_treatment_verbalized": "Parsed samples, verbalized treatment.",
    "bias_score": "Primary DV: k * (mean_treatment - mean_control) / "
                  "(n_options - 1), bounded [-1, +1]. Positive = shift in "
                  "the bias-predicted (human-like) direction; negative = "
                  "shift against it (reversed).",
    "bias_score_verbalized": "Same metric computed with the verbalized "
                             "treatment condition (RQ4).",
    "abs_bias": "Derived in analysis: |bias_score|, the susceptibility "
                "magnitude irrespective of direction.",
    "flip_treatment": "Benchmark flag: treatment answer scale is mirrored "
                      "before scoring.",
}


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    master = ROOT / cfg["paths"]["processing_dir"] / "master_scores.csv"
    out_dir = ROOT / cfg["paths"]["codebook_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not master.exists():
        print("[ERROR] master_scores.csv not found -- run 02_data_processing "
              "first.")
        return 1

    df = pd.read_csv(master)
    print("=" * 70)
    print(" ANCHORED MINDS -- GENERATE CODEBOOK")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print(f" Source: {master.relative_to(ROOT)} "
          f"({df.shape[0]} rows x {df.shape[1]} cols)")
    print("=" * 70)

    rows = []
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in "if":
            rng = f"[{s.min():.3f}, {s.max():.3f}]"
        else:
            vals = s.dropna().unique()
            rng = (", ".join(map(str, sorted(vals)[:6]))
                   + (", ..." if len(vals) > 6 else ""))
        rows.append({
            "variable": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "range_or_values": rng,
            "description": DESCRIPTIONS.get(col, "(add description)"),
        })
        print(f"  {col:<28} {str(s.dtype):<9} missing={s.isna().sum():<4} "
              f"{rng[:44]}")

    cb = pd.DataFrame(rows)
    cb.to_csv(out_dir / "codebook.csv", index=False)

    md = ["# Anchored Minds -- Variable Codebook",
          f"\nGenerated {datetime.now(timezone.utc).date()} from "
          f"`master_scores.csv` ({df.shape[0]} cells).\n",
          "| Variable | Type | Missing | Range / values | Description |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['variable']}` | {r['dtype']} | {r['n_missing']} | "
                  f"{r['range_or_values']} | {r['description']} |")
    (out_dir / "codebook.md").write_text("\n".join(md) + "\n",
                                         encoding="utf-8")

    print("\n Saved: codebook.csv, codebook.md ->",
          out_dir.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
