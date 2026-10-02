#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 02_data_processing/compute_bias_scores.py
=============================================================================
 Computes the normalized bias-magnitude score for every design cell and
 produces the MASTER ANALYSIS FILE that all four RQ scripts read.

 Metric (transparent, direction-adjusted, bounded):

     bias_score = k * (mean(answer_treatment) - mean(answer_control))
                  / (n_options - 1)                              in [-1, +1]

 where k in {-1, +1} is the benchmark's per-item direction parameter
 (metric_params from Malberg et al., arXiv:2410.15413) so that POSITIVE
 scores always mean "shifted in the bias-predicted direction". Items with
 flip_treatment=True have their treatment scale reversed before scoring,
 per the benchmark's specification.

 Output granularity: one row per model x budget x item (aggregating the
 n_samples repeated draws), with cell means, SDs, ns, and the bias score.
 A second file keeps the sample-level parsed data for mixed-effects models.

 RQ4 note: for Anchoring items, the same computation is repeated with the
 'treatment_verbalized' condition -> bias_score_verbalized column.

 Usage:
   python 02_data_processing/compute_bias_scores.py

 Saves (in 02_data_processing/outputs/):
   master_scores.csv / master_scores.json   (cell-level: THE analysis file)
   scoring_report.txt                       (descriptives log)
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


def main() -> int:
    import yaml
    import numpy as np
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_dir = ROOT / cfg["paths"]["processing_dir"]
    parsed_path = out_dir / "parsed_responses.csv"
    if not parsed_path.exists():
        log("[ERROR] parsed_responses.csv not found. Run parse_responses.py "
            "first.")
        return 1

    log("=" * 70)
    log(" ANCHORED MINDS -- COMPUTE BIAS SCORES")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 70)

    df = pd.read_csv(parsed_path)
    df = df[df["parsed"]].copy()
    log(f"\n[1/3] Loaded {len(df):,} parsed responses")

    # Exclude budget-0 cells for models that cannot disable extended thinking
    # (flagged cannot_disable_thinking in config.yaml): for these models a
    # "0-token" condition is ill-defined -- the family's non-reasoning sibling
    # provides the zero point instead. Documented in the paper's Method.
    no_disable = [m["name"] for m in cfg["models"]
                  if m.get("cannot_disable_thinking")]
    if no_disable:
        mask = df["model"].isin(no_disable) & (df["budget"] == 0)
        log(f"      EXCLUDED {mask.sum():,} budget-0 responses from "
            f"{no_disable} (cannot disable thinking; sibling models provide "
            f"the b0 point)")
        df = df[~mask].copy()

    # flip treatment scales where the benchmark says so
    flip = df["flip_treatment"] & df["condition"].str.startswith("treatment")
    df.loc[flip, "answer_idx"] = (df.loc[flip, "n_options"] + 1
                                  - df.loc[flip, "answer_idx"])
    log(f"      Applied flip_treatment reversal to {flip.sum():,} responses")

    # ------------------------------------------------- cell-level aggregation
    log("\n[2/3] Aggregating to model x budget x item cells")
    keys = ["model", "reasoning", "pair", "budget", "item_id", "bias",
            "n_options", "k"]
    agg = (df.groupby(keys + ["condition"])["answer_idx"]
             .agg(["mean", "std", "count"]).reset_index())
    wide = agg.pivot_table(index=keys, columns="condition",
                           values=["mean", "std", "count"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    denom = wide["n_options"] - 1
    wide["bias_score"] = (wide["k"] * (wide["mean_treatment"]
                                       - wide["mean_control"]) / denom)
    if "mean_treatment_verbalized" in wide.columns:
        wide["bias_score_verbalized"] = (
            wide["k"] * (wide["mean_treatment_verbalized"]
                         - wide["mean_control"]) / denom)

    n_cells = len(wide)
    log(f"      {n_cells:,} cells "
        f"({wide['model'].nunique()} models x budgets x "
        f"{wide['item_id'].nunique()} items)")

    # ------------------------------------------------------------ descriptives
    log("\n      Mean bias score by model x budget "
        "(positive = bias-direction shift):")
    desc = (wide.groupby(["model", "budget"])["bias_score"]
                .agg(["mean", "std", "count"]))
    for (model, budget), row in desc.iterrows():
        log(f"        {model:<30} b{budget:<6} "
            f"M={row['mean']:+.3f}  SD={row['std']:.3f}  n={int(row['count'])}")

    log("\n      Mean bias score by bias type (all models/budgets pooled):")
    for bias, row in (wide.groupby("bias")["bias_score"]
                          .agg(["mean", "std"]).iterrows()):
        log(f"        {bias:<28} M={row['mean']:+.3f}  SD={row['std']:.3f}")

    # -------------------------------------------------------------------- save
    csv_path = out_dir / "master_scores.csv"
    wide.to_csv(csv_path, index=False)
    (out_dir / "master_scores.json").write_text(
        wide.to_json(orient="records", indent=2), encoding="utf-8")
    log(f"\n[3/3] MASTER FILE saved -> {csv_path.relative_to(ROOT)}")
    log("      (all rq*_analysis.py scripts read only this file)")

    log("\n" + "=" * 70)
    log(" RESULT: SCORING COMPLETE \u2713")
    log(" Next step: python 02_data_processing/validate_dataset.py")
    log("=" * 70)

    (out_dir / "scoring_report.txt").write_text("\n".join(LOG_LINES) + "\n",
                                                encoding="utf-8")
    print(f"\nReport saved -> {out_dir.relative_to(ROOT)}/scoring_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
