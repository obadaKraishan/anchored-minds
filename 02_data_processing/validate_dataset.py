#!/usr/bin/env python3
"""
02_data_processing/validate_dataset.py

Final data-quality audit before the master file is frozen for analysis.

Checks:
  1. GRID COMPLETENESS -- every expected model x budget x item cell exists
     (reasoning models: all budgets; non-reasoning: budget 0 only; models
     that cannot disable thinking: budgets > 0)
  2. CELL SIZE -- cells below 80% of n_samples_per_cell in control or
     treatment are reported; fatal only if a cell has < 2 samples in a
     condition or undersized cells exceed 5% of the grid
  3. RANGE -- all bias scores within [-1, +1]; answer means within scale
  4. VARIANCE -- flags degenerate cells (SD == 0 in both conditions may
     indicate deterministic collapse; informational, not fatal)
  5. RQ4 COVERAGE -- verbalized condition present for all Anchoring cells

Exit code 0 = dataset frozen and analysis-ready. Non-zero = fix first.

Usage:
  python 02_data_processing/validate_dataset.py

Saves:
  02_data_processing/outputs/validation_report.txt
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_LINES = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "\u2713" if ok else "\u2717"
    log(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_dir = ROOT / cfg["paths"]["processing_dir"]
    master_path = out_dir / "master_scores.csv"
    if not master_path.exists():
        log("[ERROR] master_scores.csv not found. Run compute_bias_scores.py "
            "first.")
        return 1

    log("=" * 70)
    log(" ANCHORED MINDS -- VALIDATE DATASET")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 70)

    m = pd.read_csv(master_path)
    n_samples = cfg["experiment"]["n_samples_per_cell"]
    min_n = int(0.8 * n_samples)
    all_ok = True

    # ------------------------------------------------------ 1 grid completeness
    log("\n[1/5] Grid completeness")
    battery_n = cfg["experiment"]["n_items_per_bias"] * len(cfg["biases"])
    for mod in cfg["models"]:
        budgets = cfg["budgets"] if mod["reasoning"] else [0]
        if mod.get("cannot_disable_thinking"):
            budgets = [b for b in budgets if b != 0]
        expected = len(budgets) * battery_n
        got = len(m[m["model"] == mod["name"]])
        all_ok &= check(f"{mod['name']:<30} {got}/{expected} cells",
                        got == expected)

    # ------------------------------------------------------------- 2 cell sizes
    # Undersized cells (below 80% of target samples) are a WARNING as long as
    # (a) every cell keeps >= 2 samples per condition and (b) undersized cells
    # are <= 5% of the grid. Beyond that they are fatal. Exclusions (e.g. a
    # raw quantity instead of an option index) are reported in the paper.
    log(f"\n[2/5] Cell sizes (target >= {min_n}/{n_samples} per condition)")
    low = m[(m["count_control"] < min_n) | (m["count_treatment"] < min_n)]
    fatal_low = m[(m["count_control"] < 2) | (m["count_treatment"] < 2)]
    share = len(low) / max(len(m), 1)
    if len(low):
        log(f"      {len(low)} undersized cells ({share:.1%} of grid):")
        for _, r in low.head(10).iterrows():
            log(f"        {r['model']:<26} b{r['budget']:<6} {r['item_id']} "
                f"(ctrl n={int(r['count_control'])}, "
                f"trt n={int(r['count_trt'] if 'count_trt' in r else r['count_treatment'])})")
    all_ok &= check(f"no cell below 2 samples/condition (fatal threshold)",
                    len(fatal_low) == 0,
                    str(fatal_low["item_id"].tolist()[:5]) if len(fatal_low) else "")
    all_ok &= check(f"undersized cells <= 5% of grid",
                    share <= 0.05, f"{share:.1%}")
    if len(low) and share <= 0.05 and len(fatal_low) == 0:
        log("      -> treated as WARNING")

    # ------------------------------------------------------------------ 3 range
    log("\n[3/5] Value ranges")
    oob = m[(m["bias_score"] < -1) | (m["bias_score"] > 1)]
    all_ok &= check("bias_score within [-1, +1]", len(oob) == 0,
                    f"{len(oob)} out of range" if len(oob) else "")
    bad_mean = m[(m["mean_control"] < 1)
                 | (m["mean_control"] > m["n_options"])]
    all_ok &= check("answer means within option scale", len(bad_mean) == 0)

    # --------------------------------------------------------------- 4 variance
    log("\n[4/5] Degenerate cells (informational)")
    degen = m[(m["std_control"].fillna(0) == 0)
              & (m["std_treatment"].fillna(0) == 0)]
    check(f"{len(degen)}/{len(m)} cells with zero variance in both conditions",
          True, "deterministic answers; informational")

    # ------------------------------------------------------------ 5 rq4 coverage
    log("\n[5/5] RQ4 coverage (verbalized condition on Anchoring)")
    if "bias_score_verbalized" in m.columns:
        anch = m[m["bias"] == "Anchoring"]
        n_missing = anch["bias_score_verbalized"].isna().sum()
        all_ok &= check(f"Anchoring cells with verbalized score: "
                        f"{len(anch) - n_missing}/{len(anch)}", n_missing == 0)
    else:
        all_ok &= check("bias_score_verbalized column present", False,
                        "RQ4 condition missing from data")

    # ---------------------------------------------------------------- summary
    log("\n" + "=" * 70)
    if all_ok:
        log(" RESULT: DATASET VALID \u2713 -- master_scores.csv is FROZEN for "
            "analysis")
        log(" Next steps: python 03_codebook/generate_codebook.py")
        log("             python 04_analysis/rq1_reasoning_vs_bias/"
            "rq1_analysis.py")
    else:
        log(" RESULT: VALIDATION FAILED \u2717 -- fix the flagged issues, or if "
            "cells are")
        log(" missing, rerun run_experiments.py (checkpointing fills only "
            "gaps).")
    log("=" * 70)

    (out_dir / "validation_report.txt").write_text("\n".join(LOG_LINES) + "\n",
                                                   encoding="utf-8")
    print(f"\nReport saved -> {out_dir.relative_to(ROOT)}/validation_report.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())