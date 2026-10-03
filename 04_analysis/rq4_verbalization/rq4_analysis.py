#!/usr/bin/env python3
"""
04_analysis/rq4_verbalization/rq4_analysis.py

RQ4: Does forcing the model to restate the anchor before answering change
anchoring?

Every Anchoring cell (model x budget x item, 85 cells) has a standard
treatment score s and a verbalized-treatment score s_v against the same
control. The effect is s_v - s per cell (negative = verbalization reduces
anchoring). Reported:
  - pooled cell mean with a 95% bootstrap CI over cells;
  - item level (the independent unit): the five item means, t(4) against 0,
    and a two-sided sign test on the number of negative items;
  - per model: mean s and s_v, and how many models move in the negative
    direction.

Input:  02_data_processing/outputs/master_scores.csv
Usage:  python 04_analysis/rq4_verbalization/rq4_analysis.py

Saves (in outputs/): rq4_items.csv, rq4_models.csv, rq4_summary.csv,
                     rq4_report.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from scipy import stats

from analysis_utils import load_master, bootstrap_ci_mean, fmt_p, table, \
    Report

OUT = HERE / "outputs"


def main():
    df, _ = load_master()
    rep = Report()

    rep("=" * 76)
    rep(" RQ4 -- FORCED ANCHOR VERBALIZATION (Anchoring items)")
    rep("=" * 76)

    a = df[(df["bias"] == "Anchoring")
           & df["bias_score_verbalized"].notna()].copy()
    if not len(a):
        rep("[ERROR] No verbalized scores found.")
        return 1
    a["effect"] = a["bias_score_verbalized"] - a["bias_score"]

    # ------------------------------------------------------- pooled cells --
    ci = bootstrap_ci_mean(a["effect"])
    rep(f"\n Cells with both conditions: {len(a)}")
    rep(f" Pooled over cells: Delta = {a['effect'].mean():+.3f}, 95% "
        f"bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # ---------------------------------------------------------- item level --
    items = (a.groupby("item_id")
              .agg(n_cells=("effect", "size"),
                   mean_standard=("bias_score", "mean"),
                   mean_verbalized=("bias_score_verbalized", "mean"),
                   effect=("effect", "mean"))
              .reset_index())
    t, p = stats.ttest_1samp(items["effect"], 0)
    n_neg = int((items["effect"] < 0).sum())
    sign_p = stats.binomtest(n_neg, len(items), 0.5).pvalue
    rep("\n Item level:")
    rep("\n" + table([[r.item_id, r.n_cells, f"{r.mean_standard:+.3f}",
                       f"{r.mean_verbalized:+.3f}", f"{r.effect:+.3f}"]
                      for r in items.itertuples()],
                     ["item", "cells", "M_standard", "M_verbalized",
                      "Delta"]))
    rep(f"\n   item means range {items['effect'].min():+.3f} to "
        f"{items['effect'].max():+.3f}; t({len(items) - 1}) = {t:.2f}, "
        f"{fmt_p(p)}")
    rep(f"   sign test: {n_neg}/{len(items)} items negative, two-sided "
        f"{fmt_p(sign_p)}")

    # ---------------------------------------------------------- per model --
    models = (a.groupby("model")
               .agg(n_cells=("effect", "size"),
                    mean_standard=("bias_score", "mean"),
                    mean_verbalized=("bias_score_verbalized", "mean"),
                    effect=("effect", "mean"))
               .reset_index().sort_values("effect"))
    rep("\n Per model:")
    rep("\n" + table([[r.model, r.n_cells, f"{r.mean_standard:+.3f}",
                       f"{r.mean_verbalized:+.3f}", f"{r.effect:+.3f}"]
                      for r in models.itertuples()],
                     ["model", "cells", "M_standard", "M_verbalized",
                      "Delta"]))
    n_neg_models = int((models["effect"] < 0).sum())
    rep(f"\n   {n_neg_models}/{len(models)} models lower with verbalization")

    summary = pd.DataFrame([dict(
        n_cells=len(a), pooled_mean=a["effect"].mean(),
        pooled_ci_lo=ci[0], pooled_ci_hi=ci[1], n_items=len(items),
        item_min=items["effect"].min(), item_max=items["effect"].max(),
        t=float(t), df=len(items) - 1, p=float(p),
        n_items_negative=n_neg, sign_test_p=float(sign_p),
        n_models=len(models), n_models_negative=n_neg_models)])

    OUT.mkdir(parents=True, exist_ok=True)
    items.to_csv(OUT / "rq4_items.csv", index=False)
    models.to_csv(OUT / "rq4_models.csv", index=False)
    summary.to_csv(OUT / "rq4_summary.csv", index=False)
    rep("\nSaved: rq4_items.csv, rq4_models.csv, rq4_summary.csv, "
        "rq4_report.txt")
    rep.save(OUT / "rq4_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
