#!/usr/bin/env python3
"""
04_analysis/rq3_bias_taxonomy/rq3_analysis.py

RQ3: Which biases are present in the human direction, which are reversed,
and does any respond to deliberation?

Only the five items of a bias are independent, so every per-bias test is a
t-test with 4 degrees of freedom on the five item means, Holm-corrected
across the six biases.

Part A (presence). Signed score s over all model x budget cells of the bias
(85 cells). Reported: cell mean M, item-clustered bootstrap 95% CI, t(4) on
the five item means, Holm p, and d = cell-level standardized mean (M / SD).
Classification:
  Present      p_Holm < .05 and M > 0 (human direction)
  Reversed     p_Holm < .05 and M < 0
  Directional  not significant, but the mean score has the same sign in
               every model
  Absent       otherwise

Part B (deliberation sensitivity). Within each reasoning model, the per-item
OLS slope of |s| on log2(realized reasoning tokens + 1); slopes are averaged
over models within each item, and the five item means are tested against 0
with t(4), Holm-corrected across the six biases.

Inputs: 02_data_processing/outputs/master_scores.csv, data/per_call_usage.csv
Usage:  python 04_analysis/rq3_bias_taxonomy/rq3_analysis.py

Saves (in outputs/): rq3_presence.csv, rq3_sensitivity.csv,
                     rq3_report.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import (load_master, load_usage, attach_realized_tokens,
                            item_clustered_ci, holm_correction,
                            per_item_slopes, fmt_p_bare, table,
                            Report)

OUT = HERE / "outputs"


def classify(p_holm, mean, model_means):
    if p_holm < .05:
        return "Present" if mean > 0 else "Reversed"
    if (np.sign(model_means) == np.sign(mean)).all():
        return "Directional"
    return "Absent"


def main():
    df, cfg = load_master()
    df = attach_realized_tokens(df, load_usage(cfg))
    biases = cfg["biases"]
    rep = Report()

    rep("=" * 76)
    rep(" RQ3 -- BIAS TAXONOMY: PRESENCE AND DELIBERATION SENSITIVITY")
    rep("=" * 76)

    # ----------------------------------------------------- Part A: presence --
    rep("\n--- Part A: signed score vs 0 per bias (t(4) on five item means) ---")
    pres, model_means_by_bias = [], {}
    for bias in biases:
        cells = df[df["bias"] == bias].dropna(subset=["bias_score"])
        item_means = cells.groupby("item_id")["bias_score"].mean()
        model_means = cells.groupby("model")["bias_score"].mean()
        model_means_by_bias[bias] = model_means.values
        t, p = stats.ttest_1samp(item_means, 0)
        lo, hi = item_clustered_ci(cells, "bias_score")
        mean = cells["bias_score"].mean()
        pres.append(dict(
            bias=bias, n_cells=len(cells), n_items=len(item_means),
            mean=mean, ci_lo=lo, ci_hi=hi, t=float(t),
            df=len(item_means) - 1, p=float(p),
            d=mean / cells["bias_score"].std(ddof=1),
            n_models=len(model_means),
            n_models_same_sign=int((np.sign(model_means)
                                    == np.sign(mean)).sum()),
            model_min=model_means.min(), model_max=model_means.max()))
    pres = pd.DataFrame(pres)
    pres["p_holm"] = holm_correction(pres["p"])
    pres["classification"] = [
        classify(r.p_holm, r.mean, model_means_by_bias[r.bias])
        for r in pres.itertuples()]

    rows = [[r.bias, f"{r.mean:+.3f}", f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
             f"{r.t:.2f}", r.df, fmt_p_bare(r.p), fmt_p_bare(r.p_holm),
             f"{r.d:+.2f}",
             f"{r.n_models_same_sign}/{r.n_models}",
             f"{r.model_min:+.2f}..{r.model_max:+.2f}", r.classification]
            for r in pres.itertuples()]
    rep("\n" + table(rows, ["bias", "M", "95% CI (item-clustered)", "t", "df",
                            "p", "p_holm", "d", "models same sign",
                            "model means", "class"]))

    # ---------------------------------------- Part B: deliberation slopes --
    rep("\n--- Part B: slope of |s| on log2(realized tokens + 1) per bias ---")
    rep(" (per model x item slopes within reasoning models, averaged over")
    rep("  models per item; t(4) on the five item means; Holm across biases)")
    sens = []
    reasoning = df[df["reasoning"] & df["realized_tokens"].notna()]
    for bias in biases:
        sub = reasoning[reasoning["bias"] == bias]
        slopes = per_item_slopes(sub, "log_realized", "abs_bias",
                                 by=("model", "item_id"))
        item_means = slopes.groupby(level=1).mean()
        t, p = stats.ttest_1samp(item_means, 0)
        sens.append(dict(bias=bias, n_slopes=len(slopes),
                         n_items=len(item_means),
                         slope_mean=float(item_means.mean()),
                         t=float(t), df=len(item_means) - 1, p=float(p)))
    sens = pd.DataFrame(sens)
    sens["p_holm"] = holm_correction(sens["p"])
    sens["classification"] = np.where(
        sens["p_holm"] < .05,
        np.where(sens["slope_mean"] < 0, "decreases with deliberation",
                 "increases with deliberation"),
        "not significant")
    rows = [[r.bias, f"{r.slope_mean:+.4f}", f"{r.t:.2f}", r.df,
             fmt_p_bare(r.p), fmt_p_bare(r.p_holm), r.classification]
            for r in sens.itertuples()]
    rep("\n" + table(rows, ["bias", "slope", "t", "df", "p", "p_holm",
                            "result"]))
    n_null = (sens["classification"] == "not significant").sum()
    rep(f"\n {n_null}/{len(sens)} per-bias slope tests null after Holm "
        "correction.")

    OUT.mkdir(parents=True, exist_ok=True)
    pres.to_csv(OUT / "rq3_presence.csv", index=False)
    sens.to_csv(OUT / "rq3_sensitivity.csv", index=False)
    rep("\nSaved: rq3_presence.csv, rq3_sensitivity.csv, rq3_report.txt")
    rep.save(OUT / "rq3_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
