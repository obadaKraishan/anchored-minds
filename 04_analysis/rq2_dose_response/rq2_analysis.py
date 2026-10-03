#!/usr/bin/env python3
"""
04_analysis/rq2_dose_response/rq2_analysis.py

RQ2: Does bias fall as realized deliberation grows (dose-response)?

The dose is the REALIZED reasoning tokens of each model x budget x item cell
(mean over the cell's calls; see the manipulation check), not the requested
ceiling, analyzed on log2(realized tokens + 1) within the range each model
actually produced. Per reasoning model:
  1. per-item OLS slope of |s| on log2(realized + 1), tested against 0
     across the 30 items (one-sample t(29));
  2. mixed models |s| ~ log2(realized + 1) and s ~ log2(realized + 1) with a
     random item intercept (beta per doubling of realized tokens).
Budget-0 cells of models that cannot disable thinking are not in the scored
dataset, so those models contribute budgets > 0 only.

Also reports the descriptive mean signed score per model x budget.

Inputs: 02_data_processing/outputs/master_scores.csv, data/per_call_usage.csv
Usage:  python 04_analysis/rq2_dose_response/rq2_analysis.py

Saves (in outputs/): rq2_results.csv, signed_by_model_budget.csv,
                     rq2_report.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from scipy import stats

from analysis_utils import (load_master, load_usage, attach_realized_tokens,
                            reasoning_models, per_item_slopes, mixed_slope,
                            fmt_p, fmt_p_bare, table, Report)

OUT = HERE / "outputs"


def main():
    df, cfg = load_master()
    df = attach_realized_tokens(df, load_usage(cfg))
    rep = Report()

    rep("=" * 76)
    rep(" RQ2 -- DOSE-RESPONSE: BIAS vs log2(REALIZED REASONING TOKENS + 1)")
    rep("=" * 76)

    # ------------------------------------------------------- descriptives --
    desc = (df.groupby(["model", "budget"])["bias_score"].mean()
              .rename("mean_signed_score").reset_index())
    rep("\n Mean signed score per model x budget: range "
        f"{desc['mean_signed_score'].min():+.3f} to "
        f"{desc['mean_signed_score'].max():+.3f} "
        f"({(desc['mean_signed_score'] < 0).sum()}/{len(desc)} negative)")

    # ------------------------------------------------------- dose-response --
    rows, rows_csv = [], []
    for m in reasoning_models(cfg):
        sub = df[(df["model"] == m) & df["realized_tokens"].notna()].copy()
        if sub["budget"].nunique() < 2:
            continue
        lo, hi = sub["realized_tokens"].min(), sub["realized_tokens"].max()
        slopes = per_item_slopes(sub, "log_realized", "abs_bias").values
        t, p = stats.ttest_1samp(slopes, 0)
        b_abs, p_abs, _ = mixed_slope(sub, "abs_bias ~ log_realized",
                                      "log_realized")
        b_sig, p_sig, _ = mixed_slope(sub, "bias_score ~ log_realized",
                                      "log_realized")
        rows.append([m, f"{lo:.0f}-{hi:.0f}", f"{slopes.mean():+.4f}",
                     f"{t:.2f}", len(slopes) - 1, fmt_p_bare(p),
                     f"{b_abs:+.4f}", fmt_p_bare(p_abs),
                     f"{b_sig:+.4f}", fmt_p_bare(p_sig)])
        rows_csv.append(dict(model=m, tok_min=lo, tok_max=hi,
                             n_items=len(slopes), slope_mean=slopes.mean(),
                             t=float(t), df=len(slopes) - 1, p=float(p),
                             beta_abs=b_abs, p_abs=p_abs,
                             beta_signed=b_sig, p_signed=p_sig))
        rep(f"\n {m}: realized tokens span {lo:.0f} to {hi:.0f} per cell")
        rep(f"   per-item OLS slope of |s|: mean = {slopes.mean():+.4f}, "
            f"t({len(slopes) - 1}) = {t:.2f}, {fmt_p(p)}")
        rep(f"   mixed model |s|: beta = {b_abs:+.4f}, {fmt_p(p_abs)}")
        rep(f"   mixed model  s : beta = {b_sig:+.4f}, {fmt_p(p_sig)}")

    rep("\n" + table(rows, ["model", "range", "slope |s|", "t", "df", "p",
                            "beta |s|", "p", "beta s", "p"]))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_csv).to_csv(OUT / "rq2_results.csv", index=False)
    desc.to_csv(OUT / "signed_by_model_budget.csv", index=False)
    rep("\nSaved: rq2_results.csv, signed_by_model_budget.csv, rq2_report.txt")
    rep.save(OUT / "rq2_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
