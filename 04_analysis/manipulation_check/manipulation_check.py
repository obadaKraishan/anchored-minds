#!/usr/bin/env python3
"""
04_analysis/manipulation_check/manipulation_check.py

Manipulation check: did the requested thinking ceiling change how much the
model actually deliberated? Summarizes realized reasoning tokens (as reported
in each response's usage field) per model x requested budget, over every call
in the grid, including the budget-0 calls of models that cannot disable
thinking (these cells are excluded from the bias analyses but show that the
models thought anyway).

Input:  data/per_call_usage.csv
Usage:  python 04_analysis/manipulation_check/manipulation_check.py

Saves (in outputs/):
  realized_tokens.csv           count, mean, median, SD, max of reasoning
                                tokens per call, and share of calls above the
                                requested ceiling, per model x budget
  manipulation_check_report.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

from analysis_utils import load_config, load_usage, Report

OUT = HERE / "outputs"


def main():
    cfg = load_config()
    usage = load_usage(cfg)
    rep = Report()

    rep("=" * 76)
    rep(" MANIPULATION CHECK -- REALIZED REASONING TOKENS PER MODEL x BUDGET")
    rep("=" * 76)
    rep(f" {len(usage):,} calls; reasoning_tokens reported for "
        f"{usage['reasoning_tokens'].notna().sum():,}")

    tok = (usage.groupby(["model", "budget"])["reasoning_tokens"]
                .agg(["count", "mean", "median", "std", "max"])
                .reset_index())
    tok["pct_over_ceiling"] = np.nan
    for i, r in tok.iterrows():
        if r["budget"] > 0:
            calls = usage[(usage["model"] == r["model"])
                          & (usage["budget"] == r["budget"])]
            tok.loc[i, "pct_over_ceiling"] = \
                (calls["reasoning_tokens"] > r["budget"]).mean()

    rep(f"\n {'model':<28}{'budget':>8}{'mean':>8}{'median':>8}{'max':>8}"
        f"{'>ceiling':>10}")
    for _, r in tok.sort_values(["model", "budget"]).iterrows():
        oc = ("" if np.isnan(r["pct_over_ceiling"])
              else f"{100 * r['pct_over_ceiling']:.0f}%")
        rep(f" {r['model']:<28}{int(r['budget']):>8}{r['mean']:>8.0f}"
            f"{r['median']:>8.0f}{r['max']:>8.0f}{oc:>10}")

    no_disable = [m["name"] for m in cfg["models"]
                  if m.get("cannot_disable_thinking")]
    if no_disable:
        span = tok[tok["model"].isin(no_disable)]["mean"]
        rep(f"\n Models that cannot disable thinking ({', '.join(no_disable)}):"
            f" mean consumption {span.min():,.0f} to {span.max():,.0f} tokens "
            f"per call across all requested budgets, including budget 0.")

    OUT.mkdir(parents=True, exist_ok=True)
    tok.to_csv(OUT / "realized_tokens.csv", index=False)
    rep("\nSaved: realized_tokens.csv, manipulation_check_report.txt")
    rep.save(OUT / "manipulation_check_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
