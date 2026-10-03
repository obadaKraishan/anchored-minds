#!/usr/bin/env python3
"""
04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py

RQ1: Are reasoning models less biased than matched non-reasoning models of
the same family?

Per family, each item's bias for the reasoning model (averaged over its
budgets > 0) is paired with the same item's bias for the family comparator at
budget 0: the non-reasoning sibling, or for OpenAI (no sibling) the same
model with thinking disabled. Paired t(29), Wilcoxon signed-rank, Cohen's dz,
and a bootstrap CI of the mean difference; Holm across the four families.

Pooled contrast: each item's four family differences are averaged, so the 30
items remain the independent units -> one-sample t(29), bootstrap CI,
Wilcoxon, and a mixed model (difference ~ 1, random item intercept).

The primary measure is magnitude |s|; signed s is reported secondarily.

Input:  02_data_processing/outputs/master_scores.csv
Usage:  python 04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py

Saves (in outputs/): rq1_results.csv, rq1_report.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from scipy import stats

from analysis_utils import (load_master, family_comparators, paired_tests,
                            bootstrap_ci_mean, holm_correction, mixed_slope,
                            fmt_p, fmt_p_bare, fmt_stat, table, Report)

OUT = HERE / "outputs"
MEASURES = [("abs_bias", "magnitude |s|"), ("bias_score", "signed s")]


def family_pairs(df, col):
    """family -> (DataFrame[r, c] indexed by item_id, comparator label)."""
    out = {}
    for fam, (comp_df, comp_label) in family_comparators(df).items():
        reason = df[(df["pair"] == fam) & (df["reasoning"])
                    & (df["budget"] > 0)]
        if not len(reason):
            continue
        r_items = reason.groupby("item_id")[col].mean()
        c_items = comp_df.groupby("item_id")[col].mean()
        out[fam] = (pd.concat([r_items, c_items], axis=1,
                              keys=["r", "c"]).dropna(), comp_label)
    return out


def main():
    df, _ = load_master()
    rep = Report()
    rows_csv = []

    rep("=" * 76)
    rep(" RQ1 -- REASONING vs MATCHED NON-REASONING MODELS (paired by item)")
    rep("=" * 76)

    for col, label in MEASURES:
        rep(f"\n--- Measure: {label} ---")
        pairs = family_pairs(df, col)
        fams = sorted(pairs)
        res = {f: paired_tests(pairs[f][0]["r"], pairs[f][0]["c"])
               for f in fams}
        for f, a in zip(fams, holm_correction([res[f]["p_t"] for f in fams])):
            res[f]["p_holm"] = float(a)

        hdr = ["family", "comparator", "M_reas", "M_comp", "diff", "95% CI",
               "t", "df", "p", "p_holm", "dz", "p_wilcoxon"]
        rows = []
        for f in fams:
            j, comp_label = pairs[f]
            r = res[f]
            rows.append([f, comp_label, fmt_stat(j["r"].mean(), 3),
                         fmt_stat(j["c"].mean(), 3),
                         f"{r['mean_diff']:+.3f}",
                         f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]",
                         fmt_stat(r["t"]), r["df"],
                         fmt_p_bare(r["p_t"]),
                         fmt_p_bare(r["p_holm"]),
                         fmt_stat(r["dz"]),
                         fmt_p_bare(r["p_w"])])
            rows_csv.append({"measure": col, "family": f,
                             "comparator": comp_label, "n_items": r["n"],
                             "mean_reasoning": float(j["r"].mean()),
                             "mean_comparator": float(j["c"].mean()),
                             "mean_diff": r["mean_diff"],
                             "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
                             "t": r["t"], "df": r["df"], "p": r["p_t"],
                             "p_holm": r["p_holm"], "dz": r["dz"],
                             "W": r["W"], "p_wilcoxon": r["p_w"]})
        rep("\n" + table(rows, hdr))

        # pooled: one mean difference per item, averaged over families
        diffs = pd.concat({f: pairs[f][0]["r"] - pairs[f][0]["c"]
                           for f in fams}, axis=1)
        item_means = diffs.mean(axis=1).sort_index().values
        t, p = stats.ttest_1samp(item_means, 0)
        w = stats.wilcoxon(item_means)
        ci = bootstrap_ci_mean(item_means)
        dz = item_means.mean() / item_means.std(ddof=1)
        long = (diffs.stack().rename("diff").reset_index()
                     .rename(columns={"level_1": "family"}))
        beta, p_mm, z_mm = mixed_slope(long, "diff ~ 1", "Intercept")
        rep(f"\n POOLED at the item level (n = {len(item_means)} items, each "
            f"the mean of its {len(fams)} family differences):")
        rep(f"   Delta = {item_means.mean():+.3f}, 95% bootstrap CI "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}], t({len(item_means) - 1}) = "
            f"{t:.2f}, {fmt_p(p)}, dz = {dz:.2f}; Wilcoxon {fmt_p(w.pvalue)}")
        rep(f"   mixed model (random item intercept): beta = {beta:+.4f}, "
            f"z = {z_mm:.2f}, {fmt_p(p_mm)}")
        rows_csv.append({"measure": col, "family": "pooled_item_level",
                         "comparator": "", "n_items": len(item_means),
                         "mean_diff": float(item_means.mean()),
                         "ci_lo": ci[0], "ci_hi": ci[1], "t": float(t),
                         "df": len(item_means) - 1, "p": float(p),
                         "dz": float(dz), "W": float(w.statistic),
                         "p_wilcoxon": float(w.pvalue), "lmm_beta": beta,
                         "lmm_z": z_mm, "lmm_p": p_mm})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_csv).to_csv(OUT / "rq1_results.csv", index=False)
    rep("\nSaved: rq1_results.csv, rq1_report.txt")
    rep.save(OUT / "rq1_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
