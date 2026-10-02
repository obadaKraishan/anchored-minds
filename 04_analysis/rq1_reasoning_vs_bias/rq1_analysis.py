#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py
=============================================================================
 RQ1: Do reasoning models show smaller cognitive-bias magnitudes than
      matched non-reasoning models of the same family?

 Design: within each model family (pair), the reasoning model's item-level
 bias magnitude |bias_score| (averaged over its thinking budgets > 0) is
 compared against the family's non-reasoning comparator at budget 0,
 PAIRED BY ITEM (n = 30 battery items). Signed scores are analyzed
 secondarily. For families without a dedicated instruct sibling (OpenAI),
 the reasoning model at budget 0 with thinking disabled is the comparator.

 Statistics: paired t-test, Wilcoxon signed-rank, Cohen's dz, 10,000-sample
 bootstrap CIs; a pooled analysis across families; Holm correction across
 the four family tests.

 Usage:
   python 04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py

 Saves (in outputs/): figures/, rq1_results.csv, rq1_results.json,
                      _rq1_apa_report_.txt
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (load_master, family_comparators, paired_tests,
                            holm_correction, apa_paired, fmt_p, fmt_stat,
                            table, save_fig)

OUT = HERE / "outputs"
FIG = OUT / "figures"
REPORT = []


def rep(line=""):
    print(line, flush=True)
    REPORT.append(line)


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)

    rep("=" * 74)
    rep(" RQ1 -- REASONING vs NON-REASONING BIAS MAGNITUDE (matched pairs)")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 74)

    comps = family_comparators(df)
    results, rows_csv = {}, []

    for measure, col in [("magnitude |bias|", "abs_bias"),
                         ("signed bias", "bias_score")]:
        rep(f"\n--- Measure: {measure} ---")
        fam_res = {}
        for fam, (comp_df, comp_label) in comps.items():
            reason = df[(df["pair"] == fam) & (df["reasoning"])
                        & (df["budget"] > 0)]
            if not len(reason):
                continue
            r_items = reason.groupby("item_id")[col].mean()
            c_items = comp_df.set_index("item_id")[col]
            joined = pd.concat([r_items, c_items], axis=1, keys=["r", "c"])
            res = paired_tests(joined["r"], joined["c"])
            res["family"] = fam
            res["comparator"] = comp_label
            res["mean_reasoning"] = float(joined["r"].mean())
            res["mean_nonreasoning"] = float(joined["c"].mean())
            fam_res[fam] = res

        # Holm across families
        fams = list(fam_res)
        adj = holm_correction([fam_res[f]["p_t"] for f in fams])
        for f, a in zip(fams, adj):
            fam_res[f]["p_holm"] = float(a) if a == a else np.nan

        hdr = ["family", "M_reason", "M_nonreason", "diff", "t", "df",
               "p", "p_holm", "dz", "n"]
        rows = []
        for f in fams:
            r = fam_res[f]
            rows.append([f, fmt_stat(r["mean_reasoning"], 3),
                         fmt_stat(r["mean_nonreasoning"], 3),
                         fmt_stat(r["mean_diff"], 3), fmt_stat(r["t"]),
                         r["df"], fmt_p(r["p_t"]).replace("p ", ""),
                         fmt_p(r.get("p_holm", np.nan)).replace("p ", ""),
                         fmt_stat(r["dz"]), r["n"]])
            rows_csv.append({"measure": col, **{k: r[k] for k in
                            ("family", "comparator", "mean_reasoning",
                             "mean_nonreasoning", "mean_diff", "t", "df",
                             "p_t", "p_holm", "W", "p_w", "dz", "n")},
                            "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
        rep("\n" + table(rows, hdr))

        # pooled across families (item x family observations)
        pooled_r, pooled_c = [], []
        for fam, (comp_df, _) in comps.items():
            reason = df[(df["pair"] == fam) & (df["reasoning"])
                        & (df["budget"] > 0)]
            if not len(reason):
                continue
            r_items = reason.groupby("item_id")[col].mean()
            c_items = comp_df.set_index("item_id")[col]
            joined = pd.concat([r_items, c_items], axis=1,
                               keys=["r", "c"]).dropna()
            pooled_r += list(joined["r"])
            pooled_c += list(joined["c"])
        pooled = paired_tests(pooled_r, pooled_c)
        rep("\nAPA sentences:")
        for f in fams:
            rep("  " + apa_paired(f"Family {f} ({measure})", fam_res[f]))
        rep("  " + apa_paired(f"POOLED across families ({measure}, "
                              f"item x family pairs)", pooled))
        results[col] = {"families": fam_res, "pooled": pooled}

    # ------------------------------------------------------------- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, title) in zip(axes, [("abs_bias", "Bias magnitude |score|"),
                                       ("bias_score", "Signed bias score")]):
        fams, r_means, c_means = [], [], []
        for fam, (comp_df, _) in comps.items():
            reason = df[(df["pair"] == fam) & (df["reasoning"])
                        & (df["budget"] > 0)]
            if not len(reason):
                continue
            fams.append(fam)
            r_means.append(reason.groupby("item_id")[col].mean().mean())
            c_means.append(comp_df[col].mean())
        x = np.arange(len(fams))
        ax.bar(x - 0.2, c_means, 0.4, label="non-reasoning (b0)",
               color="#B0B7C3")
        ax.bar(x + 0.2, r_means, 0.4, label="reasoning (b>0)",
               color="#1F4E79")
        ax.set_xticks(x, fams)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("RQ1: Reasoning vs non-reasoning bias by model family",
                 fontsize=11)
    save_fig(fig, FIG, "rq1_family_comparison")
    plt.close(fig)

    # --------------------------------------------------------------- save ---
    pd.DataFrame(rows_csv).to_csv(OUT / "rq1_results.csv", index=False)
    (OUT / "rq1_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "_rq1_apa_report_.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq1_results.csv, rq1_results.json, _rq1_apa_report_.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
