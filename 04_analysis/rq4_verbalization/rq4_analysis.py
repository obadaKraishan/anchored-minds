#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 04_analysis/rq4_verbalization/rq4_analysis.py
=============================================================================
 RQ4: Does forced verbalization of the anchor change susceptibility?

 Design: on Anchoring items only, every model x budget cell has both a
 standard treatment score (bias_score) and a verbalized-treatment score
 (bias_score_verbalized), where the model was instructed to explicitly
 restate every number in the prompt before answering. Comparison is PAIRED
 by model x budget x item cell; per-model breakdowns and a pooled test are
 reported with paired t, Wilcoxon, Cohen's dz, and bootstrap CIs.

 Usage:
   python 04_analysis/rq4_verbalization/rq4_analysis.py

 Saves (in outputs/): figures/, rq4_results.csv, rq4_results.json,
                      _rq4_apa_report_.txt
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

from analysis_utils import (load_master, paired_tests, holm_correction,
                            apa_paired, fmt_p, fmt_stat, table, save_fig)

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
    rep(" RQ4 -- FORCED ANCHOR VERBALIZATION AND SUSCEPTIBILITY (Anchoring)")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 74)

    a = df[(df["bias"] == "Anchoring")
           & df["bias_score_verbalized"].notna()].copy()
    if not len(a):
        rep("[ERROR] No verbalized scores found -- was the RQ4 condition "
            "collected?")
        return 1
    rep(f"\n Cells with both conditions: {len(a)} "
        f"(models: {a['model'].nunique()}, budgets incl.)")

    models = sorted(a["model"].unique())
    per_model, rows_csv = {}, []
    for model in models:
        sub = a[a["model"] == model]
        per_model[model] = paired_tests(sub["bias_score_verbalized"],
                                        sub["bias_score"])
    adj = holm_correction([per_model[m]["p_t"] for m in models])
    for m, v in zip(models, adj):
        per_model[m]["p_holm"] = float(v) if v == v else np.nan

    hdr = ["model", "M_standard", "M_verbalized", "diff", "t", "df", "p",
           "p_holm", "dz", "n cells"]
    rows = []
    for m in models:
        sub = a[a["model"] == m]
        r = per_model[m]
        rows.append([m, fmt_stat(sub["bias_score"].mean(), 3),
                     fmt_stat(sub["bias_score_verbalized"].mean(), 3),
                     fmt_stat(r["mean_diff"], 3), fmt_stat(r["t"]),
                     r["df"], fmt_p(r["p_t"]).replace("p ", ""),
                     fmt_p(r["p_holm"]).replace("p ", ""),
                     fmt_stat(r["dz"]), r["n"]])
        rows_csv.append({"model": m,
                         "mean_standard": float(sub["bias_score"].mean()),
                         "mean_verbalized":
                             float(sub["bias_score_verbalized"].mean()),
                         **{k: r[k] for k in ("mean_diff", "t", "df", "p_t",
                                              "p_holm", "W", "p_w", "dz",
                                              "n")},
                         "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
    rep("\n" + table(rows, hdr))

    pooled = paired_tests(a["bias_score_verbalized"], a["bias_score"])
    rep("\nAPA sentences:")
    rep("  (positive difference = verbalization INCREASES anchoring; "
        "negative = decreases)")
    for m in models:
        rep("  " + apa_paired(f"{m} (verbalized - standard)", per_model[m]))
    rep("  " + apa_paired("POOLED across models x budgets x items "
                          "(verbalized - standard)", pooled))

    # budget interaction: does verbalization effect change with budget?
    rep("\n Verbalization effect by budget (pooled over reasoning models):")
    rea = a[a["reasoning"]]
    for b, g in rea.groupby("budget"):
        d = (g["bias_score_verbalized"] - g["bias_score"])
        rep(f"   b{b:<6} M diff = {fmt_stat(d.mean(), 3)}  "
            f"(n = {len(d)} cells)")

    # ------------------------------------------------------------- figure ---
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(models))
    ms = [a[a["model"] == m]["bias_score"].mean() for m in models]
    mv = [a[a["model"] == m]["bias_score_verbalized"].mean() for m in models]
    ax.bar(x - 0.2, ms, 0.4, label="standard treatment", color="#B0B7C3")
    ax.bar(x + 0.2, mv, 0.4, label="verbalized treatment", color="#C15F3C")
    ax.set_xticks(x, models, rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean anchoring bias score")
    ax.legend(fontsize=8)
    ax.set_title("RQ4: Anchoring with vs without forced anchor "
                 "verbalization", fontsize=10)
    save_fig(fig, FIG, "rq4_verbalization")
    plt.close(fig)

    pd.DataFrame(rows_csv).to_csv(OUT / "rq4_results.csv", index=False)
    (OUT / "rq4_results.json").write_text(
        json.dumps({"per_model": per_model, "pooled": pooled}, indent=2,
                   default=float), encoding="utf-8")
    (OUT / "_rq4_apa_report_.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq4_results.csv, rq4_results.json, _rq4_apa_report_.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
