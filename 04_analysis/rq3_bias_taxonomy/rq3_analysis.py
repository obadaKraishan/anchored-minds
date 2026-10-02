#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 04_analysis/rq3_bias_taxonomy/rq3_analysis.py
=============================================================================
 RQ3: Which biases are reasoning-resistant vs reasoning-invariant -- and,
      more fundamentally, which classic human biases are even PRESENT
      (positive), ABSENT (null), or REVERSED (negative) in current LLMs?

 Two-part analysis:
   Part A (presence): per bias, one-sample tests of the signed bias score
   against 0 over model x budget x item cells, with item-clustered
   bootstrap CIs and Holm correction -> classification PRESENT / ABSENT /
   REVERSED.
   Part B (reasoning sensitivity): per bias, within reasoning models, the
   per-item budget slope (log2(budget+1)) tested against 0 -> classification
   REASONING-SENSITIVE (slope != 0) vs REASONING-INVARIANT.

 Usage:
   python 04_analysis/rq3_bias_taxonomy/rq3_analysis.py

 Saves (in outputs/): figures/, rq3_results.csv, rq3_results.json,
                      _rq3_apa_report_.txt
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

from analysis_utils import (load_master, one_sample_test, holm_correction,
                            fmt_p, fmt_stat, table, save_fig, N_BOOT,
                            RNG_SEED)

OUT = HERE / "outputs"
FIG = OUT / "figures"
REPORT = []


def rep(line=""):
    print(line, flush=True)
    REPORT.append(line)


def cluster_boot_ci(sub, col, cluster="item_id", n_boot=N_BOOT,
                    seed=RNG_SEED):
    """Bootstrap CI of the mean, resampling item clusters (accounts for the
    non-independence of cells sharing a battery item)."""
    groups = [g[col].values for _, g in sub.groupby(cluster)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        boots[i] = np.concatenate([groups[j] for j in idx]).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)

    rep("=" * 74)
    rep(" RQ3 -- BIAS TAXONOMY: PRESENCE AND REASONING-SENSITIVITY")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 74)

    biases = cfg["biases"]
    results = {"presence": {}, "sensitivity": {}}
    rows_csv = []

    # ---------------------------------------------------- Part A: presence --
    rep("\n--- Part A: is each bias PRESENT, ABSENT, or REVERSED? ---")
    rep(" (signed score over all model x budget x item cells; CI is item-")
    rep("  clustered bootstrap; p-values Holm-corrected across 6 biases)")
    pres = {}
    for bias in biases:
        sub = df[df["bias"] == bias]
        t = one_sample_test(sub["bias_score"])
        t["ci_cluster"] = cluster_boot_ci(sub, "bias_score")
        pres[bias] = t
    adj = holm_correction([pres[b]["p"] for b in biases])
    hdr = ["bias", "M", "SD", "95% CI (clustered)", "t", "df", "p_holm",
           "d", "classification"]
    rows = []
    for b, a in zip(biases, adj):
        r = pres[b]
        r["p_holm"] = float(a) if a == a else np.nan
        if r["p_holm"] == r["p_holm"] and r["p_holm"] < .05:
            cls = "PRESENT" if r["mean"] > 0 else "REVERSED"
        else:
            cls = "ABSENT (null)"
        r["classification"] = cls
        rows.append([b, fmt_stat(r["mean"], 3), fmt_stat(r["sd"], 3),
                     f"[{fmt_stat(r['ci_cluster'][0], 3)}, "
                     f"{fmt_stat(r['ci_cluster'][1], 3)}]",
                     fmt_stat(r["t"]), r["df"],
                     fmt_p(r["p_holm"]).replace("p ", ""),
                     fmt_stat(r["d"]), cls])
        rows_csv.append({"part": "presence", "bias": b, **{k: r[k] for k in
                        ("n", "mean", "sd", "t", "df", "p", "p_holm", "d",
                         "classification")},
                        "ci_lo": r["ci_cluster"][0],
                        "ci_hi": r["ci_cluster"][1]})
    rep("\n" + table(rows, hdr))
    results["presence"] = pres

    rep("\nAPA sentences (Part A):")
    for b in biases:
        r = pres[b]
        rep(f"  {b}: M = {fmt_stat(r['mean'], 3)}, SD = "
            f"{fmt_stat(r['sd'], 3)}, item-clustered 95% CI "
            f"[{fmt_stat(r['ci_cluster'][0], 3)}, "
            f"{fmt_stat(r['ci_cluster'][1], 3)}], t({r['df']}) = "
            f"{fmt_stat(r['t'])}, Holm-corrected {fmt_p(r['p_holm'])}, "
            f"Cohen's d = {fmt_stat(r['d'])} -> {r['classification']}.")

    # ---------------------------------------- Part B: reasoning sensitivity --
    rep("\n--- Part B: is each bias sensitive to reasoning budget? ---")
    rep(" (per-item slopes of |bias| on log2(budget+1) within reasoning")
    rep("  models, pooled across models; Holm-corrected)")
    sens = {}
    for bias in biases:
        sub = df[(df["bias"] == bias) & (df["reasoning"])]
        slopes = []
        for (_, _), g in sub.groupby(["model", "item_id"]):
            if g["budget"].nunique() < 2:
                continue
            x = np.log2(g["budget"].values + 1)
            slopes.append(np.polyfit(x, g["abs_bias"].values, 1)[0])
        sens[bias] = one_sample_test(slopes)
    adj = holm_correction([sens[b]["p"] for b in biases])
    hdr = ["bias", "slope M", "95% CI", "t", "df", "p_holm",
           "classification"]
    rows = []
    for b, a in zip(biases, adj):
        r = sens[b]
        r["p_holm"] = float(a) if a == a else np.nan
        if r["p_holm"] == r["p_holm"] and r["p_holm"] < .05:
            cls = ("REASONING-RESPONSIVE (decreases)" if r["mean"] < 0
                   else "REASONING-RESPONSIVE (increases)")
        else:
            cls = "REASONING-INVARIANT"
        r["classification"] = cls
        rows.append([b, fmt_stat(r["mean"], 4),
                     f"[{fmt_stat(r['ci'][0], 4)}, {fmt_stat(r['ci'][1], 4)}]",
                     fmt_stat(r["t"]), r["df"],
                     fmt_p(r["p_holm"]).replace("p ", ""), cls])
        rows_csv.append({"part": "sensitivity", "bias": b,
                         **{k: r[k] for k in ("n", "mean", "t", "df", "p",
                                              "p_holm", "classification")},
                         "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
    rep("\n" + table(rows, hdr))
    results["sensitivity"] = sens

    rep("\nAPA sentences (Part B):")
    for b in biases:
        r = sens[b]
        rep(f"  {b}: mean per-item budget slope = {fmt_stat(r['mean'], 4)} "
            f"(95% CI [{fmt_stat(r['ci'][0], 4)}, "
            f"{fmt_stat(r['ci'][1], 4)}]), t({r['df']}) = {fmt_stat(r['t'])}, "
            f"Holm-corrected {fmt_p(r['p_holm'])} -> {r['classification']}.")

    # ------------------------------------------------------------ figure ----
    piv = (df.groupby(["bias", "model"])["bias_score"].mean()
             .unstack("model").loc[biases])
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30,
                  ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center",
                    va="center", fontsize=7.5)
    fig.colorbar(im, label="mean signed bias score")
    ax.set_title("RQ3: Signed bias score by bias type x model "
                 "(red = human-like bias, blue = reversed)", fontsize=10)
    save_fig(fig, FIG, "rq3_bias_model_heatmap")
    plt.close(fig)

    pd.DataFrame(rows_csv).to_csv(OUT / "rq3_results.csv", index=False)
    (OUT / "rq3_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "_rq3_apa_report_.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq3_results.csv, rq3_results.json, _rq3_apa_report_.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
