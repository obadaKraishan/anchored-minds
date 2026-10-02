#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 04_analysis/rq2_dose_response/rq2_analysis.py
=============================================================================
 RQ2: Does bias decrease monotonically with reasoning-token budget
      (dose-response), or does it plateau?

 Design: within each reasoning model, item-level bias scores are analyzed
 across the available budgets (b0 included only for models that can disable
 thinking). Two dose-response tests per model x measure:
   1. Spearman rank trend of item-level scores against budget
   2. Repeated-measures linear trend: per-item OLS slope of score on
      log2(budget+1), tested against 0 across items (one-sample t)
 Plus a pooled mixed-effects model: score ~ log_budget + (1 | item) using
 statsmodels MixedLM, per model and across all reasoning models.

 Usage:
   python 04_analysis/rq2_dose_response/rq2_analysis.py

 Saves (in outputs/): figures/, rq2_results.csv, rq2_results.json,
                      _rq2_apa_report_.txt
=============================================================================
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import (load_master, one_sample_test, spearman_trend,
                            bootstrap_ci_mean, fmt_p, fmt_stat, table,
                            save_fig)

OUT = HERE / "outputs"
FIG = OUT / "figures"
REPORT = []


def rep(line=""):
    print(line, flush=True)
    REPORT.append(line)


def mixed_trend(sub, col):
    """MixedLM score ~ log_budget with random intercept per item."""
    import statsmodels.formula.api as smf
    d = sub[[col, "budget", "item_id"]].dropna().copy()
    d["log_budget"] = np.log2(d["budget"] + 1)
    if d["log_budget"].nunique() < 2 or len(d) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(f"{col} ~ log_budget", d,
                            groups=d["item_id"]).fit(reml=False)
            beta = m.params["log_budget"]
            se = m.bse["log_budget"]
            z = beta / se if se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
            return dict(beta=float(beta), se=float(se), z=float(z),
                        p=float(p), n_obs=int(len(d)))
        except Exception:
            return None


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)

    rep("=" * 74)
    rep(" RQ2 -- DOSE-RESPONSE: BIAS vs REASONING BUDGET")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 74)
    rep("\n Budget coding: linear trend uses log2(budget+1); Spearman uses "
        "budget rank.")
    rep(" Models that cannot disable thinking contribute budgets > 0 only "
        "(by design).")

    reasoning = df[df["reasoning"]]
    models = sorted(reasoning["model"].unique())
    results, rows_csv = {}, []

    for measure, col in [("magnitude |bias|", "abs_bias"),
                         ("signed bias", "bias_score")]:
        rep(f"\n--- Measure: {measure} ---")
        hdr = ["model", "budgets", "rho", "p_rho", "slope/log2",
               "t(items)", "p_slope", "mixedLM beta", "p_LMM"]
        rows = []
        res_m = {}
        for model in models:
            sub = reasoning[reasoning["model"] == model]
            budgets = [int(b) for b in sorted(sub["budget"].unique())]
            cell_means = sub.groupby("budget")[col].mean()

            # Spearman on all item x budget cells
            rho, p_rho = spearman_trend(sub["budget"], sub[col])

            # per-item slope on log2(budget+1)
            slopes = []
            for _, g in sub.groupby("item_id"):
                if g["budget"].nunique() < 2:
                    continue
                x = np.log2(g["budget"].values + 1)
                y = g[col].values
                slopes.append(np.polyfit(x, y, 1)[0])
            slope_test = one_sample_test(slopes)

            lmm = mixed_trend(sub, col)
            res_m[model] = dict(budgets=budgets,
                                cell_means={int(b): float(v) for b, v
                                            in cell_means.items()},
                                spearman_rho=rho, spearman_p=p_rho,
                                slope_mean=slope_test["mean"],
                                slope_t=slope_test["t"],
                                slope_df=slope_test["df"],
                                slope_p=slope_test["p"],
                                slope_ci=slope_test["ci"], lmm=lmm)
            rows.append([model, str(budgets),
                         fmt_stat(rho, 3), fmt_p(p_rho).replace("p ", ""),
                         fmt_stat(slope_test["mean"], 4),
                         fmt_stat(slope_test["t"]),
                         fmt_p(slope_test["p"]).replace("p ", ""),
                         fmt_stat(lmm["beta"], 4) if lmm else "n/a",
                         fmt_p(lmm["p"]).replace("p ", "") if lmm else "n/a"])
            rows_csv.append({"measure": col, "model": model,
                             "spearman_rho": rho, "spearman_p": p_rho,
                             "slope_mean": slope_test["mean"],
                             "slope_t": slope_test["t"],
                             "slope_df": slope_test["df"],
                             "slope_p": slope_test["p"],
                             "slope_ci_lo": slope_test["ci"][0],
                             "slope_ci_hi": slope_test["ci"][1],
                             "lmm_beta": lmm["beta"] if lmm else np.nan,
                             "lmm_p": lmm["p"] if lmm else np.nan})
        rep("\n" + table(rows, hdr))

        rep("\nAPA sentences:")
        for model in models:
            r = res_m[model]
            direction = ("decreased" if r["slope_mean"] < 0 else "increased")
            sig = ("significantly " if r["slope_p"] == r["slope_p"]
                   and r["slope_p"] < .05 else "non-significantly ")
            rep(f"  For {model}, {measure} {sig}{direction} with reasoning "
                f"budget: per-item linear slope on log2(budget+1) M = "
                f"{fmt_stat(r['slope_mean'], 4)} (95% CI "
                f"[{fmt_stat(r['slope_ci'][0], 4)}, "
                f"{fmt_stat(r['slope_ci'][1], 4)}]), "
                f"t({r['slope_df']}) = {fmt_stat(r['slope_t'])}, "
                f"{fmt_p(r['slope_p'])}; Spearman rho = "
                f"{fmt_stat(r['spearman_rho'], 3)}, "
                f"{fmt_p(r['spearman_p'])}"
                + (f"; mixed-effects beta = {fmt_stat(r['lmm']['beta'], 4)}, "
                   f"{fmt_p(r['lmm']['p'])}." if r["lmm"] else "."))
        results[col] = res_m

    # ------------------------------------------------------------- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    colors = {"claude-haiku-4.5-thinking": "#C15F3C",
              "gpt-5.1-reasoning": "#1F4E79",
              "deepseek-r1": "#3B7A57", "qwen3-thinking": "#7D3C98"}
    for ax, (col, title) in zip(axes, [("abs_bias", "Bias magnitude |score|"),
                                       ("bias_score", "Signed bias score")]):
        for model in models:
            sub = reasoning[reasoning["model"] == model]
            g = sub.groupby("budget")[col]
            means = g.mean()
            cis = np.array([bootstrap_ci_mean(v) for _, v in g])
            x = np.log2(means.index.values + 1)
            c = colors.get(model, None)
            ax.plot(x, means.values, "o-", label=model, color=c, ms=4)
            ax.fill_between(x, cis[:, 0], cis[:, 1], alpha=0.15, color=c)
        ax.set_xticks(np.log2(np.array([0, 1024, 4096, 8192]) + 1))
        ax.set_xticklabels(["0", "1k", "4k", "8k"])
        ax.set_xlabel("reasoning budget (tokens)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("RQ2: Dose-response of bias vs reasoning budget "
                 "(95% bootstrap CI)", fontsize=11)
    save_fig(fig, FIG, "rq2_dose_response")
    plt.close(fig)

    pd.DataFrame(rows_csv).to_csv(OUT / "rq2_results.csv", index=False)
    (OUT / "rq2_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "_rq2_apa_report_.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq2_results.csv, rq2_results.json, _rq2_apa_report_.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
