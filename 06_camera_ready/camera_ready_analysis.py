#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 06_camera_ready/camera_ready_analysis.py
=============================================================================
 Re-analysis for the IEEE CogMI 2026 camera-ready, addressing the reviews.

   A. REALIZED REASONING TOKENS. The reviewers asked whether the requested
      thinking ceiling produced more deliberation. We read the per-call
      usage logged by the gateway and report mean reasoning tokens per
      model x budget, then use realized tokens (not the requested ceiling)
      as the dose for RQ2.
   B. RQ1 AT THE ITEM LEVEL. The submitted pooled test treated 30 items x
      4 families as 120 independent pairs. We average the reasoning-minus-
      instruct difference over families within each item, giving 30
      independent pairs, and report t(29).
   C. RQ3 WITH ITEM CLUSTERING. The submitted test reported an unclustered
      t(84) alongside item-clustered CIs. We replace it with a mixed model
      (random item intercept) and keep the clustered bootstrap CI.

 Run from the repository root:
   python 06_camera_ready/camera_ready_analysis.py

 Saves (in 06_camera_ready/outputs/):
   realized_tokens.csv                 model x budget consumption
   table_realized_tokens.tex           camera-ready table
   rq2_realized_dose.csv               slopes on realized tokens
   rq1_item_level.csv
   rq3_mixed.csv
   _camera_ready_apa_report_.txt       every number needed for the text
=============================================================================
"""

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_camera_ready" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
REP = []


def rep(s=""):
    print(s, flush=True)
    REP.append(s)


def fmt_p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def holm(ps):
    p = np.asarray(ps, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    prev = 0
    for rank, i in enumerate(order):
        val = min((m - rank) * p[i], 1.0)
        prev = max(prev, val)
        adj[i] = prev
    return adj


def boot_ci(x, n=10000, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


# --------------------------------------------------------------------------
# Load config, master scores, raw responses
# --------------------------------------------------------------------------
import yaml  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
master = pd.read_csv(ROOT / "02_data_processing" / "outputs" / "master_scores.csv")
rep("=" * 76)
rep(" ANCHORED MINDS -- CAMERA-READY RE-ANALYSIS")
rep("=" * 76)
rep(f" master_scores.csv: {len(master)} cells, columns: {list(master.columns)}")

raw_rows = []
for f in glob.glob(str(ROOT / "01_data_collection" / "outputs" / "raw_responses" / "*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = r.get("usage") or {}
        d = u.get("completion_tokens_details") or {}
        rt = d.get("reasoning_tokens", u.get("reasoning_tokens"))
        raw_rows.append({"model": r.get("model"), "budget": r.get("budget"),
                         "item": r.get("item_id", r.get("item", r.get("item_key"))),
                         "condition": r.get("condition"),
                         "reasoning_tokens": rt,
                         "completion_tokens": u.get("completion_tokens")})
raw = pd.DataFrame(raw_rows)
rep(f" raw responses: {len(raw)} calls; reasoning_tokens logged for "
    f"{raw['reasoning_tokens'].notna().sum()}")

# --------------------------------------------------------------------------
# A. Realized reasoning tokens
# --------------------------------------------------------------------------
rep("\n" + "-" * 76)
rep(" A. REALIZED REASONING TOKENS PER MODEL x REQUESTED BUDGET")
rep("-" * 76)
tok = (raw.groupby(["model", "budget"])["reasoning_tokens"]
          .agg(["count", "mean", "median", "std", "max"]).reset_index())
tok["pct_over_ceiling"] = np.nan
for i, r in tok.iterrows():
    if r["budget"] and r["budget"] > 0:
        sub = raw[(raw["model"] == r["model"]) & (raw["budget"] == r["budget"])]
        tok.loc[i, "pct_over_ceiling"] = (sub["reasoning_tokens"] > r["budget"]).mean()
tok.to_csv(OUT / "realized_tokens.csv", index=False)

rep(f"\n {'model':<28}{'budget':>8}{'mean':>8}{'median':>8}{'max':>8}{'>ceiling':>10}")
for _, r in tok.sort_values(["model", "budget"]).iterrows():
    oc = "" if np.isnan(r["pct_over_ceiling"]) else f"{100*r['pct_over_ceiling']:.0f}%"
    rep(f" {r['model']:<28}{int(r['budget']):>8}{r['mean']:>8.0f}{r['median']:>8.0f}"
        f"{r['max']:>8.0f}{oc:>10}")

# LaTeX table
reasoning_models = [m["name"] for m in cfg["models"] if m.get("reasoning")]
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Realized reasoning tokens per call (mean over all calls in the cell) against the requested ceiling.}",
         r"\label{tab:tokens}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrr}", r"\hline",
         r"Model & $b=0$ & $b=1{,}024$ & $b=4{,}096$ & $b=8{,}192$ \\", r"\hline"]
for m in reasoning_models:
    cells = []
    for b in [0, 1024, 4096, 8192]:
        v = tok[(tok["model"] == m) & (tok["budget"] == b)]["mean"]
        cells.append(f"{v.iloc[0]:,.0f}" if len(v) else "--")
    lines.append(f"{m.replace('_', chr(92)+'_')} & " + " & ".join(cells) + r" \\")
lines += [r"\hline", r"\end{tabular}",
          r"\par\smallskip\scriptsize The ceiling was honoured for Claude; converted to an effort tier for GPT-5.1 (flat consumption); and not enforced for DeepSeek-R1 and Qwen3, whose consumption exceeded the 1{,}024 ceiling. $b=0$ consumption is zero only where thinking could be disabled.",
          r"\end{table}"]
(OUT / "table_realized_tokens.tex").write_text("\n".join(lines) + "\n")

# merge realized tokens per cell into master (mean over the cell's calls)
item_col = next((c for c in ["item_id", "item", "item_key"] if c in master.columns), None)
cell_tok = (raw.groupby(["model", "budget", "item"])["reasoning_tokens"]
               .mean().reset_index().rename(columns={"reasoning_tokens": "realized_tokens"}))
if item_col:
    master = master.merge(cell_tok, left_on=["model", "budget", item_col],
                          right_on=["model", "budget", "item"], how="left")
else:
    mb = raw.groupby(["model", "budget"])["reasoning_tokens"].mean().reset_index() \
            .rename(columns={"reasoning_tokens": "realized_tokens"})
    master = master.merge(mb, on=["model", "budget"], how="left")
master["log_realized"] = np.log2(master["realized_tokens"].fillna(0) + 1)
master["abs_bias"] = master["bias_score"].abs()

# --------------------------------------------------------------------------
# RQ2 on realized dose
# --------------------------------------------------------------------------
rep("\n" + "-" * 76)
rep(" RQ2 RE-RUN: bias magnitude on REALIZED reasoning tokens")
rep("-" * 76)
import statsmodels.formula.api as smf  # noqa: E402

rq2_rows = []
for m in reasoning_models:
    sub = master[(master["model"] == m) & master["realized_tokens"].notna()].copy()
    if sub["budget"].nunique() < 2:
        continue
    rng_tok = (sub["realized_tokens"].min(), sub["realized_tokens"].max())
    # per-item OLS slope on log2(realized+1), then one-sample t across items
    slopes = []
    for it, g in sub.groupby(item_col or "budget"):
        if g["log_realized"].nunique() >= 2:
            slopes.append(np.polyfit(g["log_realized"], g["abs_bias"], 1)[0])
    slopes = np.array(slopes)
    t, p = stats.ttest_1samp(slopes, 0) if len(slopes) > 2 else (np.nan, np.nan)
    # mixed model with random item intercept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mm = smf.mixedlm("abs_bias ~ log_realized", sub, groups=sub[item_col]).fit()
            b_lmm, p_lmm = mm.params["log_realized"], mm.pvalues["log_realized"]
            mm_s = smf.mixedlm("bias_score ~ log_realized", sub, groups=sub[item_col]).fit()
            b_s, p_s = mm_s.params["log_realized"], mm_s.pvalues["log_realized"]
        except Exception as e:
            b_lmm = p_lmm = b_s = p_s = np.nan
    rq2_rows.append(dict(model=m, tok_min=rng_tok[0], tok_max=rng_tok[1],
                         n_items=len(slopes), slope_mean=slopes.mean(), t=t, df=len(slopes)-1,
                         p=p, beta_lmm=b_lmm, p_lmm=p_lmm, beta_signed=b_s, p_signed=p_s))
    rep(f"\n {m}: realized tokens span {rng_tok[0]:.0f} to {rng_tok[1]:.0f} per cell")
    rep(f"   per-item OLS slope of |s| on log2(realized+1): mean = {slopes.mean():+.4f}, "
        f"t({len(slopes)-1}) = {t:.2f}, {fmt_p(p)}")
    rep(f"   mixed model |s|:  beta = {b_lmm:+.4f}, {fmt_p(p_lmm)}")
    rep(f"   mixed model  s :  beta = {b_s:+.4f}, {fmt_p(p_s)}")
pd.DataFrame(rq2_rows).to_csv(OUT / "rq2_realized_dose.csv", index=False)

# --------------------------------------------------------------------------
# RQ1 at the item level
# --------------------------------------------------------------------------
rep("\n" + "-" * 76)
rep(" RQ1 RE-RUN: item-level pooled contrast (30 independent pairs)")
rep("-" * 76)
fam_of = {m["name"]: m.get("pair", m.get("family")) for m in cfg["models"]}
master["family"] = master["model"].map(fam_of)
master["reasoning"] = master["model"].map({m["name"]: bool(m.get("reasoning")) for m in cfg["models"]})

diffs_by_item = {}
fam_rows = []
for fam, g in master.groupby("family"):
    reas = g[(g["reasoning"]) & (g["budget"] > 0)].groupby(item_col)["abs_bias"].mean()
    inst = g[(~g["reasoning"]) | (g["budget"] == 0)]
    # comparator at budget 0: instruct sibling, or same model with thinking off
    inst = inst[inst["budget"] == 0].groupby(item_col)["abs_bias"].mean()
    j = pd.concat([reas, inst], axis=1, keys=["r", "i"]).dropna()
    d = j["r"] - j["i"]
    t, p = stats.ttest_rel(j["r"], j["i"])
    w = stats.wilcoxon(j["r"], j["i"])
    fam_rows.append(dict(family=fam, n=len(j), mean_reas=j["r"].mean(), mean_inst=j["i"].mean(),
                         diff=d.mean(), t=t, df=len(j)-1, p=p, dz=d.mean()/d.std(ddof=1),
                         p_w=w.pvalue))
    for it, v in d.items():
        diffs_by_item.setdefault(it, []).append(v)
fam_df = pd.DataFrame(fam_rows)
fam_df["p_holm"] = holm(fam_df["p"])
rep(f"\n {'family':<10}{'n':>4}{'diff':>9}{'t':>8}{'df':>5}{'p':>8}{'p_holm':>8}{'dz':>7}")
for _, r in fam_df.iterrows():
    rep(f" {r['family']:<10}{r['n']:>4}{r['diff']:>+9.3f}{r['t']:>8.2f}{r['df']:>5}"
        f"{r['p']:>8.3f}{r['p_holm']:>8.3f}{r['dz']:>7.2f}")

item_means = np.array([np.mean(v) for v in diffs_by_item.values()])
t_pool, p_pool = stats.ttest_1samp(item_means, 0)
w_pool = stats.wilcoxon(item_means)
ci = boot_ci(item_means)
dz_pool = item_means.mean() / item_means.std(ddof=1)
rep(f"\n POOLED, one mean difference per item (n = {len(item_means)} items, each the average over "
    f"{fam_df['n'].min()}-{len(fam_df)} families):")
rep(f"   Delta = {item_means.mean():+.3f}, 95% bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
    f"t({len(item_means)-1}) = {t_pool:.2f}, {fmt_p(p_pool)}, dz = {dz_pool:.2f}; "
    f"Wilcoxon {fmt_p(w_pool.pvalue)}")
# mixed model alternative: diff ~ 1 + (1|item) with family as fixed effect
long = pd.DataFrame([(it, f, v) for it, vs in diffs_by_item.items()
                     for f, v in zip(fam_df["family"], vs)], columns=["item", "family", "diff"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mm1 = smf.mixedlm("diff ~ 1", long, groups=long["item"]).fit()
rep(f"   mixed model intercept (random item): beta = {mm1.params['Intercept']:+.4f}, "
    f"z = {mm1.tvalues['Intercept']:.2f}, {fmt_p(mm1.pvalues['Intercept'])}")
fam_df.to_csv(OUT / "rq1_item_level.csv", index=False)

# --------------------------------------------------------------------------
# RQ3 with item clustering
# --------------------------------------------------------------------------
rep("\n" + "-" * 76)
rep(" RQ3 RE-RUN: signed score vs 0 per bias, mixed model with random item intercept")
rep("-" * 76)
rq3_rows = []
for bias, g in master.groupby("bias"):
    g = g.dropna(subset=["bias_score"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = smf.mixedlm("bias_score ~ 1", g, groups=g[item_col]).fit()
    # item-clustered bootstrap CI
    groups = [x["bias_score"].values for _, x in g.groupby(item_col)]
    rng = np.random.default_rng(42)
    bs = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
          for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    d = g["bias_score"].mean() / g["bias_score"].std(ddof=1)
    rq3_rows.append(dict(bias=bias, n_cells=len(g), n_items=g[item_col].nunique(),
                         M=g["bias_score"].mean(), ci_lo=lo, ci_hi=hi,
                         beta=mm.params["Intercept"], z=mm.tvalues["Intercept"],
                         p=mm.pvalues["Intercept"], d=d))
rq3 = pd.DataFrame(rq3_rows)
rq3["p_holm"] = holm(rq3["p"])
rq3["class"] = np.where(rq3["p_holm"] >= .05, "Absent",
                        np.where(rq3["M"] > 0, "Present", "Reversed"))
rep(f"\n {'bias':<26}{'M':>8}{'CI':>20}{'z':>8}{'p_holm':>9}{'d':>7}  class")
for _, r in rq3.iterrows():
    rep(f" {r['bias']:<26}{r['M']:>+8.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        f"{r['z']:>8.2f}{r['p_holm']:>9.3f}{r['d']:>7.2f}  {r['class']}")
rq3.to_csv(OUT / "rq3_mixed.csv", index=False)

rep("\n" + "=" * 76)
rep(" Report saved -> 06_camera_ready/outputs/_camera_ready_apa_report_.txt")
rep("=" * 76)
(OUT / "_camera_ready_apa_report_.txt").write_text("\n".join(REP) + "\n")
