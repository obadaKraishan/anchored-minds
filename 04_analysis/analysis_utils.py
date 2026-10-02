#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 04_analysis/analysis_utils.py
=============================================================================
 Shared helpers for the four RQ analysis scripts: data loading, statistics
 (paired tests, bootstrap CIs, effect sizes), and APA7/IEEE-style formatting.
 Each rq*_analysis.py imports from here but remains independently runnable.
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

RNG_SEED = 42
N_BOOT = 10000


# ---------------------------------------------------------------- data ------
def load_master():
    """Load master_scores.csv; add |bias| magnitude and family labels."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    path = ROOT / cfg["paths"]["processing_dir"] / "master_scores.csv"
    if not path.exists():
        raise SystemExit(f"[ERROR] {path} not found -- run the "
                         "02_data_processing pipeline first.")
    df = pd.read_csv(path)
    df["abs_bias"] = df["bias_score"].abs()
    return df, cfg


def family_comparators(df):
    """For each pair family: the non-reasoning comparator rows (b0).

    If the family has a dedicated non-reasoning sibling, use it; otherwise
    (e.g. OpenAI) fall back to the reasoning model at budget 0 with thinking
    disabled. Returns dict family -> (comparator_df, comparator_label).
    """
    out = {}
    for fam, sub in df.groupby("pair"):
        nonreason = sub[(~sub["reasoning"]) & (sub["budget"] == 0)]
        if len(nonreason):
            out[fam] = (nonreason, nonreason["model"].iloc[0])
        else:
            fallback = sub[(sub["reasoning"]) & (sub["budget"] == 0)]
            if len(fallback):
                out[fam] = (fallback,
                            fallback["model"].iloc[0] + " (b0, thinking off)")
    return out


# ------------------------------------------------------------ statistics ----
def bootstrap_ci_mean(x, n_boot=N_BOOT, seed=RNG_SEED, ci=95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def paired_tests(x, y):
    """Paired t + Wilcoxon + Cohen's dz for x vs y (same length, paired)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    d = x - y
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return dict(n=n, t=np.nan, df=n - 1, p_t=np.nan, W=np.nan,
                    p_w=np.nan, dz=np.nan, mean_diff=float(np.mean(d)) if n else np.nan,
                    ci=(np.nan, np.nan))
    t, p_t = stats.ttest_rel(x, y)
    try:
        W, p_w = stats.wilcoxon(x, y)
    except ValueError:
        W, p_w = np.nan, np.nan
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return dict(n=n, t=float(t), df=n - 1, p_t=float(p_t),
                W=float(W) if W == W else np.nan,
                p_w=float(p_w) if p_w == p_w else np.nan,
                dz=float(dz) if dz == dz else np.nan,
                mean_diff=float(d.mean()),
                ci=bootstrap_ci_mean(d))


def one_sample_test(x, popmean=0.0):
    """One-sample t vs popmean + bootstrap CI + Cohen's d."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return dict(n=n, mean=float(np.mean(x)) if n else np.nan,
                    sd=np.nan, t=np.nan, df=n - 1, p=np.nan, d=np.nan,
                    ci=(np.nan, np.nan))
    t, p = stats.ttest_1samp(x, popmean)
    d = (x.mean() - popmean) / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan
    return dict(n=n, mean=float(x.mean()), sd=float(x.std(ddof=1)),
                t=float(t), df=n - 1, p=float(p),
                d=float(d) if d == d else np.nan, ci=bootstrap_ci_mean(x))


def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values (order-preserving)."""
    p = np.asarray(pvals, float)
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan)
    prev = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        val = min((m - rank) * p[idx], 1.0)
        prev = max(prev, val)
        adj[idx] = prev
    return adj


def spearman_trend(budgets, values):
    """Spearman rank correlation of value against budget (dose-response)."""
    b, v = np.asarray(budgets, float), np.asarray(values, float)
    mask = ~(np.isnan(b) | np.isnan(v))
    if mask.sum() < 4 or len(set(b[mask])) < 2:
        return np.nan, np.nan
    rho, p = stats.spearmanr(b[mask], v[mask])
    return float(rho), float(p)


# ------------------------------------------------------------ formatting ----
def fmt_p(p):
    if p != p:
        return "p = n/a"
    if p < .001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def fmt_stat(v, dec=2):
    return "n/a" if v != v else f"{v:.{dec}f}"


def apa_paired(label, r):
    """One APA-style sentence for a paired comparison result dict."""
    sig = "significant" if (r["p_t"] == r["p_t"] and r["p_t"] < .05) \
        else "not significant"
    return (f"{label}: mean difference = {fmt_stat(r['mean_diff'], 3)} "
            f"(95% bootstrap CI [{fmt_stat(r['ci'][0], 3)}, "
            f"{fmt_stat(r['ci'][1], 3)}]), t({r['df']}) = {fmt_stat(r['t'])}, "
            f"{fmt_p(r['p_t'])}, Cohen's dz = {fmt_stat(r['dz'])}; "
            f"Wilcoxon W = {fmt_stat(r['W'], 1)}, {fmt_p(r['p_w'])} "
            f"(n = {r['n']} items, {sig} at alpha = .05).")


def table(rows, headers, widths=None):
    """Plain-text aligned table for the APA report files."""
    widths = widths or [max(len(str(h)), max((len(str(r[i])) for r in rows),
                        default=0)) + 2 for i, h in enumerate(headers)]
    lines = ["".join(str(h).ljust(w) for h, w in zip(headers, widths)),
             "".join("-" * w for w in widths)]
    for r in rows:
        lines.append("".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


def save_fig(fig, out_dir, name):
    """Save a figure as 300-dpi PNG + vector PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"      figure -> {name}.png/.pdf")
