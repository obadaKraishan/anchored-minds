#!/usr/bin/env python3
"""
04_analysis/analysis_utils.py

Shared helpers for the analysis scripts: data loading (scored cells and
per-call realized reasoning tokens), statistics, and report formatting.

Inference conventions used throughout (see the paper's Method):
  - The battery item is the unit of independent observation. Pooled tests
    aggregate to the item level first (30 items); per-bias tests use t(4) on
    the five item means of that bias.
  - Bootstrap CIs are percentile intervals with a fixed seed.
  - Holm correction is applied within each family of tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

RNG_SEED = 42
N_BOOT = 10000            # resamples for cell- or item-level bootstrap CIs
N_BOOT_CLUSTER = 5000     # resamples for item-clustered bootstrap CIs


# ---------------------------------------------------------------- data ------
def load_config():
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def load_master():
    """Load master_scores.csv (one row per model x budget x item cell) and
    add abs_bias = |bias_score|."""
    cfg = load_config()
    path = ROOT / cfg["paths"]["processing_dir"] / "master_scores.csv"
    if not path.exists():
        raise SystemExit(f"[ERROR] {path.relative_to(ROOT)} not found -- run "
                         "the 02_data_processing pipeline first.")
    df = pd.read_csv(path)
    df["abs_bias"] = df["bias_score"].abs()
    return df, cfg


def load_usage(cfg):
    """Load per-call token usage (data/per_call_usage.csv)."""
    path = ROOT / cfg["paths"]["usage_csv"]
    if not path.exists():
        raise SystemExit(f"[ERROR] {path.relative_to(ROOT)} not found -- run "
                         "02_data_processing/extract_usage.py first.")
    return pd.read_csv(path)


def attach_realized_tokens(df, usage):
    """Add the realized dose to each scored cell.

    realized_tokens = mean reasoning tokens over every call in the
    model x budget x item cell (all conditions); log_realized =
    log2(realized_tokens + 1).
    """
    cell = (usage.groupby(["model", "budget", "item_id"])["reasoning_tokens"]
                 .mean().rename("realized_tokens").reset_index())
    out = df.merge(cell, on=["model", "budget", "item_id"], how="left")
    out["log_realized"] = np.log2(out["realized_tokens"].fillna(0) + 1)
    return out


def reasoning_models(cfg):
    return [m["name"] for m in cfg["models"] if m.get("reasoning")]


def family_comparators(df):
    """For each family: the budget-0 comparator rows and their label.

    The comparator is the family's non-reasoning sibling at budget 0. A family
    without a sibling (OpenAI) uses its reasoning model at budget 0 with
    thinking disabled, which holds the weights constant.
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
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def item_clustered_ci(sub, col, cluster="item_id", n_boot=N_BOOT_CLUSTER,
                      seed=RNG_SEED):
    """Bootstrap CI of the cell mean, resampling whole items (cells sharing a
    battery item are not independent)."""
    groups = [g[col].values for _, g in sub.groupby(cluster)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        boots[i] = np.concatenate([groups[j] for j in idx]).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_tests(x, y):
    """Paired t + Wilcoxon + Cohen's dz + bootstrap CI of the mean difference
    for x vs y (same length, paired)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    d = x - y
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return dict(n=n, t=np.nan, df=n - 1, p_t=np.nan, W=np.nan,
                    p_w=np.nan, dz=np.nan,
                    mean_diff=float(np.mean(d)) if n else np.nan,
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


def ols_slope(x, y):
    """Least-squares slope of y on x."""
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def per_item_slopes(sub, xcol, ycol, by=("item_id",)):
    """OLS slope of ycol on xcol within each group (default: each item).
    Groups with fewer than two distinct x values are skipped."""
    out = {}
    for key, g in sub.groupby(list(by)):
        if g[xcol].nunique() < 2:
            continue
        out[key if len(by) > 1 else key[0]] = ols_slope(g[xcol], g[ycol])
    return pd.Series(out, dtype=float)


def mixed_slope(sub, formula, term, group="item_id"):
    """Mixed model with a random intercept per item; returns (beta, p) for
    `term` (REML fit, Wald p-value)."""
    import warnings
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(formula, sub, groups=sub[group]).fit()
            return float(m.params[term]), float(m.pvalues[term]), \
                float(m.tvalues[term])
        except Exception:
            return np.nan, np.nan, np.nan


# ------------------------------------------------------------ formatting ----
def fmt_p(p):
    if p != p:
        return "p = n/a"
    if p < .001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def fmt_p_bare(p):
    """p-value for table cells: '.351', '<.001', 'n/a'."""
    if p != p:
        return "n/a"
    return "<.001" if p < .001 else f"{p:.3f}".replace("0.", ".", 1)


def fmt_stat(v, dec=2):
    return "n/a" if v != v else f"{v:.{dec}f}"


def table(rows, headers, widths=None):
    """Plain-text aligned table for the report files."""
    widths = widths or [max(len(str(h)), max((len(str(r[i])) for r in rows),
                        default=0)) + 2 for i, h in enumerate(headers)]
    lines = ["".join(str(h).ljust(w) for h, w in zip(headers, widths)),
             "".join("-" * w for w in widths)]
    for r in rows:
        lines.append("".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


class Report:
    """Collects printed lines and writes them to a text report."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line, flush=True)
        self.lines.append(line)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
