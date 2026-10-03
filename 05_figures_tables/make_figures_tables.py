#!/usr/bin/env python3
"""
05_figures_tables/make_figures_tables.py

Writes every value plotted in the paper's three figures and the five LaTeX
tables. Run after the 04_analysis scripts.

Figures: the PDFs in this folder were styled interactively from the CSVs
below; this script writes the plotted values, not the artwork.
  fig1_dose_response_values.csv
      Fig. 1: mean |s| and s per reasoning model x requested budget, with
      95% bootstrap CIs over item cells
  fig2_bias_model_heatmap_values.csv
      Fig. 2: mean signed score per bias x model
  fig3_verbalization_dumbbell_values.csv
      Fig. 3: mean anchoring score per model, standard vs verbalized

Tables (numbered as in the paper):
  table1_design.tex           I    model panel and design cells
  table2_realized_tokens.tex  II   realized reasoning tokens per call
  table3_rq1.tex              III  RQ1 reasoning vs non-reasoning
  table4_rq2.tex              IV   RQ2 trend on realized tokens
  table5_rq3.tex              V    RQ3 presence classification

Usage:
  python 05_figures_tables/make_figures_tables.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "04_analysis"))

import pandas as pd

from analysis_utils import load_master, bootstrap_ci_mean, reasoning_models

ANALYSIS = ROOT / "04_analysis"

MODEL_LABELS = {
    "claude-haiku-4.5": "Claude Haiku (instruct)",
    "claude-haiku-4.5-thinking": "Claude Haiku (thinking)",
    "gpt-5.1-reasoning": "GPT-5.1 (reasoning)",
    "deepseek-v3": "DeepSeek-V3 (instruct)",
    "deepseek-r1": "DeepSeek-R1 (thinking)",
    "qwen3-instruct": "Qwen3 (instruct)",
    "qwen3-thinking": "Qwen3 (thinking)",
}
SHORT_MODEL = {"claude-haiku-4.5-thinking": "claude-th.",
               "gpt-5.1-reasoning": "gpt-5.1",
               "qwen3-thinking": "qwen3-th."}
BIAS_LABELS = {
    "Anchoring": "Anchoring",
    "Framing Effect": "Framing",
    "Loss Aversion": "Loss aversion",
    "Escalation Of Commitment": "Escalation",
    "Availability Heuristic": "Availability",
    "Confirmation Bias": "Confirmation",
}
CLASS_ORDER = {"Present": 0, "Reversed": 1, "Directional": 2, "Absent": 3}
BUDGETS = [0, 1024, 4096, 8192]


def signed_nolead(v, dec=3):
    """+0.163 -> '+.163'"""
    return f"{v:+.{dec}f}".replace("0.", ".", 1)


def p_bare(p):
    return "<.001" if p < .001 else f"{p:.3f}".replace("0.", ".", 1)


def tex_table(caption, label, colspec, header, rows, note, env="table",
              tabcolsep=None):
    lines = [rf"\begin{{{env}}}[!t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\footnotesize"]
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\hline",
              " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}",
              rf"\par\smallskip\footnotesize {note}", rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def write(name, text):
    (HERE / name).write_text(text, encoding="utf-8")
    print(f"  {name}")


def main():
    df, cfg = load_master()
    print("Writing figure values and tables to", HERE.relative_to(ROOT))

    # ---------------------------------------------------------- Fig. 1 -----
    rows = []
    for m in reasoning_models(cfg):
        for b, g in df[df["model"] == m].groupby("budget"):
            for col in ("abs_bias", "bias_score"):
                lo, hi = bootstrap_ci_mean(g[col])
                rows.append(dict(model=m, label=MODEL_LABELS[m],
                                 budget=int(b), measure=col,
                                 mean=g[col].mean(), ci_lo=lo, ci_hi=hi,
                                 n_cells=len(g)))
    pd.DataFrame(rows).to_csv(HERE / "fig1_dose_response_values.csv",
                              index=False)
    print("  fig1_dose_response_values.csv")

    # ---------------------------------------------------------- Fig. 2 -----
    heat = (df.groupby(["bias", "model"])["bias_score"].mean()
              .rename("mean_signed_score").reset_index())
    heat["bias_label"] = heat["bias"].map(BIAS_LABELS)
    heat["model_label"] = heat["model"].map(MODEL_LABELS)
    heat["bias"] = pd.Categorical(heat["bias"], cfg["biases"], ordered=True)
    heat["model"] = pd.Categorical(heat["model"], list(MODEL_LABELS),
                                   ordered=True)
    heat.sort_values(["bias", "model"]).to_csv(
        HERE / "fig2_bias_model_heatmap_values.csv", index=False)
    print("  fig2_bias_model_heatmap_values.csv")

    # ---------------------------------------------------------- Fig. 3 -----
    verb = pd.read_csv(ANALYSIS / "rq4_verbalization/outputs/rq4_models.csv")
    verb.insert(1, "label", verb["model"].map(MODEL_LABELS))
    verb.to_csv(HERE / "fig3_verbalization_dumbbell_values.csv", index=False)
    print("  fig3_verbalization_dumbbell_values.csv")

    # --------------------------------------------------------- Table I -----
    rows = []
    for m in cfg["models"]:
        sub = df[df["model"] == m["name"]]
        budgets = sorted(int(b) for b in sub["budget"].unique())
        kind = ("no-disable" if m.get("cannot_disable_thinking") else
                "reasoning" if m["reasoning"] else "instruct")
        rows.append([m["name"], m["pair"], kind,
                     ", ".join(f"{b // 1024}k" if b else "0"
                               for b in budgets), len(sub)])
    write("table1_design.tex", tex_table(
        "Model panel and design cells per model. Budgets are requested "
        "extended-thinking token ceilings.", "tab:design", "lllll",
        ["Model", "Family", "Type", "Budgets", "Cells"], rows,
        "``no-disable'' models cannot switch thinking off; their instruct "
        "siblings provide the family's budget-0 point.", tabcolsep="3.5pt"))

    # -------------------------------------------------------- Table II -----
    tok = pd.read_csv(ANALYSIS / "manipulation_check/outputs/"
                      "realized_tokens.csv")
    rows = []
    for m in reasoning_models(cfg):
        cells = []
        for b in BUDGETS:
            v = tok[(tok["model"] == m) & (tok["budget"] == b)]["mean"]
            cells.append(f"{v.iloc[0]:,.0f}" if len(v) else "--")
        rows.append([m] + cells)
    over = tok[tok["budget"] == 1024].set_index("model")["pct_over_ceiling"]
    write("table2_realized_tokens.tex", tex_table(
        "Realized reasoning tokens per call (cell means) against the "
        "requested ceiling.", "tab:tokens", "lrrrr",
        ["Model", "$b=0$", "$b=1{,}024$", "$b=4{,}096$", "$b=8{,}192$"],
        rows,
        "Honoured for Claude; converted to an effort tier for GPT-5.1 (flat "
        "consumption); not enforced for DeepSeek-R1 and Qwen3, where "
        f"{100 * over['deepseek-r1']:.0f}\\% and "
        f"{100 * over['qwen3-thinking']:.0f}\\% of calls exceeded the 1,024 "
        "ceiling. The $b=0$ cells for the two no-disable models are the "
        "excluded conditions described in the text.", tabcolsep="4pt"))

    # ------------------------------------------------------- Table III -----
    rq1 = pd.read_csv(ANALYSIS / "rq1_reasoning_vs_bias/outputs/"
                      "rq1_results.csv")
    r = rq1[rq1["measure"] == "abs_bias"]
    fam = r[r["family"] != "pooled_item_level"]
    pooled = r[r["family"] == "pooled_item_level"].iloc[0]
    rows = [[x.family, f"{x.mean_reasoning:.3f}", f"{x.mean_comparator:.3f}",
             f"${x.mean_diff:+.3f}$", f"$[{x.ci_lo:.3f}, {x.ci_hi:.3f}]$",
             f"{x.t:.2f}", p_bare(x.p_holm), f"{x.dz:.2f}"]
            for x in fam.itertuples()]
    write("table3_rq1.tex", tex_table(
        "RQ1: bias magnitude $|s|$ for reasoning models (budgets $>0$) "
        "versus matched non-reasoning comparators, paired by item ($n=30$).",
        "tab:rq1", "lccccccc",
        ["Family", r"$M_{\text{reas}}$", r"$M_{\text{instr}}$", r"$\Delta$",
         r"95\% CI", "$t(29)$", r"$p_{\text{Holm}}$", "$d_z$"], rows,
        "Positive $\\Delta$ = reasoning model more biased. Pooled at the item "
        f"level: $\\Delta = {pooled.mean_diff:+.3f}$, "
        f"$t({int(pooled.df)})={pooled.t:.2f}$, $p={p_bare(pooled.p)}$, "
        f"$d_z={pooled.dz:.2f}$.", tabcolsep="2.6pt"))

    # -------------------------------------------------------- Table IV -----
    rq2 = pd.read_csv(ANALYSIS / "rq2_dose_response/outputs/rq2_results.csv")
    rows = [[SHORT_MODEL.get(x.model, x.model),
             f"{x.tok_min:.0f}--{x.tok_max:.0f}",
             f"${x.t:.2f}$, {p_bare(x.p)}",
             f"${x.beta_abs:+.3f}$ ({p_bare(x.p_abs)})",
             f"${x.beta_signed:+.3f}$ ({p_bare(x.p_signed)})"]
            for x in rq2.itertuples()]
    write("table4_rq2.tex", tex_table(
        r"RQ2: trend of bias on $\log_2(\text{realized reasoning tokens}+1)$, "
        "per model, within the realized range.", "tab:rq2", "lcccc",
        ["Model", "Range (tokens)", "slope $t(29)$, $p$",
         r"$\beta_{|s|}$ ($p$)", r"$\beta_{s}$ ($p$)"], rows,
        "Range is the span of cell-mean realized tokens. ``slope'' tests the "
        "per-item OLS slope of $|s|$ against zero; $\\beta$ coefficients are "
        "from mixed models with random item intercepts, per doubling of "
        "realized tokens.", tabcolsep="3pt"))

    # --------------------------------------------------------- Table V -----
    rq3 = pd.read_csv(ANALYSIS / "rq3_bias_taxonomy/outputs/rq3_presence.csv")
    rq3 = rq3.assign(_o=rq3["classification"].map(CLASS_ORDER)) \
             .sort_values(["_o", "p_holm"], kind="stable")
    rows = [[BIAS_LABELS[x.bias], f"${signed_nolead(x.mean)}$",
             f"$[{signed_nolead(x.ci_lo)}, {signed_nolead(x.ci_hi)}]$",
             f"${x.t:.2f}$", p_bare(x.p_holm), f"${x.d:+.2f}$",
             x.classification] for x in rq3.itertuples()]
    write("table5_rq3.tex", tex_table(
        "RQ3: presence classification per bias. Signed score vs.\\ 0 over 85 "
        "cells; $t(4)$ on the five item means; item-clustered bootstrap "
        "CIs; Holm-corrected.", "tab:rq3", "lcccccl",
        ["Bias", "$M$", r"95\% CI", "$t(4)$", r"$p_{\text{H}}$", "$d$",
         "Class"], rows,
        "$d$ is the cell-level standardized mean. ``Present'' = human-like "
        "direction; ``Reversed'' = reliable shift against it at the item "
        "level; ``Directional'' = same sign in all seven models, but the "
        "item-level test does not reach .05 after correction with five items "
        "per bias.", tabcolsep="1.9pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
