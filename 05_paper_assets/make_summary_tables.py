#!/usr/bin/env python3
"""
=============================================================================
 ANCHORED MINDS | 05_paper_assets/make_summary_tables.py
=============================================================================
 Builds camera-ready tables for the IEEE two-column template from the
 RQ analysis outputs. Every table is saved twice: .tex (booktabs-free,
 IEEEtran-compatible \\begin{tabular}) and .md (for the repo README /
 quick inspection).

 Tables produced:
   table1_design.tex/.md          model panel + design descriptives
   table2_rq1_pairs.tex/.md       reasoning vs non-reasoning by family
   table3_rq3_taxonomy.tex/.md    bias presence classification
   table4_rq2_dose.tex/.md        dose-response trend statistics
   table5_rq4_verbalization.tex/.md

 Usage:
   python 05_paper_assets/make_summary_tables.py
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "outputs"


def fmt_p(p):
    if p is None or p != p:
        return "--"
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def save(name, tex_lines, md_lines):
    (OUT / f"{name}.tex").write_text("\n".join(tex_lines) + "\n",
                                     encoding="utf-8")
    (OUT / f"{name}.md").write_text("\n".join(md_lines) + "\n",
                                    encoding="utf-8")
    print(f"  saved {name}.tex / .md")


def tex_table(caption, label, colspec, header, rows, notes=""):
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small",
             rf"\begin{{tabular}}{{{colspec}}}", r"\hline",
             " & ".join(header) + r" \\", r"\hline"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\footnotesize {notes}")
    lines.append(r"\end{table}")
    return lines


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def main():
    import pandas as pd
    import yaml

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    master = pd.read_csv(ROOT / cfg["paths"]["processing_dir"]
                         / "master_scores.csv")
    print("=" * 70)
    print(" ANCHORED MINDS -- MAKE SUMMARY TABLES")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # ------------------------------------------------------- Table 1: design
    rows, mdrows = [], []
    for m in cfg["models"]:
        name = m["name"]
        sub = master[master["model"] == name]
        if not len(sub):
            continue
        budgets = sorted(int(b) for b in sub["budget"].unique())
        note = ("cannot disable thinking"
                if m.get("cannot_disable_thinking") else
                ("reasoning" if m["reasoning"] else "instruct"))
        b_str = ", ".join(f"{b//1024}k" if b else "0" for b in budgets)
        rows.append([name.replace("_", r"\_"), m["pair"], note, b_str,
                     len(sub)])
        mdrows.append([name, m["pair"], note, b_str, len(sub)])
    save("table1_design",
         tex_table("Model panel and design cells per model.",
                   "tab:design", "lllll",
                   ["Model", "Family", "Type", "Budgets", "Cells"], rows,
                   "Budgets are extended-thinking token ceilings; models "
                   "that cannot disable thinking contribute budgets $>0$ "
                   "only, with the instruct sibling providing the family's "
                   "zero point."),
         md_table(["Model", "Family", "Type", "Budgets", "Cells"], mdrows))

    # ---------------------------------------------------------- Table 2: RQ1
    rq1 = pd.read_csv(ROOT / "04_analysis/rq1_reasoning_vs_bias/outputs/"
                      "rq1_results.csv")
    r = rq1[rq1["measure"] == "abs_bias"]
    rows = [[x["family"], f"{x['mean_reasoning']:.3f}",
             f"{x['mean_nonreasoning']:.3f}", f"{x['mean_diff']:+.3f}",
             f"[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]",
             f"{x['t']:.2f}", fmt_p(x["p_holm"]), f"{x['dz']:.2f}"]
            for _, x in r.iterrows()]
    hdr = ["Family", "$M_{reason}$", "$M_{instruct}$", "$\\Delta$",
           "95\\% CI", "$t(29)$", "$p_{Holm}$", "$d_z$"]
    save("table2_rq1_pairs",
         tex_table("RQ1: bias magnitude $|s|$ for reasoning models "
                   "(budgets $>0$) vs matched non-reasoning comparators, "
                   "paired by item.", "tab:rq1", "lccccccc",
                   hdr, rows,
                   "Positive $\\Delta$ = reasoning model MORE biased. "
                   "Pooled across families: $\\Delta=+0.031$, "
                   "$t(119)=2.00$, $p=.048$, $d_z=0.18$."),
         md_table(["Family", "M_reason", "M_instruct", "diff", "95% CI",
                   "t(29)", "p_Holm", "dz"],
                  [[x["family"], f"{x['mean_reasoning']:.3f}",
                    f"{x['mean_nonreasoning']:.3f}",
                    f"{x['mean_diff']:+.3f}",
                    f"[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]",
                    f"{x['t']:.2f}", fmt_p(x["p_holm"]), f"{x['dz']:.2f}"]
                   for _, x in r.iterrows()]))

    # ---------------------------------------------------------- Table 3: RQ3
    rq3 = pd.read_csv(ROOT / "04_analysis/rq3_bias_taxonomy/outputs/"
                      "rq3_results.csv")
    pres = rq3[rq3["part"] == "presence"]
    rows = [[x["bias"], f"{x['mean']:+.3f}",
             f"[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]",
             f"{x['t']:.2f}", fmt_p(x["p_holm"]), f"{x['d']:.2f}",
             x["classification"]] for _, x in pres.iterrows()]
    save("table3_rq3_taxonomy",
         tex_table("RQ3: presence classification of six classic biases "
                   "(signed score vs 0; item-clustered bootstrap CIs; "
                   "Holm-corrected).", "tab:rq3", "lcccccl",
                   ["Bias", "$M$", "95\\% CI", "$t(84)$", "$p_{Holm}$",
                    "$d$", "Class"], rows,
                   "PRESENT = human-like direction; REVERSED = "
                   "significant shift against the classic direction."),
         md_table(["Bias", "M", "95% CI", "t(84)", "p_Holm", "d", "Class"],
                  rows))

    # ---------------------------------------------------------- Table 4: RQ2
    rq2 = pd.read_csv(ROOT / "04_analysis/rq2_dose_response/outputs/"
                      "rq2_results.csv")
    r = rq2[rq2["measure"] == "abs_bias"]
    rows = [[x["model"].replace("_", r"\_"), f"{x['spearman_rho']:+.3f}",
             fmt_p(x["spearman_p"]), f"{x['slope_mean']:+.4f}",
             f"[{x['slope_ci_lo']:.4f}, {x['slope_ci_hi']:.4f}]",
             fmt_p(x["slope_p"]),
             (f"{x['lmm_beta']:+.4f}" if x["lmm_beta"] == x["lmm_beta"]
              else "--"), fmt_p(x["lmm_p"])] for _, x in r.iterrows()]
    save("table4_rq2_dose",
         tex_table("RQ2: dose-response of bias magnitude on reasoning "
                   "budget (log$_2$(budget+1)).", "tab:rq2", "lccccccc",
                   ["Model", r"$\rho$", "$p_\\rho$", "slope",
                    "95\\% CI", "$p$", r"$\beta_{LMM}$", "$p_{LMM}$"],
                   rows,
                   "Per-item OLS slopes tested against 0; mixed model has "
                   "random item intercepts."),
         md_table(["Model", "rho", "p_rho", "slope", "95% CI", "p",
                   "beta_LMM", "p_LMM"], rows))

    # ---------------------------------------------------------- Table 5: RQ4
    rq4 = pd.read_csv(ROOT / "04_analysis/rq4_verbalization/outputs/"
                      "rq4_results.csv")
    rows = [[x["model"].replace("_", r"\_"), f"{x['mean_standard']:.3f}",
             f"{x['mean_verbalized']:.3f}", f"{x['mean_diff']:+.3f}",
             f"{x['t']:.2f}", int(x["df"]), fmt_p(x["p_holm"]),
             f"{x['dz']:.2f}"] for _, x in rq4.iterrows()]
    save("table5_rq4_verbalization",
         tex_table("RQ4: anchoring score with vs without forced anchor "
                   "verbalization (paired by model$\\times$budget$\\times$"
                   "item cell).", "tab:rq4", "lccccccc",
                   ["Model", "$M_{std}$", "$M_{verb}$", "$\\Delta$",
                    "$t$", "$df$", "$p_{Holm}$", "$d_z$"], rows,
                   "Negative $\\Delta$ = verbalization reduces anchoring."),
         md_table(["Model", "M_std", "M_verb", "diff", "t", "df",
                   "p_Holm", "dz"], rows))

    print("\n All tables ->", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
