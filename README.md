# The Long Road to the Same Answer

Code and data for **"The Long Road to the Same Answer: Cognitive Bias Under
Escalating Reasoning Budgets in Large Language Models"**, Obada Kraishan,
*8th IEEE International Conference on Cognitive Machine Intelligence (CogMI
2026)*, to appear.

Paper: *link to be added on publication.*

## Citation

```bibtex
@inproceedings{kraishan2026longroad,
  title     = {The Long Road to the Same Answer: Cognitive Bias Under
               Escalating Reasoning Budgets in Large Language Models},
  author    = {Kraishan, Obada},
  booktitle = {Proceedings of the 8th IEEE International Conference on
               Cognitive Machine Intelligence (CogMI)},
  year      = {2026},
  publisher = {IEEE},
  note      = {To appear}
}
```

A machine-readable citation is in [CITATION.cff](CITATION.cff).

## Summary

If the deliberation of reasoning models works like deliberate human thought,
longer thinking should weaken classic decision biases. We test that
prediction with a dose-response study. A frozen battery of 30 vignettes
covering six biases (anchoring, framing, loss aversion, escalation of
commitment, availability, confirmation), drawn from the benchmark of Malberg
et al. (2025), is given to seven models from four families
(Anthropic, OpenAI, DeepSeek, Qwen). Each reasoning model is paired with a
matched non-reasoning comparator and run at requested thinking ceilings of
0, 1,024, 4,096, and 8,192 tokens, ten samples per cell, for 12,350 API
calls. Because a requested ceiling is not the same as realized deliberation,
the reasoning tokens consumed by every call are logged and used as the dose.
Reasoning models are not less biased than their comparators, and no model
shows a reliable decline in bias magnitude as realized deliberation grows (no
significantly negative slope).
Anchoring is the only bias in the human direction. Framing is reliably
reversed. Escalation,
confirmation, and loss aversion lean against the human direction in every
model, but not reliably at the item level, and availability is absent. A
one-line instruction to restate the anchor lowered anchoring on all five
anchoring items.

## Repository structure

```
config.yaml                 design parameters, model panel, paths (single source of truth)
requirements.txt            pinned dependencies (Python 3.12)
.env.example                API key template (only needed to re-collect data)

00_setup/                   pre-flight check and cost estimate for data collection
01_data_collection/
  download_benchmark.py     fetch the benchmark (pinned HuggingFace revision)
  build_test_battery.py     sample and freeze the 30-item battery
  run_experiments.py        collection harness (OpenRouter, async, checkpointed)
  outputs/battery/          the frozen battery (CSV + JSON) and selection log
  outputs/raw_responses/    one JSONL per model x budget: every API call
02_data_processing/
  parse_responses.py        extract the chosen option from each response
  compute_bias_scores.py    cell-level bias scores -> outputs/master_scores.csv
  validate_dataset.py       grid completeness, cell sizes, ranges
  extract_usage.py          per-call token usage -> data/per_call_usage.csv
  purge_truncated.py        collection utility (see its docstring; not needed to reproduce)
03_codebook/                variable codebook for master_scores.csv
04_analysis/
  analysis_utils.py         shared loading, statistics, formatting
  manipulation_check/       realized reasoning tokens per model x budget
  rq1_reasoning_vs_bias/    reasoning vs matched comparator, items as the unit
  rq2_dose_response/        bias vs log2(realized tokens + 1)
  rq3_bias_taxonomy/        per-bias presence and deliberation sensitivity, t(4)
  rq4_verbalization/        forced anchor verbalization, item level and sign test
05_figures_tables/          plotted-value CSVs, the three figures, LaTeX tables
data/per_call_usage.csv     token usage for all 12,350 calls (no response text)
```

Every analysis script writes its results as CSV plus a plain-text report
(`*_report.txt`) in its own `outputs/` folder.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions require Python 3.12 (numpy 1.26 has no wheels for later
versions). All results were produced with these versions.

## Reproducing the paper

### From the released data (no API key, a few minutes)

Run from the repository root:

```bash
# 1. processing: raw responses -> scored cells
python 02_data_processing/parse_responses.py
python 02_data_processing/compute_bias_scores.py
python 02_data_processing/validate_dataset.py
python 02_data_processing/extract_usage.py
python 03_codebook/generate_codebook.py

# 2. analysis
python 04_analysis/manipulation_check/manipulation_check.py
python 04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py
python 04_analysis/rq2_dose_response/rq2_analysis.py
python 04_analysis/rq3_bias_taxonomy/rq3_analysis.py
python 04_analysis/rq4_verbalization/rq4_analysis.py

# 3. figure values and LaTeX tables
python 05_figures_tables/make_figures_tables.py
```

`parse_responses.py` prints one expected warning: Qwen3-Instruct is below the
95% parse-rate target (92.8%) because it sometimes answers with a raw
quantity instead of an option index, mostly on anchoring items. Those
responses are excluded from scoring, as described in the paper. Every script
exits with status 0 on success and 1 only on a genuine failure (missing or
unreadable input).

Steps 1-3 are deterministic: every bootstrap uses a fixed seed (42), and
rerunning them reproduces every committed CSV exactly (the text reports in
`02_data_processing/outputs/` and `codebook.md` differ only in their
timestamps).

### Re-collecting the data (requires an OpenRouter key; costs money)

```bash
cp .env.example .env                     # add your OPENROUTER_API_KEY
python 00_setup/setup_check.py --ping
python 00_setup/cost_estimator.py
python 01_data_collection/download_benchmark.py
python 01_data_collection/build_test_battery.py   # refuses to overwrite the frozen battery
python 01_data_collection/run_experiments.py --smoke-test
python 01_data_collection/run_experiments.py
```

The harness skips calls already present in `outputs/raw_responses/`, so to
collect a fresh dataset, point `paths.raw_responses_dir` in `config.yaml` to
an empty folder. Sampling is at temperature 1.0 and provider-side models
change over time, so a new collection will not reproduce the released
responses exactly.

## Paper tables and figures

| Paper | Content | Produced by | Output |
|---|---|---|---|
| Table I | Model panel and design cells | `05_figures_tables/make_figures_tables.py` | `05_figures_tables/table1_design.tex` |
| Table II | Realized reasoning tokens per call | `04_analysis/manipulation_check/manipulation_check.py` | `04_analysis/manipulation_check/outputs/realized_tokens.csv` → `05_figures_tables/table2_realized_tokens.tex` |
| Table III | RQ1: reasoning vs matched comparator | `04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py` | `04_analysis/rq1_reasoning_vs_bias/outputs/rq1_results.csv` → `05_figures_tables/table3_rq1.tex` |
| Table IV | RQ2: trend on realized tokens | `04_analysis/rq2_dose_response/rq2_analysis.py` | `04_analysis/rq2_dose_response/outputs/rq2_results.csv` → `05_figures_tables/table4_rq2.tex` |
| Table V | RQ3: presence classification per bias | `04_analysis/rq3_bias_taxonomy/rq3_analysis.py` | `04_analysis/rq3_bias_taxonomy/outputs/rq3_presence.csv` → `05_figures_tables/table5_rq3.tex` |
| Fig. 1 | Bias vs requested reasoning ceiling | `05_figures_tables/make_figures_tables.py` | `05_figures_tables/fig1_dose_response_values.csv`, `fig1_dose_response.pdf` |
| Fig. 2 | Signed bias by bias type and model | `05_figures_tables/make_figures_tables.py` | `05_figures_tables/fig2_bias_model_heatmap_values.csv`, `fig2_bias_model_heatmap.pdf` |
| Fig. 3 | Anchoring with vs without verbalization | `04_analysis/rq4_verbalization/rq4_analysis.py` + `05_figures_tables/make_figures_tables.py` | `05_figures_tables/fig3_verbalization_dumbbell_values.csv`, `fig3_verbalization_dumbbell.pdf` |

Statistics reported in the text, not in a table, are in the analysis reports:

| Text | Source |
|---|---|
| Realized-token statements (Sec. on requested vs realized deliberation) | `04_analysis/manipulation_check/outputs/manipulation_check_report.txt` |
| Parse rate, exclusions, undersized cells | `02_data_processing/outputs/parse_report.txt`, `validation_report.txt` |
| RQ1 pooled item-level contrast, Wilcoxon, mixed model; signed-score contrasts | `04_analysis/rq1_reasoning_vs_bias/outputs/rq1_report.txt` |
| Range of mean signed scores per model x budget | `04_analysis/rq2_dose_response/outputs/signed_by_model_budget.csv` |
| RQ3 per-bias slope tests on realized tokens | `04_analysis/rq3_bias_taxonomy/outputs/rq3_sensitivity.csv` |
| RQ4 pooled effect, item-level t(4), sign test, per-model direction | `04_analysis/rq4_verbalization/outputs/rq4_summary.csv`, `rq4_items.csv`, `rq4_models.csv` |

**Figures.** The three figure PDFs were styled interactively from the CSVs in
`05_figures_tables/`; the code writes the plotted values, not the artwork.
Every printed label and plotted point in the figures matches the CSVs. The one
exception is the Fig. 1 error bars: they came from an earlier unseeded
bootstrap and differ from the seeded CIs in `fig1_dose_response_values.csv` by
at most 0.007.

## Data

| File | Rows | Description |
|---|---|---|
| `01_data_collection/outputs/battery/final_battery.{csv,json}` | 30 | Frozen stimulus battery: item ID, bias, scenario, control and treatment prompt text, number of options, option labels, direction parameter `k`, `flip_treatment` |
| `01_data_collection/outputs/raw_responses/*.jsonl` | 12,350 | One record per API call: model, budget, item, condition, sample index, response text (visible answer only), API usage, latency, timestamp |
| `02_data_processing/outputs/parsed_responses.csv` | 12,350 | Extracted answer and parse method per call |
| `02_data_processing/outputs/master_scores.csv` | 510 | Scored dataset: one row per model x budget x item cell; variables documented in `03_codebook/outputs/codebook.md` |
| `data/per_call_usage.csv` | 12,350 | Per-call token usage: model, budget, item_id, condition, sample_idx, prompt_tokens, completion_tokens, reasoning_tokens |

Conditions are `control`, `treatment`, and `treatment_verbalized` (anchoring
items only). Budget-0 responses of DeepSeek-R1 and Qwen3-Thinking, which
cannot disable thinking, are included in the raw and usage files but excluded
from scoring.

The raw benchmark (`benchmark_raw.json`, ~60 MB) is not included; run
`01_data_collection/download_benchmark.py` to fetch the same revision
(`98fd1fd7`) from
[HuggingFace](https://huggingface.co/datasets/tum-nlp/cognitive-biases-in-llms).
It is only needed to rebuild the battery.

## License

- **Code:** MIT ([LICENSE](LICENSE)).
- **Data:** all data files in this repository (the battery, raw responses,
  parsed and scored data, per-call usage, and analysis outputs) are released
  under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Attribution.** The stimulus battery is adapted from the dataset
*cognitive-biases-in-llms* by Simon Malberg, Roman Poletukhin, Carolin M.
Schuster, and Georg Groh, described in "A Comprehensive Evaluation of
Cognitive Biases in LLMs," *Proceedings of the 5th International Conference
on Natural Language Processing for Digital Humanities (NLP4DH)*, 2025
(arXiv:2410.15413). Source:
<https://huggingface.co/datasets/tum-nlp/cognitive-biases-in-llms>, revision
`98fd1fd7728b320bb0690c51e72ca190191f2358`, licensed CC BY-SA 4.0.

Changes made: 30 items were sampled with seed 42, five from each of six
biases (Anchoring, Framing Effect, Loss Aversion, Escalation Of Commitment,
Availability Heuristic, Confirmation Bias), one per distinct scenario. No item
text was modified. Each item was given an ID (e.g. `ANCH-001`) and its answer
options were parsed from the prompt into separate fields. At collection time
a fixed answer-format instruction was appended to each prompt (see
`01_data_collection/run_experiments.py`); the stored item text is
unchanged.

## Contact

Obada Kraishan, College of Media and Communication, Texas Tech University:
omareikr@ttu.edu
