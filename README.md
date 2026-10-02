# Anchored Minds

**Does test-time reasoning immunize LLMs against human cognitive biases?**

A controlled dose-response study: we measure the magnitude of six classic
cognitive biases (anchoring, framing, decoy, sunk cost, availability,
confirmation) across matched reasoning / non-reasoning model pairs and across
increasing reasoning-token budgets (0 → 1k → 4k → 16k), testing whether
test-time compute functions as a "System 2" debiasing mechanism.

Submitted to **IEEE CogMI 2026** (8th IEEE International Conference on
Cognitive Machine Intelligence).

## Research questions

| RQ | Question | Analysis dir |
|----|----------|--------------|
| RQ1 | Do reasoning models show smaller bias magnitudes than matched non-reasoning models? | `04_analysis/rq1_reasoning_vs_bias/` |
| RQ2 | Does bias decrease monotonically with reasoning budget (dose-response)? | `04_analysis/rq2_dose_response/` |
| RQ3 | Which biases are reasoning-resistant vs. reasoning-invariant? | `04_analysis/rq3_bias_taxonomy/` |
| RQ4 | Does forced verbalization of the anchor change susceptibility? | `04_analysis/rq4_verbalization/` |

## Pipeline

```
config.yaml  ──►  00_setup (checks, cost estimate)
                   │
                   ▼
             01_data_collection
             download_benchmark ► build_test_battery ► run_experiments
                   │                                        │
                   ▼                                        ▼
             frozen battery (CSV/JSON)              raw responses (JSONL)
                                                          │
                   ┌──────────────────────────────────────┘
                   ▼
             02_data_processing
             parse_responses ► compute_bias_scores ► validate_dataset
                   │
                   ▼
             master_scores.csv  ──►  03_codebook (variable documentation)
                   │
                   ▼
             04_analysis (rq1..rq4, each standalone:
                          figures + CSV + JSON + _rq*_apa_report_.txt)
                   │
                   ▼
             05_paper_assets (camera-ready LaTeX tables)
```

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. keys
cp .env.example .env        # then fill in your key(s)

# 3. pre-flight
python 00_setup/setup_check.py --ping
python 00_setup/cost_estimator.py

# 4. data collection (Day 1-2)
python 01_data_collection/download_benchmark.py
python 01_data_collection/build_test_battery.py
python 01_data_collection/run_experiments.py --smoke-test   # 20-item dry run
python 01_data_collection/run_experiments.py                # full grid

# 5. processing (Day 3)
python 02_data_processing/parse_responses.py
python 02_data_processing/compute_bias_scores.py
python 02_data_processing/validate_dataset.py
python 03_codebook/generate_codebook.py

# 6. analysis (Day 4) - each script is standalone
python 04_analysis/rq1_reasoning_vs_bias/rq1_analysis.py
python 04_analysis/rq2_dose_response/rq2_analysis.py
python 04_analysis/rq3_bias_taxonomy/rq3_analysis.py
python 04_analysis/rq4_verbalization/rq4_analysis.py

# 7. paper tables (Day 5)
python 05_paper_assets/make_summary_tables.py
```

Every script prints verbose progress and saves its results as **CSV + JSON +
TXT**; each RQ analysis additionally writes a full APA7/IEEE statistical
report (`_rq1_apa_report_.txt`, ...) with sample sizes, test statistics,
p-values, effect sizes, and formatted tables.

## Data

Bias test items are drawn from the cognitive-bias benchmark of
Malberg, Poletukhin, Schuster & Groh (2024), *A Comprehensive Evaluation of
Cognitive Biases in LLMs* (arXiv:2410.15413) — 30,000 tests covering 30
biases. The frozen 6-bias battery used in this study ships in
`01_data_collection/outputs/battery/`.

## Reproducibility

- All randomness is seeded (`project.seed` in `config.yaml`).
- The stimulus battery is frozen before the full experimental run.
- Analysis scripts read only `02_data_processing/outputs/master_scores.csv`.
- Raw API responses are excluded from the repo (see `.gitignore`) but the
  full pipeline regenerates them from the frozen battery.

## License / citation

To be added after the review period (double-blind).
