# Anchored Minds -- Variable Codebook

Generated 2026-10-03 from `master_scores.csv` (510 cells).

| Variable | Type | Missing | Range / values | Description |
|---|---|---|---|---|
| `model` | object | 0 | claude-haiku-4.5, claude-haiku-4.5-thinking, deepseek-r1, deepseek-v3, gpt-5.1-reasoning, qwen3-instruct, ... | Model panel name from config.yaml (7 models, 4 families). |
| `reasoning` | bool | 0 | False, True | True if the model supports a controllable thinking budget. |
| `pair` | object | 0 | claude, deepseek, openai, qwen | Model family grouping a reasoning model with its non-reasoning sibling (claude / openai / deepseek / qwen). |
| `budget` | int64 | 0 | [0.000, 8192.000] | Reasoning-token ceiling for the cell (0 = extended thinking disabled; models flagged cannot_disable_thinking contribute budgets > 0 only). |
| `item_id` | object | 0 | ANCH-001, ANCH-002, ANCH-003, ANCH-004, ANCH-005, AVAI-001, ... | Frozen battery item identifier (e.g. ANCH-003). |
| `bias` | object | 0 | Anchoring, Availability Heuristic, Confirmation Bias, Escalation Of Commitment, Framing Effect, Loss Aversion | Cognitive bias targeted by the item (Malberg et al. 2024 benchmark label). |
| `n_options` | int64 | 0 | [7.000, 11.000] | Number of ordinal answer options on the item's scale (7 or 11). |
| `k` | int64 | 0 | [-1.000, 1.000] | Direction parameter from the benchmark's metric_params: +1/-1 so that positive bias_score always means a shift in the bias-predicted direction. |
| `count_control` | float64 | 0 | [3.000, 10.000] | Parsed samples in the control condition (target 10). |
| `count_treatment` | float64 | 0 | [2.000, 10.000] | Parsed samples in the treatment condition (target 10). |
| `count_treatment_verbalized` | float64 | 425 | [5.000, 10.000] | Parsed samples, verbalized treatment. |
| `mean_control` | float64 | 0 | [1.000, 11.000] | Mean chosen option index across samples, control condition (no bias manipulation). |
| `mean_treatment` | float64 | 0 | [1.000, 11.000] | Mean chosen option index across samples, treatment condition (bias manipulation present; scale reversed first when flip_treatment is set). |
| `mean_treatment_verbalized` | float64 | 425 | [2.900, 7.000] | Mean chosen option index, verbalized treatment (RQ4; Anchoring items only). |
| `std_control` | float64 | 0 | [0.000, 2.452] | SD of chosen option index, control condition. |
| `std_treatment` | float64 | 0 | [0.000, 5.270] | SD of chosen option index, treatment condition. |
| `std_treatment_verbalized` | float64 | 425 | [0.000, 2.079] | SD, verbalized treatment condition. |
| `bias_score` | float64 | 0 | [-1.000, 0.967] | Primary DV: k * (mean_treatment - mean_control) / (n_options - 1), bounded [-1, +1]. Positive = shift in the bias-predicted (human-like) direction; negative = shift against it (reversed). |
| `bias_score_verbalized` | float64 | 425 | [-0.020, 0.400] | Same metric computed with the verbalized treatment condition (RQ4). |
