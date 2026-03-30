# Qwen2.5-3B No-UF Baseline Snapshot

Date: 2026-03-29

Configuration:

- backbone: `Qwen/Qwen2.5-3B-Instruct`
- prompt style: `anonymous_slots`
- readout: `symmetric_scoring`
- UF warm-up: `no`
- architectures:
  - `vanilla`
  - `setllm`
  - `setcausal`

Longleaf jobs:

- `40027862` = `vanilla`
- `40027863` = `setllm`
- `40027864` = `setcausal`
- `40027865` = dependent table-generation job

Remote artifact root:

- `~/setllm_runs/setLLM-master/artifacts/judge_mtbench_main_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring`

## Table 1. Dataset Statistics

| Dataset | Split | Rows | Unique Questions | A | B | Tie | A% | B% | Tie% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MT-Bench human pairwise | Train | 2739 | 64 | 1059 | 1065 | 615 | 38.66 | 38.88 | 22.45 |
|  | Eval | 616 | 16 | 234 | 217 | 165 | 37.99 | 35.23 | 26.79 |
|  | Total | 3355 | 80 | 1293 | 1282 | 780 | 38.54 | 38.21 | 23.25 |

## Table 2. Main Results

| Backbone | Readout | Arch | UF | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency | First-Pos Win Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | vanilla | no | 47.08 | 64.30 | 0.00 | 29.87 | 79.87 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setllm | no | 38.64 | 52.77 | 0.00 | 99.35 | 50.32 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setcausal | no | 54.38 | 74.28 | 0.00 | 99.35 | 51.14 |

## Table 3. Rigorous Human-Alignment Results

| Backbone | Readout | Arch | UF | Row-Level Agreement | Majority-Vote Agreement | Unanimous-Only Agreement | Swap Consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | vanilla | no | 47.08 | 45.77 | 45.05 | 29.87 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setllm | no | 38.64 | 40.05 | 39.86 | 99.35 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setcausal | no | 54.38 | 54.92 | 55.90 | 99.35 |

## Per-Architecture Metrics

### vanilla

```json
{
  "accuracy": 0.4707792207792208,
  "human_agreement": 0.4707792207792208,
  "no_tie_accuracy": 0.6430155210643016,
  "tie_precision": 0.0,
  "tie_recall": 0.0,
  "tie_f1": 0.0,
  "swap_consistency": 0.2987012987012987,
  "first_position_win_rate": 0.7987012987012987,
  "swapped_first_position_win_rate": 0.8116883116883117,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```

### setllm

```json
{
  "accuracy": 0.38636363636363635,
  "human_agreement": 0.38636363636363635,
  "no_tie_accuracy": 0.5277161862527716,
  "tie_precision": 0.0,
  "tie_recall": 0.0,
  "tie_f1": 0.0,
  "swap_consistency": 0.9935064935064936,
  "first_position_win_rate": 0.5032467532467533,
  "swapped_first_position_win_rate": 0.5032467532467533,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```

### setcausal

```json
{
  "accuracy": 0.5438311688311688,
  "human_agreement": 0.5438311688311688,
  "no_tie_accuracy": 0.7427937915742794,
  "tie_precision": 0.0,
  "tie_recall": 0.0,
  "tie_f1": 0.0,
  "swap_consistency": 0.9935064935064936,
  "first_position_win_rate": 0.5113636363636364,
  "swapped_first_position_win_rate": 0.48214285714285715,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```
