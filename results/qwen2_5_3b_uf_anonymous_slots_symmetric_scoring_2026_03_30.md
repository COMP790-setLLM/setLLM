# Qwen2.5-3B +UF Snapshot

Date: 2026-03-30

Configuration:

- backbone: `Qwen/Qwen2.5-3B-Instruct`
- prompt style: `anonymous_slots`
- readout: `symmetric_scoring`
- UF warm-up: `yes`
- tie head: `input-dependent symmetric tie head`
- architectures:
  - `vanilla`
  - `setllm`
  - `setcausal`

Longleaf jobs:

- `40096207` = `vanilla`
- `40096208` = `setllm`
- `40096209` = `setcausal`
- `40096210` = dependent table-generation job

Remote artifact root:

- `~/setllm_runs/setLLM-master/artifacts/judge_mtbench_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring`

## Table 1. Dataset Statistics

| Dataset | Split | Rows | Unique Questions | A | B | Tie | A% | B% | Tie% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MT-Bench human pairwise | Train | 2739 | 64 | 1059 | 1065 | 615 | 38.66 | 38.88 | 22.45 |
|  | Eval | 616 | 16 | 234 | 217 | 165 | 37.99 | 35.23 | 26.79 |
|  | Total | 3355 | 80 | 1293 | 1282 | 780 | 38.54 | 38.21 | 23.25 |

## Table 2. Main Results

| Backbone | Readout | Arch | UF | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency | First-Pos Win Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | vanilla | yes | 46.59 | 63.41 | 1.18 | 44.64 | 69.32 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setllm | yes | 42.05 | 53.88 | 15.02 | 99.35 | 43.18 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setcausal | yes | 55.68 | 76.05 | 0.00 | 99.35 | 50.81 |

## Table 3. Rigorous Human-Alignment Results

| Backbone | Readout | Arch | UF | Row-Level Agreement | Majority-Vote Agreement | Unanimous-Only Agreement | Swap Consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | vanilla | yes | 46.59 | 45.08 | 44.58 | 44.64 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setllm | yes | 42.05 | 41.65 | 41.27 | 99.35 |
| Qwen/Qwen2.5-3B-Instruct | symmetric_scoring | setcausal | yes | 55.68 | 55.61 | 56.13 | 99.35 |

## Per-Architecture Metrics

### vanilla

```json
{
  "accuracy": 0.4659090909090909,
  "human_agreement": 0.4659090909090909,
  "no_tie_accuracy": 0.6341463414634146,
  "tie_precision": 0.25,
  "tie_recall": 0.006060606060606061,
  "tie_f1": 0.01183431952662722,
  "swap_consistency": 0.44642857142857145,
  "first_position_win_rate": 0.6931818181818182,
  "swapped_first_position_win_rate": 0.6818181818181818,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```

### setllm

```json
{
  "accuracy": 0.42045454545454547,
  "human_agreement": 0.42045454545454547,
  "no_tie_accuracy": 0.5388026607538803,
  "tie_precision": 0.3333333333333333,
  "tie_recall": 0.09696969696969697,
  "tie_f1": 0.15023474178403756,
  "swap_consistency": 0.9935064935064936,
  "first_position_win_rate": 0.4318181818181818,
  "swapped_first_position_win_rate": 0.4837662337662338,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```

### setcausal

```json
{
  "accuracy": 0.5568181818181818,
  "human_agreement": 0.5568181818181818,
  "no_tie_accuracy": 0.7605321507760532,
  "tie_precision": 0.0,
  "tie_recall": 0.0,
  "tie_f1": 0.0,
  "swap_consistency": 0.9935064935064936,
  "first_position_win_rate": 0.5081168831168831,
  "swapped_first_position_win_rate": 0.49837662337662336,
  "num_examples": 616,
  "num_no_tie_examples": 451
}
```
