# Experiment Record (No W&B Needed)

Date: 2026-03-30

This file is the self-contained experiment record for the current SetLLM / SetCausal project status. It is meant to answer the practical collaborator question:

> If I do not have a W&B link, how do I know the exact command, configuration, and result?

The answer is:

1. Use the `sbatch` submission command recorded below.
2. Use the referenced Longleaf job script to see the full Python command.
3. Use the Longleaf log file to see the echoed runtime configuration.
4. Use the result JSON path to see the final metrics.

## Longleaf Repo Root

Remote repo root:

```bash
/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master
```

Main logs directory:

```bash
/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/logs
```

Main artifacts directory:

```bash
/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/artifacts
```

## How To Recover The Exact Run

For any completed job:

1. Read the `sbatch` command in this file.
2. Open the referenced script in `scripts/`.
3. Open the corresponding log file:
   - logs print `model=...`, `architecture=...`, `judge_prompt_style=...`, `judge_readout=...`, dataset paths, and warm-up settings at the top.
4. Open the corresponding `judge_pairwise-metrics.json`.

This is enough to reconstruct the actual run even without W&B.

## Script Reference

### MT-Bench, No UF

Script:

```bash
scripts/longleaf_run_judge_mtbench_main_a100.sbatch
```

This script runs:

- dataset: `MT-Bench human pairwise`
- backbone: configurable
- prompt style: configurable
- readout: configurable
- train mode: LoRA finetuning
- eval mode: final evaluation with saved metrics

### MT-Bench, +UF

Script:

```bash
scripts/longleaf_run_judge_mtbench_uf_a100.sbatch
```

This script adds:

- `UltraFeedback-style warm-up`
- `--extra-pretrain-samples 10000`

### Generic +UF Runs

Script:

```bash
scripts/longleaf_run_judge_generic_uf_a100.sbatch
```

This is used for:

- `LLMBar Natural`
- `Qwen-7B` retry
- tie-weight and tie-diff ablations

### Automatic Table Generation

Script:

```bash
scripts/longleaf_generate_judge_tables_generic.sbatch
```

This now runs on CPU (`general` partition), not GPU.

## Completed Core Runs

### A. MT-Bench, Qwen/Qwen2.5-3B-Instruct, No UF

Submission commands:

```bash
sbatch --parsable scripts/longleaf_run_judge_mtbench_main_a100.sbatch \
  vanilla Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0

sbatch --parsable scripts/longleaf_run_judge_mtbench_main_a100.sbatch \
  setllm Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0

sbatch --parsable scripts/longleaf_run_judge_mtbench_main_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0
```

Job IDs:

- `40027862`: vanilla
- `40027863`: setllm
- `40027864`: setcausal

Log files:

```bash
logs/judge-main-a100-40027862.out
logs/judge-main-a100-40027863.out
logs/judge-main-a100-40027864.out
```

Result files:

```bash
artifacts/judge_mtbench_main_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/vanilla/judge_pairwise-metrics.json
artifacts/judge_mtbench_main_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setllm/judge_pairwise-metrics.json
artifacts/judge_mtbench_main_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setcausal/judge_pairwise-metrics.json
```

Headline results:

| Architecture | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency |
| --- | ---: | ---: | ---: | ---: |
| `vanilla` | `47.08` | `64.30` | `0.00` | `29.87` |
| `setllm` | `38.64` | `52.77` | `0.00` | `99.35` |
| `setcausal` | `54.38` | `74.28` | `0.00` | `99.35` |

Saved local snapshot:

```bash
saved_results/qwen2_5_3b_no_uf_anonymous_slots_symmetric_scoring_2026_03_29.md
```

### B. MT-Bench, Qwen/Qwen2.5-3B-Instruct, +UF

Submission commands:

```bash
sbatch --parsable scripts/longleaf_run_judge_mtbench_uf_a100.sbatch \
  vanilla Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0

sbatch --parsable scripts/longleaf_run_judge_mtbench_uf_a100.sbatch \
  setllm Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0

sbatch --parsable scripts/longleaf_run_judge_mtbench_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring 1.0
```

Job IDs:

- `40096207`: vanilla
- `40096208`: setllm
- `40096209`: setcausal

Log files:

```bash
logs/judge-uf-a100-40096207.out
logs/judge-uf-a100-40096208.out
logs/judge-uf-a100-40096209.out
```

Result files:

```bash
artifacts/judge_mtbench_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/vanilla/judge_pairwise-metrics.json
artifacts/judge_mtbench_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setllm/judge_pairwise-metrics.json
artifacts/judge_mtbench_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setcausal/judge_pairwise-metrics.json
```

Headline results:

| Architecture | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency |
| --- | ---: | ---: | ---: | ---: |
| `vanilla` | `46.59` | `63.41` | `1.18` | `44.64` |
| `setllm` | `42.05` | `53.88` | `15.02` | `99.35` |
| `setcausal` | `55.68` | `76.05` | `0.00` | `99.35` |

Saved local snapshot:

```bash
saved_results/qwen2_5_3b_uf_anonymous_slots_symmetric_scoring_2026_03_30.md
```

### C. MT-Bench, Tie-Weight Ablation on setcausal

Submission commands:

```bash
sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/eval.jsonl \
  mtbench 2.0 10000 0.0

sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/eval.jsonl \
  mtbench 4.0 10000 0.0
```

Job IDs:

- `40281633`: `judge_tie_weight = 2.0`
- `40281634`: `judge_tie_weight = 4.0`

Result:

- identical to the earlier `+UF setcausal` result
- conclusion: `Tie` class reweighting alone did not help

### D. LLMBar Natural, Qwen/Qwen2.5-3B-Instruct, +UF

Submission commands:

```bash
sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  vanilla Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/eval.jsonl \
  llmbar_natural 1.0 10000 0.0

sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setllm Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/eval.jsonl \
  llmbar_natural 1.0 10000 0.0

sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/eval.jsonl \
  llmbar_natural 1.0 10000 0.0
```

Job IDs:

- `40281816`: vanilla
- `40281817`: setllm
- `40281818`: setcausal

Result files:

```bash
artifacts/judge_llmbar_natural_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setcausal/judge_pairwise-metrics.json
artifacts/judge_llmbar_natural_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/setllm/judge_pairwise-metrics.json
artifacts/judge_llmbar_natural_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0/vanilla/judge_pairwise-metrics.json
```

Headline results:

| Architecture | Human Agreement | No-Tie Agreement | Tie F1 | Swap Consistency |
| --- | ---: | ---: | ---: | ---: |
| `vanilla` | `40.00` | `40.00` | `0.00` | `20.00` |
| `setllm` | `40.00` | `40.00` | `0.00` | `100.00` |
| `setcausal` | `70.00` | `70.00` | `0.00` | `100.00` |

Saved local snapshot:

```bash
saved_results/qwen2_5_3b_uf_llmbar_natural_anonymous_slots_symmetric_scoring_2026_03_30.md
```

## Ongoing Runs

### E. Qwen/Qwen2.5-7B-Instruct Retry

Submission command:

```bash
sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-7B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/eval.jsonl \
  mtbench 1.0 10000 0.0
```

Current retry job:

- `40343093`

Why this retry exists:

- the earlier `7B` run failed only at final eval due to CUDA OOM
- we fixed that by offloading the training model before loading the separate eval model

### F. Tie-Diff-Penalty Ablation

Submission commands:

```bash
sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/eval.jsonl \
  mtbench 1.0 10000 0.5

sbatch --parsable scripts/longleaf_run_judge_generic_uf_a100.sbatch \
  setcausal Qwen/Qwen2.5-3B-Instruct anonymous_slots symmetric_scoring \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/mt_bench_human_pairwise/eval.jsonl \
  mtbench 1.0 10000 1.0
```

Job IDs:

- `40345804`: `judge_tie_diff_penalty = 0.5`
- `40345805`: `judge_tie_diff_penalty = 1.0`

Purpose:

- class weighting did not fix `Tie`
- this ablation directly penalizes `|score_1 - score_2|` on gold tie examples

### G. LLMBar Automatic Table Regeneration

Submission command:

```bash
sbatch --parsable scripts/longleaf_generate_judge_tables_generic.sbatch \
  "LLMBar Natural" \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/train.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/eval.jsonl \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/data/llmbar_natural_pairwise/stats.json \
  /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/artifacts/judge_llmbar_natural_uf_a100_Qwen_Qwen2.5-3B-Instruct_anonymous_slots_symmetric_scoring_tw1p0 \
  Qwen/Qwen2.5-3B-Instruct \
  symmetric_scoring \
  anonymous_slots \
  yes
```

Job ID:

- `40345806`

## Where To Look First

If a collaborator wants the minimum set of files:

1. This file:

```bash
experiment_record_2026_03_30.md
```

2. The concise result snapshots:

```bash
saved_results/qwen2_5_3b_no_uf_anonymous_slots_symmetric_scoring_2026_03_29.md
saved_results/qwen2_5_3b_uf_anonymous_slots_symmetric_scoring_2026_03_30.md
saved_results/qwen2_5_3b_uf_llmbar_natural_anonymous_slots_symmetric_scoring_2026_03_30.md
```

3. The evolving lab notebook:

```bash
records.md
```

4. The current paper-oriented summary:

```bash
results_section_draft.md
```
