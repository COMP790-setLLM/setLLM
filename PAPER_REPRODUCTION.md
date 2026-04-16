# Paper Reproduction Guide

This branch is a minimal reproduction branch for the retained paper main-table
results on:

- `Qwen/Qwen2.5-3B-Instruct`
- `google/gemma-7b`

The retained paper protocol in this branch is:

- task: `judge_pairwise`
- prompt style: `anonymous_slots`
- readout: `symmetric_scoring`
- main system: **tie-head enabled**
- fixed calibration protocol: **`unanimous_all`**
- architectures:
  - `vanilla`
  - `setllm`
  - `setcausal`

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install peft accelerate sentencepiece
```

If you plan to reproduce the retained `Gemma-7B` runs, make sure the Hugging
Face account used on the cluster has already accepted the `google/gemma-7b`
license.

## 2. Prepare Data

This step creates:

- `data/mt_bench_human_pairwise/train.jsonl`
- `data/mt_bench_human_pairwise/eval.jsonl`
- `data/cleaned_ultrafeedback_top10k.jsonl`
- `data/mt_bench_human_pairwise/clean_calibration_majority_all.jsonl`
- `data/mt_bench_human_pairwise/clean_calibration_unanimous_all.jsonl`
- `data/mt_bench_human_pairwise/clean_calibration_unanimous_ties.jsonl`

```bash
source .venv/bin/activate
bash scripts/prepare_paper_judge_data.sh
```

Notes:

- MT-Bench train/eval is split by `question_id`, not by row.
- Calibration data is derived from the **train split only**.
- The paper main table uses a fixed `unanimous_all` calibration protocol.

## 3. Reproduce the Main Table

This branch keeps only the retained main-table setting:

- seed `0`
- tie-head enabled
- fixed `unanimous_all`
- `Qwen-3B` and `Gemma-7B`

On Longleaf:

```bash
SEED_ARG=0 \
FIXED_MAIN_CALIBRATION_SET=unanimous_all \
bash scripts/submit_mtbench_qwen_gemma_main.sh \
  /users/y/a/yangliu1/setllm_runs/setLLM-master
```

Each backbone runs:

1. `+UF` warm-up
2. MT-Bench judge fine-tuning
3. eval-only passes on:
   - `majority_all`
   - `unanimous_all`
   - `unanimous_ties`
   - held-out `eval`
4. threshold selection on calibration data
5. final calibrated comparison render

## 4. Important Notes

- This branch is intentionally narrower than our full working directory.
- It does **not** include paper LaTeX or PDF files.
- It does **not** include the broader Mistral or multi-seed experiments.
- It does **not** include `d-only` ablation scripts.

The goal here is just to reproduce the retained `Qwen + Gemma` main-table code
path cleanly.
