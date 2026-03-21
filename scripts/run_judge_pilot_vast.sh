#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <architecture> <model_path> <train_jsonl> <eval_jsonl> <output_dir>"
  exit 1
fi

ARCH="$1"
MODEL_PATH="$2"
TRAIN_JSONL="$3"
EVAL_JSONL="$4"
OUTPUT_DIR="$5"

python reproduce_setllm.py \
  --model "$MODEL_PATH" \
  --task judge_pairwise \
  --judge-train-jsonl "$TRAIN_JSONL" \
  --judge-eval-jsonl "$EVAL_JSONL" \
  --architecture "$ARCH" \
  --output-dir "$OUTPUT_DIR" \
  --do-train \
  --do-eval \
  --max-train-samples 64 \
  --max-eval-samples 32 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lora-r 4 \
  --lora-alpha 4 \
  --gradient-checkpointing \
  --update-steps 8 \
  --warmup-steps 1 \
  --save-every 0 \
  --eval-every 0 \
  --bf16 \
  --fp32-eval \
  --wandb-project setllm-judge \
  --wandb-run-name "pilot-${ARCH}"
