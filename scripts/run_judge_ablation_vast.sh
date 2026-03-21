#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <model_path> <train_jsonl> <eval_jsonl> <output_root> <wandb_project> [max_seq_len] [update_steps]"
  exit 1
fi

MODEL_PATH="$1"
TRAIN_JSONL="$2"
EVAL_JSONL="$3"
OUTPUT_ROOT="$4"
WANDB_PROJECT="$5"
MAX_SEQ_LEN="${6:-1024}"
UPDATE_STEPS="${7:-100}"

mkdir -p "$OUTPUT_ROOT"

for ARCH in vanilla setllm setcausal; do
  python reproduce_setllm.py \
    --model "$MODEL_PATH" \
    --task judge_pairwise \
    --judge-train-jsonl "$TRAIN_JSONL" \
    --judge-eval-jsonl "$EVAL_JSONL" \
    --architecture "$ARCH" \
    --output-dir "$OUTPUT_ROOT/$ARCH" \
    --do-train \
    --do-eval \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r 4 \
    --lora-alpha 4 \
    --gradient-checkpointing \
    --update-steps "$UPDATE_STEPS" \
    --warmup-steps 10 \
    --save-every 0 \
    --eval-every 25 \
    --bf16 \
    --fp32-eval \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "judge-${ARCH}"
done
