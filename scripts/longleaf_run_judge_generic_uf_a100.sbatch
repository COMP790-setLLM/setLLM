#!/bin/bash
#SBATCH -J judge-generic-uf-a100
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master}"
CLEANED_ULTRA_JSONL="${CLEANED_ULTRA_JSONL:-$REPO_ROOT/data/cleaned_ultrafeedback_top10k.jsonl}"
ARCHITECTURE="${1:-setcausal}"
MODEL_PATH="${2:-Qwen/Qwen2.5-3B-Instruct}"
PROMPT_STYLE="${3:-anonymous_slots}"
READOUT="${4:-symmetric_scoring}"
TRAIN_JSONL="${5:?missing train_jsonl}"
EVAL_JSONL="${6:?missing eval_jsonl}"
DATASET_TAG="${7:-custom}"
TIE_WEIGHT="${8:-1.0}"
UF_SAMPLES="${9:-10000}"
TIE_DIFF_PENALTY="${10:-0.0}"
TIE_THRESHOLD="${11:-}"
ATTN_IMPLEMENTATION="${12:-}"
SEED_ARG="${13:-42}"
MAX_TRAIN_SAMPLES_ARG="${14:-}"
MAX_EVAL_SAMPLES_ARG="${15:-}"
JUDGE_SCORE_ANCHOR_POSITION="${JUDGE_SCORE_ANCHOR_POSITION:-last}"
JUDGE_SCORE_ANCHOR_POOLING="${JUDGE_SCORE_ANCHOR_POOLING:-single}"
JUDGE_D_ONLY="${JUDGE_D_ONLY:-0}"
JUDGE_MAX_RESPONSE_TOKENS="${JUDGE_MAX_RESPONSE_TOKENS:-}"
USE_FP32_EVAL="${USE_FP32_EVAL:-1}"
DO_FINAL_EVAL="${DO_FINAL_EVAL:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
MODEL_TAG="${MODEL_PATH//\//_}"
TIE_TAG="${TIE_WEIGHT//./p}"
TIE_DIFF_TAG="${TIE_DIFF_PENALTY//./p}"
if [ -n "$TIE_THRESHOLD" ]; then
  TIE_THRESHOLD_TAG="${TIE_THRESHOLD//./p}"
else
  TIE_THRESHOLD_TAG="none"
fi
SEED_TAG="${SEED_ARG//./p}"
SCORE_ANCHOR_TAG=""
if [ "$JUDGE_SCORE_ANCHOR_POOLING" = "single" ] && [ "$JUDGE_SCORE_ANCHOR_POSITION" != "last" ]; then
  SCORE_ANCHOR_TAG="_sap${JUDGE_SCORE_ANCHOR_POSITION}"
fi
SCORE_POOL_TAG=""
if [ "$JUDGE_SCORE_ANCHOR_POOLING" != "single" ]; then
  SCORE_POOL_TAG="_sp${JUDGE_SCORE_ANCHOR_POOLING}"
fi
OUTPUT_ROOT="$REPO_ROOT/artifacts/judge_${DATASET_TAG}_uf_a100_${MODEL_TAG}_${PROMPT_STYLE}_${READOUT}_tw${TIE_TAG}_tdp${TIE_DIFF_TAG}_tth${TIE_THRESHOLD_TAG}_seed${SEED_TAG}${SCORE_ANCHOR_TAG}${SCORE_POOL_TAG}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
RUN_STORAGE_ROOT="${RUN_STORAGE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-$RUN_STORAGE_ROOT/.cache}"

cd "$REPO_ROOT"
source .venv/bin/activate

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_ROOT" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/wandb"

export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Running generic +UF judge experiment on a100-gpu"
echo "model=$MODEL_PATH"
echo "architecture=$ARCHITECTURE"
echo "judge_prompt_style=$PROMPT_STYLE"
echo "judge_readout=$READOUT"
echo "cleaned_ultra_jsonl=$CLEANED_ULTRA_JSONL"
echo "train_jsonl=$TRAIN_JSONL"
echo "eval_jsonl=$EVAL_JSONL"
echo "dataset_tag=$DATASET_TAG"
echo "judge_tie_weight=$TIE_WEIGHT"
echo "judge_tie_diff_penalty=$TIE_DIFF_PENALTY"
echo "judge_tie_threshold=${TIE_THRESHOLD:-<disabled>}"
echo "seed=$SEED_ARG"
echo "max_train_samples=${MAX_TRAIN_SAMPLES_ARG:-<all>}"
echo "max_eval_samples=${MAX_EVAL_SAMPLES_ARG:-<all>}"
echo "judge_score_anchor_position=$JUDGE_SCORE_ANCHOR_POSITION"
echo "judge_score_anchor_pooling=$JUDGE_SCORE_ANCHOR_POOLING"
echo "judge_d_only=$JUDGE_D_ONLY"
echo "judge_max_response_tokens=${JUDGE_MAX_RESPONSE_TOKENS:-<disabled>}"
echo "use_fp32_eval=$USE_FP32_EVAL"
echo "do_final_eval=$DO_FINAL_EVAL"
echo "hf_hub_offline=$HF_HUB_OFFLINE"
echo "transformers_offline=$TRANSFORMERS_OFFLINE"
echo "attn_implementation=${ATTN_IMPLEMENTATION:-<default>}"
echo "extra_pretrain_samples=$UF_SAMPLES"
echo "run_storage_root=$RUN_STORAGE_ROOT"
echo "cache_root=$CACHE_ROOT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
  if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME="${DATASET_TAG}-uf-${ARCHITECTURE}-${MODEL_TAG}-${PROMPT_STYLE}-${READOUT}-tw${TIE_TAG}-tdp${TIE_DIFF_TAG}-${SLURM_JOB_ID:-manual}"
  fi
  echo "wandb_project=$WANDB_PROJECT"
  echo "wandb_entity=${WANDB_ENTITY:-<default>}"
  echo "wandb_run_name=$WANDB_RUN_NAME"
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME")
  if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  echo "wandb_project=<disabled>"
fi

TIE_THRESHOLD_ARGS=()
if [ -n "$TIE_THRESHOLD" ]; then
  TIE_THRESHOLD_ARGS+=(--judge-tie-threshold "$TIE_THRESHOLD")
fi

ATTN_ARGS=()
if [ -n "$ATTN_IMPLEMENTATION" ]; then
  ATTN_ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi

SAMPLE_ARGS=()
if [ -n "$MAX_TRAIN_SAMPLES_ARG" ]; then
  SAMPLE_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES_ARG")
fi
if [ -n "$MAX_EVAL_SAMPLES_ARG" ]; then
  SAMPLE_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES_ARG")
fi

JUDGE_LENGTH_ARGS=()
if [ -n "$JUDGE_MAX_RESPONSE_TOKENS" ]; then
  JUDGE_LENGTH_ARGS+=(--judge-max-response-tokens "$JUDGE_MAX_RESPONSE_TOKENS")
fi

JUDGE_D_ONLY_ARGS=()
if [ "$JUDGE_D_ONLY" = "1" ]; then
  JUDGE_D_ONLY_ARGS+=(--judge-d-only)
fi

EVAL_PRECISION_ARGS=()
if [ "$USE_FP32_EVAL" = "1" ]; then
  EVAL_PRECISION_ARGS+=(--fp32-eval)
fi

LOCAL_FILES_ARGS=()
if [ "$HF_HUB_OFFLINE" = "1" ] || [ "$TRANSFORMERS_OFFLINE" = "1" ]; then
  LOCAL_FILES_ARGS+=(--local-files-only)
fi

TRAIN_EVAL_ARGS=(--do-train)
if [ "$DO_FINAL_EVAL" = "1" ]; then
  TRAIN_EVAL_ARGS+=(--do-eval)
fi

python reproduce_setllm.py \
  --model "$MODEL_PATH" \
  --task judge_pairwise \
  --cleaned-ultra-jsonl "$CLEANED_ULTRA_JSONL" \
  --extra-pretrain-samples "$UF_SAMPLES" \
  --judge-train-jsonl "$TRAIN_JSONL" \
  --judge-eval-jsonl "$EVAL_JSONL" \
  --judge-prompt-style "$PROMPT_STYLE" \
  --judge-readout "$READOUT" \
  --judge-tie-weight "$TIE_WEIGHT" \
  --judge-tie-diff-penalty "$TIE_DIFF_PENALTY" \
  --seed "$SEED_ARG" \
  --judge-score-anchor-position "$JUDGE_SCORE_ANCHOR_POSITION" \
  --judge-score-anchor-pooling "$JUDGE_SCORE_ANCHOR_POOLING" \
  "${JUDGE_D_ONLY_ARGS[@]}" \
  "${JUDGE_LENGTH_ARGS[@]}" \
  "${LOCAL_FILES_ARGS[@]}" \
  "${TIE_THRESHOLD_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --write-judge-predictions \
  --architecture "$ARCHITECTURE" \
  "${WANDB_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT/$ARCHITECTURE" \
  "${TRAIN_EVAL_ARGS[@]}" \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lora-r 4 \
  --lora-alpha 4 \
  --gradient-checkpointing \
  --learning-rate 2e-5 \
  --update-steps 256 \
  --warmup-steps 16 \
  --save-every 0 \
  --eval-every 0 \
  --bf16 \
  "${EVAL_PRECISION_ARGS[@]}"
