#!/bin/bash
#SBATCH -J judge-evalonly-a100
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -o /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master}"
ARCHITECTURE="${1:?missing architecture}"
MODEL_PATH="${2:?missing model path}"
PROMPT_STYLE="${3:?missing prompt style}"
READOUT="${4:?missing readout}"
TRAIN_JSONL="${5:?missing train_jsonl}"
EVAL_JSONL="${6:?missing eval_jsonl}"
ADAPTER_DIR="${7:?missing adapter_dir}"
OUTPUT_DIR="${8:?missing output_dir}"
TIE_THRESHOLD="${9:-}"
ATTN_IMPLEMENTATION="${10:-}"
JUDGE_SCORE_ANCHOR_POSITION="${JUDGE_SCORE_ANCHOR_POSITION:-last}"
JUDGE_SCORE_ANCHOR_POOLING="${JUDGE_SCORE_ANCHOR_POOLING:-single}"
JUDGE_D_ONLY="${JUDGE_D_ONLY:-0}"
JUDGE_MAX_RESPONSE_TOKENS="${JUDGE_MAX_RESPONSE_TOKENS:-}"
USE_FP32_EVAL="${USE_FP32_EVAL:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
MODEL_TAG="${MODEL_PATH//\//_}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
RUN_STORAGE_ROOT="${RUN_STORAGE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-$RUN_STORAGE_ROOT/.cache}"

cd "$REPO_ROOT"
source .venv/bin/activate

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/wandb"

export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Running judge eval-only experiment on a100-gpu"
echo "model=$MODEL_PATH"
echo "architecture=$ARCHITECTURE"
echo "judge_prompt_style=$PROMPT_STYLE"
echo "judge_readout=$READOUT"
echo "train_jsonl=$TRAIN_JSONL"
echo "eval_jsonl=$EVAL_JSONL"
echo "adapter_dir=$ADAPTER_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "judge_tie_threshold=${TIE_THRESHOLD:-<disabled>}"
echo "judge_score_anchor_position=$JUDGE_SCORE_ANCHOR_POSITION"
echo "judge_score_anchor_pooling=$JUDGE_SCORE_ANCHOR_POOLING"
echo "judge_d_only=$JUDGE_D_ONLY"
echo "judge_max_response_tokens=${JUDGE_MAX_RESPONSE_TOKENS:-<disabled>}"
echo "use_fp32_eval=$USE_FP32_EVAL"
echo "hf_hub_offline=$HF_HUB_OFFLINE"
echo "transformers_offline=$TRANSFORMERS_OFFLINE"
echo "attn_implementation=${ATTN_IMPLEMENTATION:-<default>}"
echo "run_storage_root=$RUN_STORAGE_ROOT"
echo "cache_root=$CACHE_ROOT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
  if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME="evalonly-${ARCHITECTURE}-${MODEL_TAG}-${PROMPT_STYLE}-${READOUT}-${SLURM_JOB_ID:-manual}"
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

python reproduce_setllm.py \
  --model "$MODEL_PATH" \
  --task judge_pairwise \
  --judge-train-jsonl "$TRAIN_JSONL" \
  --judge-eval-jsonl "$EVAL_JSONL" \
  --judge-prompt-style "$PROMPT_STYLE" \
  --judge-readout "$READOUT" \
  --judge-score-anchor-position "$JUDGE_SCORE_ANCHOR_POSITION" \
  --judge-score-anchor-pooling "$JUDGE_SCORE_ANCHOR_POOLING" \
  "${JUDGE_D_ONLY_ARGS[@]}" \
  "${JUDGE_LENGTH_ARGS[@]}" \
  "${LOCAL_FILES_ARGS[@]}" \
  --eval-adapter-dir "$ADAPTER_DIR" \
  "${TIE_THRESHOLD_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  --write-judge-predictions \
  --architecture "$ARCHITECTURE" \
  "${WANDB_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --do-eval \
  --batch-size 1 \
  --eval-batch-size 1 \
  --bf16 \
  "${EVAL_PRECISION_ARGS[@]}"
