#!/bin/bash
set -euo pipefail

REPO_ROOT="${1:-/users/y/a/yangliu1/setllm_runs/setLLM-master}"
SEED_ARG="${SEED_ARG:-0}"
FIXED_MAIN_CALIBRATION_SET="${FIXED_MAIN_CALIBRATION_SET:-unanimous_all}"
RUN_STORAGE_ROOT="${RUN_STORAGE_ROOT:-/users/y/a/yangliu1/setllm_runs}"
BACKBONES="${BACKBONES:-qwen3b gemma7b}"

cd "$REPO_ROOT"

model_for_key() {
  case "$1" in
    qwen3b) echo "Qwen/Qwen2.5-3B-Instruct" ;;
    gemma7b) echo "google/gemma-7b" ;;
    *) echo "unknown backbone key: $1" >&2; return 1 ;;
  esac
}

dataset_tag_for_key() {
  case "$1" in
    qwen3b) echo "mtbench_qwen3b_tiehead_main" ;;
    gemma7b) echo "mtbench_gemma7b_tiehead_main" ;;
    *) echo "unknown backbone key: $1" >&2; return 1 ;;
  esac
}

gpu_profile_for_key() {
  case "$1" in
    qwen3b) echo "small" ;;
    gemma7b) echo "large" ;;
    *) echo "unknown backbone key: $1" >&2; return 1 ;;
  esac
}

cache_root_for_key() {
  case "$1" in
    gemma7b) echo "/nas/longleaf/home/yangliu1/.cache" ;;
    qwen3b) echo "$RUN_STORAGE_ROOT/.cache" ;;
    *) echo "unknown backbone key: $1" >&2; return 1 ;;
  esac
}

echo "Submitting Qwen/Gemma retained main-table runs"
echo "repo_root=$REPO_ROOT"
echo "seed=$SEED_ARG"
echo "fixed_main_calibration_set=$FIXED_MAIN_CALIBRATION_SET"

for key in $BACKBONES; do
  model="$(model_for_key "$key")"
  dataset_tag="$(dataset_tag_for_key "$key")"
  gpu_profile="$(gpu_profile_for_key "$key")"
  cache_root="$(cache_root_for_key "$key")"

  echo "backbone=$key model=$model"
  GPU_PROFILE="$gpu_profile" \
  JUDGE_D_ONLY="0" \
  JUDGE_MAX_RESPONSE_TOKENS="${JUDGE_MAX_RESPONSE_TOKENS:-512}" \
  USE_FP32_EVAL="${USE_FP32_EVAL:-0}" \
  TRAIN_HF_HUB_OFFLINE="${TRAIN_HF_HUB_OFFLINE:-1}" \
  EVAL_HF_HUB_OFFLINE="${EVAL_HF_HUB_OFFLINE:-1}" \
  RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT" \
  CACHE_ROOT="$cache_root" \
  FIXED_MAIN_CALIBRATION_SET="$FIXED_MAIN_CALIBRATION_SET" \
  SEED_ARG="$SEED_ARG" \
  bash scripts/submit_mtbench_uf_calibrated_pipeline.sh \
    "$REPO_ROOT" \
    "$model" \
    "$dataset_tag"
done
