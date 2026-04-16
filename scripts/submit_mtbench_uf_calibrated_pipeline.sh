#!/bin/bash
set -euo pipefail

REPO_ROOT="${1:-/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master}"
MODEL_PATH="${2:-Qwen/Qwen2.5-7B-Instruct}"
DATASET_TAG="${3:-mtbench_pipeline}"
cd "$REPO_ROOT"
source scripts/slurm_submit_utils.sh

TRAIN_JSONL="$REPO_ROOT/data/mt_bench_human_pairwise/train.jsonl"
EVAL_JSONL="$REPO_ROOT/data/mt_bench_human_pairwise/eval.jsonl"
PROMPT_STYLE="anonymous_slots"
READOUT="symmetric_scoring"
GPU_PROFILE="${GPU_PROFILE:-large}"
WANDB_PROJECT="${WANDB_PROJECT:-setllm-judge}"
WANDB_ENTITY="${WANDB_ENTITY:-ko-the-university-of-north-carolina-at-chapel-hill}"
UF_SAMPLES="${UF_SAMPLES:-10000}"
SEED_ARG="${SEED_ARG:-42}"
SEED_TAG="${SEED_ARG//./p}"
JUDGE_MAX_RESPONSE_TOKENS="${JUDGE_MAX_RESPONSE_TOKENS:-512}"
USE_FP32_EVAL="${USE_FP32_EVAL:-0}"
JUDGE_D_ONLY="${JUDGE_D_ONLY:-0}"
OBJECTIVE="${OBJECTIVE:-tie_aware}"
TIE_F1_WEIGHT="${TIE_F1_WEIGHT:-0.5}"
AGREEMENT_FLOOR="${AGREEMENT_FLOOR:-}"
FIXED_MAIN_CALIBRATION_SET="${FIXED_MAIN_CALIBRATION_SET:-unanimous_all}"
TRAIN_HF_HUB_OFFLINE="${TRAIN_HF_HUB_OFFLINE:-0}"
TRAIN_TRANSFORMERS_OFFLINE="${TRAIN_TRANSFORMERS_OFFLINE:-$TRAIN_HF_HUB_OFFLINE}"
EVAL_HF_HUB_OFFLINE="${EVAL_HF_HUB_OFFLINE:-1}"
EVAL_TRANSFORMERS_OFFLINE="${EVAL_TRANSFORMERS_OFFLINE:-$EVAL_HF_HUB_OFFLINE}"
MODEL_TAG="${MODEL_PATH//\//_}"
PIPE_ROOT="$REPO_ROOT/artifacts/${DATASET_TAG}_${MODEL_TAG}_seed${SEED_TAG}_uf_calibrated_pipeline"
RUN_STORAGE_ROOT="${RUN_STORAGE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-$RUN_STORAGE_ROOT/.cache}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"

mkdir -p "$LOG_ROOT" "$PIPE_ROOT" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/wandb"

anchor_pos_for_arch() {
  local arch="$1"
  if [[ "$MODEL_PATH" == "google/gemma-7b" && "$arch" == "setcausal" ]]; then
    echo "first"
  else
    echo "last"
  fi
}

anchor_pool_for_arch() {
  local arch="$1"
  if [[ "$MODEL_PATH" == "google/gemma-2b" && "$arch" == "setcausal" ]]; then
    echo "single"
  else
    echo "single"
  fi
}

attn_impl_for_arch() {
  local arch="$1"
  if [[ "$MODEL_PATH" == "google/gemma-7b" && "$arch" == "setllm" ]]; then
    echo "eager"
  else
    echo ""
  fi
}

score_anchor_suffix() {
  local pos="$1"
  local pool="$2"
  local out=""
  if [ "$pool" = "single" ] && [ "$pos" != "last" ]; then
    out="_sap${pos}"
  fi
  if [ "$pool" != "single" ]; then
    out="${out}_sp${pool}"
  fi
  echo "$out"
}

training_output_root() {
  local pos="$1"
  local pool="$2"
  local suffix
  suffix="$(score_anchor_suffix "$pos" "$pool")"
  echo "$REPO_ROOT/artifacts/judge_${DATASET_TAG}_uf_a100_${MODEL_TAG}_${PROMPT_STYLE}_${READOUT}_tw1p0_tdp0p0_tthnone_seed${SEED_ARG}${suffix}"
}

submit_variant() {
  local arch="$1"
  local pos pool attn_impl train_root adapter_dir calib_root finalize_root sbatch_cmd train_job
  pos="$(anchor_pos_for_arch "$arch")"
  pool="$(anchor_pool_for_arch "$arch")"
  attn_impl="$(attn_impl_for_arch "$arch")"
  train_root="$(training_output_root "$pos" "$pool")"
  adapter_dir="$train_root/$arch/judge_pairwise-finetune"
  calib_root="$PIPE_ROOT/$arch/calibration_eval"
  finalize_root="$PIPE_ROOT/$arch/finalized"
  sbatch_cmd="$(slurm_submit_prefix "$GPU_PROFILE")"

  train_job=$(
    REPO_ROOT="$REPO_ROOT" \
    CLEANED_ULTRA_JSONL="$REPO_ROOT/data/cleaned_ultrafeedback_top10k.jsonl" \
    RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    HF_HUB_OFFLINE="$TRAIN_HF_HUB_OFFLINE" \
    TRANSFORMERS_OFFLINE="$TRAIN_TRANSFORMERS_OFFLINE" \
    JUDGE_SCORE_ANCHOR_POSITION="$pos" \
    JUDGE_SCORE_ANCHOR_POOLING="$pool" \
    JUDGE_D_ONLY="$JUDGE_D_ONLY" \
    JUDGE_MAX_RESPONSE_TOKENS="$JUDGE_MAX_RESPONSE_TOKENS" \
    USE_FP32_EVAL="$USE_FP32_EVAL" \
    DO_FINAL_EVAL="0" \
    WANDB_RUN_NAME="${DATASET_TAG}-uf-${arch}-${MODEL_TAG}" \
    $sbatch_cmd --output "$LOG_ROOT/%x-%j.out" scripts/longleaf_run_judge_generic_uf_a100.sbatch \
      "$arch" \
      "$MODEL_PATH" \
      "$PROMPT_STYLE" \
      "$READOUT" \
      "$TRAIN_JSONL" \
      "$EVAL_JSONL" \
      "$DATASET_TAG" \
      1.0 \
      "$UF_SAMPLES" \
      0.0 \
      "" \
      "$attn_impl" \
      "$SEED_ARG"
  )

  submit_eval_split() {
    local split_name="$1"
    local eval_jsonl="$2"
    local output_dir="$3"
    REPO_ROOT="$REPO_ROOT" \
    RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    HF_HUB_OFFLINE="$EVAL_HF_HUB_OFFLINE" \
    TRANSFORMERS_OFFLINE="$EVAL_TRANSFORMERS_OFFLINE" \
    JUDGE_SCORE_ANCHOR_POSITION="$pos" \
    JUDGE_SCORE_ANCHOR_POOLING="$pool" \
    JUDGE_D_ONLY="$JUDGE_D_ONLY" \
    JUDGE_MAX_RESPONSE_TOKENS="$JUDGE_MAX_RESPONSE_TOKENS" \
    USE_FP32_EVAL="$USE_FP32_EVAL" \
    WANDB_RUN_NAME="${DATASET_TAG}-calib-${arch}-${split_name}-${MODEL_TAG}" \
    sbatch --parsable --dependency=afterok:${train_job} --output "$LOG_ROOT/%x-%j.out" \
      scripts/longleaf_run_judge_evalonly_a100.sbatch \
      "$arch" "$MODEL_PATH" "$PROMPT_STYLE" "$READOUT" \
      "$TRAIN_JSONL" "$eval_jsonl" "$adapter_dir" "$output_dir" \
      "" "$attn_impl"
  }

  local calib_majority_job calib_unanimous_job calib_ties_job eval_job finalize_job
  calib_majority_job=$(submit_eval_split "majority_all" "$REPO_ROOT/data/mt_bench_human_pairwise/clean_calibration_majority_all.jsonl" "$calib_root/majority_all")
  calib_unanimous_job=$(submit_eval_split "unanimous_all" "$REPO_ROOT/data/mt_bench_human_pairwise/clean_calibration_unanimous_all.jsonl" "$calib_root/unanimous_all")
  calib_ties_job=$(submit_eval_split "unanimous_ties" "$REPO_ROOT/data/mt_bench_human_pairwise/clean_calibration_unanimous_ties.jsonl" "$calib_root/unanimous_ties")
  eval_job=$(submit_eval_split "eval" "$EVAL_JSONL" "$calib_root/eval")

  finalize_job=$(
    REPO_ROOT="$REPO_ROOT" \
    RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    sbatch --parsable --dependency=afterok:${calib_majority_job}:${calib_unanimous_job}:${calib_ties_job}:${eval_job} --output "$LOG_ROOT/%x-%j.out" \
      scripts/longleaf_finalize_tie_calibration.sbatch \
      "$REPO_ROOT" "$calib_root" "$finalize_root" "$PROMPT_STYLE" "$OBJECTIVE" "$TIE_F1_WEIGHT" "$AGREEMENT_FLOOR"
  )

  echo "${arch}:${train_job}:${finalize_job}:$finalize_root/calibration_ablation.json"
}

echo "Submitting MT-Bench +UF calibrated pipeline"
echo "repo_root=$REPO_ROOT"
echo "model=$MODEL_PATH"
echo "dataset_tag=$DATASET_TAG"
echo "gpu_profile=$GPU_PROFILE"
echo "gpu_partition=${GPU_PARTITION:-<default>}"
echo "judge_max_response_tokens=$JUDGE_MAX_RESPONSE_TOKENS"
echo "use_fp32_eval=$USE_FP32_EVAL"
echo "judge_d_only=$JUDGE_D_ONLY"
echo "fixed_main_calibration_set=$FIXED_MAIN_CALIBRATION_SET"
echo "objective=$OBJECTIVE"
echo "tie_f1_weight=$TIE_F1_WEIGHT"
echo "train_hf_hub_offline=$TRAIN_HF_HUB_OFFLINE"
echo "eval_hf_hub_offline=$EVAL_HF_HUB_OFFLINE"
echo "run_storage_root=$RUN_STORAGE_ROOT"
echo "cache_root=$CACHE_ROOT"
echo "log_root=$LOG_ROOT"

VANILLA_INFO="$(submit_variant vanilla)"
SETLLM_INFO="$(submit_variant setllm)"
SETCAUSAL_INFO="$(submit_variant setcausal)"

VANILLA_FINAL_JSON="${VANILLA_INFO##*:}"
SETLLM_FINAL_JSON="${SETLLM_INFO##*:}"
SETCAUSAL_FINAL_JSON="${SETCAUSAL_INFO##*:}"
VANILLA_FINAL_JOB="$(echo "$VANILLA_INFO" | cut -d: -f3)"
SETLLM_FINAL_JOB="$(echo "$SETLLM_INFO" | cut -d: -f3)"
SETCAUSAL_FINAL_JOB="$(echo "$SETCAUSAL_INFO" | cut -d: -f3)"

COMPARE_MD="$PIPE_ROOT/${MODEL_TAG}_calibrated_comparison.md"
COMPARE_JOB=$(
  REPO_ROOT="$REPO_ROOT" \
  RUN_STORAGE_ROOT="$RUN_STORAGE_ROOT" \
  CACHE_ROOT="$CACHE_ROOT" \
  FIXED_MAIN_CALIBRATION_SET="$FIXED_MAIN_CALIBRATION_SET" \
  sbatch --parsable --dependency=afterok:${VANILLA_FINAL_JOB}:${SETLLM_FINAL_JOB}:${SETCAUSAL_FINAL_JOB} --output "$LOG_ROOT/%x-%j.out" \
    scripts/longleaf_render_calibrated_comparison.sbatch \
    "$REPO_ROOT" \
    "MT-Bench +UF Calibrated Comparison" \
    "$MODEL_PATH" \
    "$COMPARE_MD" \
    --variant "vanilla=$VANILLA_FINAL_JSON" \
    --variant "setllm=$SETLLM_FINAL_JSON" \
    --variant "setcausal=$SETCAUSAL_FINAL_JSON"
)

echo "VANILLA_INFO=$VANILLA_INFO"
echo "SETLLM_INFO=$SETLLM_INFO"
echo "SETCAUSAL_INFO=$SETCAUSAL_INFO"
echo "COMPARE_JOB=$COMPARE_JOB"
echo "COMPARE_MD=$COMPARE_MD"
