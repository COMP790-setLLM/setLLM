#!/bin/bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
UF_ROWS="${UF_ROWS:-10000}"
MTBENCH_DIR="${MTBENCH_DIR:-$REPO_ROOT/data/mt_bench_human_pairwise}"
UF_JSONL="${UF_JSONL:-$REPO_ROOT/data/cleaned_ultrafeedback_top10k.jsonl}"

cd "$REPO_ROOT"

mkdir -p "$MTBENCH_DIR"

echo "Preparing MT-Bench human pairwise train/eval split"
"$PYTHON_BIN" prepare_judge_data.py \
  --source mt_bench_human \
  --split-name human \
  --output-dir "$MTBENCH_DIR" \
  --seed "$SEED"

echo "Preparing cleaned UltraFeedback SFT warm-up data"
"$PYTHON_BIN" scripts/prepare_ultrafeedback_sft.py \
  --output-jsonl "$UF_JSONL" \
  --max-rows "$UF_ROWS" \
  --seed "$SEED"

build_split() {
  local name="$1"
  local selection="$2"
  local label_filter="$3"

  echo "Preparing calibration split: $name"
  "$PYTHON_BIN" scripts/build_tie_calibration_split.py \
    --input-jsonl "$MTBENCH_DIR/train.jsonl" \
    --output-jsonl "$MTBENCH_DIR/clean_calibration_${name}.jsonl" \
    --stats-json "$MTBENCH_DIR/clean_calibration_${name}.stats.json" \
    --selection "$selection" \
    --label-filter "$label_filter" \
    --min-votes 2 \
    --calibration-question-fraction 0.2 \
    --seed "$SEED"
}

build_split "majority_all" "majority" "all"
build_split "unanimous_all" "unanimous" "all"
build_split "unanimous_ties" "unanimous" "ties_only"

echo "Done."
echo "train_jsonl=$MTBENCH_DIR/train.jsonl"
echo "eval_jsonl=$MTBENCH_DIR/eval.jsonl"
echo "ultrafeedback_jsonl=$UF_JSONL"
echo "calibration_majority=$MTBENCH_DIR/clean_calibration_majority_all.jsonl"
echo "calibration_unanimous=$MTBENCH_DIR/clean_calibration_unanimous_all.jsonl"
echo "calibration_ties=$MTBENCH_DIR/clean_calibration_unanimous_ties.jsonl"
