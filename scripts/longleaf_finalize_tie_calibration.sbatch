#!/bin/bash
#SBATCH -J judge-calibration-finalize
#SBATCH -p general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${1:-/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master}"
CALIB_EVAL_ROOT="${2:?missing calibration eval root}"
OUTPUT_ROOT="${3:?missing output root}"
PROMPT_STYLE="${4:-anonymous_slots}"
OBJECTIVE="${5:-tie_aware}"
TIE_F1_WEIGHT="${6:-0.5}"
AGREEMENT_FLOOR="${7:-}"

cd "$REPO_ROOT"
source .venv/bin/activate

mkdir -p "$OUTPUT_ROOT"

ARGS=(
  --eval-predictions-jsonl "$CALIB_EVAL_ROOT/eval/judge_pairwise-predictions.jsonl"
  --calibration "majority_all=$CALIB_EVAL_ROOT/majority_all/judge_pairwise-predictions.jsonl"
  --calibration "unanimous_all=$CALIB_EVAL_ROOT/unanimous_all/judge_pairwise-predictions.jsonl"
  --calibration "unanimous_ties=$CALIB_EVAL_ROOT/unanimous_ties/judge_pairwise-predictions.jsonl"
  --prompt-style "$PROMPT_STYLE"
  --objective "$OBJECTIVE"
  --tie-f1-weight "$TIE_F1_WEIGHT"
  --output-json "$OUTPUT_ROOT/calibration_ablation.json"
  --output-markdown "$OUTPUT_ROOT/calibration_ablation.md"
)

if [ -n "$AGREEMENT_FLOOR" ]; then
  ARGS+=(--agreement-floor "$AGREEMENT_FLOOR")
fi

python scripts/run_tie_calibration_ablation.py "${ARGS[@]}"
