#!/bin/bash
#SBATCH -J judge-calibration-render
#SBATCH -p general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /nas/longleaf/home/yangliu1/setllm_runs/setLLM-master/logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${1:-/nas/longleaf/home/yangliu1/setllm_runs/setLLM-master}"
TITLE="${2:?missing title}"
MODEL="${3:?missing model}"
OUTPUT_MD="${4:?missing output markdown}"
shift 4

cd "$REPO_ROOT"
source .venv/bin/activate

ARGS=(
  --title "$TITLE"
  --model "$MODEL"
  --dataset "MT-Bench human pairwise"
  --exclude-tie-only-main
)

if [ -n "${FIXED_MAIN_CALIBRATION_SET:-}" ]; then
  ARGS+=(--fixed-calibration-set "$FIXED_MAIN_CALIBRATION_SET")
fi

ARGS+=("$@" --output-md "$OUTPUT_MD")

python scripts/render_mtbench_calibrated_comparison.py "${ARGS[@]}"
