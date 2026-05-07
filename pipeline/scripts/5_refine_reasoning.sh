#!/usr/bin/env bash
#SBATCH --job-name=step5_refine
#SBATCH --output=logs/step5_refine_%A_%a.out
#SBATCH --error=logs/step5_refine_%A_%a.err
#SBATCH --array=0-15             # 16 parallel CPU tasks – adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# No GPU required – this step calls the OpenAI API over HTTPS.

set -euo pipefail

# --- Paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/pipeline"
CONFIG="${PIPELINE_DIR}/cfg/5_refine_reasoning.yaml"

# --- Activate your environment ---
# source /path/to/your/venv/bin/activate

# --- OpenAI API key ---
# Set this before submitting, or add it to your cluster's secret manager:
# export OPENAI_API_KEY="sk-..."
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

# --- Python path ---
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

# Enable SLURM-array row sharding (reads SLURM_ARRAY_TASK_ID / TASK_COUNT)
export STEP5_SPLIT_BY_SLURM_ARRAY=1

mkdir -p logs

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}"

python "${PIPELINE_DIR}/5_refine_reasoning.py" --config "${CONFIG}"
