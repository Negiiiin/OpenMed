#!/usr/bin/env bash
#SBATCH --job-name=step4_reasoning
#SBATCH --output=logs/step4_reasoning_%A_%a.out
#SBATCH --error=logs/step4_reasoning_%A_%a.err
#SBATCH --array=0-7              # 8 parallel tasks – adjust to match your worker count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # OctoMed-7B fits on a single A100 / H100
#SBATCH --mem=40G
#SBATCH --time=24:00:00

set -euo pipefail

# --- Paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/pipeline"
CONFIG="${PIPELINE_DIR}/cfg/4_generate_reasoning.yaml"

# --- Activate your environment ---
# source /path/to/your/venv/bin/activate
# module load cuda/12.1   # or whichever CUDA version your cluster provides

# --- Python path ---
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

# OctoMed is downloaded from Hugging Face at runtime (or from a local cache).
# Set HF_HOME if you want weights cached in a non-default location:
# export HF_HOME=/path/to/your/hf_cache

# vLLM spawn method (required for multi-GPU or array jobs)
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}"

python "${PIPELINE_DIR}/4_generate_reasoning.py" --config "${CONFIG}"
