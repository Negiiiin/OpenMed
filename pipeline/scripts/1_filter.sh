#!/bin/bash
#SBATCH --job-name=pipeline_step1_filter
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/pipeline_1_%A_%a.log
#SBATCH --error=logs/pipeline_1_%A_%a.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# Activate your Python environment
# source /path/to/your/env/bin/activate

export PYTHONPATH="$REPO_ROOT/pipeline:${PYTHONPATH:-}"
export SLURM_ARRAY_TASK_COUNT=6

python pipeline/1_filter.py --config pipeline/cfg/1_filter.yaml
