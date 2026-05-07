#!/bin/bash
#SBATCH --job-name=pipeline_step2_select_context
#SBATCH --array=0-100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_2_%A_%a.log
#SBATCH --error=logs/pipeline_2_%A_%a.err

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# Activate your Python environment
# source /path/to/your/env/bin/activate

# Set your OpenAI API key (do not hard-code; use an environment variable or secret manager)
# export OPENAI_API_KEY="your-api-key-here"

export PYTHONPATH="$REPO_ROOT/pipeline:${PYTHONPATH:-}"
export SLURM_ARRAY_TASK_COUNT=101

python pipeline/2_select_relevant_context.py --config pipeline/cfg/2_select_relevant_context.yaml
