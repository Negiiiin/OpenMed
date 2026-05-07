#!/usr/bin/env bash
#SBATCH --job-name=step7_judge
#SBATCH --output=logs/step7_%A_%a.out
#SBATCH --error=logs/step7_%A_%a.err
#SBATCH --array=0-7              # one shard per GPU; adjust to match your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # remove this line if using backend=openai or anthropic
#SBATCH --mem=40G
#SBATCH --time=48:00:00

set -euo pipefail

# --- Paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/pipeline"
CONFIG="${PIPELINE_DIR}/cfg/7_judge_unit_questions.yaml"

# --- Activate your environment ---
# source /path/to/your/venv/bin/activate

# --- API keys (judges always need OPENAI_API_KEY; inference may need others) ---
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi
# export ANTHROPIC_API_KEY="..."   # only needed when inference_backend=anthropic
# export GEMINI_API_KEY="..."      # only needed when using Gemini as judge

# --- vLLM env vars (ignored for openai/anthropic backends) ---
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- Python path ---
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

mkdir -p logs

# Compute row slice for this task.
TOTAL_ROWS=$(python3 -c "
import json
count = 0
with open('/data/outputs/step6/unit_questions.jsonl') as f:
    for line in f:
        if line.strip(): count += 1
print(count)
" 2>/dev/null || echo 0)

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
CHUNK=$(( (TOTAL_ROWS + TASK_COUNT - 1) / TASK_COUNT ))
FROM_ROW=$(( TASK_ID * CHUNK ))
TO_ROW=$(( FROM_ROW + CHUNK ))

BASE_OUT=$(python3 -c "
import yaml, os
c = yaml.safe_load(open('${CONFIG}'))
base, ext = os.path.splitext(c['output_path'])
print(f'{base}_task${TASK_ID}{ext}')
" 2>/dev/null || echo "/data/outputs/step7/runs_task${TASK_ID}.jsonl")

SUMMARY_OUT=$(python3 -c "
import yaml, os
c = yaml.safe_load(open('${CONFIG}'))
base, ext = os.path.splitext(c['output_summary_path'])
print(f'{base}_task${TASK_ID}{ext}')
" 2>/dev/null || echo "/data/outputs/step7/summary_task${TASK_ID}.json")

echo "Task ${TASK_ID}/${TASK_COUNT}: rows [${FROM_ROW}, ${TO_ROW}) -> ${BASE_OUT}"

python "${PIPELINE_DIR}/7_judge_unit_questions.py" \
    --config              "${CONFIG}" \
    --from_row            "${FROM_ROW}" \
    --to_row              "${TO_ROW}" \
    --output_path         "${BASE_OUT}" \
    --output_summary_path "${SUMMARY_OUT}"

echo "Done. Merge outputs with:"
echo "  cat /data/outputs/step7/runs_task*.jsonl > /data/outputs/step7/runs.jsonl"
