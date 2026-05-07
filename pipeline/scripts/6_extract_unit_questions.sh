#!/usr/bin/env bash
#SBATCH --job-name=step6_unit_q
#SBATCH --output=logs/step6_%A_%a.out
#SBATCH --error=logs/step6_%A_%a.err
#SBATCH --array=0-15             # 16 parallel CPU tasks -- adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# No GPU required -- this step calls the OpenAI API over HTTPS.

set -euo pipefail

# --- Paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_DIR="${REPO_ROOT}/pipeline"
CONFIG="${PIPELINE_DIR}/cfg/6_extract_unit_questions.yaml"

# --- Activate your environment ---
# source /path/to/your/venv/bin/activate

# --- OpenAI API key ---
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

# --- Python path ---
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

mkdir -p logs

# Compute row slice for this array task.
TOTAL_ROWS=$(python3 -c "
import json, sys
count = 0
with open('$(python3 -c "import yaml; c=yaml.safe_load(open(\"${CONFIG}\")); print(c[\"input_path\"])")') as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
" 2>/dev/null || echo 0)

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}

CHUNK=$(( (TOTAL_ROWS + TASK_COUNT - 1) / TASK_COUNT ))
FROM_ROW=$(( TASK_ID * CHUNK ))
TO_ROW=$(( FROM_ROW + CHUNK ))

# Per-shard output file (merge shards after all tasks finish).
BASE_OUTPUT=$(python3 -c "
import yaml, os
c = yaml.safe_load(open('${CONFIG}'))
base, ext = os.path.splitext(c['output_path'])
print(f'{base}_task${TASK_ID}{ext}')
" 2>/dev/null || echo "/data/outputs/step6/unit_questions_task${TASK_ID}.jsonl")

echo "Task ${TASK_ID}/${TASK_COUNT}: rows [${FROM_ROW}, ${TO_ROW}) -> ${BASE_OUTPUT}"

python "${PIPELINE_DIR}/6_extract_unit_questions.py" \
    --config      "${CONFIG}" \
    --from_row    "${FROM_ROW}" \
    --to_row      "${TO_ROW}" \
    --output_path "${BASE_OUTPUT}"

echo "Done. Merge with:"
echo "  cat /data/outputs/step6/unit_questions_task*.jsonl > /data/outputs/step6/unit_questions.jsonl"
