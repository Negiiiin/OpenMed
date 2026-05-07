#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=20:00:00
#SBATCH --job-name=train_grpo
#SBATCH --output=logs/train/grpo_%j.out
#SBATCH --error=logs/train/grpo_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs/train"

# Activate your Python environment
# FINETUNE_ENV="${FINETUNE_ENV:-/path/to/your/env}"
# if [ -f "$FINETUNE_ENV/bin/activate" ]; then
#   source "$FINETUNE_ENV/bin/activate"
# fi

TRAIN_DIR="$REPO_ROOT/train"
export PYTHONPATH="$TRAIN_DIR/src:$TRAIN_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Base model (or path to a pre-merged SFT checkpoint)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Optional: initialize GRPO from a merged SFT checkpoint
# GRPO_INIT_SFT_CHECKPOINT="/path/to/merged/sft-checkpoint"
# if [ -n "${GRPO_INIT_SFT_CHECKPOINT:-}" ]; then
#   MODEL_NAME="$GRPO_INIT_SFT_CHECKPOINT"
# fi

DATA_PATH="${DATA_PATH:-/data/grpo_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/grpo_run_${RUN_ID}}"

NUM_DEVICES=${SLURM_GPUS_ON_NODE:-4}
NUM_NODES="${SLURM_NNODES:-1}"
TOTAL_GPUS=$((NUM_NODES * NUM_DEVICES))

echo "Run ID: $RUN_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Data path: $DATA_PATH"
echo "Num nodes: $NUM_NODES | GPUs per node: $NUM_DEVICES (total: $TOTAL_GPUS)"

cd "$TRAIN_DIR"

TRAIN_ARGS=(
  src/train/train_grpo.py
  --deepspeed scripts/zero3.json
  --use_liger_loss True
  --model_id "$MODEL_NAME"
  --data_path "$DATA_PATH"
  --image_folder "$IMAGE_FOLDER"
  --lora_enable True
  --use_dora False
  --lora_namespan_exclude "['lm_head', 'embed_tokens']"
  --lora_rank 128
  --lora_alpha 256
  --lora_dropout 0.05
  --num_lora_modules -1
  --freeze_vision_tower True
  --freeze_llm True
  --freeze_merger False
  --enable_reasoning True
  --bf16 True
  --fp16 False
  --disable_flash_attn2 True
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs 1
  --num_generations 16
  --generation_batch_size 64
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --max_completion_length 512
  --max_prompt_length 1024
  --image_min_pixels $((128 * 28 * 28))
  --image_max_pixels $((256 * 28 * 28))
  --learning_rate 5e-5
  --merger_lr 1e-4
  --remove_unused_columns False
  --weight_decay 0.01
  --warmup_ratio 0.05
  --lr_scheduler_type "cosine"
  --logging_steps 1
  --tf32 True
  --gradient_checkpointing False
  --report_to wandb
  --lazy_preprocess True
  --save_strategy "steps"
  --save_steps 300
  --save_total_limit 5
  --dataloader_num_workers 4
)

if [ "$NUM_NODES" -gt 1 ]; then
  if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "ERROR: multi-node launch requires a Slurm allocation. Submit with sbatch or use salloc."
    exit 1
  fi

  HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  MASTER_ADDR="${MASTER_ADDR:-$HEAD_NODE}"
  MASTER_PORT="${MASTER_PORT:-29500}"

  echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  echo "Launching one torchrun agent per node via srun."

  SRUN_ARGS=(
    --nodes "$NUM_NODES"
    --ntasks "$NUM_NODES"
    --ntasks-per-node 1
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-8}"
  )

  export MASTER_ADDR MASTER_PORT NUM_NODES NUM_DEVICES
  srun "${SRUN_ARGS[@]}" bash -lc '
    set -euo pipefail
    cd "'"$TRAIN_DIR"'"
    export PYTHONPATH="'"$PYTHONPATH"'"
    export MASTER_ADDR="'"$MASTER_ADDR"'"
    export MASTER_PORT="'"$MASTER_PORT"'"
    NODE_RANK="${SLURM_PROCID}"
    echo "[$(hostname)] NODE_RANK=$NODE_RANK launching torchrun"
    torchrun \
      --nnodes="'"$NUM_NODES"'" \
      --nproc_per_node="'"$NUM_DEVICES"'" \
      --node_rank="$NODE_RANK" \
      --master_addr="'"$MASTER_ADDR"'" \
      --master_port="'"$MASTER_PORT"'" \
      '"$(printf '%q ' "${TRAIN_ARGS[@]}")"'
  '
else
  echo "Launching single-node torchrun."
  torchrun --standalone --nproc_per_node "$NUM_DEVICES" "${TRAIN_ARGS[@]}"
fi
