#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --job-name=train_sft
#SBATCH --output=logs/train/sft_%j.out
#SBATCH --error=logs/train/sft_%j.err

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

# NCCL / distributed settings
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export NCCL_DEBUG TORCH_DISTRIBUTED_DEBUG NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ENABLE_MONITORING TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC NCCL_SOCKET_IFNAME

if [ -n "${SLURM_TMPDIR:-}" ]; then
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SLURM_TMPDIR/triton-autotune}"
  mkdir -p "$TRITON_CACHE_DIR"
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-/data/sft_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints/sft_run_${RUN_ID}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$TRAIN_DIR/scripts/zero3.json}"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=4
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
NUM_DEVICES="${SLURM_GPUS_ON_NODE:-${NUM_DEVICES:-4}}"
NUM_NODES="${SLURM_NNODES:-1}"
TOTAL_GPUS=$((NUM_NODES * NUM_DEVICES))
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * TOTAL_GPUS)))
if [ "$GRAD_ACCUM_STEPS" -lt 1 ]; then
  GRAD_ACCUM_STEPS=1
fi

echo "Run ID: $RUN_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Data path: $DATA_PATH"
echo "Image folder: $IMAGE_FOLDER"
echo "Num nodes: $NUM_NODES | GPUs per node: $NUM_DEVICES (total: $TOTAL_GPUS)"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"

cd "$TRAIN_DIR"

TRAIN_ARGS=(
  src/train/train_sft.py
  --use_liger_kernel True
  --lora_enable True
  --use_dora False
  --lora_namespan_exclude "['lm_head', 'embed_tokens']"
  --lora_rank 128
  --lora_alpha 256
  --lora_dropout 0.05
  --num_lora_modules -1
  --deepspeed "$DEEPSPEED_CONFIG"
  --model_id "$MODEL_NAME"
  --data_path "$DATA_PATH"
  --image_folder "$IMAGE_FOLDER"
  --remove_unused_columns False
  --freeze_vision_tower False
  --freeze_llm True
  --freeze_merger False
  --bf16 True
  --fp16 False
  --disable_flash_attn2 True
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs 2
  --per_device_train_batch_size "$BATCH_PER_DEVICE"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --max_seq_length "$MAX_SEQ_LENGTH"
  --image_min_pixels $((256 * 32 * 32))
  --image_max_pixels $((768 * 32 * 32))
  --learning_rate 5e-7
  --merger_lr 5e-7
  --vision_lr 1e-8
  --weight_decay 0.01
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 1
  --tf32 True
  --gradient_checkpointing True
  --report_to wandb
  --lazy_preprocess True
  --save_strategy steps
  --save_steps 300
  --save_total_limit 5
  --dataloader_num_workers 4
  --enable_reasoning True
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
    --kill-on-bad-exit=1
    --export=ALL
  )

  export MASTER_ADDR MASTER_PORT NUM_NODES NUM_DEVICES

  srun "${SRUN_ARGS[@]}" bash -lc '
    set -euo pipefail
    cd "'"$TRAIN_DIR"'"
    export PYTHONPATH="'"$PYTHONPATH"'"
    export MASTER_ADDR="'"$MASTER_ADDR"'"
    export MASTER_PORT="'"$MASTER_PORT"'"
    export NCCL_DEBUG="'"$NCCL_DEBUG"'"
    export TORCH_DISTRIBUTED_DEBUG="'"$TORCH_DISTRIBUTED_DEBUG"'"
    export NCCL_ASYNC_ERROR_HANDLING="'"$NCCL_ASYNC_ERROR_HANDLING"'"
    export TORCH_NCCL_ENABLE_MONITORING="'"$TORCH_NCCL_ENABLE_MONITORING"'"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="'"$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"'"
    export NCCL_SOCKET_IFNAME="'"$NCCL_SOCKET_IFNAME"'"
    NODE_RANK="${SLURM_PROCID}"
    echo "[$(hostname)] NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
    python -m torch.distributed.run \
      --nnodes="'"$NUM_NODES"'" \
      --nproc_per_node="'"$NUM_DEVICES"'" \
      --node_rank="$NODE_RANK" \
      --master_addr="'"$MASTER_ADDR"'" \
      --master_port="'"$MASTER_PORT"'" \
      '"$(printf '%q ' "${TRAIN_ARGS[@]}")"'
  '
else
  echo "Launching single-node torchrun."
  python -m torch.distributed.run --standalone --nproc_per_node "$NUM_DEVICES" "${TRAIN_ARGS[@]}"
fi
