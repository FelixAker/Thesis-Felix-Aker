#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

# Multi-GPU (DDP) student training via torchrun -- general form.
#
# Effective batch MUST stay 64 to remain comparable with the single-GPU runs:
#   NPROC x PER_DEV x ACCUM = 64
# The script asserts this and refuses to start otherwise.
#
# Per-device batch is architecture-dependent because the KD loss materialises a
# [batch, seq, vocab] fp32 log-softmax: Gemma-3's 262k vocab needs ~15 GB at
# per-device batch 8, so it cannot use 16 like GPT-2/Qwen2 on a 40 GB card.

set -e

MODE=$1
DATASET_PATH=$2
EXP_DIR=$3
ENTROPY=${4:-1.5}
MODEL_CONFIG=$5
EPOCHS=${6:-40}
NPROC=${7:-4}
PER_DEV=${8:-16}
ACCUM=${9:-1}

EFF=$(( NPROC * PER_DEV * ACCUM ))
if [ "$EFF" -ne 64 ]; then
    echo "FATAL: effective batch = NPROC($NPROC) x PER_DEV($PER_DEV) x ACCUM($ACCUM) = $EFF, expected 64"
    exit 1
fi

echo "=========================================================="
echo "DDP Student Training (embedding inheritance) — Alex"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_JOB_NODELIST"
echo "Mode: $MODE | Teacher: $MODEL_CONFIG | Entropy: $ENTROPY | Epochs: $EPOCHS"
echo "GPUs: $NPROC | per-device batch: $PER_DEV | accum: $ACCUM | effective: $EFF"
echo "Date: $(date)"
echo "=========================================================="

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
mkdir -p "$HF_HOME"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export WANDB_MODE="offline"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

MODEL_OUT="${EXP_DIR}/model"
mkdir -p "$MODEL_OUT"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

apptainer exec --nv --bind "$BIND_PATH:$BIND_PATH" \
    "$CONTAINER" \
    torchrun --standalone --nproc_per_node="$NPROC" \
    src/training/train_student.py \
    --model_config "$MODEL_CONFIG" \
    --hidden_size 384 \
    --num_hidden_layers 6 \
    --num_attention_heads 6 \
    --baseline_mode "$MODE" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$MODEL_OUT" \
    --alpha 0.5 \
    --temperature 1.0 \
    --batch_size "$PER_DEV" \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate 2e-4 \
    --epochs "$EPOCHS" \
    --seed 42 \
    --entropy_threshold "$ENTROPY" \
    --dataloader_workers 8 \
    --inherit_embeddings \
    --wandb_project "selective-kd-ddp-40ep-inherit"

echo "=========================================================="
echo "DDP training completed at $(date)"
echo "Final model: ${MODEL_OUT}/final_model"
echo "=========================================================="
