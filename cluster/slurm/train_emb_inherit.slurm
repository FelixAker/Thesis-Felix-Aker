#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

set -e

MODE=$1
DATASET_PATH=$2
EXP_DIR=$3
ENTROPY=$4
TEACHER_NAME=$5
MODEL_CONFIG=$6
FREEZE_EMBEDDINGS=${7:-false}

echo "=========================================================="
echo "Embedding Inheritance Training on ALEX Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Mode: $MODE"
echo "Teacher/Config: $TEACHER_NAME"
echo "Model Config: $MODEL_CONFIG"
echo "Dataset: $DATASET_PATH"
echo "Experiment Dir: $EXP_DIR"
echo "Entropy Threshold: $ENTROPY"
echo "Inherit Embeddings: YES"
echo "Freeze Embeddings: $FREEZE_EMBEDDINGS"
echo "Date: $(date)"
echo "=========================================================="

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
mkdir -p "$HF_HOME"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export WANDB_MODE="offline"

MODEL_OUT="${EXP_DIR}/model"
mkdir -p "$MODEL_OUT"

# Build optional --freeze_embeddings flag
FREEZE_FLAG=""
if [ "$FREEZE_EMBEDDINGS" = "true" ]; then
    FREEZE_FLAG="--freeze_embeddings"
fi

echo "----------------------------------------------------------"
echo "TRAINING WITH EMBEDDING INHERITANCE"
echo "----------------------------------------------------------"

apptainer exec --nv --bind "$BIND_PATH:$BIND_PATH" "$CONTAINER" \
    python src/training/train_student.py \
    --model_config "$MODEL_CONFIG" \
    --baseline_mode "$MODE" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$MODEL_OUT" \
    --alpha 0.5 \
    --temperature 1.0 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --epochs 40 \
    --entropy_threshold "$ENTROPY" \
    --inherit_embeddings \
    $FREEZE_FLAG \
    --wandb_project "selective-kd-emb-inherit-alex"

echo "=========================================================="
echo "Embedding Inheritance Training completed successfully at $(date)"
echo "Final model: ${MODEL_OUT}/final_model"
echo "=========================================================="
