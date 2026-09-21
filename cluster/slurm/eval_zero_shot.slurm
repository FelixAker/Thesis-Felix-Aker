#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

set -e

MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=$3

echo "=========================================================="
echo "2025 Evaluation Pipeline on ALEX Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Model Path: $MODEL_PATH"
echo "Backend: $BACKEND"
echo "Date: $(date)"
echo "=========================================================="

cd "$REPO_ROOT"/evaluation-pipeline-2025

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
# Do NOT set PYTHONUSERBASE - container has correct torch 2.9.0 + torchvision 0.24.0
# Small extras (nltk, nlopt, etc.) are installed into container's default --user site
export NLTK_DATA="${HOME}/nltk_data"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Run evaluation inside the container (mounting the eval_env to use installed packages)
apptainer exec --nv \
  --bind "$BIND_PATH:$BIND_PATH" \
  "$CONTAINER" \
  bash eval_custom_zero_shot.sh "$MODEL_PATH" "$BACKEND" "$EVAL_DIR"

echo "Evaluation Complete: $(date)"
