#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Phase 1: cache the teacher's top-K logits and per-token entropy once.
# Runs on a single A100; takes 2 to 5 minutes per teacher on the 10M corpus.
#
# Usage: sbatch phase1_precompute.slurm <TEACHER_MODEL> <INPUT_DATASET> <OUTPUT_DATASET>
#   e.g. sbatch phase1_precompute.slurm openai-community/gpt2 \
#              data/cleaned_10m data/augmented_gpt2base_10m

set -e

TEACHER_MODEL=$1
INPUT_DATASET=$2
OUTPUT_DATASET=$3

if [ -z "$TEACHER_MODEL" ] || [ -z "$INPUT_DATASET" ] || [ -z "$OUTPUT_DATASET" ]; then
    echo "Usage: sbatch phase1_precompute.slurm <TEACHER_MODEL> <INPUT_DATASET> <OUTPUT_DATASET>"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
mkdir -p "$HF_HOME"
# Compute nodes on Alex reach the internet through a proxy.
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

echo "=========================================================="
echo "Phase 1: Teacher Logit Precomputation"
echo "Teacher: $TEACHER_MODEL"
echo "Input:   $INPUT_DATASET"
echo "Output:  $OUTPUT_DATASET"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_JOB_NODELIST | Date: $(date)"
echo "=========================================================="

rm -rf "$OUTPUT_DATASET"
mkdir -p "$OUTPUT_DATASET"

apptainer exec --nv --bind "$BIND_PATH:$BIND_PATH" "$CONTAINER" \
    python src/data/compute_teacher_logits.py \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_path "$INPUT_DATASET" \
    --output_path "$OUTPUT_DATASET" \
    --batch_size 16 \
    --max_length 512 \
    --k 5

echo "=========================================================="
echo "Phase 1 complete at $(date)"
echo "=========================================================="
