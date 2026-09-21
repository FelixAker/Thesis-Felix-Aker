#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=direction_coherence

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

set -e

echo "=========================================================="
echo "Embedding Direction Coherence Analysis on ALEX Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================================="

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
export NLTK_DATA="${HOME}/nltk_data"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

apptainer exec --nv \
  --bind "$BIND_PATH:$BIND_PATH" \
  "$CONTAINER" \
  python3 src/analysis/direction_coherence.py

echo "Direction Coherence Analysis Complete: $(date)"
