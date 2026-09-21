#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONTAINER="${CONTAINER:-/home/atuin/b279bb/shared/images/flash_attn_3.sif}"
BIND_PATH="${BIND_PATH:-$HOME}"

set -e

MODEL_PATH=$1
EVAL_OUT=$2

echo "=========================================================="
echo "GLUE-Only Evaluation on ALEX"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_PATH"
echo "Output: $EVAL_OUT"
echo "Date: $(date)"
echo "=========================================================="

cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${REPO_ROOT}/.cache"
export NLTK_DATA="${HOME}/nltk_data"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

mkdir -p "$EVAL_OUT"

for SUBSET in boolq mrpc multirc qqp; do
    echo "  -> GLUE Subset: $SUBSET"
    apptainer exec --nv \
        --bind "$BIND_PATH:$BIND_PATH" \
        --env http_proxy=http://proxy.nhr.fau.de:80 \
        --env https_proxy=http://proxy.nhr.fau.de:80 \
        "$CONTAINER" \
        python src/evaluation/evaluate_glue.py \
        --subset "$SUBSET" \
        --model_type decoder \
        --model_path "$MODEL_PATH" \
        --bs 64 \
        --max_epochs 10 2>&1 | tee "${EVAL_OUT}/glue_${SUBSET}.txt"
done

echo "GLUE Complete: $(date)"
