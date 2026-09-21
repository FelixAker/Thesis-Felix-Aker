#!/bin/bash

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# cluster/submit/full_eval.sh — Evaluate one trained checkpoint on everything: BLiMP,
# BLiMP Supplement, EWoK (zero-shot, via the official evaluation-pipeline-2025)
# and GLUE (BoolQ, MRPC, MultiRC, QQP fine-tuning, via evaluate_glue.py).
#
# This does not reimplement any evaluation logic — it submits the two existing,
# already-proven SLURM pipelines (cluster/slurm/eval_zero_shot.slurm and
# cluster/slurm/eval_glue.slurm) for the same checkpoint, so both run in
# parallel on separate GPU allocations rather than serialized in one long job.
#
# Usage:
#   bash cluster/submit/full_eval.sh <MODEL_PATH> [BACKEND]
#
#   MODEL_PATH  Path to a trained checkpoint's final_model directory, e.g.
#               experiments/2026-08-22_12-03-27_scratch_gpt2base_10m_384/model/final_model
#   BACKEND Model backend for the zero-shot pipeline (default: causal).
#               All decoder-only teachers used in this thesis (GPT-2, Qwen2,
#               Gemma-3) are "causal".
#
# Example:
#   bash cluster/submit/full_eval.sh experiments/2026-08-22_12-03-27_scratch_gpt2base_10m_384/model/final_model

set -e

MODEL_PATH=$1
BACKEND=${2:-causal}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash cluster/submit/full_eval.sh <MODEL_PATH> [BACKEND]"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Resolve to an absolute path. A relative path here (e.g. from running this
# script with `for d in experiments/*; do ...`) gets passed straight through
# to AutoModelForCausalLM.from_pretrained() inside the eval container, which
# then treats anything that isn't an absolute path as a Hugging Face Hub repo
# ID instead of a local directory and fails with an HFValidationError.
MODEL_PATH=$(cd "$MODEL_PATH" && pwd)

EVAL_ZERO_SHOT_DIR="${REPO_ROOT}/evaluation-pipeline-2025/evaluation_data/full_eval"

# Experiment directory = two levels up from .../model/final_model
EXP_DIR=$(dirname "$(dirname "$MODEL_PATH")")
EXP_NAME=$(basename "$EXP_DIR")
LOG_DIR="${EXP_DIR}/logs"
GLUE_OUT_DIR="${EXP_DIR}/eval"
mkdir -p "$LOG_DIR" "$GLUE_OUT_DIR"

echo "=========================================================="
echo "Submitting full evaluation for: $EXP_NAME"
echo "Model path : $MODEL_PATH"
echo "Backend    : $BACKEND"
echo "=========================================================="

echo "-> Submitting zero-shot pipeline (BLiMP + BLiMP Supplement + EWoK)"
ZS_JOB=$(sbatch --parsable \
    --job-name="zeroshot_${EXP_NAME}" \
    --output="${LOG_DIR}/eval2025_%j.out" \
    --error="${LOG_DIR}/eval2025_%j.err" \
    "${REPO_ROOT}/cluster/slurm/eval_zero_shot.slurm" "$MODEL_PATH" "$BACKEND" "$EVAL_ZERO_SHOT_DIR")
echo "   job ${ZS_JOB} -> results in ${EXP_DIR}/eval_2025/"

echo "-> Submitting GLUE fine-tuning (BoolQ, MRPC, MultiRC, QQP)"
GLUE_JOB=$(sbatch --parsable \
    --job-name="glue_${EXP_NAME}" \
    --output="${LOG_DIR}/glue_%j.out" \
    --error="${LOG_DIR}/glue_%j.err" \
    "${REPO_ROOT}/cluster/slurm/eval_glue.slurm" "$MODEL_PATH" "$GLUE_OUT_DIR")
echo "   job ${GLUE_JOB} -> results in ${GLUE_OUT_DIR}/"

echo "=========================================================="
echo "Submitted. Track with: squeue -u \$USER"
echo "Zero-shot job: ${ZS_JOB}  |  GLUE job: ${GLUE_JOB}"
echo "=========================================================="
