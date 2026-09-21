#!/bin/bash

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# Submit script for the embedding-inheritance ablation: full 6-condition sweep
# (scratch, full_kd, selective 0.5/1.0/1.5/2.0) x 3 teachers, with
# --inherit_embeddings enabled (embeddings left trainable, not frozen),
# directly comparable to the existing 18-run baseline table.
#
# Training only -- evaluation is handled separately via cluster/submit/full_eval.sh once
# each run's final_model exists, to keep evaluation methodology identical to
# every other number already in the thesis.

set -e

cd "$REPO_ROOT"

declare -A TEACHER_NAME=(
  ["gpt2base"]="openai-community/gpt2"
  ["qwen2_0.5b"]="Qwen/Qwen2-0.5B"
  ["gemma3_270m"]="google/gemma-3-270m"
)
declare -A DATASET_PATH=(
  ["gpt2base"]="${REPO_ROOT}/data/augmented_gpt2base_10m"
  ["qwen2_0.5b"]="${REPO_ROOT}/data/augmented_qwen2_0.5b_10m"
  ["gemma3_270m"]="${REPO_ROOT}/data/augmented_gemma3_270m_10m"
)

# mode:entropy pairs -- entropy is "N/A" for scratch/full_kd (script ignores it
# for those modes, but train_student.py still wants a value)
CONDITIONS=(
  "scratch:1.5"
  "full_kd:1.5"
  "selective:0.5"
  "selective:1.0"
  "selective:1.5"
  "selective:2.0"
)

FREEZE_EMBEDDINGS="false"

echo "=========================================================="
echo "Submitting embedding-inheritance sweep (18 runs) on ALEX"
echo "=========================================================="

for ARCH in "${!TEACHER_NAME[@]}"; do
    TS=$(date +%Y-%m-%d_%H-%M-%S)
    sleep 1  # ensure distinct timestamps across archs

    for COND in "${CONDITIONS[@]}"; do
        MODE="${COND%%:*}"
        ENTROPY="${COND##*:}"

        if [ "$MODE" == "selective" ]; then
            LABEL="selective_${ENTROPY}"
        else
            LABEL="$MODE"
        fi

        FOLDER_NAME="${TS}_${LABEL}_${ARCH}_10m_384_inherit"
        EXP_DIR="${REPO_ROOT}/experiments/${FOLDER_NAME}"
        mkdir -p "${EXP_DIR}/logs"

        echo "Submitting: $ARCH / $LABEL"

        sbatch \
            --job-name="inherit_${ARCH}_${LABEL}" \
            --output="${EXP_DIR}/logs/train_%j.out" \
            --error="${EXP_DIR}/logs/train_%j.err" \
            cluster/slurm/train_emb_inherit.slurm \
            "$MODE" \
            "${DATASET_PATH[$ARCH]}" \
            "$EXP_DIR" \
            "$ENTROPY" \
            "${TEACHER_NAME[$ARCH]}" \
            "${TEACHER_NAME[$ARCH]}" \
            "$FREEZE_EMBEDDINGS"

        sleep 1
    done
done

echo "Done submitting embedding-inheritance sweep on ALEX!"
