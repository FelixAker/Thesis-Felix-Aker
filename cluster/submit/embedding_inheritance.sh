#!/bin/bash
# Repeat the 18-condition main study with embedding inheritance: the student's
# embedding table is initialized from the teacher's pretrained embeddings
# instead of randomly.
#
# Identical to main_sweep_40ep.sh except for the job script, which adds
# --inherit_embeddings. Effective batch stays 64 for every run, so the two arms
# are directly comparable.

set -e

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
R="$REPO_ROOT"
cd "$R"

declare -A TEACHER=(
  [gpt2base]="openai-community/gpt2"
  [qwen2_0.5b]="Qwen/Qwen2-0.5B"
  [gemma3_270m]="google/gemma-3-270m"
)
declare -A DATA=(
  [gpt2base]="${R}/data/augmented_gpt2base_10m"
  [qwen2_0.5b]="${R}/data/augmented_qwen2_0.5b_10m"
  [gemma3_270m]="${R}/data/augmented_gemma3_270m_10m"
)
# Per-device batch differs by architecture for GPU-memory reasons only:
# Gemma-3's 262k-vocab log-softmax needs ~15 GB at per-device batch 8.
declare -A PERDEV=( [gpt2base]=16 [qwen2_0.5b]=16 [gemma3_270m]=8 )
declare -A ACCUM=(  [gpt2base]=1  [qwen2_0.5b]=1  [gemma3_270m]=2 )

CONDITIONS=( "scratch:1.5" "full_kd:1.5" "selective:0.5" "selective:1.0" "selective:1.5" "selective:2.0" )

EPOCHS=40
NPROC=4

echo "=========================================================="
echo "Submitting 18 x 40-epoch runs with embedding inheritance"
echo "=========================================================="

for ARCH in gpt2base qwen2_0.5b gemma3_270m; do
  TS=$(date +%Y-%m-%d_%H-%M-%S)
  sleep 1
  for C in "${CONDITIONS[@]}"; do
    MODE="${C%%:*}"; ENT="${C##*:}"
    if [ "$MODE" == "selective" ]; then LABEL="selective_${ENT}"; else LABEL="$MODE"; fi

    E="${R}/experiments/${TS}_${LABEL}_${ARCH}_10m_384_ddp40_inherit"
    mkdir -p "${E}/logs"
    echo "  -> $ARCH / $LABEL  (perdev=${PERDEV[$ARCH]} accum=${ACCUM[$ARCH]})"

    sbatch --job-name="inh40_${ARCH}_${LABEL}" \
      --output="${E}/logs/train_%j.out" \
      --error="${E}/logs/train_%j.err" \
      cluster/slurm/train_emb_inherit.slurm \
      "$MODE" "${DATA[$ARCH]}" "$E" "$ENT" "${TEACHER[$ARCH]}" \
      "$EPOCHS" "$NPROC" "${PERDEV[$ARCH]}" "${ACCUM[$ARCH]}"
    sleep 1
  done
done

echo "Done. Track with: squeue -u \$USER"
