#!/bin/bash

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# Re-run the full 18-condition main study (3 teachers x 6 modes) at 40 epochs
# using 4-GPU DDP and the corrected 5% warmup.
#
# Effective batch is 64 for every run, matching the original single-GPU study.
# Per-device batch differs by architecture purely for GPU-memory reasons:
#   GPT-2 (50k vocab)      -> 16 x 1
#   Qwen2-0.5B (152k vocab)-> 16 x 1
#   Gemma-3-270m (262k)    ->  8 x 2   (262k-vocab fp32 log-softmax needs ~15 GB at bs8)

set -e

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
declare -A PERDEV=( [gpt2base]=16 [qwen2_0.5b]=16 [gemma3_270m]=8 )
declare -A ACCUM=(  [gpt2base]=1  [qwen2_0.5b]=1  [gemma3_270m]=2 )

CONDITIONS=( "scratch:1.5" "full_kd:1.5" "selective:0.5" "selective:1.0" "selective:1.5" "selective:2.0" )

EPOCHS=40
NPROC=4

echo "=========================================================="
echo "Submitting 18 x 40-epoch DDP runs (corrected 5% warmup)"
echo "=========================================================="

for ARCH in gpt2base qwen2_0.5b gemma3_270m; do
  TS=$(date +%Y-%m-%d_%H-%M-%S)
  sleep 1
  for C in "${CONDITIONS[@]}"; do
    MODE="${C%%:*}"; ENT="${C##*:}"
    if [ "$MODE" == "selective" ]; then LABEL="selective_${ENT}"; else LABEL="$MODE"; fi

    E="${R}/experiments/${TS}_${LABEL}_${ARCH}_10m_384_ddp40"
    mkdir -p "${E}/logs"
    echo "  -> $ARCH / $LABEL  (perdev=${PERDEV[$ARCH]} accum=${ACCUM[$ARCH]})"

    sbatch --job-name="ddp40_${ARCH}_${LABEL}" \
      --output="${E}/logs/train_%j.out" \
      --error="${E}/logs/train_%j.err" \
      cluster/slurm/train_ddp.slurm \
      "$MODE" "${DATA[$ARCH]}" "$E" "$ENT" "${TEACHER[$ARCH]}" \
      "$EPOCHS" "$NPROC" "${PERDEV[$ARCH]}" "${ACCUM[$ARCH]}"
    sleep 1
  done
done

echo "Done. Track with: squeue -u \$USER"
