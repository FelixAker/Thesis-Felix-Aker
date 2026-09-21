#!/bin/bash

# Paths. Override REPO_ROOT/CONTAINER when running outside the Alex cluster.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# Re-run the GPT-2 compute-budget ablation (6 conditions) at 10 epochs with the
# CORRECTED loss code, so the thesis no longer has to report pre-fix numbers.
#
# Identical to cluster/submit/main_sweep_40ep.sh except: GPT-2 only, EPOCHS=10, and
# the experiment directories carry a _ddp10 suffix. Effective batch stays 64
# (4 GPUs x 16 x 1), so the only difference to the 40-epoch runs is the budget.

set -e

R="$REPO_ROOT"
cd "$R"

TEACHER="openai-community/gpt2"
DATA="${R}/data/augmented_gpt2base_10m"
ARCH=gpt2base
PERDEV=16
ACCUM=1
EPOCHS=10
NPROC=4

CONDITIONS=( "scratch:1.5" "full_kd:1.5" "selective:0.5" "selective:1.0" "selective:1.5" "selective:2.0" )

echo "=========================================================="
echo "Submitting 6 x 10-epoch DDP runs (GPT-2, corrected code)"
echo "=========================================================="

TS=$(date +%Y-%m-%d_%H-%M-%S)

for C in "${CONDITIONS[@]}"; do
  MODE="${C%%:*}"; ENT="${C##*:}"
  if [ "$MODE" == "selective" ]; then LABEL="selective_${ENT}"; else LABEL="$MODE"; fi

  E="${R}/experiments/${TS}_${LABEL}_${ARCH}_10m_384_ddp10"
  mkdir -p "${E}/logs"
  echo "  -> $ARCH / $LABEL  (perdev=${PERDEV} accum=${ACCUM} epochs=${EPOCHS})"

  sbatch --job-name="ddp10_${ARCH}_${LABEL}" \
    --time=01:30:00 \
    --output="${E}/logs/train_%j.out" \
    --error="${E}/logs/train_%j.err" \
    cluster/slurm/train_ddp.slurm \
    "$MODE" "$DATA" "$E" "$ENT" "$TEACHER" \
    "$EPOCHS" "$NPROC" "$PERDEV" "$ACCUM"
  sleep 1
done

echo "Done. Track with: squeue -u \$USER"
