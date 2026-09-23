# Entropy-Based Selective Knowledge Distillation

Code and results for the bachelor thesis *Entropy-Based Selective Knowledge
Distillation for Data-Constrained Language Model Training* (TUM, 2026).

The thesis asks whether a student trained under the 10-million-word cap of the
[BabyLM Challenge](https://babylm.github.io/) should learn from a teacher's
entire output distribution, or only from the positions where the teacher is
genuinely uncertain. The distillation loss is therefore applied only where the
teacher's predictive entropy exceeds a threshold `ε`.

Main findings: selective distillation beats training from scratch but not
distillation at every position; what decides student quality is how much of the
teacher's vocabulary the training corpus actually covers; and initializing the
student's embedding table from the teacher recovers part of the resulting gap.

## Pipeline

```
corpus ──► [Phase 1] teacher forward pass ──► cached top-K logits + entropy (disk)
                                                        │
                                            [Phase 2] student training (40 epochs)
                                                        │
                                   ┌────────────────────┴────────────────────┐
                            zero-shot eval                              GLUE fine-tuning
                     (BLiMP, Supplement, EWoK)                  (BoolQ, MRPC, MultiRC, QQP)
```

The teacher runs **once**, in Phase 1, and is never loaded during training.
Phase 2 reads the cached logits, so all six conditions of a teacher (scratch,
full KD, and four entropy thresholds) reuse one cache.

## Layout

```
src/
  data/compute_teacher_logits.py      Phase 1: cache top-K logits + entropy
  training/train_student.py           Phase 2: student training, all conditions
  training/custom_loss.py             the selective KD loss
  training/train_bpe_tokenizer.py     10,000-token tokenizer (Section 5.13)
  training/train_teacher_from_scratch.py   teacher for the 10k-vocabulary run
  training/morphology_tokenizer.py    tokenizer wrapper for that run
  evaluation/collate_zero_shot.py     score BLiMP/Supplement/EWoK predictions
  evaluation/evaluate_glue.py         GLUE fine-tuning and scoring
  analysis/embedding_drift.py         how far embeddings move during training
  analysis/direction_coherence.py     whether unseen embeddings collapse together
cluster/
  slurm/                              one job script per pipeline stage
  submit/                             wrappers that submit a whole sweep
results/                              the numbers reported in the thesis
```

## Results

| File | Thesis |
|---|---|
| `results/main_results_40epoch.csv` | Tables 5.1 and 5.8, the 18-condition main study (zero-shot and GLUE) |
| `results/embedding_inheritance.csv` | Table D.1, the same 18 conditions with inherited embeddings (Section 5.6) |
| `results/ablation_10epoch_gpt2.csv` | Tables 5.9 and D.2, GPT-2 at a quarter of the training budget |
| `results/runtimes.csv` | Table 5.10, measured run times |

Every value in these files is the one printed in the thesis; they were checked against
the tables programmatically.

Zero-shot scores are the average accuracy reported by the official
`evaluation-pipeline-2025` package; GLUE scores are the best validation score
over at most 10 fine-tuning epochs (accuracy, or F1 for MRPC).

## Reproducing a run

Requires a CUDA GPU, the BabyLM 10M corpus, and the packages in
`requirements.txt`. Paths below are relative to the repository root.

```bash
# Phase 1: cache the teacher's top-K logits and entropy (2 to 5 minutes on one A100)
python src/data/compute_teacher_logits.py \
    --teacher_model openai-community/gpt2 \
    --dataset_path data/cleaned_10m \
    --output_path data/augmented_gpt2base_10m \
    --batch_size 16 --max_length 512 --k 5

# Phase 2: train the student. --baseline_mode selects the condition:
#   scratch  (no teacher), full_kd (KD everywhere), selective (KD above the threshold)
torchrun --standalone --nproc_per_node=4 src/training/train_student.py \
    --model_config openai-community/gpt2 \
    --hidden_size 384 --num_hidden_layers 6 --num_attention_heads 6 \
    --baseline_mode selective --entropy_threshold 1.0 \
    --dataset_path data/augmented_gpt2base_10m \
    --output_dir experiments/selective_1.0/model \
    --alpha 0.5 --temperature 1.0 --learning_rate 2e-4 --epochs 40 \
    --batch_size 16 --gradient_accumulation_steps 1 --seed 42

# Evaluation: zero-shot via evaluation-pipeline-2025, then collate
python src/evaluation/collate_zero_shot.py \
    --experiments_dir experiments \
    --eval_data_dir evaluation-pipeline-2025/evaluation_data/full_eval \
    --output_csv zero_shot.csv

# GLUE, one task at a time
python src/evaluation/evaluate_glue.py \
    --subset boolq --model_type decoder \
    --model_path experiments/selective_1.0/model/final_model
```

Add `--inherit_embeddings` to Phase 2 to initialize the student's embedding
table from the teacher's embeddings instead of randomly.

## Cluster

The reported runs were executed on the FAU NHR **Alex** cluster with SLURM and
an Apptainer image. Phase 1 used one A100 (40 GB), 16 CPU cores and 120 GB RAM;
training used four A100s, 64 CPU cores and 480 GB RAM, at an effective batch
size of 64 (4 GPUs × 16 per device × 1 accumulation step).

The scripts in `cluster/` take the repository root from `$REPO_ROOT` (default:
the current directory) and the container from `$CONTAINER`, so they can be
pointed at another machine without editing them. The submit wrappers use
associative arrays and therefore need bash 4 or newer (macOS ships bash 3.2):

```bash
REPO_ROOT=$PWD sbatch cluster/slurm/phase1_precompute.slurm \
    openai-community/gpt2 data/cleaned_10m data/augmented_gpt2base_10m

REPO_ROOT=$PWD bash cluster/submit/main_sweep_40ep.sh      # 18 runs, 40 epochs
REPO_ROOT=$PWD bash cluster/submit/ablation_10ep_gpt2.sh   # GPT-2 at 10 epochs
REPO_ROOT=$PWD bash cluster/submit/embedding_inheritance.sh
REPO_ROOT=$PWD bash cluster/submit/full_eval.sh <path/to/final_model>
```

## Notes

- Every reported condition was trained with a single seed (42). Differences
  below roughly one percentage point should not be read as real effects.
- The corpus, checkpoints and cached logits are not in this repository. The
  corpus comes from the BabyLM Challenge; the teachers are the public Hugging
  Face checkpoints `openai-community/gpt2`, `Qwen/Qwen2-0.5B` and
  `google/gemma-3-270m`.
