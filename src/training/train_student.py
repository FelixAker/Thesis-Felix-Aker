"""
Phase 2 of the pipeline: train the student with entropy-based selective
knowledge distillation (Sections 3.4 to 3.6 of the thesis).

Reads the cached teacher logits written by src/data/compute_teacher_logits.py
and applies the KD loss only at positions whose teacher entropy exceeds
--entropy_threshold. The two baselines are selected with --baseline_mode:
"scratch" (no teacher signal) and "full_kd" (KD at every position).

--inherit_embeddings initializes the student embedding table from the teacher's
pretrained embeddings instead of randomly (Section 3.8).
"""

import os
import json
import argparse
import collections
import numpy as np
import torch

# Allowlist numpy reconstruct for RNG state loading
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except Exception:
    pass
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except Exception:
    pass

# Monkey-patch torch.load to bypass weights_only restrictions on older PyTorch versions
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed
)

# Allow torch.load on the cluster image, which ships PyTorch < 2.6
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.trainer as trainer_module
    trainer_module.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from datasets import load_from_disk, load_dataset
from custom_loss import EntropySelectiveKDLoss


class StatsCallback(TrainerCallback):
    """
    Accumulates per-step KD metrics and saves a comprehensive JSON summary
    at the end of training. Captures:
      - Per-step: ce_loss, kd_loss, kd_fraction, active_kd_tokens
      - Per-epoch averages of all the above
      - Global summary: mean/std/min/max kd_fraction, total token counts
    Saves to {output_dir}/training_stats.json
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.step_history = []      # list of dicts, one per logging step
        self._current = collections.defaultdict(list)  # buffer within a log interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called each time the Trainer logs. Pick up our custom keys if present."""
        if logs is None:
            return
        entry = {"global_step": state.global_step, "epoch": state.epoch}
        for key in ("train/ce_loss", "train/kd_loss", "train/kd_fraction", "train/active_kd_tokens"):
            if key in logs:
                entry[key] = logs[key]
        # Only append if we captured at least one KD metric
        if len(entry) > 2:
            self.step_history.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        """Save full stats JSON when training finishes."""
        # Under DDP every rank runs this callback; only rank 0 may write, or the
        # ranks race on the same file and can corrupt it.
        if not state.is_world_process_zero:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        kd_fractions   = [e["train/kd_fraction"]       for e in self.step_history if "train/kd_fraction"       in e]
        ce_losses      = [e["train/ce_loss"]            for e in self.step_history if "train/ce_loss"            in e]
        kd_losses      = [e["train/kd_loss"]            for e in self.step_history if "train/kd_loss"            in e]
        active_tokens  = [e["train/active_kd_tokens"]   for e in self.step_history if "train/active_kd_tokens"   in e]

        # Per-epoch averages
        epoch_stats = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in self.step_history:
            ep = int(e.get("epoch", 0))
            for key in ("train/ce_loss", "train/kd_loss", "train/kd_fraction", "train/active_kd_tokens"):
                if key in e:
                    epoch_stats[ep][key].append(e[key])
        epoch_averages = {
            ep: {k: float(np.mean(v)) for k, v in metrics.items()}
            for ep, metrics in sorted(epoch_stats.items())
        }

        summary = {
            "total_steps": state.global_step,
            "total_epochs": state.num_train_epochs,

            # KD token selection summary
            "kd_fraction_mean":   float(np.mean(kd_fractions))   if kd_fractions else None,
            "kd_fraction_std":    float(np.std(kd_fractions))    if kd_fractions else None,
            "kd_fraction_min":    float(np.min(kd_fractions))    if kd_fractions else None,
            "kd_fraction_max":    float(np.max(kd_fractions))    if kd_fractions else None,
            "kd_fraction_final":  kd_fractions[-1]               if kd_fractions else None,

            # Loss summary
            "ce_loss_final":      ce_losses[-1]  if ce_losses  else None,
            "kd_loss_final":      kd_losses[-1]  if kd_losses  else None,
            "ce_loss_mean":       float(np.mean(ce_losses))  if ce_losses  else None,
            "kd_loss_mean":       float(np.mean(kd_losses))  if kd_losses  else None,

            # Token counts
            "total_active_kd_tokens_logged": int(sum(active_tokens)) if active_tokens else None,

            # Full step-by-step history and per-epoch breakdown
            "epoch_averages": epoch_averages,
            "step_history":   self.step_history,
        }

        out_path = os.path.join(self.output_dir, "training_stats.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining stats saved to {out_path}")
        if summary['kd_fraction_mean'] is not None:
            print(f"   Mean KD fraction (teacher helped): {summary['kd_fraction_mean']:.1%}")
            print(f"   KD fraction range: {summary['kd_fraction_min']:.1%} – {summary['kd_fraction_max']:.1%}")
        else:
            print("   KD fraction metrics not captured via HF logs (check wandb for full_kd/scratch runs)")
        ce_final = summary['ce_loss_final']
        kd_final = summary['kd_loss_final']
        if ce_final is not None:
            print(f"   Final CE loss: {ce_final:.4f} | Final KD loss: {kd_final:.4f}")


def compute_dataset_stats(dataset, output_dir, entropy_threshold=1.0):
    """
    Scan the precomputed dataset to report token-level mask statistics.
    Shows how many tokens the teacher's KD signal was applied to.
    Saves to {output_dir}/dataset_stats.json
    """
    print("\nComputing dataset mask statistics...")
    train_split = dataset["train"]

    # Check if this dataset has entropy_score or entropy_mask (augmented datasets do; raw datasets don't)
    has_entropy = "entropy_score" in train_split.features or "entropy_mask" in train_split.features
    if len(train_split) == 0 or not has_entropy:
        print("   No entropy_score/entropy_mask field found — skipping dataset stats (raw or full_kd dataset).")
        return {}

    # To avoid memory blowing up on massive datasets, sample up to 100k rows
    sample_size = min(len(train_split), 100000)
    print(f"   Analyzing a random sample of {sample_size} sequences...")
    # Use torch for random choice instead of np.random to respect seed
    indices = torch.randperm(len(train_split))[:sample_size].numpy().tolist()
    
    total_tokens = 0
    total_masked = 0
    all_fractions = []

    for idx in indices:
        example = train_split[idx]
        entropy = np.array(example.get("entropy_score", example.get("entropy_mask", [])), dtype=np.float32)
        mask = (entropy > entropy_threshold).astype(np.float32)
        attn = np.array(example.get("attention_mask", np.ones_like(mask)), dtype=np.float32)
        valid = attn.sum()
        active = (mask * attn).sum()
        if valid > 0:
            total_tokens += int(valid)
            total_masked += int(active)
            all_fractions.append(active / valid)

    dataset_kd_fraction = total_masked / total_tokens if total_tokens > 0 else 0.0

    stats = {
        "sampled_examples":      sample_size,
        "total_valid_tokens":    total_tokens,
        "total_masked_tokens":   total_masked,
        "dataset_kd_fraction":   float(dataset_kd_fraction),
        "per_example_kd_fraction_mean": float(np.mean(all_fractions)) if all_fractions else 0.0,
        "per_example_kd_fraction_std":  float(np.std(all_fractions))  if all_fractions else 0.0,
        "per_example_kd_fraction_min":  float(np.min(all_fractions))  if all_fractions else 0.0,
        "per_example_kd_fraction_max":  float(np.max(all_fractions))  if all_fractions else 0.0,
        "per_example_kd_fraction_p25":  float(np.percentile(all_fractions, 25)) if all_fractions else 0.0,
        "per_example_kd_fraction_p75":  float(np.percentile(all_fractions, 75)) if all_fractions else 0.0,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"   Dataset KD fraction (mask=1 tokens): {dataset_kd_fraction:.1%}")
    print(f"   Sampled {sample_size} examples → {total_masked:,} / {total_tokens:,} tokens masked")
    print(f"   Per-example KD fraction: mean={stats['per_example_kd_fraction_mean']:.1%}, "
          f"std={stats['per_example_kd_fraction_std']:.1%}, "
          f"p25={stats['per_example_kd_fraction_p25']:.1%}, "
          f"p75={stats['per_example_kd_fraction_p75']:.1%}")
    print(f"   Dataset stats saved to {out_path}\n")
    return stats

class SelectiveKDTrainer(Trainer):
    """
    Custom Trainer subclassing Hugging Face Trainer to support Selective Knowledge Distillation.
    Overrides compute_loss to inject custom pre-computed teacher logits and entropy masks.
    """
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute custom selective distillation loss.
        """
        # Extract custom fields from input batch
        teacher_topk_logits = inputs.pop("teacher_topk_logits")
        teacher_topk_indices = inputs.pop("teacher_topk_indices")

        # Support both old datasets (pre-binarized 'entropy_mask') and new ones ('entropy_score').
        # If the dataset has a pre-binarized mask, pass it with is_premask=True so the loss
        # function skips dynamic thresholding (the mask was already baked in at Phase 1 time).
        if "entropy_score" in inputs:
            entropy_score = inputs.pop("entropy_score")
            is_premask = False
        else:
            entropy_score = inputs.pop("entropy_mask")
            is_premask = True

        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        
        # Mask padding tokens so they are ignored by the loss function (-100)
        if labels is not None and attention_mask is not None:
            labels = labels.clone()
            labels[attention_mask == 0] = -100
            
        # Forward pass through student model
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Compute combined Cross-Entropy and Top-K KL Divergence loss
        loss, metrics = self.loss_fn(
            student_logits=student_logits,
            teacher_topk_logits=teacher_topk_logits,
            teacher_topk_indices=teacher_topk_indices,
            targets=labels,
            entropy_score=entropy_score,
            is_premask=is_premask
        )
        
        # loss_fn returns detached tensors; converting them to Python floats forces a
        # CUDA sync, so only do it on steps we actually report. Doing it every
        # micro-batch cost ~60 ms/step of pipeline stall.
        should_report = (self.state.global_step % self.args.logging_steps == 0)

        if should_report:
            scalars = {k: float(v) for k, v in metrics.items()}
            print(f"\n[DEBUG Step {self.state.global_step}] Raw total_loss: {float(loss):.4f} | "
                  f"ce_loss: {scalars.get('train/ce_loss', 0):.4f} | "
                  f"kd_loss: {scalars.get('train/kd_loss', 0):.4f}", flush=True)

            # Log custom metrics to wandb
            if self.args.report_to and "wandb" in self.args.report_to:
                import wandb
                if wandb.run is not None:
                    # Log metrics without committing to allow Trainer to sync the global step
                    wandb.log(scalars, commit=False)

        return (loss, outputs) if return_outputs else loss


def preprocess_logits_for_metrics(logits, labels):
    """
    Extract predicted token IDs during evaluation loop to save CPU/GPU memory.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """
    Calculate token prediction accuracy against true labels.
    """
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Ignore padding tokens (-100)
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    correct = (labels == predictions).sum()
    accuracy = correct / float(len(labels))
    return {"accuracy": accuracy}


def parse_args():
    parser = argparse.ArgumentParser(description="Student Model Selective KD Training Script")
    parser.add_argument("--model_config", type=str, default="openai-community/gpt2", help="Base model architecture config")
    parser.add_argument("--hidden_size", type=int, default=384, help="Student hidden layer size")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of transformer layers for student")
    parser.add_argument("--num_attention_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to precomputed huggingface dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models/student_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight balancing CE loss and Selective KD loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature scaling for distillation")
    parser.add_argument("--batch_size", type=int, default=64, help="Per device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="selective-kd-thesis", help="Wandb project name")
    parser.add_argument("--baseline_mode", type=str, choices=["selective", "full_kd", "scratch"], default="selective", help="Experimental mode. 'selective': standard selective KD. 'full_kd': standard KD without mask. 'scratch': no KD (alpha=1.0).")
    parser.add_argument("--entropy_threshold", type=float, default=1.0, help="Threshold above which to apply distillation (applied dynamically).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint folder to resume training from, or 'true' to automatically find latest")
    parser.add_argument("--inherit_embeddings", action="store_true", help="Copy teacher's pre-trained embedding table into student before training (fixes vocabulary starvation on large-vocab models).")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze the inherited embedding layer (requires --inherit_embeddings). Student learns grammar but not vocabulary semantics.")
    # Throughput knobs. These do not affect the training maths -- only how fast
    # batches are fed to the GPU and how often we pause to evaluate/checkpoint.
    parser.add_argument("--dataloader_workers", type=int, default=8, help="Dataloader worker processes. Should match --cpus-per-task; 2 starves the GPU.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between validation passes.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Steps between intermediate checkpoints (final_model is always saved at the end).")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize wandb environment variables if enabled
    if args.wandb_project and args.wandb_project.lower() != "none":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        report_to = "none"
        
    # Enforce baseline alpha overrides
    if args.baseline_mode == "scratch":
        print("Baseline Mode: SCRATCH. Forcing alpha = 1.0 (No Knowledge Distillation)")
        args.alpha = 1.0
    elif args.baseline_mode == "full_kd":
        print("Baseline Mode: FULL_KD. Ignoring entropy mask during distillation.")
    else:
        print("Mode: SELECTIVE KD.")
        
    print(f"Loading precomputed dataset from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"\n\nERROR: Augmented dataset directory '{args.dataset_path}' does not exist!\n"
            f"You must run Phase 1 first to precompute teacher logits using Gemma-7B.\n"
            f"Command: python src/data/compute_teacher_logits.py --dataset_path <input_data> --output_path {args.dataset_path}\n"
        )
        
    if os.path.exists(os.path.join(args.dataset_path, "dataset_dict.json")):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)
        
    print("Configuring student model architecture...")
    config = AutoConfig.from_pretrained(args.model_config)
    config.hidden_size = args.hidden_size
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attention_heads
    
    # Handle GQA (Grouped Query Attention) models like Gemma and Qwen
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = min(config.num_key_value_heads, args.num_attention_heads)
        
    # CRITICAL for Qwen2: Slice layer_types list to match the new number of layers
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        config.layer_types = config.layer_types[:args.num_hidden_layers]
        
    # CRITICAL: Scale head_dim correctly to prevent NaN loss in Gemma architecture
    if hasattr(config, "head_dim"):
        config.head_dim = args.hidden_size // args.num_attention_heads
        
    # CRITICAL: Scale intermediate_size to prevent MLP exploding activations
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = args.hidden_size * 4
    
    model = AutoModelForCausalLM.from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Student Model Initialized. Total Parameters: {num_params:,}")

    # ─── Embedding Inheritance ───────────────────────────────────────────────
    if args.inherit_embeddings:
        print("\n" + "=" * 60)
        print("[EMBEDDING INHERITANCE] Loading teacher model on CPU to copy embeddings...")
        print(f"[EMBEDDING INHERITANCE] Teacher: {args.model_config}")
        print("=" * 60)

        # Load teacher on CPU only — avoids competing with student for GPU VRAM
        teacher_for_emb = AutoModelForCausalLM.from_pretrained(
            args.model_config,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        # Locate embedding layers in both student and teacher
        # Gemma/Gemma2B uses model.embed_tokens; GPT-2 uses transformer.wte
        student_embed = None
        teacher_embed = None
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            student_embed = model.model.embed_tokens
            teacher_embed = teacher_for_emb.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            student_embed = model.transformer.wte
            teacher_embed = teacher_for_emb.transformer.wte
        else:
            print("[EMBEDDING INHERITANCE] WARNING: Could not locate embed_tokens layer. Skipping inheritance.")

        if student_embed is not None and teacher_embed is not None:
            student_vocab = student_embed.weight.shape[0]
            teacher_vocab = teacher_embed.weight.shape[0]
            student_dim  = student_embed.weight.shape[1]
            teacher_dim  = teacher_embed.weight.shape[1]

            if student_vocab != teacher_vocab:
                raise ValueError(
                    f"[EMBEDDING INHERITANCE] FATAL: Vocab size mismatch! "
                    f"Student={student_vocab}, Teacher={teacher_vocab}. "
                    f"Top-K KD indices would be invalid. Aborting."
                )

            print(f"[EMBEDDING INHERITANCE] Student dim={student_dim}, Teacher dim={teacher_dim}")

            with torch.no_grad():
                if student_dim == teacher_dim:
                    # Dimensions match — direct copy
                    student_embed.weight.copy_(teacher_embed.weight)
                    print(f"[EMBEDDING INHERITANCE] Direct copy ({student_vocab} × {student_dim}).")
                else:
                    # Dimensions differ — project teacher embeddings down to student size via truncated SVD
                    # This preserves semantic structure: top-K singular vectors capture principal axes
                    print(f"[EMBEDDING INHERITANCE] Dimension mismatch. Projecting {teacher_dim}→{student_dim} via truncated SVD...")
                    # Capture the student's own freshly-initialised embedding scale BEFORE it is
                    # overwritten below, so the projected embeddings are rescaled to match the
                    # distribution the rest of the (untouched) architecture was calibrated for,
                    # rather than an arbitrary unit variance. Embeddings here are weight-tied to
                    # the LM head, so an oversized embedding table directly inflates every logit;
                    # rescaling to unit std (the previous behaviour) produced embedding rows with
                    # ~50x the norm of a correctly-scaled random init, which showed up as training
                    # loss starting around 150 instead of the expected ~10-11.
                    target_std = student_embed.weight.float().std().item()
                    W = teacher_embed.weight.float()  # (vocab, teacher_dim)
                    # Center the embedding matrix before SVD for numerical stability
                    W_mean = W.mean(dim=0, keepdim=True)
                    W_centered = W - W_mean
                    # Truncated SVD: keep only top student_dim components
                    # U: (vocab, student_dim), S: (student_dim,), Vh: (student_dim, teacher_dim)
                    U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
                    U = U[:, :student_dim]      # (vocab, student_dim)
                    S = S[:student_dim]         # (student_dim,)
                    # Scale by singular values so the projected embeddings have similar norm to originals
                    W_projected = U * S.unsqueeze(0)  # (vocab, student_dim)
                    # Rescale to match the student's own random-init embedding scale
                    current_std = W_projected.std()
                    if current_std > 1e-8:
                        W_projected = W_projected * (target_std / current_std)
                    student_embed.weight.copy_(W_projected.to(student_embed.weight.dtype))
                    print(f"[EMBEDDING INHERITANCE] SVD projection complete. Projected {student_vocab} × {teacher_dim} → {student_vocab} × {student_dim} (rescaled to student init std={target_std:.4f}).")

            embed_params = student_embed.weight.numel()
            print(f"[EMBEDDING INHERITANCE] Inherited {embed_params:,} embedding parameters ({student_vocab} tokens × {student_dim} dims).")

            # Also copy/project lm_head if it is NOT weight-tied to embeddings
            # (Gemma ties them, GPT-2 does not)
            student_lm_head = getattr(model, "lm_head", None)
            teacher_lm_head = getattr(teacher_for_emb, "lm_head", None)
            if student_lm_head is not None and teacher_lm_head is not None:
                if student_lm_head.weight.data_ptr() != student_embed.weight.data_ptr():
                    with torch.no_grad():
                        student_lm_head.weight.copy_(student_embed.weight)
                    print(f"[EMBEDDING INHERITANCE] lm_head synced to match inherited embeddings.")
                else:
                    print("ℹ lm_head is weight-tied to embed_tokens — no separate copy needed.")

            if args.freeze_embeddings:
                student_embed.weight.requires_grad_(False)
                if student_lm_head is not None and student_lm_head.weight.data_ptr() != student_embed.weight.data_ptr():
                    student_lm_head.weight.requires_grad_(False)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[EMBEDDING INHERITANCE] Embedding layer FROZEN. Trainable parameters: {trainable_params:,}")
            else:
                print("[EMBEDDING INHERITANCE] Embedding layer will be fine-tuned during training.")

        # Free teacher model from CPU memory
        del teacher_for_emb
        import gc; gc.collect()
        print("[EMBEDDING INHERITANCE] Teacher model unloaded from CPU memory.")
        print("=" * 60 + "\n")
    # ────────────────────────────────────────────────────────────────────────
    
    # Initialize custom loss function
    loss_fn = EntropySelectiveKDLoss(
        alpha=args.alpha, 
        temperature=args.temperature,
        baseline_mode=args.baseline_mode,
        entropy_threshold=args.entropy_threshold
    )
    
    # Training arguments (see Table 4.3 for the values used in the thesis)
    # Compute actual steps per epoch dynamically from dataset size
    train_size = len(dataset["train"])
    # An *optimizer* step consumes batch_size x grad_accum x world_size sequences.
    # Dividing by batch_size alone (the previous behaviour) counted micro-batches,
    # so warmup_steps was inflated by grad_accum x world_size: bs8 x accum8 asked
    # for 1659 warmup steps against only 4140 real steps -- 40% warmup, not 5%.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    seqs_per_optimizer_step = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, train_size // seqs_per_optimizer_step)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))  # 5% of total optimizer steps
    print(f"   Dataset: {train_size} sequences, batch_size={args.batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}, world_size={world_size} "
          f"(effective batch {seqs_per_optimizer_step})")
    print(f"   Optimizer steps per epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=args.eval_steps if "validation" in dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        remove_unused_columns=False,  # CRITICAL: Keep custom columns (teacher logits, masks) in input batches
        seed=args.seed,
        # Dataloading was the throughput bottleneck: profiling showed 2 workers
        # deliver a batch every ~395 ms while the GPU consumes one every ~223 ms,
        # so the A100 sat idle waiting on data. 8 workers (matching --cpus-per-task)
        # drops delivery to ~68 ms, making the step GPU-bound instead.
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=4 if args.dataloader_workers > 0 else None,
        dataloader_pin_memory=True,
    )
    
    trainer = SelectiveKDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        loss_fn=loss_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )
    
    # Resolve resume_from_checkpoint parameter
    resume = args.resume_from_checkpoint
    if resume is not None:
        if resume.lower() == "true":
            resume = True
        elif resume.lower() in ("false", "none"):
            resume = None

    print("Starting student training loop...")
    stats_cb = StatsCallback(output_dir=args.output_dir)

    # Compute and save dataset-level mask statistics before training.
    # Rank-0 only under DDP: every rank would otherwise rescan the whole dataset
    # and race writing the same dataset_stats.json.
    if args.baseline_mode != "full_kd" and trainer.is_world_process_zero():
        compute_dataset_stats(dataset, output_dir=args.output_dir, entropy_threshold=args.entropy_threshold)

    trainer.add_callback(stats_cb)
    trainer.train(resume_from_checkpoint=resume)

    # Save final model. trainer.save_model() is already rank-0 safe internally,
    # but the tokenizer write is not, so guard it explicitly.
    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    if trainer.is_world_process_zero():
        # Also save the tokenizer so AutoTokenizer.from_pretrained(final_save_path) works correctly.
        # trust_remote_code=True: needed for local custom-tokenizer teachers (e.g. the
        # 10k-vocabulary from-scratch teacher); no-op for standard HF Hub teachers.
        tokenizer = AutoTokenizer.from_pretrained(args.model_config, trust_remote_code=True)
        tokenizer.save_pretrained(final_save_path)
        # save_pretrained() writes the auto_map reference but not the custom tokenizer
        # *code* module, so carry it over too when the teacher ships one -- otherwise
        # AutoTokenizer.from_pretrained(final_model, trust_remote_code=True) fails at
        # evaluation time.
        if os.path.isdir(args.model_config):
            custom_tok = os.path.join(args.model_config, "morphology_tokenizer.py")
            if os.path.exists(custom_tok):
                import shutil
                shutil.copy(custom_tok, os.path.join(final_save_path, "morphology_tokenizer.py"))
                print("Copied custom tokenizer module into final_model.")
        print(f"Training complete. Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
