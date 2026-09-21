"""
Stage 1 of the 10k-vocabulary experiment (Section 5.13 of the thesis):
pretrain a GPT-2-small-scale causal LM
from scratch (no distillation, no pretrained weights) on the BabyLM 100M-word
corpus, using the 10,000-token BPE tokenizer.

The resulting checkpoint becomes a self-trained teacher for Stage 2, where it is
run through src/data/compute_teacher_logits.py and distilled onto a small
student on the 10M-word corpus, exactly like the existing GPT-2/Qwen2/Gemma-3
teachers.

Reuses the same 512-token chunking convention as 01_compute_teacher_logits.py
(concatenate all docs, insert EOS at doc boundaries, split into fixed blocks)
so the resulting dataset is directly comparable to the rest of the pipeline.
"""
import os
import sys
import argparse

import torch
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain a GPT-2-small-scale teacher from scratch")
    p.add_argument("--tokenizer_path", type=str, required=True,
                    help="Path to the HF-loadable 10k tokenizer bundle (trust_remote_code)")
    p.add_argument("--dataset_path", type=str, required=True,
                    help="Directory containing train.txt / dev.txt (raw text, one doc per line)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_attention_heads", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_workers", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--wandb_project", type=str, default="morphology-bpe-teacher")
    p.add_argument("--tokenize_only", action="store_true",
                    help="Tokenise/chunk the corpus, save it to --output_dir/tokenized_dataset, "
                         "and exit. Run this once (single process) before the DDP training "
                         "launch so the 100M-word corpus isn't tokenised redundantly by every "
                         "rank -- that was blowing up disk quota (each of 4 ranks x 8 dataloader "
                         "workers wrote its own near-duplicate arrow cache shards).")
    p.add_argument("--pretokenized_path", type=str, default=None,
                    help="Path to a dataset already produced by --tokenize_only. If given, "
                         "tokenisation/chunking is skipped entirely and this is loaded directly.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.wandb_project and args.wandb_project.lower() != "none":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    print(f"Loading the 10k tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # GPT-2's tokenizer ships without a pad token; DataCollatorForLanguageModeling
    # needs one. The morphology tokenizer defines <pad> itself, so this is a no-op
    # there. Chunks are all exactly max_length so nothing is actually padded, but
    # the collator still requires the token to exist.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token was unset; using eos_token ({tokenizer.eos_token}) as pad")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    if args.pretokenized_path:
        print(f"Loading pre-tokenised dataset from {args.pretokenized_path}...")
        from datasets import load_from_disk
        dataset = load_from_disk(args.pretokenized_path)
        print(f"Loaded dataset: {dataset}")
    else:
        data_files = {}
        for fname in sorted(os.listdir(args.dataset_path)):
            if fname.endswith(".txt"):
                if "train" in fname:
                    data_files["train"] = os.path.join(args.dataset_path, fname)
                elif "dev" in fname or "val" in fname:
                    data_files["validation"] = os.path.join(args.dataset_path, fname)
        if "train" not in data_files:
            raise FileNotFoundError(f"No train.txt found under {args.dataset_path}")
        print(f"Detected splits: {list(data_files.keys())}")
        dataset = load_dataset("text", data_files=data_files)

        print(f"Tokenising and chunking corpus into {args.max_length}-token blocks...")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=False, padding=False)

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=args.dataloader_workers,
            desc="Tokenising",
        )

        max_length = args.max_length

        def chunk_examples(examples):
            all_ids = []
            for ids in examples["input_ids"]:
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)
            total = (len(all_ids) // max_length) * max_length
            chunks = [all_ids[i:i + max_length] for i in range(0, total, max_length)]
            return {"input_ids": chunks, "attention_mask": [[1] * max_length] * len(chunks)}

        dataset = tokenized.map(
            chunk_examples,
            batched=True,
            desc="Chunking",
        )
        print(f"Chunked dataset: {dataset}")

    if args.tokenize_only:
        out_path = os.path.join(args.output_dir, "tokenized_dataset")
        print(f"--tokenize_only set: saving chunked dataset to {out_path} and exiting.")
        os.makedirs(args.output_dir, exist_ok=True)
        dataset.save_to_disk(out_path)
        print("Done.")
        return

    print("Configuring GPT-2-small-scale architecture (from scratch, no pretrained weights)...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
        n_inner=args.hidden_size * 4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialised from scratch. Total parameters: {num_params:,}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    seqs_per_optimizer_step = args.batch_size * args.gradient_accumulation_steps * world_size
    train_size = len(dataset["train"])
    steps_per_epoch = max(1, train_size // seqs_per_optimizer_step)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))
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
        seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        dataloader_prefetch_factor=4 if args.dataloader_workers > 0 else None,
        dataloader_pin_memory=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=collator,
    )

    print("Starting from-scratch teacher pretraining...")
    trainer.train()

    final_save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_save_path)
    if trainer.is_world_process_zero():
        # The saved tokenizer must not advertise a longer context than the model
        # actually has. Downstream code (evaluation/evaluate_glue.py) truncates to
        # tokenizer.model_max_length with no explicit max_length, so a tokenizer
        # claiming 1024 (GPT-2's default) against an n_positions=512 model feeds
        # over-long sequences straight into the position embedding table and
        # crashes with a CUDA device-side assert on long-input tasks such as
        # BoolQ and MultiRC, while short-input tasks pass.
        tokenizer.model_max_length = args.max_length
        tokenizer.save_pretrained(final_save_path)

        # save_pretrained() writes the vocab/config but not the custom tokenizer
        # *code* module, so a 10k-tokenizer checkpoint needs morphology_tokenizer.py
        # copied alongside it for AutoTokenizer(..., trust_remote_code=True) to work.
        # Only do this when that tokenizer is actually in use; copying it next to a
        # standard HF tokenizer just leaves a misleading unused file.
        if type(tokenizer).__name__ == "MorphologyBPETokenizer":
            import shutil
            shutil.copy(
                os.path.join(os.path.dirname(__file__), "morphology_tokenizer.py"),
                os.path.join(final_save_path, "morphology_tokenizer.py"),
            )
        print(f"Training complete. Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
