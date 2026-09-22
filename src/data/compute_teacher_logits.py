"""
Phase 1 of the pipeline (Section 3.2 of the thesis).

Runs the teacher once over the training corpus and caches, per token position,
its top-K logits (K=5) and the entropy of its output distribution. Phase 2
reads this cache, so the teacher is never loaded during student training.

The teacher and the student must share a vocabulary: the student configuration
is derived from the teacher's, and top-K indices from a different vocabulary
would be out of range for the student's logit tensor.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def compute_entropy_and_topk(logits, k=5):
    """
    Computes distribution entropy and extracts Top-K logits/indices.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        k: Number of top logits to retain
        
    Returns:
        topk_logits: Tensor of Top-K logit values
        topk_indices: Tensor of Top-K vocabulary indices
        entropy_score: Continuous entropy values per token
    """
    # Compute probability distribution
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Calculate Shannon entropy per token: shape (batch_size, seq_len)
    entropy_score = -torch.sum(probs * log_probs, dim=-1)
    
    # Extract Top-K values and indices
    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
    
    return topk_values, topk_indices, entropy_score


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute top-K teacher logits and per-token entropy")
    parser.add_argument("--teacher_model", type=str, default="openai-community/gpt2",
                        help="HuggingFace teacher model. MUST share the same tokenizer vocabulary "
                             "as the student model so that top-K indices are compatible. "
                             "Default: gpt2 (50,257 tokens), the smallest of the three teachers "
                             "reported in the thesis.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input tokenized dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to save augmented dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--k", type=int, default=5, help="Number of Top-K logits to retain")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--quantize", action="store_true", help="Load model in 8-bit (for 16GB GPUs like V100). Halves VRAM usage.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! The script will run incredibly slowly on CPU. Please check your PyTorch installation or SLURM GPU allocation.")
        
    print(f"Loading teacher model: {args.teacher_model}...")
    # trust_remote_code=True: needed for local custom-tokenizer teachers (e.g. the
    # 10k-vocabulary from-scratch teacher), which ship a small Python module
    # (morphology_tokenizer.py) alongside the model instead of using a built-in
    # HF tokenizer class. No-op for standard HF Hub teachers (GPT-2/Qwen2/Gemma).
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Vocabulary compatibility guard ---
    # The student MUST use the same tokenizer as the teacher. 
    # Since we are expanding beyond GPT-2, we will trust the user to pair them correctly.
    print(f"Teacher '{args.teacher_model}' has vocab size {len(tokenizer):,}.")
        
    # Load model - use 8-bit quantization for smaller GPUs (e.g. V100 16GB), full float16 for A100/H100
    if args.quantize:
        print("Loading in 8-bit mode (bitsandbytes) to fit on smaller GPUs...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            device_map={"":0},
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            device_map={"":0},
            torch_dtype=torch.float16
        )
    model.eval()
    
    print(f"Loading dataset from {args.dataset_path}...")

    # Upfront check: fail fast with a clear message if the path doesn't exist
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"\n\nDataset path not found: '{args.dataset_path}'\n"
            f"Please verify the path exists on the cluster before submitting.\n"
            f"You can check with: ls {args.dataset_path}\n"
        )

    if os.path.exists(os.path.join(args.dataset_path, "dataset_dict.json")):
        # Already a saved HuggingFace dataset
        dataset = load_from_disk(args.dataset_path)
    elif os.path.isdir(args.dataset_path):
        # Directory of raw .txt files — auto-detect train/validation splits
        data_files = {}
        for fname in sorted(os.listdir(args.dataset_path)):
            if fname.endswith(".txt"):
                if "train" in fname:
                    data_files["train"] = os.path.join(args.dataset_path, fname)
                elif "dev" in fname or "val" in fname:
                    data_files["validation"] = os.path.join(args.dataset_path, fname)
        if not data_files:
            raise FileNotFoundError(
                f"\n\nNo .txt files found in '{args.dataset_path}'.\n"
                f"Files present: {os.listdir(args.dataset_path)}\n"
            )
        print(f"Detected splits: {list(data_files.keys())}")
        dataset = load_dataset("text", data_files=data_files)
    else:
        raise ValueError(f"Unrecognised dataset path format: '{args.dataset_path}'")
        
    # -----------------------------------------------------------------------
    # CHUNKING STEP: tokenize full corpus and chunk into non-padded blocks
    # -----------------------------------------------------------------------
    # We join all lines, tokenize the full text stream, and split into
    # fixed-length chunks of max_length tokens.  This avoids per-sentence
    # padding which wastes ~97 % of each sequence on <|endoftext|> tokens and
    # corrupts both the CE targets and the teacher KD signal.
    # -----------------------------------------------------------------------
    print(f"Tokenising and chunking corpus into {args.max_length}-token blocks…")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )

    def chunk_examples(examples):
        """Concatenate all token IDs and split into fixed-length chunks."""
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # sentence boundary marker

        total = (len(all_ids) // args.max_length) * args.max_length
        chunks = [all_ids[i : i + args.max_length] for i in range(0, total, args.max_length)]
        return {"input_ids": chunks, "attention_mask": [[1] * args.max_length] * len(chunks)}

    chunked = tokenized.map(
        chunk_examples,
        batched=True,
        desc="Chunking",
    )

    print(f"Chunked dataset: {chunked}")
    dataset = chunked

    def process_batch(examples):
        """Run teacher inference and extract Top-K logits + entropy mask."""
        import torch as _torch
        input_ids = _torch.tensor(examples["input_ids"], dtype=_torch.long)
        attention_mask = _torch.tensor(examples["attention_mask"], dtype=_torch.long)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with _torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            topk_logits, topk_indices, entropy_score = compute_entropy_and_topk(
                logits,
                k=args.k,
            )

        # Gate entropy_score by attention_mask so padding positions are 0
        entropy_score = entropy_score * attention_mask.float()

        # Move back to CPU. Cast to compact dtypes to minimise disk footprint.
        return {
            "input_ids": input_ids.cpu().to(_torch.int32).numpy(),
            "attention_mask": attention_mask.cpu().to(_torch.int8).numpy(),
            "teacher_topk_logits": topk_logits.cpu().to(_torch.float16).numpy(),
            "teacher_topk_indices": topk_indices.cpu().to(_torch.int32).numpy(),
            "entropy_score": entropy_score.cpu().to(_torch.float16).numpy(),
            "labels": input_ids.cpu().to(_torch.int32).numpy(),
        }

    print("Pre-computing teacher Top-K logits and entropy masks across dataset...")
    # Process dataset map with batched=True
    augmented_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,
        desc="Computing teacher logits",
        keep_in_memory=True  # Bypass HuggingFace disk cache to save 50% space
    )
    
    print(f"Saving augmented dataset to {args.output_path}...")
    augmented_dataset.save_to_disk(args.output_path)
    print("Pre-computation complete successfully!")


if __name__ == "__main__":
    main()
