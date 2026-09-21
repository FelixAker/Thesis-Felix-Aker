"""
Train the 10,000-token tokenizer used in Section 5.13 of the thesis.

Trains a byte-level BPE tokenizer at a given vocabulary size and save it as a
Hugging Face *fast* tokenizer.

Two deliberate choices:

  * Byte-level BPE (the same family GPT-2 uses) rather than SentencePiece, so
    that a comparison against the GPT-2 tokenizer varies vocabulary size and
    little else.
  * The `tokenizers` library rather than sentencepiece, so the result is a
    native fast tokenizer. Slow tokenizers cannot produce offset mappings, and
    evaluation-pipeline-2025 requires them for every zero-shot task.
"""
import argparse
import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Raw text file to train on")
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_length", type=int, default=512,
                   help="Must match the model's n_positions; the saved tokenizer "
                        "advertises this so downstream truncation cannot exceed "
                        "the position embedding table.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"training byte-level BPE, vocab_size={args.vocab_size}, on {args.corpus}")

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[args.corpus],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )

    # Serialise the trained tokenizer in full and hand that file to
    # PreTrainedTokenizerFast. Constructing GPT2TokenizerFast from vocab.json +
    # merges.txt silently produces an empty tokenizer under transformers 5.x.
    tok_json = os.path.join(args.out, "tokenizer.json")
    tok.save(tok_json)

    hf = PreTrainedTokenizerFast(
        tokenizer_file=tok_json,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=args.max_length,
    )
    hf.save_pretrained(args.out)

    probe = "The unhappiness of running quickly."
    enc = hf(probe, return_offsets_mapping=True)
    print("vocab size :", hf.vocab_size)
    print("is_fast    :", hf.is_fast)
    print("tokens     :", hf.convert_ids_to_tokens(enc["input_ids"]))
    assert "offset_mapping" in enc, "offset_mapping missing - eval pipeline would fail"
    print("offsets OK; saved to", args.out)


if __name__ == "__main__":
    main()
