"""
Tokenizer wrapper for the 10k-vocabulary experiment (Section 5.13).

Note: the filename is load-bearing. Saved checkpoints point at it through the
`auto_map` entry in their tokenizer_config.json, so renaming this file would
break loading of every checkpoint trained with it.

Thin HF-compatible wrapper around a raw SentencePiece BPE model, used because
transformers' built-in slow-tokenizer -> fast-tokenizer converters (LlamaTokenizer,
T5Tokenizer, etc.) assume a Unigram-type SentencePiece model. Our 10k
tokenizer was trained with
`--model_type=bpe`, which the built-in converters silently mis-parse into a
near-empty vocabulary instead of raising an error. This class talks to the
SentencePiece processor directly and sidesteps that conversion path entirely.

Loadable via `AutoTokenizer.from_pretrained(path, trust_remote_code=True)` once
this file, tokenizer.model, and a tokenizer_config.json with the matching
`auto_map` entry are saved alongside each other (see save_pretrained()).
"""
import os
import sentencepiece as spm
from transformers import PreTrainedTokenizer


class MorphologyBPETokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, pad_token="<pad>", unk_token="<unk>",
                 bos_token="<s>", eos_token="</s>", **kwargs):
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)
        super().__init__(pad_token=pad_token, unk_token=unk_token,
                          bos_token=bos_token, eos_token=eos_token, **kwargs)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def _offsets_for(self, text):
        """
        Character spans for each token, matching what a fast tokenizer's
        return_offsets_mapping would give.

        evaluation-pipeline-2025 uses offset_mapping to work out which tokens
        belong to the scored completion span, so it is required for every
        zero-shot task. HF's slow-tokenizer base class does not implement it,
        and converting this SentencePiece BPE model to a fast tokenizer does not
        reproduce its exact segmentation - so we compute the spans directly from
        the pieces instead, which keeps the tokenisation the models were
        actually trained with.
        """
        pieces = self.sp_model.encode(text, out_type=str)
        offsets = []
        cursor = 0
        for piece in pieces:
            # "▁" marks a preceding space (or the sentence-initial dummy prefix)
            surface = piece[1:] if piece.startswith("▁") else piece
            if surface == "":
                # bare "▁": a whitespace token, or the dummy prefix at position 0
                offsets.append((cursor, cursor))
                continue
            idx = text.find(surface, cursor)
            if idx < 0:
                # byte-fallback piece or normalisation mismatch: emit an empty
                # span at the cursor rather than desynchronising the rest
                offsets.append((cursor, cursor))
                continue
            offsets.append((idx, idx + len(surface)))
            cursor = idx + len(surface)
        return offsets

    def _aligned_offsets(self, text, ids):
        """
        Piece offsets aligned 1:1 with `ids`. build_inputs_with_special_tokens()
        wraps the pieces in <s>/</s>, so the raw piece spans are shorter than
        input_ids; special-token positions get an empty (0, 0) span, which is
        what fast tokenizers emit for them. A length mismatch here would make
        the pipeline attribute scores to the wrong tokens.
        """
        offsets = self._offsets_for(text)
        specials = set(self.all_special_ids)
        aligned, it = [], iter(offsets)
        for tok_id in ids:
            if tok_id in specials:
                aligned.append((0, 0))
            else:
                aligned.append(next(it, (0, 0)))
        return aligned

    def __call__(self, *args, **kwargs):
        want_offsets = kwargs.pop("return_offsets_mapping", False)
        encoding = super().__call__(*args, **kwargs)
        if not want_offsets:
            return encoding

        text = kwargs.get("text", args[0] if args else None)
        ids = encoding["input_ids"]
        if isinstance(text, str):
            seq = ids[0] if ids and isinstance(ids[0], list) else ids
            encoding["offset_mapping"] = self._aligned_offsets(text, seq)
        elif isinstance(text, (list, tuple)):
            encoding["offset_mapping"] = [
                self._aligned_offsets(t, i) for t, i in zip(text, ids)
            ]
        return encoding

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def save_vocabulary(self, save_directory, filename_prefix=None):
        out_name = (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model"
        out_path = os.path.join(save_directory, out_name)
        if os.path.abspath(out_path) != os.path.abspath(self.vocab_file):
            with open(self.vocab_file, "rb") as fin, open(out_path, "wb") as fout:
                fout.write(fin.read())
        return (out_path,)
