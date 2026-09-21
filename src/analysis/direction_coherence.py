#!/usr/bin/env python3
"""
Tests whether unseen token embeddings move apart or collapse together
(Section 5.6 of the thesis).

Follow-up to embedding_drift.py. That analysis found that "unseen"
(never appear in the 10M training corpus) token embeddings move MORE than
"seen" token embeddings during training -- the opposite of what a naive
"zero gradient => frozen at init" story predicts. The likely explanation is
weight tying: the embedding table doubles as the LM head, so the softmax
normalizer sends every vocabulary row a small, diffuse "be globally less
likely" gradient on every step, regardless of whether that token ever
appears as an input or target.

This script tests whether that diffuse push is *generic* (shared across
unseen tokens, i.e. they all move in roughly the same direction) versus
*differentiated* (each token developing its own distinct representation, as
seen tokens should from genuine co-occurrence-based learning), via two
independent, init-reconstruction-free diagnostics computed directly on the
FINAL trained embeddings:

  1. Mean pairwise cosine similarity within a random sample of unseen-token
     embeddings vs. within a random sample of seen-token embeddings vs. a
     freshly-initialised random baseline (expected ~0 for i.i.d. random
     vectors in high dimension).
  2. Fraction of variance explained by the top principal component(s) of
     each group (PCA) -- a dominant shared direction shows up as unusually
     high top-PC explained variance; differentiated, semantically distinct
     vectors look closer to isotropic (variance spread across dimensions).

Both diagnostics operate on the FINAL embedding matrix only, so they are
robust to any uncertainty in exactly reproducing the original random seed
used at initialisation (unlike a distance-from-init measurement).
"""
import json
import numpy as np
import torch
from safetensors.torch import load_file

ARCHS = {
    "gpt2base": {
        "final_model": "experiments/2026-08-22_12-03-27_full_kd_gpt2base_10m_384/model/final_model",
        "csv": "embedding_drift_gpt2base.csv",
        "embed_key_candidates": ["transformer.wte.weight"],
    },
    "qwen2_0.5b": {
        "final_model": "experiments/2026-08-22_12-03-34_full_kd_qwen2_0.5b_10m_384/model/final_model",
        "csv": "embedding_drift_qwen2_0.5b.csv",
        "embed_key_candidates": ["model.embed_tokens.weight"],
    },
    "gemma3_270m": {
        "final_model": "experiments/2026-08-22_12-03-40_full_kd_gemma3_270m_10m_384/model/final_model",
        "csv": "embedding_drift_gemma3_270m.csv",
        "embed_key_candidates": ["model.embed_tokens.weight"],
    },
}

SEED = 42
SAMPLE_N = 3000  # tokens sampled per group for the O(n^2) pairwise cosine check
N_PC_FOR_REPORT = 5


def get_embedding(path, candidates):
    sd = load_file(path)
    for k in candidates:
        if k in sd:
            return sd[k].float()
    raise KeyError(f"None of {candidates} found in {path}. Keys sample: {list(sd.keys())[:20]}")


def mean_pairwise_cosine(mat):
    """mat: [n, d] tensor. Returns mean off-diagonal cosine similarity."""
    normed = torch.nn.functional.normalize(mat, dim=1)
    sim = normed @ normed.T
    n = sim.shape[0]
    off_diag_sum = sim.sum() - torch.trace(sim)
    return float(off_diag_sum / (n * (n - 1)))


def top_pc_explained_variance(mat, k=N_PC_FOR_REPORT):
    """mat: [n, d] tensor, mean-centered before PCA. Returns cumulative
    explained-variance ratio for the top-1 and top-k principal components."""
    x = mat - mat.mean(dim=0, keepdim=True)
    # SVD on the (n, d) centered matrix; singular values^2 / (n-1) = eigenvalues of covariance
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = var.sum()
    top1 = float(var[0] / total)
    topk = float(var[:k].sum() / total)
    return top1, topk


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    for name, meta in ARCHS.items():
        print(f"=== {name} ===", flush=True)
        emb = get_embedding(f"{meta['final_model']}/model.safetensors", meta["embed_key_candidates"])

        import csv as csv_module
        freqs = {}
        with open(meta["csv"]) as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                freqs[int(row["token_id"])] = int(row["frequency"])
        vocab_size = emb.shape[0]
        seen_ids = np.array([i for i in range(vocab_size) if freqs.get(i, 0) > 0])
        unseen_ids = np.array([i for i in range(vocab_size) if freqs.get(i, 0) == 0])

        n = min(SAMPLE_N, len(seen_ids), len(unseen_ids))
        seen_sample = rng.choice(seen_ids, size=n, replace=False)
        unseen_sample = rng.choice(unseen_ids, size=n, replace=False)

        seen_mat = emb[torch.as_tensor(seen_sample, dtype=torch.long)]
        unseen_mat = emb[torch.as_tensor(unseen_sample, dtype=torch.long)]

        # Fresh random baseline: same shape/scale as this architecture's own
        # embedding init distribution, estimated from the empirical std of
        # the trained matrix's *unseen* rows is not appropriate (already
        # moved) -- instead just draw standard-normal vectors scaled to the
        # mean row-norm we already measured at init reconstruction time, if
        # available; otherwise fall back to N(0, 0.02) (a common transformer
        # init std) purely as an isotropic-random reference point.
        random_baseline = torch.randn(n, emb.shape[1]) * 0.02

        seen_cos = mean_pairwise_cosine(seen_mat)
        unseen_cos = mean_pairwise_cosine(unseen_mat)
        random_cos = mean_pairwise_cosine(random_baseline)

        seen_top1, seen_topk = top_pc_explained_variance(seen_mat)
        unseen_top1, unseen_topk = top_pc_explained_variance(unseen_mat)
        random_top1, random_topk = top_pc_explained_variance(random_baseline)

        summary = {
            "n_sampled_per_group": n,
            "mean_pairwise_cosine": {
                "seen": seen_cos, "unseen": unseen_cos, "random_baseline": random_cos,
            },
            "top1_pc_explained_variance": {
                "seen": seen_top1, "unseen": unseen_top1, "random_baseline": random_top1,
            },
            f"top{N_PC_FOR_REPORT}_pc_explained_variance": {
                "seen": seen_topk, "unseen": unseen_topk, "random_baseline": random_topk,
            },
        }
        print(json.dumps(summary, indent=2), flush=True)
        results[name] = summary

    with open("direction_coherence_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote direction_coherence_summary.json")


if __name__ == "__main__":
    main()
