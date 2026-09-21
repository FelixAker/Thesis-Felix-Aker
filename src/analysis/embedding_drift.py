#!/usr/bin/env python3
"""
Measures how far each token embedding moved during training (Section 5.6).

Direct empirical test of the vocabulary starvation hypothesis: instead of
inferring which embedding rows are undertrained from corpus frequency counts
and Zipf's law alone, this measures how far each token's embedding vector
actually moved from its random initialisation over the course of training,
and correlates that movement with the token's frequency in the training
corpus.

For each architecture, this:
  1. Reconstructs the exact random initial embedding matrix by replaying the
     same set_seed(42) -> AutoModelForCausalLM.from_config(config) call used
     in training/train_student.py, using the architecture's own saved
     config.json (so hidden_size/vocab_size/etc. match exactly).
  2. Loads the corresponding trained final_model's embedding matrix.
  3. Computes per-token L2 distance between initial and final embedding.
  4. Counts per-token frequency in the training split of the augmented
     dataset used for that run.
  5. Writes a per-token CSV, a frequency-decile-binned CSV (for a compact
     thesis figure/table), and a JSON summary.

Run inside the project's apptainer container (has torch/transformers/datasets):
  apptainer exec --nv --bind <home>:<home> <sif> python3 src/analysis/embedding_drift.py
"""
import json
import csv
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from safetensors.torch import load_file
from datasets import load_from_disk

ARCHS = {
    "gpt2base": {
        "final_model": "experiments/2026-08-22_12-03-27_full_kd_gpt2base_10m_384/model/final_model",
        "dataset_path": "data/augmented_gpt2base_10m",
        "embed_key_candidates": ["transformer.wte.weight"],
        "vocab_size": 50257,
    },
    "qwen2_0.5b": {
        "final_model": "experiments/2026-08-22_12-03-34_full_kd_qwen2_0.5b_10m_384/model/final_model",
        "dataset_path": "data/augmented_qwen2_0.5b_10m",
        "embed_key_candidates": ["model.embed_tokens.weight"],
        "vocab_size": 151936,
    },
    "gemma3_270m": {
        "final_model": "experiments/2026-08-22_12-03-40_full_kd_gemma3_270m_10m_384/model/final_model",
        "dataset_path": "data/augmented_gemma3_270m_10m",
        "embed_key_candidates": ["model.embed_tokens.weight"],
        "vocab_size": 262144,
    },
}

SEED = 42
N_BINS = 10


def get_embedding_from_safetensors(path, candidates):
    sd = load_file(path)
    for k in candidates:
        if k in sd:
            return sd[k]
    raise KeyError(f"None of {candidates} found in {path}. Keys sample: {list(sd.keys())[:20]}")


def reconstruct_init_embedding(final_model_dir, embed_key_candidates, seed=SEED):
    """Replays the exact model-construction call from train_student.py
    (set_seed then AutoModelForCausalLM.from_config) using the architecture's
    own saved config, to get the true pre-training embedding matrix."""
    config = AutoConfig.from_pretrained(final_model_dir)
    set_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    sd = model.state_dict()
    for k in embed_key_candidates:
        if k in sd:
            return sd[k].clone()
    raise KeyError(f"None of {embed_key_candidates} found in fresh model. Keys sample: {list(sd.keys())[:20]}")


def token_frequencies(dataset_path, vocab_size):
    ds = load_from_disk(dataset_path)
    split = ds["train"]
    counts = np.zeros(vocab_size, dtype=np.int64)
    for batch in split.iter(batch_size=2000):
        ids = np.asarray(batch["input_ids"]).reshape(-1)
        ids = ids[(ids >= 0) & (ids < vocab_size)]
        counts += np.bincount(ids, minlength=vocab_size)
    return counts


def main():
    results = {}
    for name, meta in ARCHS.items():
        print(f"=== {name} ===", flush=True)

        init_emb = reconstruct_init_embedding(meta["final_model"], meta["embed_key_candidates"])
        model_path = f"{meta['final_model']}/model.safetensors"
        final_emb = get_embedding_from_safetensors(model_path, meta["embed_key_candidates"])
        assert init_emb.shape == final_emb.shape, (name, init_emb.shape, final_emb.shape)
        assert init_emb.shape[0] == meta["vocab_size"], (name, init_emb.shape)

        dist = torch.norm(final_emb.float() - init_emb.float(), dim=1).numpy()
        init_norm = torch.norm(init_emb.float(), dim=1).numpy()

        freqs = token_frequencies(meta["dataset_path"], meta["vocab_size"])
        seen_mask = freqs > 0
        unseen_mask = ~seen_mask

        summary = {
            "vocab_size": int(meta["vocab_size"]),
            "n_unseen": int(unseen_mask.sum()),
            "pct_unseen": float(unseen_mask.mean() * 100),
            "mean_dist_unseen": float(dist[unseen_mask].mean()) if unseen_mask.any() else None,
            "median_dist_unseen": float(np.median(dist[unseen_mask])) if unseen_mask.any() else None,
            "mean_dist_seen": float(dist[seen_mask].mean()) if seen_mask.any() else None,
            "median_dist_seen": float(np.median(dist[seen_mask])) if seen_mask.any() else None,
            "mean_init_norm": float(init_norm.mean()),
            "spearman_corr_logfreq_dist": None,
        }
        # Spearman correlation between log(1+freq) and distance moved (seen tokens only,
        # to avoid the mass point at freq=0 dominating a rank correlation)
        if seen_mask.sum() > 2:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(np.log1p(freqs[seen_mask]), dist[seen_mask])
            summary["spearman_corr_logfreq_dist"] = float(rho)

        print(json.dumps(summary, indent=2), flush=True)
        results[name] = summary

        with open(f"embedding_drift_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["token_id", "frequency", "distance_moved", "init_norm"])
            for i in range(meta["vocab_size"]):
                w.writerow([i, int(freqs[i]), float(dist[i]), float(init_norm[i])])

        order = np.argsort(-freqs)
        ranked_seen = [i for i in order if freqs[i] > 0]
        n_seen = len(ranked_seen)
        bin_rows = [("unseen (freq=0)", int(unseen_mask.sum()), summary["mean_dist_unseen"])]
        for b in range(N_BINS):
            lo = int(b * n_seen / N_BINS)
            hi = int((b + 1) * n_seen / N_BINS)
            idxs = ranked_seen[lo:hi]
            if not idxs:
                continue
            mean_d = float(np.mean([dist[i] for i in idxs]))
            bin_rows.append((f"seen decile {b + 1}/10 (most->least frequent)", len(idxs), mean_d))

        with open(f"embedding_drift_{name}_binned.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "n_tokens", "mean_distance_moved"])
            w.writerows(bin_rows)

    with open("embedding_drift_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote embedding_drift_<arch>.csv, embedding_drift_<arch>_binned.csv, embedding_drift_summary.json")


if __name__ == "__main__":
    main()
