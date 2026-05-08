"""
Stage 4 — distances
For each precision level, compute:
  - intra-speaker cosine distances (same speaker, same word, different recordings)
  - inter-speaker cosine distances (different speakers, same word)
  - their ratio
Also measure computation time.
Output: data/distances.json
"""
import json
import time

import numpy as np
import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Compute in the dtype of the inputs — do NOT upcast to float64.
    # This is intentional: we want to measure the effect of reduced precision
    # on the actual distance values, not just on storage.
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(1.0 - float(np.dot(a, b)) / denom)


def compute_distances(reps: np.ndarray, meta: list[dict]) -> dict:
    """Return intra/inter distances grouped by word."""
    intra, inter = [], []

    words = sorted(set(m["word"] for m in meta))  # deterministic order for residual alignment
    for word in words:
        idx = [i for i, m in enumerate(meta) if m["word"] == word]
        for i in idx:
            for j in idx:
                if i >= j:
                    continue
                d = cosine_distance(reps[i], reps[j])
                if meta[i]["speaker"] == meta[j]["speaker"]:
                    intra.append(d)
                else:
                    inter.append(d)

    return {"intra": intra, "inter": inter}


def load_representations(precision: str) -> np.ndarray:
    if precision == "int8":
        q = np.load("data/reps_int8.npy")
        with open("data/int8_scale.json") as f:
            scale = json.load(f)["scale"]
        # Dequantise to float32 so dot products are computed in float32,
        # preserving the quantisation error but avoiding integer overflow.
        return (q.astype(np.float32) * np.float32(scale))
    return np.load(f"data/reps_{precision}.npy")


def summarise(distances: dict) -> dict:
    intra = np.array(distances["intra"])
    inter = np.array(distances["inter"])
    ratio = float(np.mean(inter) / np.mean(intra)) if np.mean(intra) > 0 else None
    return {
        "intra_mean": float(np.mean(intra)),
        "intra_std": float(np.std(intra)),
        "inter_mean": float(np.mean(inter)),
        "inter_std": float(np.std(inter)),
        "ratio_inter_intra": ratio,
        "n_intra": len(intra),
        "n_inter": len(inter),
    }


def main():
    with open("data/meta.json") as f:
        meta = json.load(f)

    precisions = ["float64", "float32", "float16", "int8"]
    results = {}

    ref_intra, ref_inter = None, None
    for prec in precisions:
        t0 = time.perf_counter()
        reps = load_representations(prec)
        dists = compute_distances(reps, meta)
        elapsed = time.perf_counter() - t0

        summary = summarise(dists)
        summary["compute_time_s"] = elapsed
        summary["raw"] = dists
        results[prec] = summary

        if prec == "float64":
            ref_intra = summary["intra_mean"]
            ref_inter = summary["inter_mean"]

        delta_intra = summary["intra_mean"] - ref_intra if ref_intra else 0.0
        delta_inter = summary["inter_mean"] - ref_inter if ref_inter else 0.0
        print(
            f"{prec:10s}  intra={summary['intra_mean']:.6f} (Δ={delta_intra:+.2e})  "
            f"inter={summary['inter_mean']:.6f} (Δ={delta_inter:+.2e})  "
            f"ratio={summary['ratio_inter_intra']:.5f}  time={elapsed:.2f}s"
        )

    with open("data/distances.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
