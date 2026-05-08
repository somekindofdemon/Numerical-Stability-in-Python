"""
Stage 3 — convert
Convert float64 representations to float32, float16, and 8-bit quantised.

8-bit quantisation scheme (symmetric per-tensor):
  scale = max(|X|) / 127
  q = clip(round(X / scale), -128, 127)  stored as int8
  dequantise: X_hat = q * scale           recovered as float64

This is the simplest form of uniform symmetric quantisation.
It maps the full dynamic range of the tensor to [-127, 127],
minimising the maximum absolute error for a given bit width.
"""
import json
import os

import numpy as np
import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def quantise_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (int8 array, scale)."""
    scale = float(np.max(np.abs(x))) / 127.0
    if scale == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def dequantise_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float64) * scale


def main():
    data = np.load("data/representations.npz")
    reps_f64 = data["reps"]  # (N, D) float64

    conversions = {
        "float32": reps_f64.astype(np.float32),
        "float16": reps_f64.astype(np.float16),
    }

    os.makedirs("data", exist_ok=True)

    # Save float32 and float16
    for name, arr in conversions.items():
        np.save(f"data/reps_{name}.npy", arr)
        size_mb = arr.nbytes / 1e6
        print(f"{name}: {arr.dtype}, {size_mb:.2f} MB")

    # Save int8 + scale
    q, scale = quantise_int8(reps_f64)
    np.save("data/reps_int8.npy", q)
    with open("data/int8_scale.json", "w") as f:
        json.dump({"scale": scale}, f)
    size_mb = q.nbytes / 1e6
    print(f"int8: {q.dtype}, {size_mb:.2f} MB, scale={scale:.6e}")

    # Reference float64
    np.save("data/reps_float64.npy", reps_f64)
    size_mb = reps_f64.nbytes / 1e6
    print(f"float64 (reference): {reps_f64.dtype}, {size_mb:.2f} MB")

    # Quantisation error statistics
    reps_int8_back = dequantise_int8(q, scale)
    err = np.abs(reps_f64 - reps_int8_back)
    print(f"\nInt8 dequantisation error — mean: {err.mean():.4e}, max: {err.max():.4e}")


if __name__ == "__main__":
    main()
