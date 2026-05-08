"""
Stage 2 — extract
Load wav2vec2 and extract frame-level representations for each word occurrence.
Aggregate via mean pooling over time. All stored in float64.
Output: data/representations.npz
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from transformers import Wav2Vec2Model, Wav2Vec2Processor

DEVICE = "cpu"  # wav2vec inference on CPU for reproducibility


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling (sufficient for feature extraction)."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, audio)


def extract_representation(
    wav_path: str,
    start: float,
    end: float,
    processor,
    model,
    target_sr: int,
) -> np.ndarray:
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Slice to word boundaries
    s = int(start * sr)
    e = int(end * sr)
    segment = audio[s:e]
    if len(segment) == 0:
        return None

    segment = resample(segment.astype(np.float64), sr, target_sr)

    inputs = processor(
        segment.astype(np.float32),
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states: (1, T, D) — mean pool over time → (D,)
    rep = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
    return rep.astype(np.float64)


def main():
    params = load_params()
    target_sr = params["target_sr"]
    model_name = params["wav2vec_model"]

    print(f"Loading model {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
    model.eval()

    with open("data/manifest.json") as f:
        manifest = json.load(f)

    reps = []    # (D,) float64
    meta = []    # {word, speaker, wav, start, end}

    t0 = time.time()
    total = sum(
        len(occs) for spk_dict in manifest.values() for occs in spk_dict.values()
    )
    done = 0
    for word, spk_dict in manifest.items():
        for speaker, occurrences in spk_dict.items():
            for occ in occurrences:
                rep = extract_representation(
                    occ["wav"], occ["start"], occ["end"],
                    processor, model, target_sr
                )
                if rep is None:
                    continue
                reps.append(rep)
                meta.append({"word": word, "speaker": speaker,
                              "wav": occ["wav"], "start": occ["start"], "end": occ["end"]})
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {done}/{total} ({elapsed:.1f}s)")

    reps_array = np.stack(reps)  # (N, D)
    print(f"Extracted {len(reps)} representations, shape {reps_array.shape}")
    print(f"Total time: {time.time()-t0:.1f}s")

    os.makedirs("data", exist_ok=True)
    np.savez("data/representations.npz", reps=reps_array)
    with open("data/meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
