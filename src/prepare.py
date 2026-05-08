"""
Stage 1 — prepare
Parse all word-level CSV files and build a manifest of (speaker, word, wav, start, end).
Filter to words that appear across enough speakers with enough repetitions.
Output: data/manifest.json
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def parse_words_csv(path: Path) -> list[dict]:
    entries = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 3:
                continue
            word = row[0].strip('"').strip()
            if not word:
                continue
            try:
                start = float(row[1])
                end = float(row[2])
            except ValueError:
                continue
            entries.append({"word": word, "start": start, "end": end})
    return entries


def main():
    params = load_params()
    corpus_root = Path(params["corpus_root"])
    min_speakers = params["min_speakers"]
    min_reps = params["min_reps_per_spk"]
    target_words = set(params.get("target_words") or [])

    # word -> speaker -> list of occurrences
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for speaker_dir in sorted(corpus_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for csv_file in sorted(speaker_dir.glob("*_words.csv")):
            # e.g. ab_rus_list1_FRcorp10_words.csv → ab_rus_list1_FRcorp10.wav
            wav_file = csv_file.with_name(csv_file.stem.removesuffix("_words") + ".wav")
            if not wav_file.exists():
                continue
            for entry in parse_words_csv(csv_file):
                data[entry["word"]][speaker].append(
                    {
                        "speaker": speaker,
                        "wav": str(wav_file),
                        "start": entry["start"],
                        "end": entry["end"],
                    }
                )

    # Filter words
    manifest = {}
    for word, spk_dict in data.items():
        if target_words and word not in target_words:
            continue
        # keep only speakers with enough repetitions
        valid = {s: occs for s, occs in spk_dict.items() if len(occs) >= min_reps}
        if len(valid) < min_speakers:
            continue
        manifest[word] = valid

    os.makedirs("data", exist_ok=True)
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_occ = sum(
        len(occs) for spk_dict in manifest.values() for occs in spk_dict.values()
    )
    print(f"Words selected: {list(manifest.keys())}")
    print(f"Total occurrences: {total_occ}")


if __name__ == "__main__":
    main()
