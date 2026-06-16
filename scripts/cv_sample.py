#!/usr/bin/env python3
"""Pull older-female Romanian reference clips from Common Voice (ungated mirror).

Runs ON the box in fish-venv (has datasets/soundfile/librosa). Streams the RO
train split, keeps female speakers in older age buckets, resamples to 44.1k mono,
and writes them + a manifest (with transcripts for Fish --prompt-text)."""
import os
import json
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset

OUT = "/ephemeral/cv-ro"
os.makedirs(OUT, exist_ok=True)
WANT_AGE = {"fifties", "sixties", "seventies", "eighties", "nineties"}
FEMALE = {"female", "female_feminine"}

got = []
for split in ["other", "validation", "train", "test", "invalidated"]:
    ds = load_dataset("fsicoli/common_voice_19_0", "ro", split=split,
                      streaming=True, trust_remote_code=True)
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > 40000:
            break
        if row.get("gender") not in FEMALE:
            continue
        if row.get("age") not in WANT_AGE:
            continue
        a = row["audio"]
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = a["sampling_rate"]
        dur = len(arr) / sr
        if dur < 3.0:
            continue
        arr44 = librosa.resample(arr, orig_sr=sr, target_sr=44100)
        age = row.get("age")
        i = len(got)
        fn = os.path.join(OUT, "cv_{:02d}_{}_{}.wav".format(i, split, age))
        sf.write(fn, arr44, 44100)
        sentence = row.get("sentence", "")
        got.append({"file": os.path.basename(fn), "split": split, "age": age,
                    "gender": row.get("gender"), "dur": round(dur, 1),
                    "client": (row.get("client_id") or "")[:12], "text": sentence})
        print("[{}] {}s split={} age={} :: {}".format(i, round(dur, 1), split, age, sentence[:70]))
        if len(got) >= 20:
            break
    if len(got) >= 20:
        break

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(got, f, ensure_ascii=False, indent=2)
print("scanned", scanned, "saved", len(got), "->", OUT)
