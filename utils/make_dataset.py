import csv
import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf

SR = 44100                  
NUM_CLIPS = 10             
SNRS_DB = [0, 5, 10, 15, 20]  
MIN_SECONDS = 20              
SEED = 0

CLEAN_DIR = "../data/raw/clean"
NOISE_DIR = "../data/raw/noise"
OUT_DIR = "../data"
CLEAN_NAME = "/clean"
NOISE_NAME = "/noise"

random.seed(SEED)

def split_80_10_10(items):
    """Shuffle a list and cut it into train / val / test."""
    items = list(items)
    random.shuffle(items)
    n_val = max(1, round(0.1 * len(items)))
    n_test = max(1, round(0.1 * len(items)))
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return {"train": train, "val": val, "test": test}


def load_noise_slice(path, n_samples):
    """Read n_samples of mono noise starting at a random point in the file."""
    seconds = n_samples / SR
    file_seconds = librosa.get_duration(path=path)
    start = random.uniform(0, max(0, file_seconds - seconds))
    y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=seconds)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    return y[:n_samples]


clip_files = sorted(glob.glob(f"{CLEAN_DIR}/**/*.mp3", recursive=True))
demand_files = sorted(glob.glob(f"{NOISE_DIR}/demand/**/ch01.wav", recursive=True))
analog_files = sorted(glob.glob(f"{NOISE_DIR}/analog/*.wav"))
analog_set = set(analog_files)

print(len(clip_files), "clips,", len(demand_files), "demand files,", len(analog_files), "analog files")
if len(clip_files) == 0:
    raise SystemExit("No mp3 files found under " + CLEAN_DIR)
if len(demand_files) + len(analog_files) == 0:
    raise SystemExit("No noise files found under " + NOISE_DIR)

random.shuffle(clip_files)
clip_files = clip_files[:NUM_CLIPS]

clip_split = split_80_10_10(clip_files)
demand_split = split_80_10_10(demand_files)
analog_split = split_80_10_10(analog_files)

noise_split = {}
for split in ["train", "val", "test"]:
    noise_split[split] = demand_split[split] + analog_split[split]
    if len(noise_split[split]) == 0:
        raise SystemExit("No noise files left for the " + split + " split, add more noise files")

rows = []
for split in ["train", "val", "test"]:
    for clip_path in clip_split[split]:

        # load the clean clip as stereo.
        try:
            clean, _ = librosa.load(clip_path, sr=SR, mono=False)
        except Exception as err:
            print("skipping", clip_path, "-", err)
            continue
        if clean.ndim == 1:
            clean = np.stack([clean, clean])
        clean = clean[:2]
        n = clean.shape[1]
        if n < MIN_SECONDS * SR:
            continue
        clean_power = np.mean(clean ** 2)
        # picking random noise file
        noise_path = random.choice(noise_split[split])
        left = load_noise_slice(noise_path, n)
        right = load_noise_slice(noise_path, n)
        noise = np.stack([left, right])
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-12:
            continue

        # 10 * log10(clean power / noise power) = snr_db.
        snr_db = random.choice(SNRS_DB)
        gain = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
        noise = noise * gain
        mixture = clean + noise
        # handle clipping
        peak = np.max(np.abs(mixture))
        if peak > 0.99:
            scale = 0.99 / peak
            clean = clean * scale
            noise = noise * scale
            mixture = mixture * scale

        # save the files
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        example_dir = f"{OUT_DIR}/{split}/{clip_id}"
        os.makedirs(example_dir, exist_ok=True)
        sf.write(f"{example_dir}/mixture.wav", mixture.T, SR, subtype="PCM_16")
        sf.write(f"{example_dir}/clean.wav", clean.T, SR, subtype="PCM_16")
        sf.write(f"{example_dir}/noise.wav", noise.T, SR, subtype="PCM_16")

        if noise_path in analog_set:
            noise_type = "analog"
        else:
            noise_type = "demand"
        rows.append({
            "split": split,
            "clip_id": clip_id,
            "duration": round(n / SR, 3),
            "snr_db": snr_db,
            "noise_type": noise_type,
            "noise_file": noise_path,
        })

        if len(rows) % 50 == 0:
            print(len(rows), "examples written")

for split in ["train", "val", "test"]:
    with open(f"{OUT_DIR}/{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mix_path", CLEAN_NAME + "_path", NOISE_NAME + "_path", "duration"])
        for row in rows:
            if row["split"] == split:
                folder = f"{split}/{row['clip_id']}"
                writer.writerow([folder + "/mixture.wav", folder + "/clean.wav",
                                 folder + "/noise.wav", row["duration"]])

with open(f"{OUT_DIR}/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
