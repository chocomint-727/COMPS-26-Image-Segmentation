# python utils/make_dataset.py --noise-type analog

import argparse
import csv
import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf

SR = 44100
NUM_CLIPS = 4000
SNRS_DB = [0, 5, 10, 15, 20]
MIN_SECONDS = 20
SEED = 0
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(PROJECT_DIR, "data/audio/raw/clean")
NOISE_DIR = os.path.join(PROJECT_DIR, "data/audio/raw/noise")
OUT_DIR = os.path.join(PROJECT_DIR, "data/audio")
CLEAN_NAME = "/clean"
NOISE_NAME = "/noise"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--noise-type",
    choices=["analog", "demand", "both"],
    default="both",
    help="Generate an analog dataset, a demand dataset, or both."
)
args = parser.parse_args()

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

print(len(clip_files), "clean clips")
if len(clip_files) == 0:
    raise SystemExit("No mp3 files found under " + CLEAN_DIR)

random.shuffle(clip_files)
clip_files = clip_files[:NUM_CLIPS]

# split clean clips ONCE so both datasets use identical songs per split
clip_split = split_80_10_10(clip_files)

dataset_types = (
    ["analog", "demand"]
    if args.noise_type == "both"
    else [args.noise_type]
)

for dataset_type in dataset_types:
    if dataset_type == "analog":
        noise_files = sorted(glob.glob(f"{NOISE_DIR}/analog/*.wav"))
    else:
        noise_files = sorted(
            glob.glob(f"{NOISE_DIR}/demand/**/ch01.wav", recursive=True)
        )

    print(len(noise_files), dataset_type, "noise files")
    if len(noise_files) == 0:
        raise SystemExit(
            "No " + dataset_type + " noise files found under " + NOISE_DIR
        )

    noise_split = split_80_10_10(noise_files)

    for split in ["train", "val", "test"]:
        if len(clip_split[split]) > 0 and len(noise_split[split]) == 0:
            raise SystemExit(
                "No noise files left for the " + split + " split"
            )

    # each noise type gets its own output directory and CSV files
    dataset_dir = f"{OUT_DIR}/{dataset_type}"
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

            # picking random noise file from the selected noise type
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
            example_dir = f"{dataset_dir}/{split}/{clip_id}"
            os.makedirs(example_dir, exist_ok=True)
            sf.write(f"{example_dir}/mixture.wav", mixture.T, SR, subtype="PCM_16")
            sf.write(f"{example_dir}/clean.wav", clean.T, SR, subtype="PCM_16")
            sf.write(f"{example_dir}/noise.wav", noise.T, SR, subtype="PCM_16")

            rows.append({
                "split": split,
                "clip_id": clip_id,
                "duration": round(n / SR, 3),
                "snr_db": snr_db,
                "noise_type": dataset_type,
                "noise_file": noise_path,
            })
            #progress
            if len(rows) % 50 == 0:
                print(len(rows), "examples written")

    for split in ["train", "val", "test"]:
        with open(f"{dataset_dir}/{split}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mix_path", CLEAN_NAME + "_path", NOISE_NAME + "_path", "duration"])
            for row in rows:
                if row["split"] == split:
                    folder = f"{split}/{row['clip_id']}"
                    writer.writerow([folder + "/mixture.wav", folder + "/clean.wav",
                                     folder + "/noise.wav", row["duration"]])

    with open(f"{dataset_dir}/metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "clip_id", "duration", "snr_db", "noise_type", "noise_file"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Finished {dataset_type} dataset: {len(rows)} examples")
