"""Creates the necessary CSV files for Spleeter to be trained on"""
import csv
from pathlib import Path
import soundfile as sf

ROOT = Path("/ws/data/raw")
OUT = Path("/ws/configs")
NAMES = ["mixture", "clean", "noise"]

def build_csv(split, out_csv):
    fields = []
    for name in NAMES:
        fields.append(f"{name}_path")
    fields.append("duration")
    rows = []
    for song in sorted((ROOT /split).iterdir()):
        if not song.is_dir():
            continue
        row = {}
        for name in NAMES:
            path = song / f"{name}.wav"
            row[f"{name}_path"] = str(path.relative_to(ROOT))
        info = sf.info(song / "mixture.wav")
        row["duration"] = info.frames/info.samplerate
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

build_csv("train", OUT / "train.csv")
build_csv("val", OUT / "val.csv")