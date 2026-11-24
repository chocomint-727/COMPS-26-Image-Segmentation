import csv
import math
import subprocess
import sys
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (N_FFT, HOP, MULTIPLE, calculate_mse, calculate_si_sdr, calculate_snr,
                        spectrogram_mses, crop, read_mono, fit_length)

ROOT = Path(__file__).resolve().parents[1]
TEST_FOLDER = ROOT / "data" / "raw" / "test"
CONFIG_PATH = ROOT / "configs" / "clean_noise.json"
RESULTS_FOLDER = ROOT / "evaluation_results" / "spleeter"
STEMS_FOLDER = RESULTS_FOLDER / "stems"
OUTPUT_CSV = RESULTS_FOLDER / "spleeter_metrics.csv"

SAMPLE_RATE = 44100

COLUMNS = [
    "sample_id",
    "input_mse",
    "spleeter_clean_mse",
    "spleeter_noise_mse",
    "loud_input_mse",
    "loud_spleeter_mse",
    "lowband_input_mse",
    "lowband_spleeter_mse",
    "input_snr",
    "spleeter_snr",
    "input_si_sdr",
    "spleeter_si_sdr",
    "duration_seconds",
]


def unet_length(num_samples):
    """Number of samples the U-Net evaluation keeps after its spectrogram crop"""
    frames = 1 + num_samples // HOP
    width = frames // MULTIPLE * MULTIPLE
    return (width - 1) * HOP


def spectrogram_db(audio):
    """Magnitude spectrogram in dB, scaled to its own peak"""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)


def find_songs():
    """Sorted ids of the song folders in the test set"""
    songs = []
    for folder in sorted(TEST_FOLDER.iterdir()):
        songs.append(folder.name)
    return songs


def stems_exist(song):
    """True if Spleeter has already written both stems for this song"""
    clean_stem = STEMS_FOLDER / song / "clean.wav"
    noise_stem = STEMS_FOLDER / song / "noise.wav"
    return clean_stem.is_file() and noise_stem.is_file()


def run_spleeter(songs):
    """Run Spleeter on every song that has no stems yet"""
    inputs_folder = STEMS_FOLDER / "_inputs"
    inputs_folder.mkdir(parents=True, exist_ok=True)

    files_to_separate = []
    longest_duration = 0.0
    for song in songs:
        if stems_exist(song):
            continue
        mixture_path = TEST_FOLDER / song / "mixture.wav"
        link_path = inputs_folder / f"{song}.wav"
        if not link_path.is_symlink():
            link_path.symlink_to(mixture_path)
        duration = sf.info(str(mixture_path)).duration
        if duration > longest_duration:
            longest_duration = duration
        files_to_separate.append(str(link_path))

    if len(files_to_separate) == 0:
        return

    max_duration = math.ceil(longest_duration) + 1
    command = [
        "spleeter", "separate",
        "-p", str(CONFIG_PATH),
        "-o", str(STEMS_FOLDER),
        "-d", str(max_duration),
    ]
    command = command + files_to_separate
    subprocess.run(command, check=True)


def evaluate_song(song):
    """Score one song's Spleeter stems against the clean and noise targets"""
    song_folder = TEST_FOLDER / song
    stem_folder = STEMS_FOLDER / song

    clean = read_mono(song_folder / "clean.wav")
    mixture = read_mono(song_folder / "mixture.wav")
    noise = read_mono(song_folder / "noise.wav")
    num_samples = len(clean)

    spleeter_clean = read_mono(stem_folder / "clean.wav")
    spleeter_noise = read_mono(stem_folder / "noise.wav")
    spleeter_clean = fit_length(spleeter_clean, num_samples)
    spleeter_noise = fit_length(spleeter_noise, num_samples)

    keep = unet_length(num_samples)
    clean_crop = clean[:keep]
    mixture_crop = mixture[:keep]
    spleeter_crop = spleeter_clean[:keep]

    c, m, nz, sc, sn = crop(*[spectrogram_db(audio) for audio in
                              (clean, mixture, noise, spleeter_clean, spleeter_noise)])
    input_mse, loud_input_mse, lowband_input_mse = spectrogram_mses(c, m)
    spleeter_mse, loud_spleeter_mse, lowband_spleeter_mse = spectrogram_mses(c, sc)

    return {
        "sample_id": song,
        "input_mse": input_mse,
        "spleeter_clean_mse": spleeter_mse,
        "spleeter_noise_mse": calculate_mse(nz, sn),
        "loud_input_mse": loud_input_mse,
        "loud_spleeter_mse": loud_spleeter_mse,
        "lowband_input_mse": lowband_input_mse,
        "lowband_spleeter_mse": lowband_spleeter_mse,
        "input_snr": calculate_snr(clean_crop, mixture_crop),
        "spleeter_snr": calculate_snr(clean_crop, spleeter_crop),
        "input_si_sdr": calculate_si_sdr(clean_crop, mixture_crop),
        "spleeter_si_sdr": calculate_si_sdr(clean_crop, spleeter_crop),
        "duration_seconds": num_samples / SAMPLE_RATE,
    }


def main():
    """Main function to run the evaluation.
    Separates the test set with Spleeter and writes per song metrics to a CSV"""
    songs = find_songs()
    run_spleeter(songs)

    rows = []
    for song in songs:
        rows.append(evaluate_song(song))

    with OUTPUT_CSV.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for row in rows:
            values = []
            for column in COLUMNS:
                values.append(row[column])
            writer.writerow(values)


if __name__ == "__main__":
    main()