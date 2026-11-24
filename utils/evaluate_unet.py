"""Evaluate U-Net predictions on test recordings."""
from pathlib import Path
import sys
import csv
import numpy as np
import torch
import librosa
import soundfile as sf
from unet_model import UNet
from eval_utils import N_FFT, HOP, EPS, calculate_mse, calculate_si_sdr, calculate_snr, spectrogram_mses, crop, read_mono

ROOT = Path(__file__).resolve().parents[1]
TEST_FOLDER = ROOT / "data/raw/test"
SPEC_FOLDER = ROOT / "data/spectrogram-magnitude/numpy-files"
CHECKPOINT = ROOT / "src/unet/checkpoints/full/spectrogram_7.4644.pt"
OUTPUT_FOLDER = ROOT / "evaluation_results"

MAX_SAMPLES = 0
SAVE_AUDIO = True
GRIFFIN_LIM_ITERS = 32

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src/unet"))


def evaluate_audio(raw_folder, sample_id, predicted_clean, predicted_noise, clean_spec, mixture_spec):
    clean_audio = read_mono(raw_folder / "clean.wav")
    mixture_audio = read_mono(raw_folder / "mixture.wav")
    sr = sf.info(str(raw_folder / "clean.wav")).samplerate
    mixture_sr = sf.info(str(raw_folder / "mixture.wav")).samplerate
    if sr != mixture_sr:
        raise ValueError(f"Sample rates do not match for {sample_id}: {sr} vs {mixture_sr}")

    stft = librosa.stft(mixture_audio, n_fft=N_FFT, hop_length=HOP)
    phase = np.exp(1j * np.angle(stft))
    reference = np.max(np.abs(stft))

    def reconstruct(spectrogram):
        height, width = spectrogram.shape
        magnitude = librosa.db_to_amplitude(spectrogram, ref=reference)
        expected_bins = N_FFT // 2 + 1
        magnitude = np.pad(magnitude, ((0, expected_bins - height), (0, 0)))

        X = magnitude * phase[:, :width]
        for _ in range(GRIFFIN_LIM_ITERS):
            audio = librosa.istft(X, n_fft=N_FFT, hop_length=HOP)
            rebuilt = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)[:, :width]
            X = magnitude * np.exp(1j * np.angle(rebuilt))
        return librosa.istft(X, n_fft=N_FFT, hop_length=HOP)
    
    predicted_audio = reconstruct(predicted_clean)
    reconstructed_clean = reconstruct(clean_spec)
    baseline_audio = reconstruct(mixture_spec)
    clean_audio = np.squeeze(clean_audio)
    predicted_audio = np.squeeze(predicted_audio)
    mixture_audio = np.squeeze(mixture_audio)
    reconstructed_clean = np.squeeze(reconstructed_clean)
    baseline_audio = np.squeeze(baseline_audio)

    min_len = min(len(clean_audio), len(predicted_audio), len(mixture_audio), 
                  len(reconstructed_clean), len(baseline_audio))
    
    clean_audio = clean_audio[:min_len]
    predicted_audio = predicted_audio[:min_len]
    mixture_audio = mixture_audio[:min_len]
    reconstructed_clean = reconstructed_clean[:min_len]
    baseline_audio = baseline_audio[:min_len]

    reconstruction_snr = calculate_snr(clean_audio, reconstructed_clean)
    input_snr = calculate_snr(clean_audio, mixture_audio)
    output_snr = calculate_snr(clean_audio, predicted_audio)
    
    input_sdr = calculate_si_sdr(clean_audio, mixture_audio)
    output_sdr = calculate_si_sdr(clean_audio, predicted_audio)
    
    baseline_snr = calculate_snr(clean_audio, baseline_audio)
    baseline_sdr = calculate_si_sdr(clean_audio, baseline_audio)

    original_duration = len(clean_audio) / sr

    return (input_snr, output_snr, reconstruction_snr, original_duration, 
            input_sdr, output_sdr, baseline_snr, baseline_sdr)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    channels = [32, 64, 128, 256, 384, 512]
    model = UNet(2, len(channels), 1, channels + channels[::-1]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    results = []

    for raw_folder in sorted(TEST_FOLDER.iterdir()):
        if not raw_folder.is_dir():
            continue

        sample_id = raw_folder.name
        spec_folder = SPEC_FOLDER / sample_id
        paths = [spec_folder / f"{name}.npy" for name in ("mixture", "clean", "noise")]

        if not all(path.is_file() for path in paths):
            print(f"Skipping {sample_id}: missing matching spectrograms")
            continue

        mixture, clean, noise = [np.load(path) for path in paths]

        mixture, clean, noise = [x.astype(np.float32) for x in crop(mixture, clean, noise)]
        if clean.size == 0:
            print(f"Skipping {sample_id}: spectrogram too small")
            continue

        model_input = torch.from_numpy(mixture)[None, None].to(device)
        with torch.inference_mode():
            predicted_clean, predicted_noise = model(model_input)[0].cpu().numpy()

        input_mse, loud_input_mse, lowband_input_mse = spectrogram_mses(clean, mixture)
        clean_mse, loud_clean_mse, lowband_clean_mse = spectrogram_mses(clean, predicted_clean)
        noise_mse = calculate_mse(noise, predicted_noise)

        reduction = (input_mse - clean_mse) / (input_mse + EPS) * 100
        print(f"\nTest recording: {sample_id}")

        snr_results = evaluate_audio(raw_folder, sample_id, predicted_clean, predicted_noise, clean, mixture)

        if snr_results is not None:
            (input_snr, output_snr, reconstruction_snr, original_duration, 
             input_sdr, output_sdr, baseline_snr, baseline_sdr) = snr_results

            results.append([
                sample_id, input_mse, clean_mse, noise_mse, reduction,
                input_snr, output_snr, output_snr - input_snr,
                input_sdr, output_sdr, output_sdr - input_sdr,
                reconstruction_snr, original_duration,
                baseline_snr, baseline_sdr,
                loud_input_mse, loud_clean_mse,
                lowband_input_mse, lowband_clean_mse
            ])

        if MAX_SAMPLES and len(results) >= MAX_SAMPLES:
            break

    if not results:
        print("No matching test recordings found.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if GRIFFIN_LIM_ITERS > 0:
        csv_path = OUTPUT_FOLDER / f"spectrogram_metrics_griffinlim{GRIFFIN_LIM_ITERS}.csv"
    else:
        csv_path = OUTPUT_FOLDER / "spectrogram_metrics.csv"
    
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_id", "input_mse", "clean_mse", "noise_mse", "error_reduction_percent",
            "input_snr", "output_snr", "snr_improvement",
            "input_si_sdr", "output_si_sdr", "si_sdr_improvement",
            "reconstruction_snr", "original_duration_seconds",
            "pipeline_only_snr", "pipeline_only_si_sdr",
            "loud_input_mse", "loud_clean_mse",
            "lowband_input_mse", "lowband_clean_mse"
        ])
        writer.writerows(results)

    scores = np.array([row[1:] for row in results], dtype=float)
    averages = scores.mean(axis=0)

    metric_names = [
        "input MSE", "clean MSE", "noise MSE", "error reduction (%)",
        "input SNR", "output SNR", "SNR improvement",
        "input SI-SDR", "output SI-SDR", "SI-SDR improvement",
        "reconstruction SNR", "original duration (s)",
        "pipeline-only SNR", "pipeline-only SI-SDR",
        "loud-bin input MSE", "loud-bin clean MSE",
        "low-band input MSE", "low-band clean MSE"
    ]

    print(f"\nEvaluated {len(results)} test recordings. Results saved to: {csv_path}")
    print("\nMetrics (on avg):")
    for name, avg in zip(metric_names, averages):
        print(f"{name:<25}: {avg:>8.4f}")


if __name__ == "__main__":
    main()