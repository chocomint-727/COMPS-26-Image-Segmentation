"""Settings and metrics shared by the U-Net, Spleeter and figure evaluation scripts"""
import numpy as np
import soundfile as sf

N_FFT = 2048
HOP = 512
MULTIPLE = 64
LOW_BAND_BINS = 512
LOUD_PERCENTILE = 80
EPS = 1e-10


def calculate_mse(actual, predicted):
    """Mean squared error between two arrays"""
    return np.mean((actual.astype(np.float64) - predicted.astype(np.float64)) ** 2)


def calculate_snr(actual, predicted):
    """Signal-to-noise ratio of a prediction against the reference (in decibels)"""
    actual = actual.astype(np.float64)
    predicted = predicted.astype(np.float64)
    return 10 * np.log10(np.mean(actual ** 2) / (np.mean((actual - predicted) ** 2) + EPS))


def calculate_si_sdr(actual, predicted):
    """Scale invariant SNR of a prediction against the reference (also in decibels)"""
    actual = actual.astype(np.float64) - np.mean(actual.astype(np.float64))
    predicted = predicted.astype(np.float64) - np.mean(predicted.astype(np.float64))
    target = np.sum(actual * predicted) / (np.sum(actual ** 2) + EPS) * actual
    error = predicted - target
    return 10 * np.log10(np.sum(target ** 2) / (np.sum(error ** 2) + EPS))


def spectrogram_mses(clean, estimate):
    """MSE over all bins, the loudest 20% of the clean bins, and the 0-11 kHz bins
    We take all of this data because Spleeter is unfairly penalized as it does not
    process more than 11 kHz of audio
    """
    loud = clean > np.percentile(clean, LOUD_PERCENTILE)
    if not loud.any():
        loud = np.ones_like(clean, dtype=bool)
    return (calculate_mse(clean, estimate),
            calculate_mse(clean[loud], estimate[loud]),
            calculate_mse(clean[:LOW_BAND_BINS], estimate[:LOW_BAND_BINS]))


def crop(*spectrograms):
    """Crop spectrograms to their shared shape"""
    height = min(s.shape[0] for s in spectrograms) // MULTIPLE * MULTIPLE
    width = min(s.shape[1] for s in spectrograms) // MULTIPLE * MULTIPLE
    return [s[:height, :width] for s in spectrograms]


def read_mono(path):
    """Load a wav file as mono"""
    audio, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1)


def fit_length(audio, length):
    """Trim audio, or pad it with silence, because we need it to be exact"""
    return np.pad(audio, (0, max(0, length - len(audio))))[:length]
