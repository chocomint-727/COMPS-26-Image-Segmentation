import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the audio file
# file_path = "wav-files/lucki.wav"  # Replace with your actual file path

# Create the output directory if it doesn't exist
input_path = Path("../data/spectrogram/wav-files")  # Replace with your actual input directory path
output_dir_numpy = Path("../data/spectrogram/numpy-files")  # Replace with your actual output directory path
output_dir_spectrogram = Path("../data/spectrogram/spectrogram-images")  # Replace with your actual output directory path
output_dir_numpy.mkdir(parents=True, exist_ok=True)
output_dir_spectrogram.mkdir(parents=True, exist_ok=True)
audio_files = list(input_path.glob("*.wav"))  # Get all .wav files in the input directory


for files in audio_files:
    try:
        # Preserves the original sample rate and converts to mono
        # 1D-array representing amplitude values of the audio signal, and sr is the sample rate
        y, sr = librosa.load(files, sr=None, mono=True)

        # Compute the Short-Time Fourier Transform (STFT)
        # Finds out which frequencies exist in an audio signal and when they occur.
        D = librosa.stft(y)
        # Computes the magnitde/amplitude of each recorded frequency in the audio signal.
        magnitude = np.abs(D)

        # Convert magnitude to decibels (dB)
        # Human hearing is logarithmic, so we convert the amplitude to decibels for better visualization. (Loudness)
        #Example: Contains Decibels and height of row can be used to deduce frequency. The higher the row, the higher the frequency.
        # <----------------- TIME (Columns) ----------------->
        #                    Frame 0      Frame 1      Frame 2      ...   Frame 430
        #               ┌──────────────────────────────────────────────────────────┐
        # Row 1024 (High)│ -60.2 dB     -55.1 dB     -58.4 dB       ...    -70.0 dB │
        # Row 1023       │ -50.4 dB     -48.2 dB     -52.0 dB       ...    -68.5 dB │
        #    ...         │    ...          ...          ...         ...       ...   │
        # Row 20         │ -12.1 dB      -4.2 dB      -6.8 dB       ...    -45.1 dB │
        # Row 0   (Low)  │ -75.0 dB     -72.3 dB     -74.1 dB       ...    -80.0 dB │
        #               └──────────────────────────────────────────────────────────┘
        S_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        full_save_path = output_dir_numpy / f"{files.stem}_spectrogram.npy"
        np.save(full_save_path, S_db)

    except Exception as e:
        print(f"Error processing {files}: {e}")

    # Plot and display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {files}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    # To save instead of showing, use: plt.savefig("spectrogram.png", dpi=300)
    # plt.show()
    plt.savefig(f"{output_dir_spectrogram}/{files.stem}_spectrogram.png", dpi=300)