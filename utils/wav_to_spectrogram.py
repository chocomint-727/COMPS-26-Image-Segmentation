import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

COMPLEX_FLAG = False
# file_path = "wav-files/lucki.wav"

input_path = Path("../data/raw/test")  
if COMPLEX_FLAG:
    output_dir_numpy_complex = Path("../data/spectrogram-complex-log/numpy-files")  
# output_dir_spectrogram_complex = Path("../data/spectrogram-complex-log/spectrogram-images")
output_dir_numpy_magnitude = Path("../data/spectrogram-magnitude-test-set/numpy-files")


# rglob searches recursively across all subfolders
audio_files = list(input_path.rglob("*.wav"))

for files in audio_files:
    try:
        relative_dir = files.relative_to(input_path).parent

        if COMPLEX_FLAG:
            target_numpy_dir = output_dir_numpy_complex / relative_dir
        # target_spec_dir = output_dir_spectrogram_complex / relative_dir
        target_numpy_magnitude_dir = output_dir_numpy_magnitude / relative_dir

        if COMPLEX_FLAG:
            target_numpy_dir.mkdir(parents=True, exist_ok=True)
        # target_spec_dir.mkdir(parents=True, exist_ok=True)
        target_numpy_magnitude_dir.mkdir(parents=True, exist_ok=True)

        # load audio files in mono
        y, sr = librosa.load(files, sr=None, mono=True)
        
        # stft!
        D = librosa.stft(y)        
        if COMPLEX_FLAG:
            real_part = np.real(D)  # shape: (1025, T)
            imag_part = np.imag(D)  # shape: (1025, T)
            # stack into a 2-channel 3D array of shape (2, 1025, T)
            two_channel_stft = np.stack([real_part, imag_part], axis=0)
            np.save(target_numpy_dir / f"{files.stem}_complex_stft.npy", two_channel_stft)

            print(f"Shape of D: {two_channel_stft.shape}")
            print(f"Data type: {two_channel_stft.dtype}")

        magnitude = np.abs(D)
        # magnitude to db
        S_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        np.save(target_numpy_magnitude_dir / f"{files.stem}.npy", S_db)

    except Exception as e:
        print(f"Error processing {files}: {e}")

    