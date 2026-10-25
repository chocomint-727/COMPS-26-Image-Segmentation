
import numpy as np
import soundfile as sf
from pathlib import Path


clean_folder = Path("../data/spectrogram/clean")
noisy_folder = Path("../data/spectrogram/noisy")
denoised_folder = Path("../data/spectrogram/denoised")


def load_spectrogram(filename):
    # Load spectrogram from .npy file
    return np.load(filename)

#mean squared erroe
def calculate_mse(clean, audio):
    return np.mean((clean - audio) ** 2)

#c alculate error reduction percentage
def calculate_error_reduction(clean, noisy, denoised):
    before = calculate_mse(clean, noisy)
    after = calculate_mse(clean, denoised)
    return (before - after) / (before + 1e-10) *100

input_mse_results = [] 
output_mse_results = []
error_reduction_results = []


for clean_file in sorted(clean_folder.glob("*.npy")): # Iterate through all clean spectrogram files
    noisy_file = noisy_folder / clean_file.name
    denoised_file = denoised_folder / clean_file.name

    if not noisy_file.exists() or not denoised_file.exists(): # Check if corresponding noisy and denoised files exist
        print(f"Skipping {clean_file.name}, Missing files.")
        continue

    clean_spectrogram = load_spectrogram(clean_file) # Load clean spectrogram
    noisy_spectrogram = load_spectrogram(noisy_file) # Load noisy spectrogram
    denoised_spectrogram = load_spectrogram(denoised_file) # Load denoised spectrogram

    if clean_spectrogram.shape != noisy_spectrogram.shape or clean_spectrogram.shape != denoised_spectrogram.shape: 
        print(f"Skipping {clean_file.name}, Shape mismatch.")
        continue

    input_mse = calculate_mse(clean_spectrogram, noisy_spectrogram) # Calculate MSE between clean and noisy spectrogram
    output_mse = calculate_mse(clean_spectrogram, denoised_spectrogram) # Calculate MSE between clean and denoised spectrogram
    error_reduction = calculate_error_reduction(clean_spectrogram, noisy_spectrogram, denoised_spectrogram) # Calculate error reduction percentage

    input_mse_results.append(input_mse)  
    output_mse_results.append(output_mse)
    error_reduction_results.append(error_reduction)

    print(f"{clean_file.name}: Error Reduction: {error_reduction:.2f}% errror reduction")

if len(error_reduction_results)>0: 
    print("Dataset Evaluation Results:")

    print(f"Song evaluated: {len(error_reduction_results)}") #Prints out total number of songs

    print(f"Average Input MSE: {np.mean(input_mse_results):.4f}")
    print(f"Average Output MSE: {np.mean(output_mse_results):.4f}") #Note .4f just means display the number with 4 decimal places

    print(f"Average Error Reduction: {np.mean(error_reduction_results):.2f}%")

else:
    print("No valid files were evaluated.")
