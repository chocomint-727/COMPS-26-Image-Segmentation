
import numpy as np
import soundfile as sf


# Load audio files
def load_audio(filename):
    audio, sr = sf.read(filename) #Load audio file using soundfile

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1) #Convert channels to mono by averaging channels if mutiple channels are present

    return audio, sr


# Calculate Signal-to-Noise Ratio
def calculate_snr(clean, audio): 
    signal = np.sum(clean ** 2) # ccalculate the power of the clean signal 
    error = np.sum((clean - audio) ** 2) # calcualte the power of the error signal the differece 

    return 10 * np.log10(signal / (error + 1e-10)) 


# Calculate Scale-Invariant Signal-to-Distortion Ratio
def calculate_si_sdr(clean, audio):
    clean = clean - np.mean(clean)
    audio = audio - np.mean(audio)

    # Find the scaling factor that best matches the clean audio
    scale = np.dot(audio, clean) / (np.sum(clean ** 2) + 1e-10) # 1e-10 is added to avoid division by zero
    target = scale * clean
    error = audio - target

    return 10 * np.log10( 
        #ratio is converted into decibels using 10 * np.log10()
        (np.sum(target ** 2) + 1e-10) /
        (np.sum(error ** 2) + 1e-10)
    )


# Calculate percentage error reduction
def calculate_error_reduction(clean, noisy, denoised):
    before = np.sum((clean - noisy) ** 2)
    after = np.sum((clean - denoised) ** 2)
 
    return (before - after) / (before + 1e-10) * 100


# Load recordings
clean, sr1 = load_audio("clean.wav")
noisy, sr2 = load_audio("noisy.wav")
denoised, sr3 = load_audio("denoised.wav")

# Check audio compatibility
if sr1 != sr2 or sr1 != sr3:
    raise ValueError("Audio sample rates must match.")

if len(clean) != len(noisy) or len(clean) != len(denoised):
    raise ValueError("Audio lengths must match.")

# Calculate metrics
input_snr = calculate_snr(clean, noisy)
output_snr = calculate_snr(clean, denoised)

input_sdr = calculate_si_sdr(clean, noisy)
output_sdr = calculate_si_sdr(clean, denoised)

error_reduction = calculate_error_reduction(clean, noisy, denoised)

# Display results
print(f"Input SNR: {input_snr:.2f} dB")
print(f"Output SNR: {output_snr:.2f} dB")
print(f"SNR Improvement: {output_snr - input_snr:.2f} dB")

print(f"Input SI-SDR: {input_sdr:.2f} dB")
print(f"Output SI-SDR: {output_sdr:.2f} dB")
print(f"SI-SDR Improvement: {output_sdr - input_sdr:.2f} dB")

print(f"Error Reduction: {error_reduction:.2f}%")
