"""Generate synthetic vinyl crackle and tape hiss as mono 44.1 kHz wav files."""
import os
import numpy as np
import soundfile as sf

SR = 44100
SECONDS = 60
VARIANTS = 6 
OUT_DIR = "raw/noise/analog"

os.makedirs(OUT_DIR, exist_ok=True)
n = SR * SECONDS
t = np.arange(n) / SR

for i in range(VARIANTS):
    rng = np.random.default_rng(i)

    # vinyl crackle
    crackle = np.zeros(n)
    clicks_per_second = rng.uniform(5, 40)
    n_clicks = int(clicks_per_second * SECONDS)
    for pos in rng.integers(0, n - 500, size=n_clicks):
        length = int(rng.integers(20, 400))
        amplitude = rng.exponential(0.3)
        decay = np.exp(-np.arange(length) / (length / 4))
        crackle[pos:pos + length] += amplitude * rng.standard_normal(length) * decay
    surface = 0.01 * np.diff(rng.standard_normal(n + 1))
    crackle = crackle + surface
    crackle = 0.9 * crackle / np.max(np.abs(crackle))
    sf.write(f"{OUT_DIR}/vinyl_crackle_{i:02d}.wav", crackle, SR, subtype="PCM_16")

    # tape hiss
    white = rng.standard_normal(n + 1)
    hiss = np.diff(white)
    drift = 1 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
    hiss = hiss * drift
    hiss = 0.9 * hiss / np.max(np.abs(hiss))
    sf.write(f"{OUT_DIR}/tape_hiss_{i:02d}.wav", hiss, SR, subtype="PCM_16")

print("wrote", 2 * VARIANTS, "files to", OUT_DIR)