"""Plot the stored U-Net and Spleeter evaluation results into evaluation_results/figures/."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import N_FFT, HOP, spectrogram_mses, crop, read_mono, fit_length

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation_results"
OUTPUT_DIR = RESULTS / "figures"
TEST_FOLDER = ROOT / "data/raw/test"
SPEC_FOLDER = ROOT / "data/spectrogram-magnitude/numpy-files"
STEMS_FOLDER = RESULTS / "spleeter/stems"
SPLEETER_MSE_CACHE = RESULTS / "spleeter/spleeter_spectrogram_mse.csv"

LEVELS = [0, 5, 10, 15, 20]
MONO_OFFSET_DB = 2.5

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
UNET = "U-Net + mixture phase"
UNET_GL = "U-Net + Griffin-Lim"
SPLEETER = "Spleeter"
COLOR = {UNET: "#2a78d6", UNET_GL: "#1baf7a", SPLEETER: "#eb6834"}
MARKER = {UNET: "o", UNET_GL: "^", SPLEETER: "s"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def read(path):
    """Read a results CSV, keeping sample ids as strings."""
    return pd.read_csv(path, dtype={"sample_id": str})


def load_unet():
    """Return the first mixture-phase U-Net run whose rebuild step reproduces the input SNR."""
    for name in ("spectrogram_metrics.csv", "spectrogram_metrics_istft.csv"):
        path = RESULTS / name
        if path.is_file():
            df = read(path)
            if (df["pipeline_only_snr"] - df["input_snr"]).abs().mean() < 1:
                return df
    raise SystemExit("No usable mixture-phase U-Net CSV in " + str(RESULTS))


def load_scores(unet):
    """Return per-song SNR and SI-SDR for every model, with each song's mixing level."""
    gl = read(RESULTS / "spectrogram_metrics_griffinlim32.csv")
    spleeter = read(RESULTS / "spleeter/spleeter_metrics.csv").rename(
        columns={"spleeter_snr": "output_snr", "spleeter_si_sdr": "output_si_sdr"})
    columns = ["sample_id", "input_snr", "input_si_sdr", "output_snr", "output_si_sdr"]
    data = pd.concat([df[columns].assign(model=model)
                      for model, df in [(UNET, unet), (UNET_GL, gl), (SPLEETER, spleeter)]], ignore_index=True)
    data["improvement"] = data["output_snr"] - data["input_snr"]
    data["level"] = np.clip(np.round((data["input_snr"] - MONO_OFFSET_DB) / 5) * 5, 0, 20).astype(int)
    return data


def spleeter_mse():
    """Return Spleeter's spectrogram MSE against the U-Net's targets, computed from its stems once and cached."""
    if SPLEETER_MSE_CACHE.is_file():
        return read(SPLEETER_MSE_CACHE)
    import librosa

    rows = []
    for song in sorted(p.name for p in STEMS_FOLDER.iterdir() if (p / "clean.wav").is_file()):
        clean_audio = read_mono(TEST_FOLDER / song / "clean.wav")
        peak = np.abs(librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP)).max()
        stem = fit_length(read_mono(STEMS_FOLDER / song / "clean.wav"), len(clean_audio))
        stem_db = librosa.amplitude_to_db(np.abs(librosa.stft(stem, n_fft=N_FFT, hop_length=HOP)), ref=peak, top_db=None)
        clean, stem_db = crop(np.load(SPEC_FOLDER / song / "clean.npy").astype(np.float64), np.maximum(stem_db, -80))
        mse, loud_mse, lowband_mse = spectrogram_mses(clean, stem_db)
        rows.append({"sample_id": song, "spleeter_mse": mse, "loud_spleeter_mse": loud_mse,
                     "lowband_spleeter_mse": lowband_mse})
    df = pd.DataFrame(rows)
    df.to_csv(SPLEETER_MSE_CACHE, index=False)
    return df


def bar_group(ax, values, names, colours, labels, digits=1):
    """Draw skinny grouped bars with a value label on each bar."""
    x = np.arange(len(labels))
    slot = 0.8 / len(names)
    for i, (name, colour, series) in enumerate(zip(names, colours, values)):
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * slot, series, slot * 0.62, color=colour,
                      label=name, linewidth=0, zorder=2)
        for bar, v in zip(bars, series):
            ax.annotate(f"{v:.{digits}f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))


def save(fig, name):
    """Save a figure to the output folder and close it."""
    fig.savefig(OUTPUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUTPUT_DIR / name)


def fig_average(data):
    """Plot mean SNR and SI-SDR of the noisy input and each model."""
    inputs = data[data["model"] == UNET]
    values = [[inputs["input_snr"].mean(), inputs["input_si_sdr"].mean()]]
    values += [[g["output_snr"].mean(), g["output_si_sdr"].mean()] for _, g in data.groupby("model", sort=False)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_group(ax, values, ["Noisy input"] + list(COLOR), [MUTED] + list(COLOR.values()), ["SNR", "SI-SDR"])
    ax.set_ylabel("dB")
    ax.set_title("Average signal quality")
    save(fig, "average_snr_si_sdr.png")


def fig_snr_by_level(data):
    """Plot mean and interquartile output SNR of each model per mixing level."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    inputs = data[data["model"] == UNET].groupby("level")["input_snr"].mean()
    ax.plot(inputs.index, inputs.values, color=MUTED, marker="D", markersize=6, linewidth=1.5,
            label="Noisy input (no change)")
    for model, colour in COLOR.items():
        group = data[data["model"] == model].groupby("level")["output_snr"]
        mean = group.mean()
        ax.fill_between(mean.index, group.quantile(0.25).values, group.quantile(0.75).values,
                        color=colour, alpha=0.12, linewidth=0)
        ax.plot(mean.index, mean.values, color=colour, marker=MARKER[model], markersize=7, linewidth=2, label=model)
    ax.set_xticks(LEVELS, [f"{lv} dB" for lv in LEVELS])
    ax.set_xlabel("Mixing SNR")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Output SNR by mixing level")
    ax.set_xlim(LEVELS[0] - 1.5, LEVELS[-1] + 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.annotate("lines: mean\nbands: middle 50% of songs", (1.02, 0), xycoords="axes fraction",
                fontsize=8, color=MUTED, va="bottom")
    save(fig, "snr_by_input_level.png")


def fig_scatter(data):
    """Plot every song's input SNR against its output SNR with a no-change line."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lo = min(data["input_snr"].min(), data["output_snr"].min()) - 2
    hi = max(data["input_snr"].max(), data["output_snr"].max()) + 2
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, label="No change")
    for model, colour in COLOR.items():
        d = data[data["model"] == model]
        ax.scatter(d["input_snr"], d["output_snr"], s=34, color=colour, marker=MARKER[model],
                   edgecolors=SURFACE, linewidths=1.2, alpha=0.9, label=model)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Per-song SNR")
    ax.legend(loc="upper left")
    save(fig, "snr_per_song_scatter.png")


def fig_improvement_distribution(data):
    """Plot the distribution of per-song SNR improvement for each model."""
    values = [data.loc[data["model"] == m, "improvement"].values for m in COLOR]
    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 1.1 * len(COLOR)))
    ax.axvline(0, color=MUTED, linewidth=1.2)
    bodies = ax.violinplot(values, vert=False, showextrema=False, widths=0.8)["bodies"]
    rng = np.random.default_rng(0)
    for i, (model, v, body) in enumerate(zip(COLOR, values, bodies), start=1):
        body.set_facecolor(COLOR[model])
        body.set_alpha(0.25)
        body.set_edgecolor("none")
        ax.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), s=10, color=COLOR[model], marker=MARKER[model],
                   alpha=0.7, linewidths=0)
        median = np.median(v)
        ax.plot([median, median], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(-0.03, i - 0.1, model, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(-0.03, i - 0.06, f"median {median:+.1f} dB\nworse on {(v < 0).sum()} of {len(v)} songs".replace("-", "−"),
                transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=8.5, color=INK_2, linespacing=1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0, 0.5, "worse than input ←  ", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.text(0, 0.5, "  → better than input", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(len(COLOR) + 0.5, 0.25)
    ax.set_xlabel("SNR improvement (dB)")
    ax.set_title("Per-song SNR improvement")
    ax.grid(axis="y", visible=False)
    save(fig, "snr_improvement_distribution.png")


def fig_mse(table, groups, title, filename):
    """Plot mean spectrogram MSE of the noisy input, the U-Net and Spleeter for each group of bins."""
    values = [[table[prefix + column].mean() for _, prefix in groups]
              for column in ("input_mse", "clean_mse", "spleeter_mse")]
    fig, ax = plt.subplots(figsize=(2.6 + 2.2 * len(groups), 4))
    bar_group(ax, values, ["Noisy input", "U-Net", SPLEETER], [MUTED, COLOR[UNET], COLOR[SPLEETER]],
              [label for label, _ in groups], digits=0)
    ax.set_ylabel("MSE (dB²)")
    ax.set_title(title)
    save(fig, filename)


def main():
    """Make every evaluation figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unet = load_unet()
    data = load_scores(unet)
    fig_average(data)
    fig_snr_by_level(data)
    fig_scatter(data)
    fig_improvement_distribution(data)
    table = unet.merge(spleeter_mse(), on="sample_id", suffixes=("", "_spleeter"))
    fig_mse(table, [("All bins", ""), ("Loudest 20%", "loud_")], "Spectrogram error", "spectrogram_mse.png")
    fig_mse(table, [("0–11 kHz", "lowband_")], "Spectrogram error, 0–11 kHz", "spectrogram_mse_below_11khz.png")


if __name__ == "__main__":
    main()
