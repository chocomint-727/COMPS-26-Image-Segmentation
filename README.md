# COMPS-26-Image-Segmentation
## Denoising Spectrograms: Segmenting Noise Artifacts from Song
Team: Omar, Christian, Anthony, and Jeremiah.

We remove synthetic analog noise (vinyl crackle and tape hiss) from music using image segmentation techniques on spectrograms. We made our very own custom PyTorch U-Net that predicts the clean and noise magnitude spectrograms directly. We compare it with a Spleeter we trained on the same data and score both with spectrogram MSE, SNR and SI-SDR on the test set.

## File structure

```
configs/             Spleeter config (clean_noise.json) and its train/val CSVs
data/                audio clips (raw/{train,val,test}/<id>/{clean,mixture,noise}.wav) and spectrograms
src/unet/            U-Net training script and notebooks, checkpoints/full/*.pt
src/spleeter/        trained Spleeter model (clean_noise/)
utils/               dataset, spectrogram, evaluation and figure scripts
  eval_figs.py           figures from the evaluation CSVs
  eval_utils.py          shared metrics (MSE, SNR, SI-SDR) and helpers
  evaluate_unet.py       U-Net test-set evaluation
  evaluate_spleeter.py   Spleeter test-set evaluation
  make_analog_noise.py   generate crackle / hiss noise files
  make_dataset.py        mix clean songs with noise at 0-20 dB SNR, split 80/10/10
  make_spleeter_csv.py   write Spleeter's train/val CSVs
  unet_model.py          U-Net architecture
  wav_to_spectrogram.py  wav to dB magnitude spectrogram .npy files
evaluation_results/  metric CSVs, Spleeter stems and figures
site/                project website
Dockerfile.unet      PyTorch image (U-Net)
Dockerfile.spleeter  TensorFlow + Spleeter image
```

## Building

There are two Docker images, one for the U-Net and one for Spleeter:

```
docker build -f Dockerfile.unet -t comps-unet .
docker build -f Dockerfile.spleeter -t comps-spleeter .
```

## Running

Start a container with the repo mounted (`run.sh` does the same for an image tagged `comps-project`, provided for your convenience):

```
docker run --gpus all --ipc=host -it -v "$(pwd)":/ws comps-unet bash
docker run --gpus all --ipc=host -it -v "$(pwd)":/ws comps-spleeter bash
```

## Data

1. Put the clean songs in `data/audio/raw/clean/`.
2. Generate the noise by running make_analog_noise.py. This writes the noise files to `data/audio/raw/noise/analog/`.
3. Build the mixtures by running `make_dataset.py --noise-type analog`. This writes noise into `data/audio/analog/`. The other scripts read these clips from `data/raw/{train,val,test}/`.
4. Make the spectrograms by running wav_to_spectrogram.py. 

## Training

**U-Net** (`comps-unet`):

```
cd src/unet && python U_net_full.py
```

This trains for 50 epochs and saves the model to `src/unet/checkpoints/full/spectrogram_<loss>.pt`.

**Spleeter** (`comps-spleeter`):

```
python utils/make_spleeter_csv.py
spleeter train -p configs/clean_noise.json -d data/raw
```

Spleeter saves the model to `src/spleeter/clean_noise/`  (can be changed in config).

## Inference

**Spleeter:** separate any song into `clean` and `noise` stems:

```
spleeter separate -p /ws/configs/clean_noise.json -o /ws/outputs /path/audio.wav
```

**U-Net:** `evaluate_unet.py` runs the model on every test clip but we did not make a standalone inference script. Instead, the model definition can be loaded from `utils/unet_model.py` and checkpoint loaded from `/src/unet/checkpoints/full` for future use.

## Evaluation

```
python utils/evaluate_unet.py       # run on comps-unet
python utils/evaluate_spleeter.py   # run on comps-spleeter
python utils/eval_figs.py
```

- **`evaluate_unet.py`:** writes `evaluation_results/spectrogram_metrics*.csv`. `GRIFFIN_LIM_ITERS` sets the phase reconstruction. Use `0` to keep the mixture's phase and `32` to run Griffin-Lim.
- **`evaluate_spleeter.py`:** runs Spleeter on any test clip that has no stems yet then writes `evaluation_results/spleeter/spleeter_metrics.csv`.
- **`eval_figs.py`:** saves the figures to `evaluation_results/figures/`.

## AI Disclosure

We made use of AI tools in the completion of this project. However, AI was not used to replace any of the learning needed in order to produce results, as this would have violated our stated goal of prioritizing learning. It was instead used to debug code in the case of our U-Net implementation and to produce code that members of the group could have produced with minimal cognitive effort, such as the noise generation script. When any group member's knowledge of topics was insufficient, we used AI to research, but primarily to find relevant resources which we read and understood ourselves.