# Creativity in Convolutional Diffusion Models

A small, fully transparent playground that combines two ideas:

1. **Creativity from locality and equivariance** (Kamb & Ganguli, 2024).
   A diffusion model whose denoiser is *local* (it only sees a small patch around each pixel) and
   *translation-equivariant* cannot simply memorise its training set. Instead it produces
   *locally consistent patch mosaics*: new images in which every patch looks like some patch of
   the training data, but the image as a whole does not.
2. **Feynman-Kac (FK) steering** (Singhal et al., 2025).
   An inference-time method that runs a population of particles through the reverse
   process and resamples them according to a reward, steering samples toward a target without
   retraining.

The setup is deliberately minimal. A one-layer periodic CNN is trained on just **two 16×16 binary
images, all-black and all-white**. A model that memorised the data would only ever output these two
images. The local model instead generates new black-and-white mosaics. FK steering is then used to
push those mosaics toward chosen target patterns (half-planes, quadrants, a checkerboard, ...).

## How it works

| Component | File | Details |
|---|---|---|
| Data | `ccdm/dataset.py` | Each item is the all-black (`0`) or all-white (`1`) 16×16 image, chosen uniformly at random. |
| Denoiser | `ccdm/model.py` | `PeriodicConv2D(1→C, k×k)` → `+` sinusoidal time embedding → BatchNorm → `tanh` → per-pixel linear `C→1` → `tanh`. Circular padding makes the network exactly equivariant to translations on the torus, and its receptive field is `k×k`. |
| Diffusion | `ccdm/diffusion.py` | DDPM with the cosine β-schedule (Nichol & Dhariwal, 2021). The network predicts the clean image **x₀** and is trained with MSE. Sampling uses the posterior q(x_{t-1} \| x_t, x̂₀). |
| FK steering | `ccdm/fk_steering.py` | Particle resampling with `diff`, `max` or `add` potentials, optional ESS-based adaptive resampling. Adapted from the authors' reference implementation. |
| Rewards | `ccdm/rewards.py` | Target patterns and reward functions evaluated on the (binarised) x̂₀ predictions. |
| Metric | `ccdm/metrics.py` | **Local consistency ratio**: the fraction of pixels that equal the majority colour of their k×k periodic neighbourhood. |

## Repository layout

```
ccdm/
  model.py         periodic convolution + denoiser
  diffusion.py     cosine schedule, forward noising, reverse step
  dataset.py       all-black / all-white dataset
  fk_steering.py   Feynman-Kac steering
  rewards.py       target images and reward functions
  metrics.py       local consistency ratio
  plotting.py      figure helpers
  utils.py         seeding, device, run naming
train.py           train a model
generate.py        sample (plain DDPM or FK-steered)
tests/             unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train

```bash
python train.py
```

Defaults: 100 diffusion steps, 3×3 kernel, 4 channels, 3000 epochs, batch size 100, Adam with lr 1e-4.
This writes `checkpoints/step100_perioconv3_channel4.pth` and a loss curve in
`outputs/training_plots/`. Run `python train.py -h` to see all options, for example
`--kernel-size 5 --channels 8`.

### 2. Sample (plain DDPM)

```bash
python generate.py
```

This runs `--num-runs` (10) independent runs of `--batch-size` (100) samples. For each run it
saves the denoising trajectory and the binarised final image of the first sample. At the end it
prints the local consistency ratio of the initial noise (binarised at 0.5) and of the generated
images (binarised at `--threshold`, 0.5 by default).

Pass the same `--n-steps / --kernel-size / --channels` that you trained with, or point
`--checkpoint` at a file directly.

### 3. Sample with FK steering

Pass a reward with `--reward` to turn on FK steering. The batch is then the particle population:

```bash
python generate.py --reward top-black
python generate.py --reward quadrant --lmbda 5 --potential max --adaptive-resampling
```

| `--reward` | Target | Reward |
|---|---|---|
| `top-white` | top half white, bottom half black | −(number of mismatched pixels) |
| `top-black` | top half black, bottom half white | −(number of mismatched pixels) |
| `quadrant` | top-right and bottom-left quadrants white | −(number of mismatched pixels) |
| `checkerboard` | alternating pixels | −(number of mismatched pixels) |
| `black` / `white` | uniform image | −(number of mismatched pixels) |
| `block` | exactly one fully white 3×3 block (periodic) | 100 if satisfied, else 0 |

FK options: `--lmbda` (steering strength, default 10), `--potential {diff,max,add}` (default `diff`),
`--adaptive-resampling`, and `--resample-frequency`. By default the population is resampled at
every step. The script also prints the mean final reward.

With FK steering, `--threshold` defaults to **0.25**. In the trained model, white pixels top out
around 0.76 and the pixels on the boundary of a white region land around 0.3–0.4. Binarising at
0.5 would therefore strip one pixel from every edge of a white region.

### Outputs

```
outputs/
  training_plots/loss_<run>.png
  <run>/                       # plain sampling
    denoising_sequences/sample_<i>.png
    samples/sample_<i>.png
  fk_<reward>_<run>/           # FK-steered sampling
    target_<reward>.png
    denoising_sequences/sample_<i>.png
    samples/sample_<i>.png
```

where `<run>` is `step{n_steps}_perioconv{kernel_size}_channel{channels}`.

### Tests

```bash
pytest
```

## Results

These numbers come from one model trained with the default settings (`python train.py`) and 300 FK
samples per target (3 runs × 100 particles, `diff` potential). They are a single run, not averages
over seeds.

| Sampling | Outcome |
|---|---|
| Plain | 99% of samples are new black-and-white mosaics (neither all-black nor all-white). Local consistency rises from 0.71 for the initial noise to about 0.985. |
| FK `top-black` / `top-white` (λ=10) | 100% / 97% of samples match the target exactly at threshold 0.25. At 0.5 the white half loses one pixel on each edge, so 0% match. |
| FK `quadrant` (λ=10) | The quadrant layout is right, but on average about 13 of 256 pixels differ from the target. |
| FK `checkerboard` (λ=10) | No improvement over plain sampling (about 126 of 256 pixels wrong). |
| FK `block` | The reward is sparse (0 or 100). Success varied from about 1% to 20% between runs at λ=30. |

The checkerboard failure is what the theory predicts. Every 3×3 patch of the training data is
uniform, so a local denoiser cannot produce the alternating patches a checkerboard needs. FK
steering can only reweight samples the model already produces.

## Notes

- Images live in `{0, 1}` (0 = black, 1 = white). Pure Gaussian noise binarised at 0.5 therefore
  serves as the baseline for the local consistency metric.
- `ccdm/rewards.py` also contains an experimental `projection_reward` (it matches the row and
  column occupancy of the image to a target). It is not exposed through the CLI.
- Checkpoints saved by earlier versions of this repo (which had an unused `time_emb` layer) still
  load through `--checkpoint`.

## References

- M. Kamb and S. Ganguli. *An analytic theory of creativity in convolutional diffusion models.*
  arXiv:2412.20292, 2024.
- R. Singhal, Z. Horvitz, R. Teehan, M. Ren, Z. Yu, K. McKeown, R. Ranganath.
  *A General Framework for Inference-time Scaling and Steering of Diffusion Models.*
  arXiv:2501.06848, 2025. Code: https://github.com/zacharyhorvitz/Fk-Diffusion-Steering
- A. Nichol and P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
