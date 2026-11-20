# Experiments

## Fixed 3 / 18 / 30 matrix

The repository describes one closed experiment matrix:

- **3 dataset manifests:** Kolmogorov flow, Kuramoto–Sivashinsky (KS), and
  `era5_small`.
- **3 prior configurations:** one frozen autoregressive diffusion prior per
  dataset.
- **18 controller configurations:** each dataset has DS-2, DS-4, DS-8, MS-2,
  MS-4, and MS-8.
- **30 evaluation configurations:** Kolmogorov has six regimes at horizons 60
  and 180 (12); KS has six at horizons 140 and 640 (12); ERA5-small has six at
  horizon 60 (6).

DS uses average-pool downsampling and nearest-neighbor lifting at every
physical step. MS uses a regular-grid mask with observations every fourth
step. The suffix 2, 4, or 8 is the spatial factor.

## Core settings

Following the paper, the configs use:

- Prior U-Net width 64, multipliers `(1, 2, 4, 8)`, learned sinusoidal
  embedding dimension 128, `pred_v`, sigmoid noise schedule, and three DDIM
  sampling steps.
- Prior training with batch size 32, learning rate `3.2e-4`, one million
  steps, gradient accumulation 1, fp16, EMA decay `0.995`, and EMA update
  interval 10.
- Controller architecture `film_controlnet`, hidden width 768, and eight
  group-normalization groups.
- Preview length 16 for Kolmogorov and ERA5-small, and 54 for KS.
- Control scale `gamma=0.1`.
- Zero observation noise for the reported matrix.
- Eight trajectories for the reported ERA5 timing comparison. The repository
  also uses eight as its evaluation default for every configuration.

## Training and evaluation settings

The configs set controller training and evaluation as follows:

- Adam, learning rate `5e-5`, zero weight decay, constant schedule.
- Batch size 8, accumulation 1, fp16, max gradient norm 1.0, intermediate
  loss weight 1.0.
- Controller steps: 200,000 for Kolmogorov, 100,000 for KS, and 150,000 for
  ERA5-small.
- At every physical step, `beta=0.01` weights full-state RMSE supervision on
  the final and intermediate Tweedie estimates.
- Seeds: training 0, data order 1, diffusion 2, observation 3.
- Lower-index-origin alignment and zero mask phase.

## Running one cell

```bash
cada train-prior \
  --config configs/priors/ks.yaml \
  --device cuda

cada train-controller \
  --config configs/controllers/ks__ms_8.yaml \
  --device cuda

cada evaluate \
  --config configs/evaluations/ks__ms_8__h640.yaml \
  --device cuda
```

## Evaluation-only ablation

TTO-DA reuses any of the 30 evaluation configs and does not add controller
configs or evaluation YAMLs:

```bash
cada evaluate \
  --config configs/evaluations/ks__ms_8__h640.yaml \
  --method tto-da \
  --tto-steps 50 \
  --tto-lr 0.01 \
  --device cuda
```

TTO-DA creates a free latent control for every physical transition and DDIM
step. Within each configured preview chunk, Adam minimizes the same final and
intermediate observation objective plus `beta`-weighted control energy, with
the config's `gamma`, `beta`, and diffusion seed. The default repository
settings are 50 iterations per observed chunk and learning rate `0.01`. Prior
parameters remain outside the optimizer and receive no gradients. TTO-DA
requires only the prior checkpoint; CADA still requires the controller
checkpoint.

Method name and method parameters are part of the resolved run identity, so
CADA and TTO-DA outputs cannot share a run path. Both methods write the same
artifact fields and core metrics; `controlled` denotes the evaluated CADA or
optimized TTO-DA trajectory according to artifact metadata.

The default checkpoint paths in downstream configs point to
`$CADA_ROOT/runs/<domain>/{prior,controller}/.../checkpoints/final.pt`.
Command-line checkpoint overrides are available when actual checkpoints use a
different location.

Full training runs require GPU nodes. Adding `--max-steps 2` to a training
command gives a quick smoke test.

The repository implements CADA and the TTO-DA ablation.

## GenCast experiment

The GenCast experiment is configured separately under `gencast_da/configs`
because it uses a frozen JAX GenCast Mini prior, native 1-degree ERA5 fields,
and a Haiku controller. Its controller, evaluation, and figure configs follow
the same dry-run and run-identity rules but do not enlarge the PyTorch
3/18/30 matrix above.
