# Architecture

CADA is packaged from `src/cada` and exposed through the `cada` console
command. The executable path is deliberately configuration-driven:

1. `cli.py` parses `data`, `train-prior`, `train-controller`, `evaluate`, and
   `figure`.
2. `config.py` validates one standalone YAML document and resolves
   `${CADA_ROOT}`.
3. `data.py` opens lazy HDF5 trajectory sources; `preprocessing.py` generates
   KS data, computes normalization statistics, and converts ERA5 netCDF input.
4. `models.py` builds the 1D or 2D U-Net and `diffusion.py` supplies the
   autoregressive diffusion transition.
5. `observations.py` represents spatial operators and arrival cadence.
6. `control.py` produces FiLM-conditioned denoising controls, while
   `assimilation.py` implements fixed, anchored preview windows.
7. `training.py` trains either the prior or controller. The prior is frozen
   during controller training.
8. `evaluation.py` runs controlled and uncontrolled trajectories with shared
   initial noise, then delegates scalar calculations to `metrics.py`.
9. `checkpoint.py`, `artifacts.py`, and `runtime.py` provide atomic outputs,
   checksums, resolved configuration, run IDs, seeds, and runtime provenance.
10. `figures.py` plots saved evaluation artifacts; it does not train or
    evaluate models.

## Controlled rollout

For each physical transition, the prior performs three DDIM denoising
sub-steps. The controller receives the current latent, previous forecast,
selected preview observation and mask, previous control, preview lag, local
frame index, and diffusion signal-to-noise information. It shifts the noisy
state before the frozen denoiser is applied.

At every physical step, controller training uses `beta`-weighted full-state
RMSE against the corresponding training state on the final and intermediate
Tweedie estimates. At observation times, observation-space MSE on those same
estimates is added to the loss. Dense supervision is averaged over physical
steps, while observation supervision is averaged over actual arrivals, so a
sparser cadence does not silently reduce its weight.

Training uses an exact finite preview window. Evaluation partitions longer
horizons into preview-sized chunks, passing the model-generated final state of
one chunk into the next without projecting it onto ground truth. Observations
beyond the active chunk boundary remain invisible until the next chunk.

## Boundaries

The maintained PyTorch package is `cada`, not the legacy
`denoising_diffusion_pytorch` import surface. Portions of the diffusion
implementation are adapted from the lucidrains project as recorded in
[`NOTICE`](../NOTICE), but the CADA-specific package and experiment system are
new project work.

`gencast_da` is a second, self-contained JAX backend. It has its own CLI,
configs, environment, checkpoint serializer, and model code. It mirrors the
same run directories, resolved configuration, metadata, checksums, optional
tracking, and dry-run behavior without importing the PyTorch package.

The filesystem is the backend boundary:

```text
prediff process ────────┐
                       ├── $CADA_ROOT/runs/...
gencast-da process ─────┘
```

Native Torch and JAX checkpoints are intentionally not interchangeable.
