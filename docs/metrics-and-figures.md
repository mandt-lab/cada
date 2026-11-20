# Metrics and figures

## Evaluation metrics

Metrics exclude the supplied initial state and cover forecast states only.
Controlled and uncontrolled rollouts use the same initial diffusion noise.

- **RMSE:** first computed per trajectory and forecast time over the configured
  event dimensions, then averaged. KS uses the state; Kolmogorov uses spectral
  vorticity; ERA5-small uses all three physical channels.
- **High correlation time (HCT):** the last one-based forecast step whose
  per-trajectory Pearson correlation is at least `0.9`. The JSON records both
  forecast-step and physical-time forms. A trajectory that never meets the
  threshold is null in serialized output.
- **KS total variation:** periodic discrete total variation in physical space;
  reporting uses the mean absolute error from truth.
- **Kolmogorov dissipation:** physical-space kinetic-energy dissipation with
  viscosity `1e-3` on a `2π × 2π` domain; reporting uses mean absolute error.
- **ERA5 vorticity:** the finite-difference curl of the two wind channels on
  the 60° by 30° grid, with derivatives taken per degree.

Evaluation RMSE is in `normalized_model_space`; physical diagnostics are in
`physical_space`. The exact variable, domain, and aggregation string are
embedded in each evaluation YAML and repeated in the metrics JSON.

## Figures

`cada figure` plots one saved evaluation artifact as ground truth,
uncontrolled prior, and CADA panels. It does not plot external baselines.
Snapshot and spacetime configs require a strict `.npz` or `.pt` artifact
containing `truth`, `uncontrolled`, `controlled`, and metadata. The four
included configs cover MS-4 examples:

- `kolmogorov__ms_4__h180__vorticity.yaml`: Kolmogorov vorticity snapshots
  over the 180-step rollout.
- `ks__ms_4__h640__spacetime.yaml`: KS spacetime over the 640-step rollout.
- `era5_small__ms_4__h60__temperature.yaml`: ERA5-small temperature snapshots.
- `era5_small__ms_4__h60__vorticity.yaml`: ERA5-small vorticity snapshots.

The configs explicitly select trajectory indices, frames, color limits, and
layout choices.

Validate a config without requiring its input:

```bash
cada figure \
  --config configs/figures/ks__ms_4__h640__spacetime.yaml \
  --dry-run
```

Render after staging the artifact at the path printed by dry-run:

```bash
cada figure \
  --config configs/figures/ks__ms_4__h640__spacetime.yaml
```

Every rendered figure receives a deterministic `.provenance.json` sidecar
with resolved inputs and hashes.

## GenCast metrics and figures

GenCast evaluation records cosine-latitude-weighted full-field temperature
RMSE, clean-truth RMSE on observed and unobserved cells, and a separate
innovation RMSE against noisy observations for each 12-hour lead. Its native
NPZ artifact keeps full-grid truth, CADA prediction, noisy observations,
observation masks, coordinates, lead times, and metadata.

`gencast_da/configs/figure.yaml` renders CADA truth, prediction, and absolute
error only. It does not stage or render external baseline outputs.
