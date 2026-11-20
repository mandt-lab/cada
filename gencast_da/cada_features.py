"""Helpers to build the per-step training/eval features for CADA-on-GenCast.

For each `target_time_idx` in the example trajectory we need:

  - `inputs, targets_template, forcings`  — built from the example's own
    frames `[target-2, target-1]`.
  - `obs_full_jax, obs_mask_jax`           — normalized-residual obs arrays
    consumed by the CADA wrapper's controller (`mask*y + (1-mask)*Tweedie`
    is built inside the wrapper, but `mask*y` part starts here).
  - `obs_phys_full_jax`                    — physical-space (K) obs zero-
    filled at unobserved cells; needed by the obs MSE loss term.
  - `obs_mask_phys_jax`                    — same shape and 1.0 at
    observed cells; used by both physical and normalized loss weights.
  - `truth_T_jax`                          — (B, 1, L, lat, lon) physical
    truth temperature at the target step, for the full-field MSE term.
  - `lat_w_jax`                            — `cos(lat)` weights normalised
    to unit mean.

The default observation regime uses `temperature` on a 4° grid at all 13
levels with sigma_y=0.1 K. Observation RNG is seeded deterministically per
(epoch, target_idx).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import xarray

from gencast_da.obs_operator import (
    ObsConfig,
    build_obs_indices,
    build_obs_mask,
)
from gencast_da.obs_operator import (
    predicted as obs_predicted,
)
from gencast_da.observation_arrays import make_obs_arrays
from gencast_da.rollout_utils import build_step_args

if TYPE_CHECKING:
    from gencast_da.gencast_runner import GenCastBundle


@dataclass
class CADAStepArrays:
    """All JAX arrays needed by one CADA training/eval step.

    Shapes for the default observation regime:
        obs_full_jax, obs_mask_jax: (B=1, 1, L=13, lat=181, lon=360) float32
            — normalized residual space.
        obs_phys_full_jax:          same shape, physical K, zero at unobserved.
        truth_T_jax:                (B, 1, L, lat, lon) float32, physical K.
        lat_w_jax:                  (lat,) float32, sum = lat (mean = 1).
    """

    inputs: xarray.Dataset
    targets_template: xarray.Dataset
    forcings: xarray.Dataset
    obs_full_jax: jnp.ndarray
    obs_mask_jax: jnp.ndarray
    obs_phys_full_jax: jnp.ndarray
    truth_T_jax: jnp.ndarray
    lat_w_jax: jnp.ndarray
    target_idx: int
    observation_mask: np.ndarray


def _build_step_args_from_example(
    example: xarray.Dataset,
    target_time_idx: int,
    task_config,
):
    """Build (inputs, targets_template, forcings) from `example`'s own frames.

    Uses `[target-2, target-1, target]` slice — the first two as inputs,
    the third as the target template. No particle memory (training data
    starts from ground truth at every step).
    """
    from graphcast import data_utils

    slice_indices = [target_time_idx - 2, target_time_idx - 1, target_time_idx]
    base = example.isel(time=slice_indices)
    inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
        base,
        target_lead_times=slice("12h", "12h"),
        **dataclasses.asdict(task_config),
    )
    return inputs, targets_template, forcings


def _build_step_data_from_args(
    bundle: GenCastBundle,
    example: xarray.Dataset,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    target_time_idx: int,
    obs_cfg: ObsConfig,
    rng_np: np.random.Generator,
) -> CADAStepArrays:
    """Common tail shared by `build_step` and `build_step_with_memory`.

    Pulls truth from `example` at `target_time_idx`, draws one obs
    realisation, and packages everything (inputs/targets/forcings, obs
    arrays, truth, lat weights) into a `CADAStepArrays`. `inputs` is the
    only thing that differs between train (truth-from-example) and
    rollout (rolled-forward memory) — see callers.
    """
    truth_slice = example.isel(time=[target_time_idx])
    var = obs_cfg.var

    truth_T_np = (
        truth_slice[var].transpose("batch", "time", "level", "lat", "lon").values
    ).astype(np.float32)

    truth_at_obs = obs_predicted(truth_slice, obs_cfg)
    obs_phys = truth_at_obs + rng_np.normal(
        0.0, obs_cfg.noise_std, size=truth_at_obs.shape
    ).astype(truth_at_obs.dtype)

    obs_full_np, obs_mask_np, _ = make_obs_arrays(
        inputs, obs_phys, bundle.diffs_stddev_by_level, obs_cfg
    )

    n_batch = inputs.sizes["batch"]
    n_lat = inputs.sizes["lat"]
    n_lon = inputs.sizes["lon"]
    n_level = inputs.sizes["level"]

    obs_phys_full_np = np.zeros((n_batch, 1, n_level, n_lat, n_lon), dtype=np.float32)
    obs_phys_arr = obs_phys.transpose("batch", "time", "level", "lat", "lon").values
    if obs_phys.sizes["lat"] == n_lat and obs_phys.sizes["lon"] == n_lon:
        obs_phys_full_np = obs_phys_arr.astype(np.float32)
    else:
        lat_idx, lon_idx = build_obs_indices(inputs, obs_cfg)
        obs_phys_full_np[:, :, :, lat_idx[:, None], lon_idx[None, :]] = obs_phys_arr

    obs_mask_2d = build_obs_mask(inputs, obs_cfg)
    obs_phys_full_np *= obs_mask_2d[None, None, None, :, :]

    lat_deg = inputs.coords["lat"].values
    lat_w_np = np.cos(np.deg2rad(lat_deg)).astype(np.float32)
    lat_w_np = lat_w_np / lat_w_np.mean()

    return CADAStepArrays(
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        obs_full_jax=jnp.asarray(obs_full_np),
        obs_mask_jax=jnp.asarray(obs_mask_np),
        obs_phys_full_jax=jnp.asarray(obs_phys_full_np),
        truth_T_jax=jnp.asarray(truth_T_np),
        lat_w_jax=jnp.asarray(lat_w_np),
        target_idx=target_time_idx,
        observation_mask=obs_mask_2d.copy(),
    )


def build_step(
    bundle: GenCastBundle,
    example: xarray.Dataset,
    target_time_idx: int,
    obs_cfg: ObsConfig,
    rng_np: np.random.Generator,
) -> CADAStepArrays:
    """Build all arrays needed for one CADA training/eval step.

    `rng_np` is consumed deterministically for the obs noise draw, so two
    calls with the same `np.random.default_rng(seed)` yield identical
    obs arrays. Pass a freshly-seeded generator if you want to fix
    obs across runs.
    """
    inputs, targets_template, forcings = _build_step_args_from_example(
        example, target_time_idx, bundle.task_config
    )
    return _build_step_data_from_args(
        bundle,
        example,
        inputs,
        targets_template,
        forcings,
        target_time_idx,
        obs_cfg,
        rng_np,
    )


def build_step_with_memory(
    bundle: GenCastBundle,
    example: xarray.Dataset,
    memory: xarray.Dataset,
    target_time_idx: int,
    obs_cfg: ObsConfig,
    rng_np: np.random.Generator,
) -> CADAStepArrays:
    """Like `build_step` but uses a 2-frame `memory` window for `inputs`.

    Same per-step obs/truth/last_input scaffolding as `build_step`, but
    the predictor's input frames come from `memory` instead of from the
    example trajectory. Use this for K_phys > 1 rollout training where
    physical step k > 0 must condition on the controller's own (detached)
    predictions from earlier steps — exactly the same memory contract
    used in `run_cada_gencast.py`.
    """
    inputs, targets_template, forcings = build_step_args(
        example, memory, target_time_idx, bundle.task_config
    )
    return _build_step_data_from_args(
        bundle,
        example,
        inputs,
        targets_template,
        forcings,
        target_time_idx,
        obs_cfg,
        rng_np,
    )


def cos_lat_full_mse(
    pred_T: jnp.ndarray,
    truth_T: jnp.ndarray,
    lat_w: jnp.ndarray,
) -> jnp.ndarray:
    """Cos(lat)-weighted full-field MSE, average over levels.

    pred_T, truth_T: (B, 1, L, lat, lon) physical K.
    lat_w:           (lat,) normalised cos-lat weights (mean = 1).
    """
    err = (pred_T - truth_T) ** 2
    w = lat_w[None, None, None, :, None]
    return jnp.mean(err * w)


def masked_obs_mse(
    pred_T: jnp.ndarray,
    obs_phys_full: jnp.ndarray,
    obs_mask: jnp.ndarray,
) -> jnp.ndarray:
    """MSE only at observed cells, in physical units.

    pred_T, obs_phys_full, obs_mask: (B, 1, L, lat, lon).
    """
    diff = jnp.where(obs_mask > 0, pred_T - obs_phys_full, 0.0)
    n_obs = jnp.maximum(jnp.sum(obs_mask), 1.0)
    return jnp.sum(diff**2) / n_obs
