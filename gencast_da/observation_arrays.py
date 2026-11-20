"""Convert physical observations into GenCast controller arrays."""

from __future__ import annotations

import numpy as np
import xarray

from gencast_da.obs_operator import (
    ObsConfig,
    build_obs_indices,
    build_obs_mask,
)


def make_obs_arrays(
    inputs: xarray.Dataset,
    obs_phys: xarray.DataArray,
    diffs_stddev_by_level: xarray.Dataset,
    obs_cfg: ObsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized observations, mask, and per-level noise."""

    variable = obs_cfg.var
    if variable not in inputs or variable not in diffs_stddev_by_level:
        raise KeyError(f"missing observation variable {variable!r}")
    if obs_phys.sizes.get("time") != 1:
        raise ValueError("physical observations must contain exactly one time")
    if obs_phys.sizes.get("batch") != inputs.sizes["batch"]:
        raise ValueError("observation batch size does not match inputs")
    observation_mask = build_obs_mask(inputs, obs_cfg)

    expected_levels = inputs.coords["level"].values
    expected_batches = inputs.coords["batch"].values
    if obs_cfg.mask_mode == "stride":
        latitude_indices, longitude_indices = build_obs_indices(inputs, obs_cfg)
        expected_latitudes = inputs.coords["lat"].values[latitude_indices]
        expected_longitudes = inputs.coords["lon"].values[longitude_indices]
    else:
        latitude_indices = longitude_indices = None
        expected_latitudes = inputs.coords["lat"].values
        expected_longitudes = inputs.coords["lon"].values
    try:
        aligned_observation = obs_phys.sel(
            batch=expected_batches,
            level=expected_levels,
            lat=expected_latitudes,
            lon=expected_longitudes,
        )
        aligned_differences = diffs_stddev_by_level[variable].sel(level=expected_levels)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "observation/statistic coordinates do not match GenCast inputs"
        ) from error
    if aligned_differences.dims != ("level",):
        raise ValueError("diffs_stddev_by_level must have only a level dimension")

    last_input = (
        inputs[variable].isel(time=-1).transpose("batch", "level", "lat", "lon").values
    )
    physical_observation = aligned_observation.transpose(
        "batch", "time", "level", "lat", "lon"
    ).values
    differences = aligned_differences.values.astype(np.float32)
    if (
        not np.isfinite(physical_observation).all()
        or not np.isfinite(differences).all()
        or np.any(differences <= 0)
    ):
        raise ValueError("observations and normalization differences must be finite")
    normalized_noise = (obs_cfg.noise_std / differences).astype(np.float32)

    shape = (
        inputs.sizes["batch"],
        1,
        inputs.sizes["level"],
        inputs.sizes["lat"],
        inputs.sizes["lon"],
    )
    mask = np.zeros(shape, dtype=np.float32)
    if (
        aligned_observation.sizes["lat"] == inputs.sizes["lat"]
        and aligned_observation.sizes["lon"] == inputs.sizes["lon"]
    ):
        full_observation = physical_observation.astype(np.float32)
    else:
        assert latitude_indices is not None and longitude_indices is not None
        full_observation = np.zeros(shape, dtype=np.float32)
        full_observation[
            :, :, :, latitude_indices[:, None], longitude_indices[None, :]
        ] = physical_observation

    normalized_observation = (
        (full_observation - last_input[:, None, :, :, :])
        / differences[None, None, :, None, None]
    ).astype(np.float32)
    normalized_observation *= observation_mask[None, None, None, :, :]
    mask[...] = observation_mask[None, None, None, :, :]
    return normalized_observation, mask, normalized_noise
