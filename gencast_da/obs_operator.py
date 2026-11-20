"""Single-variable observation operators for CADA-on-GenCast."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray


@dataclass
class ObsConfig:
    """Temperature observation geometry and noise."""

    var: str = "temperature"
    mask_mode: str = "stride"
    lat_stride_deg: int = 4
    lon_stride_deg: int = 4
    noise_std: float = 0.1
    seed: int = 0
    random_mask_seed: int = 0
    random_mask_fraction: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.var, str) or not self.var:
            raise ValueError("var must be a non-empty string")
        if self.mask_mode not in {"stride", "random"}:
            raise ValueError("mask_mode must be 'stride' or 'random'")
        for name, value in (
            ("lat_stride_deg", self.lat_stride_deg),
            ("lon_stride_deg", self.lon_stride_deg),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.noise_std)) or self.noise_std <= 0:
            raise ValueError("noise_std must be finite and positive")
        for name, value in (
            ("seed", self.seed),
            ("random_mask_seed", self.random_mask_seed),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 2**63
            ):
                raise ValueError(f"{name} must lie in [0, 2**63)")
        if self.random_mask_fraction is not None and (
            not math.isfinite(float(self.random_mask_fraction))
            or not 0 < self.random_mask_fraction <= 1
        ):
            raise ValueError("random_mask_fraction must lie in (0, 1]")


def _grid_step_deg(coordinate: np.ndarray) -> float:
    if coordinate.size < 2:
        raise ValueError("at least two grid points are required")
    differences = np.diff(coordinate.astype(np.float64))
    if (
        not np.isfinite(differences).all()
        or not np.all(differences > 0)
        or not np.allclose(differences, differences[0])
    ):
        raise ValueError("grid coordinates must be finite, ascending, and uniform")
    return float(differences[0])


def build_obs_indices(
    state: xarray.Dataset,
    config: ObsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude/longitude indices for a regular observation grid."""

    if config.mask_mode != "stride":
        raise ValueError(
            "build_obs_indices requires mask_mode='stride'; "
            "use build_obs_mask for random masks"
        )
    latitude = state.coords["lat"].values
    longitude = state.coords["lon"].values
    latitude_step = _grid_step_deg(latitude)
    longitude_step = _grid_step_deg(longitude)
    latitude_ratio = config.lat_stride_deg / latitude_step
    longitude_ratio = config.lon_stride_deg / longitude_step
    if not np.isclose(latitude_ratio, round(latitude_ratio)):
        raise ValueError("latitude stride must be a multiple of the grid step")
    if not np.isclose(longitude_ratio, round(longitude_ratio)):
        raise ValueError("longitude stride must be a multiple of the grid step")
    latitude_factor = round(latitude_ratio)
    longitude_factor = round(longitude_ratio)
    return (
        np.arange(0, len(latitude), latitude_factor),
        np.arange(0, len(longitude), longitude_factor),
    )


def _stride_obs_count(state: xarray.Dataset, config: ObsConfig) -> int:
    stride_config = ObsConfig(
        var=config.var,
        mask_mode="stride",
        lat_stride_deg=config.lat_stride_deg,
        lon_stride_deg=config.lon_stride_deg,
        noise_std=config.noise_std,
        seed=config.seed,
        random_mask_seed=config.random_mask_seed,
    )
    latitude_indices, longitude_indices = build_obs_indices(
        state,
        stride_config,
    )
    return len(latitude_indices) * len(longitude_indices)


def build_obs_mask(state: xarray.Dataset, config: ObsConfig) -> np.ndarray:
    """Return a Boolean ``(lat, lon)`` observation mask."""

    shape = (state.sizes["lat"], state.sizes["lon"])
    mask = np.zeros(shape, dtype=bool)
    if config.mask_mode == "stride":
        latitude_indices, longitude_indices = build_obs_indices(state, config)
        mask[latitude_indices[:, None], longitude_indices[None, :]] = True
        return mask
    if config.mask_mode != "random":
        raise ValueError("mask_mode must be 'stride' or 'random'")

    total = shape[0] * shape[1]
    count = (
        _stride_obs_count(state, config)
        if config.random_mask_fraction is None
        else int(round(config.random_mask_fraction * total))
    )
    count = max(1, min(count, total))
    generator = np.random.default_rng(config.random_mask_seed)
    mask.reshape(-1)[generator.choice(total, size=count, replace=False)] = True
    return mask


def apply(
    state: xarray.Dataset,
    config: ObsConfig,
    rng: np.random.Generator | None = None,
) -> xarray.DataArray:
    """Generate noisy observations from a state."""

    if config.var not in state.data_vars:
        raise KeyError(f"variable {config.var!r} is absent from the state")
    values = state[config.var]
    if config.mask_mode == "stride":
        latitude_indices, longitude_indices = build_obs_indices(state, config)
        clean = values.isel(lat=latitude_indices, lon=longitude_indices)
    else:
        clean = values
    generator = rng or np.random.default_rng(config.seed)
    noise = generator.normal(0.0, config.noise_std, size=clean.shape).astype(
        clean.dtype
    )
    observation = clean + xarray.zeros_like(clean) + noise
    observation.name = config.var
    return observation


def predicted(
    state: xarray.Dataset,
    config: ObsConfig,
) -> xarray.DataArray:
    """Apply only the linear observation operator."""

    if config.mask_mode == "stride":
        latitude_indices, longitude_indices = build_obs_indices(state, config)
        return state[config.var].isel(
            lat=latitude_indices,
            lon=longitude_indices,
        )
    return state[config.var]


def log_likelihood(
    state: xarray.Dataset,
    observation: xarray.DataArray,
    config: ObsConfig,
) -> float:
    """Return Gaussian observation log likelihood up to a constant."""

    difference = (
        observation.values - predicted(state, config).values
    ) / config.noise_std
    if config.mask_mode == "random":
        mask = build_obs_mask(state, config)
        difference *= mask[None, None, None, :, :]
    return -0.5 * float(np.sum(difference * difference))


def summary(state: xarray.Dataset, config: ObsConfig) -> dict:
    """Return JSON-safe observation counts."""

    mask = build_obs_mask(state, config)
    levels = state.sizes["level"]
    return {
        "var": config.var,
        "mask_mode": config.mask_mode,
        "n_spatial_sites": int(mask.sum()),
        "n_levels": levels,
        "n_obs_per_step": int(mask.sum()) * levels,
        "noise_std": config.noise_std,
    }
