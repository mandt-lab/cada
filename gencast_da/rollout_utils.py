"""Shared autoregressive rollout helpers for CADA-on-GenCast."""

from __future__ import annotations

import dataclasses

import numpy as np
import xarray


def to_temperature_array(prediction: xarray.Dataset, variable: str) -> np.ndarray:
    """Extract a GenCast field as ``(time, lat, lon, level)`` NumPy data."""

    return (
        prediction[variable]
        .isel(batch=0)
        .transpose("time", "lat", "lon", "level")
        .values
    )


def cosine_latitude_weights(latitude_degrees: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitude_degrees))
    return weights / weights.mean()


def weighted_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    latitude_degrees: np.ndarray,
) -> float:
    """Cosine-latitude-weighted RMSE over latitude, longitude, and level."""

    error = (prediction - truth) ** 2
    weights = cosine_latitude_weights(latitude_degrees)
    return float(np.sqrt((error * weights[:, None, None]).mean()))


def partition_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    observed: bool,
) -> float | None:
    """RMSE on the observed or complementary spatial partition."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("partition RMSE inputs must share [time,lat,lon,level]")
    if mask.shape != prediction.shape[1:3]:
        raise ValueError("partition mask does not match latitude/longitude")
    selected = mask if observed else ~mask
    if not bool(selected.any()):
        return None
    difference = prediction[:, selected, :] - target[:, selected, :]
    return float(np.sqrt(np.mean(difference**2)))


def evolving_input_variables(task_config: object) -> list[str]:
    """Variables that are both predicted and used as future inputs."""

    return sorted(set(task_config.target_variables) & set(task_config.input_variables))


def initial_memory(
    example: xarray.Dataset,
    task_config: object,
) -> xarray.Dataset:
    """Return the first two frames of autoregressive model state."""

    return example.isel(time=[0, 1])[evolving_input_variables(task_config)]


def build_step_args(
    example: xarray.Dataset,
    memory: xarray.Dataset,
    target_time_index: int,
    task_config: object,
) -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """Build one GenCast transition from a two-frame evolving memory."""

    from graphcast import data_utils

    indices = [
        target_time_index - 2,
        target_time_index - 1,
        target_time_index,
    ]
    base = example.isel(time=indices).copy(deep=True)
    target_frame = base.isel(time=[2])
    aligned_memory = memory.assign_coords(time=base.time.values[:2])

    evolving = evolving_input_variables(task_config)
    for name, dataset in (
        ("memory", aligned_memory),
        ("example", base),
    ):
        missing = sorted(set(evolving) - set(dataset.data_vars))
        if missing:
            raise ValueError(f"{name} is missing evolving variables: {missing}")

    for variable in evolving:
        memory_variable = aligned_memory[variable].reset_coords(drop=True)
        target_variable = target_frame[variable].reset_coords(drop=True)
        base = base.assign(
            {
                variable: xarray.concat(
                    [memory_variable, target_variable],
                    dim="time",
                )
            }
        )

    return data_utils.extract_inputs_targets_forcings(
        base,
        target_lead_times=slice("12h", "12h"),
        **dataclasses.asdict(task_config),
    )


def update_memory(
    previous_memory: xarray.Dataset,
    prediction: xarray.Dataset,
    evolving_variables: list[str],
) -> xarray.Dataset:
    """Advance a two-frame autoregressive memory by one prediction."""

    data = {}
    for name, dataset in (
        ("memory", previous_memory),
        ("prediction", prediction),
    ):
        missing = sorted(set(evolving_variables) - set(dataset.data_vars))
        if missing:
            raise ValueError(f"{name} is missing evolving variables: {missing}")
    for variable in evolving_variables:
        previous = previous_memory[variable].isel(time=[1]).reset_coords(drop=True)
        current = prediction[variable].isel(time=[0]).reset_coords(drop=True)
        data[variable] = xarray.concat([previous, current], dim="time")
    return xarray.Dataset(data)
