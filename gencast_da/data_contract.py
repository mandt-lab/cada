"""Validate that a downloaded ERA5 NetCDF matches the contract GenCast 1deg
expects.

GenCast Mini (1deg) requires:
    - 1.0deg grid: lat in {-90, -89, …, 90} (181), lon in {0, 1, …, 359} (360)
    - 13 pressure levels: PRESSURE_LEVELS_WEATHERBENCH_13
    - input_duration='24h' (current frame + frame 12h prior)
    - target lead times in 12h steps
    - target surface vars: 2m_temperature, mean_sea_level_pressure,
      10m_u_component_of_wind, 10m_v_component_of_wind,
      total_precipitation_12hr, sea_surface_temperature
    - target atmospheric vars: temperature, geopotential, u/v wind,
      vertical_velocity, specific_humidity (× 13 levels)
    - generated forcings: year/day-of-year sin/cos
    - static vars: land/sea mask, geopotential_at_surface

Run:
    python -m gencast_da.data_contract
"""

from __future__ import annotations

import dataclasses

import numpy as np
import xarray


def validate_example_dataset(
    example: xarray.Dataset,
    *,
    expected_levels: tuple[int, ...],
    expected_variables: set[str],
) -> None:
    expected_sizes = {"batch": 1, "lat": 181, "lon": 360}
    for dimension, expected in expected_sizes.items():
        actual = example.sizes.get(dimension)
        if actual != expected:
            raise ValueError(f"{dimension} size must be {expected}, got {actual}")
    if example.sizes.get("time", 0) < 3:
        raise ValueError("GenCast trajectories require at least three frames")

    latitude = np.asarray(example.coords["lat"].values)
    longitude = np.asarray(example.coords["lon"].values)
    levels = tuple(int(value) for value in example.coords["level"].values)
    if not np.array_equal(latitude, np.arange(-90, 91, dtype=latitude.dtype)):
        raise ValueError("latitude must be ascending -90..90 at one degree")
    if not np.array_equal(longitude, np.arange(360, dtype=longitude.dtype)):
        raise ValueError("longitude must be ascending 0..359 at one degree")
    if levels != expected_levels:
        raise ValueError(
            f"pressure levels mismatch: got {levels}, expected {expected_levels}"
        )

    time = np.asarray(example.coords["time"].values)
    if not np.issubdtype(time.dtype, np.timedelta64):
        raise ValueError("time coordinates must be timedeltas")
    if not np.all(np.diff(time) == np.timedelta64(12, "h")):
        raise ValueError("time coordinates must have a 12-hour cadence")
    if time[0] != np.timedelta64(0, "h"):
        raise ValueError("time coordinates must start at zero")
    datetimes = np.asarray(example.coords["datetime"].values)
    if datetimes.shape != (1, len(time)):
        raise ValueError("datetime coordinates must have shape [batch,time]")
    if not np.all(np.diff(datetimes[0]) == np.timedelta64(12, "h")):
        raise ValueError("datetime coordinates must have a 12-hour cadence")
    datetime_offsets = datetimes[0] - datetimes[0, 0]
    if not np.array_equal(datetime_offsets, time):
        raise ValueError("datetime coordinates are inconsistent with time")
    first_hour = int(
        datetimes[0, 0].astype("datetime64[h]").astype(np.int64) % 24
    )
    if first_hour not in (0, 12):
        raise ValueError("datetime coordinates must align to 00/12 UTC")

    missing = sorted(expected_variables - set(example.data_vars))
    if missing:
        raise ValueError(f"missing GenCast variables: {missing}")
    for variable in expected_variables:
        data = example[variable]
        if variable in {"land_sea_mask", "geopotential_at_surface"}:
            expected_dims = ("lat", "lon")
        elif variable in {"year_progress_sin", "year_progress_cos"}:
            expected_dims = ("batch", "time")
        elif variable in {"day_progress_sin", "day_progress_cos"}:
            expected_dims = ("batch", "time", "lon")
        elif "level" in data.dims:
            expected_dims = ("batch", "time", "level", "lat", "lon")
        else:
            expected_dims = ("batch", "time", "lat", "lon")
        if data.dims != expected_dims:
            raise ValueError(
                f"{variable} dimensions must be {expected_dims}, got {data.dims}"
            )
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"{variable} must use a floating-point dtype")
        values = np.asarray(data.values)
        if variable == "sea_surface_temperature":
            if np.isinf(values).any():
                raise ValueError("sea_surface_temperature may use NaN, not infinity")
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                raise ValueError("sea_surface_temperature has no finite values")
            reference_mask = finite_mask[(0, 0)]
            if not np.all(finite_mask == reference_mask):
                raise ValueError(
                    "sea_surface_temperature NaN mask must be time-invariant"
                )
        elif not np.isfinite(values).all():
            raise ValueError(f"{variable} contains non-finite values")


def validate_statistics(
    statistics: tuple[xarray.Dataset, ...],
    *,
    expected_levels: tuple[int, ...],
    require_positive: bool = False,
) -> None:
    for dataset in statistics:
        if "level" not in dataset.coords:
            raise ValueError("normalization statistics have no level coordinate")
        levels = tuple(int(value) for value in dataset.coords["level"].values)
        if levels != expected_levels:
            raise ValueError("normalization statistic levels are misaligned")
        for variable, data in dataset.data_vars.items():
            if data.dims not in ((), ("level",)):
                raise ValueError(
                    f"statistic {variable} must be scalar or level-only"
                )
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError(f"statistic {variable} must be numeric")
            if not np.isfinite(data.values).all():
                raise ValueError(f"statistic {variable} contains non-finite values")
            if require_positive and np.any(data.values <= 0):
                raise ValueError(f"statistic {variable} must be strictly positive")


def main() -> None:
    from graphcast import graphcast

    from gencast_da.gencast_runner import load_bundle, load_example_batch

    bundle = load_bundle()
    example = load_example_batch()
    expected_levels = tuple(graphcast.PRESSURE_LEVELS_WEATHERBENCH_13)
    expected_vars = (
        set(bundle.task_config.input_variables)
        | set(bundle.task_config.target_variables)
        | set(bundle.task_config.forcing_variables)
    )
    validate_example_dataset(
        example,
        expected_levels=expected_levels,
        expected_variables=expected_vars,
    )
    validate_statistics(
        (
            bundle.diffs_stddev_by_level,
            bundle.stddev_by_level,
        ),
        expected_levels=expected_levels,
        require_positive=True,
    )
    validate_statistics(
        (
            bundle.mean_by_level,
            bundle.min_by_level,
        ),
        expected_levels=expected_levels,
    )

    print("=" * 60)
    print("Dataset summary")
    print("=" * 60)
    print("dims:    ", dict(example.sizes))
    print("coords:  ", list(example.coords))
    print("data_vars:")
    for name, da in example.data_vars.items():
        print(f"  - {name}: dims={tuple(da.dims)} shape={da.shape}")

    print()
    print("=" * 60)
    print("TaskConfig (from checkpoint)")
    print("=" * 60)
    for k, v in dataclasses.asdict(bundle.task_config).items():
        print(f"  {k} = {v}")

    print()
    print("=" * 60)
    print("Coordinate sanity checks")
    print("=" * 60)
    lat = example.coords["lat"].values
    lon = example.coords["lon"].values
    lvl = example.coords["level"].values
    print(
        f"  lat:   n={len(lat)} min={lat.min()} max={lat.max()} step={lat[1] - lat[0]}"
    )
    print(
        f"  lon:   n={len(lon)} min={lon.min()} max={lon.max()} step={lon[1] - lon[0]}"
    )
    print(f"  level: n={len(lvl)} values={list(lvl)}")
    print()
    print(f"All {len(expected_vars)} TaskConfig variables present in dataset.")
    print("Data contract OK.")


if __name__ == "__main__":
    main()
