"""Deterministic, import-safe data preparation for the core CADA datasets.

The module performs no work at import time.  Functions that write files use a
temporary file in the destination directory and never replace an existing
target unless ``overwrite=True`` is passed explicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import h5py
import numpy as np
import torch
from scipy.fft import irfft, rfft, rfftfreq
from scipy.integrate import solve_ivp

PathLike: TypeAlias = str | os.PathLike[str]

ERA5_PAPER_YEARS = tuple(range(2006, 2017))
ERA5_TRAIN_YEARS = tuple(range(2006, 2016))
ERA5_HOLDOUT_YEAR = 2016


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _prepare_target(path: PathLike, overwrite: bool) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True to replace it"
            )
        if target.is_dir():
            raise IsADirectoryError(target)
    return target


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_temporary_file(
    temporary: Path, target: Path, *, overwrite: bool
) -> None:
    if overwrite:
        os.replace(temporary, target)
    else:
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"{target} was created while output was being prepared"
            ) from error
        else:
            temporary.unlink()

    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@dataclass(frozen=True)
class KSConfig:
    """Numerical and initial-condition settings for periodic KS trajectories."""

    t_start: float = 0.0
    t_end: float = 100.0
    domain_length: float = 64.0
    spatial_points: int = 256
    integration_steps: int = 250
    saved_steps: int = 140
    initial_modes: int = 10
    mode_min: int = 1
    mode_max_exclusive: int = 3
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-9
    h5_dtype: str = "float64"

    def __post_init__(self) -> None:
        t_start = _finite_scalar(self.t_start, "t_start")
        t_end = _finite_scalar(self.t_end, "t_end")
        domain_length = _finite_scalar(self.domain_length, "domain_length")
        if t_end <= t_start:
            raise ValueError("t_end must be greater than t_start")
        if domain_length <= 0:
            raise ValueError("domain_length must be positive")
        _positive_integer(self.spatial_points, "spatial_points")
        _positive_integer(self.integration_steps, "integration_steps")
        _positive_integer(self.saved_steps, "saved_steps")
        _positive_integer(self.initial_modes, "initial_modes")
        _positive_integer(self.mode_min, "mode_min")
        _positive_integer(self.mode_max_exclusive, "mode_max_exclusive")
        if self.integration_steps < 2:
            raise ValueError("integration_steps must be at least two")
        if self.saved_steps > self.integration_steps:
            raise ValueError("saved_steps cannot exceed integration_steps")
        if self.mode_max_exclusive <= self.mode_min:
            raise ValueError("mode_max_exclusive must be greater than mode_min")
        if _finite_scalar(self.relative_tolerance, "relative_tolerance") <= 0:
            raise ValueError("relative_tolerance must be positive")
        if _finite_scalar(self.absolute_tolerance, "absolute_tolerance") <= 0:
            raise ValueError("absolute_tolerance must be positive")
        if self.h5_dtype not in {"float32", "float64"}:
            raise ValueError("h5_dtype must be 'float32' or 'float64'")

    @property
    def dx(self) -> float:
        return self.domain_length / self.spatial_points

    @property
    def dt(self) -> float:
        return (self.t_end - self.t_start) / (self.integration_steps - 1)

    @property
    def dataset_name(self) -> str:
        return f"pde_{self.saved_steps}-{self.spatial_points}"


@dataclass(frozen=True)
class KSSplitSeeds:
    """Independent deterministic seeds for the standard data splits."""

    train: int
    valid: int
    test: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.train, "train")
        _nonnegative_integer(self.valid, "valid")
        _nonnegative_integer(self.test, "test")
        if len({self.train, self.valid, self.test}) != 3:
            raise ValueError("train, valid, and test seeds must be distinct")


@dataclass(frozen=True)
class KSSplit:
    """A named KS split and its deterministic random seed."""

    name: str
    samples: int
    seed: int

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty HDF5 group name")
        _positive_integer(self.samples, "samples")
        _nonnegative_integer(self.seed, "seed")


@dataclass(frozen=True)
class KSTrajectory:
    """One simulated trajectory and its periodic coordinates."""

    values: np.ndarray
    time: np.ndarray
    space: np.ndarray
    dt: float
    dx: float


@dataclass(frozen=True)
class KSH5Result:
    """Description of a generated grouped KS HDF5 file."""

    path: Path
    split: str
    dataset: str
    shape: tuple[int, int, int]
    seed: int


@dataclass(frozen=True)
class KSMinMaxStats:
    """Absolute extrema computed from a grouped KS training dataset."""

    dataset: str
    shape: tuple[int, int, int]
    absolute_minimum: float
    absolute_maximum: float

    def __post_init__(self) -> None:
        minimum = _finite_scalar(self.absolute_minimum, "absolute_minimum")
        maximum = _finite_scalar(self.absolute_maximum, "absolute_maximum")
        if maximum <= minimum:
            raise ValueError("absolute_maximum must be greater than absolute_minimum")

    def as_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "shape": list(self.shape),
            "abs_min": self.absolute_minimum,
            "abs_max": self.absolute_maximum,
        }


@dataclass(frozen=True)
class DenseChannelStats:
    """Per-channel extrema for dense ``[N, T, C, H, W]`` data."""

    dataset: str
    shape: tuple[int, int, int, int, int]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 5 or any(size <= 0 for size in self.shape):
            raise ValueError(
                f"shape must be positive [N, T, C, H, W], got {self.shape}"
            )
        if len(self.minimum) != self.shape[2] or len(self.maximum) != self.shape[2]:
            raise ValueError("statistics length must equal the channel dimension")
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("channel statistics must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("each channel maximum must be greater than its minimum")


@dataclass(frozen=True)
class ERA5Config:
    """Paper ERA5 protocol: 500 hPa North Atlantic T/u/v at six-hour cadence."""

    years: tuple[int, ...] = ERA5_PAPER_YEARS
    train_years: tuple[int, ...] = ERA5_TRAIN_YEARS
    holdout_year: int = ERA5_HOLDOUT_YEAR
    pressure_level_hpa: int = 500
    area: tuple[float, float, float, float] = (60.0, -60.0, 30.0, 0.0)
    target_height: int = 64
    target_width: int = 64
    trajectory_steps: int = 64
    interval_hours: int = 6

    def __post_init__(self) -> None:
        if not self.years or tuple(sorted(self.years)) != self.years:
            raise ValueError("years must be a non-empty increasing tuple")
        if len(set(self.years)) != len(self.years):
            raise ValueError("years cannot contain duplicates")
        if not self.train_years or tuple(sorted(self.train_years)) != self.train_years:
            raise ValueError("train_years must be a non-empty increasing tuple")
        if (
            tuple(range(self.train_years[0], self.train_years[-1] + 1))
            != self.train_years
        ):
            raise ValueError("train_years must be consecutive")
        if set(self.train_years) | {self.holdout_year} != set(self.years):
            raise ValueError("train_years plus holdout_year must equal years")
        if self.holdout_year in self.train_years:
            raise ValueError("holdout_year cannot be a training year")
        _positive_integer(self.pressure_level_hpa, "pressure_level_hpa")
        _positive_integer(self.target_height, "target_height")
        _positive_integer(self.target_width, "target_width")
        _positive_integer(self.trajectory_steps, "trajectory_steps")
        _positive_integer(self.interval_hours, "interval_hours")
        north, west, south, east = (
            _finite_scalar(value, "area coordinate") for value in self.area
        )
        if north <= south:
            raise ValueError("area north must be greater than area south")
        if east <= west:
            raise ValueError("area east must be greater than area west")


@dataclass(frozen=True)
class ERA5SplitResult:
    """Description of one converted dense ERA5 split."""

    path: Path
    years: tuple[int, ...]
    shape: tuple[int, int, int, int, int]
    dropped_timesteps: int


@dataclass(frozen=True)
class ERA5ConversionResult:
    """Paths and train-only statistics produced by ERA5 conversion."""

    train: ERA5SplitResult
    holdout: ERA5SplitResult
    stats_path: Path
    train_stats: DenseChannelStats


def derive_ks_split_seeds(base_seed: int = 0) -> KSSplitSeeds:
    """Derive stable, independent train/valid/test seeds from ``base_seed``."""

    _nonnegative_integer(base_seed, "base_seed")
    children = np.random.SeedSequence(base_seed).spawn(3)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
    return KSSplitSeeds(train=seeds[0], valid=seeds[1], test=seeds[2])


def make_ks_splits(
    train_samples: int,
    valid_samples: int,
    test_samples: int,
    *,
    base_seed: int = 0,
) -> tuple[KSSplit, KSSplit, KSSplit]:
    """Construct standard split specifications with deterministic seeds."""

    seeds = derive_ks_split_seeds(base_seed)
    return (
        KSSplit("train", train_samples, seeds.train),
        KSSplit("valid", valid_samples, seeds.valid),
        KSSplit("test", test_samples, seeds.test),
    )


def ks_pseudospectral_rhs(
    _time: float, state: np.ndarray, domain_length: float
) -> np.ndarray:
    """Evaluate ``-u u_x - u_xx - u_xxxx`` on a periodic Fourier grid."""

    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or state.size < 2:
        raise ValueError(
            "state must be a one-dimensional array with at least two points"
        )
    domain_length = _finite_scalar(domain_length, "domain_length")
    if domain_length <= 0:
        raise ValueError("domain_length must be positive")
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")

    wave_numbers = 2.0 * np.pi * rfftfreq(
        state.size, d=domain_length / state.size
    )
    transformed = rfft(state)
    first_multiplier = 1j * wave_numbers
    if state.size % 2 == 0:
        first_multiplier[-1] = 0.0
    first = irfft(first_multiplier * transformed, n=state.size)
    second = irfft(-(wave_numbers**2) * transformed, n=state.size)
    fourth = irfft((wave_numbers**4) * transformed, n=state.size)
    return -state * first - second - fourth


def sample_ks_initial_condition(config: KSConfig, seed: int) -> np.ndarray:
    """Sample the legacy random sine-series initial condition deterministically."""

    _nonnegative_integer(seed, "seed")
    generator = np.random.default_rng(seed)
    amplitudes = generator.random(config.initial_modes) - 0.5
    phases = 2.0 * np.pi * generator.random(config.initial_modes)
    modes = generator.integers(
        config.mode_min,
        config.mode_max_exclusive,
        size=config.initial_modes,
    )
    space = np.arange(config.spatial_points, dtype=np.float64) * config.dx
    angles = (
        2.0
        * np.pi
        * modes[None, :]
        * space[:, None]
        / config.domain_length
        + phases[None, :]
    )
    initial = np.sum(amplitudes[None, :] * np.sin(angles), axis=1)
    if not np.isfinite(initial).all():
        raise ValueError("sampled KS initial condition is not finite")
    return initial


def simulate_ks(
    initial_condition: np.ndarray, config: KSConfig = KSConfig()
) -> KSTrajectory:
    """Integrate one periodic KS trajectory with the legacy Radau tolerances."""

    initial = np.asarray(initial_condition, dtype=np.float64)
    expected_shape = (config.spatial_points,)
    if initial.shape != expected_shape:
        raise ValueError(
            f"initial_condition must have shape {expected_shape}, got {initial.shape}"
        )
    if not np.isfinite(initial).all():
        raise ValueError("initial_condition must contain only finite values")

    full_time = np.linspace(
        config.t_start,
        config.t_end,
        config.integration_steps,
        dtype=np.float64,
    )
    solution = solve_ivp(
        ks_pseudospectral_rhs,
        (config.t_start, config.t_end),
        initial,
        method="Radau",
        t_eval=full_time,
        args=(config.domain_length,),
        rtol=config.relative_tolerance,
        atol=config.absolute_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"KS integration failed: {solution.message}")

    values = np.asarray(solution.y.T[-config.saved_steps :], dtype=np.float64)
    time = full_time[-config.saved_steps :]
    space = np.arange(config.spatial_points, dtype=np.float64) * config.dx
    expected_values_shape = (config.saved_steps, config.spatial_points)
    if values.shape != expected_values_shape:
        raise RuntimeError(
            f"KS solver returned shape {values.shape}, expected {expected_values_shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("KS solver returned non-finite values")
    return KSTrajectory(
        values=values,
        time=time,
        space=space,
        dt=config.dt,
        dx=config.dx,
    )


def generate_ks_trajectory(
    config: KSConfig = KSConfig(), *, seed: int
) -> KSTrajectory:
    """Sample and integrate one deterministic KS trajectory."""

    return simulate_ks(sample_ks_initial_condition(config, seed), config)


def _validate_ks_h5(
    path: Path, split: KSSplit, config: KSConfig
) -> tuple[int, int, int]:
    expected_shape = (split.samples, config.saved_steps, config.spatial_points)
    with h5py.File(path, "r") as file:
        if split.name not in file or not isinstance(file[split.name], h5py.Group):
            raise RuntimeError(f"missing group {split.name!r} in generated KS file")
        group = file[split.name]
        required = {config.dataset_name, "x", "dx", "t", "dt"}
        missing = required.difference(group.keys())
        if missing:
            raise RuntimeError(
                f"generated KS file is missing datasets: {sorted(missing)}"
            )
        if tuple(group[config.dataset_name].shape) != expected_shape:
            raise RuntimeError("generated KS trajectory dataset has the wrong shape")
        expected_shapes = {
            "x": (split.samples, config.spatial_points),
            "dx": (split.samples,),
            "t": (split.samples, config.saved_steps),
            "dt": (split.samples,),
        }
        for name, shape in expected_shapes.items():
            if tuple(group[name].shape) != shape:
                raise RuntimeError(f"generated KS dataset {name!r} has the wrong shape")
        for start in range(0, split.samples, 16):
            stop = min(start + 16, split.samples)
            if not np.isfinite(group[config.dataset_name][start:stop]).all():
                raise RuntimeError("generated KS trajectory dataset is not finite")
        if not np.isfinite(group["x"][:]).all():
            raise RuntimeError("generated KS space coordinates are not finite")
        if not np.isfinite(group["t"][:]).all():
            raise RuntimeError("generated KS time coordinates are not finite")
        if np.any(group["dx"][:] <= 0) or np.any(group["dt"][:] <= 0):
            raise RuntimeError("generated KS spacings must be positive")
    return expected_shape


def generate_ks_hdf5(
    path: PathLike,
    split: KSSplit,
    config: KSConfig = KSConfig(),
    *,
    overwrite: bool = False,
) -> KSH5Result:
    """Generate one grouped KS split and atomically write its HDF5 file."""

    target = _prepare_target(path, overwrite)
    temporary = _temporary_path(target)
    split_generator = np.random.default_rng(split.seed)
    sample_seeds = split_generator.integers(
        0,
        np.iinfo(np.uint64).max,
        size=split.samples,
        dtype=np.uint64,
    )

    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["equation"] = "Kuramoto-Sivashinsky"
            file.attrs["split_seed"] = split.seed
            group = file.create_group(split.name)
            values = group.create_dataset(
                config.dataset_name,
                shape=(split.samples, config.saved_steps, config.spatial_points),
                dtype=config.h5_dtype,
            )
            space = group.create_dataset(
                "x",
                shape=(split.samples, config.spatial_points),
                dtype=config.h5_dtype,
            )
            dx = group.create_dataset(
                "dx", shape=(split.samples,), dtype=config.h5_dtype
            )
            time = group.create_dataset(
                "t",
                shape=(split.samples, config.saved_steps),
                dtype=config.h5_dtype,
            )
            dt = group.create_dataset(
                "dt", shape=(split.samples,), dtype=config.h5_dtype
            )

            for index, sample_seed in enumerate(sample_seeds):
                trajectory = generate_ks_trajectory(config, seed=int(sample_seed))
                values[index] = trajectory.values
                space[index] = trajectory.space
                dx[index] = trajectory.dx
                time[index] = trajectory.time
                dt[index] = trajectory.dt

            file.flush()

        shape = _validate_ks_h5(temporary, split, config)
        _fsync_file(temporary)
        _commit_temporary_file(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()

    return KSH5Result(
        path=target,
        split=split.name,
        dataset=f"{split.name}/{config.dataset_name}",
        shape=shape,
        seed=split.seed,
    )


def _grouped_ks_dataset(
    file: h5py.File, split: str, dataset_name: str | None
) -> tuple[str, h5py.Dataset]:
    if not split or "/" in split:
        raise ValueError("split must be a non-empty HDF5 group name")
    if split not in file or not isinstance(file[split], h5py.Group):
        raise KeyError(f"group {split!r} was not found")
    group = file[split]
    if dataset_name is None:
        candidates = sorted(
            name
            for name, node in group.items()
            if name.startswith("pde_") and isinstance(node, h5py.Dataset)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one pde_* dataset in {split!r}, found {candidates}"
            )
        dataset_name = candidates[0]
    if "/" in dataset_name or dataset_name not in group:
        raise KeyError(f"dataset {split}/{dataset_name} was not found")
    dataset = group[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{split}/{dataset_name} is not a dataset")
    return f"{split}/{dataset_name}", dataset


def compute_ks_training_minmax(
    h5_path: PathLike,
    *,
    dataset_name: str | None = None,
    batch_size: int = 16,
) -> KSMinMaxStats:
    """Stream absolute min/max from the grouped ``train`` KS tensor."""

    _positive_integer(batch_size, "batch_size")
    minimum = np.inf
    maximum = -np.inf
    with h5py.File(h5_path, "r") as file:
        key, dataset = _grouped_ks_dataset(file, "train", dataset_name)
        shape = tuple(int(size) for size in dataset.shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError(f"{key} must have positive shape [N, T, X], got {shape}")
        for start in range(0, shape[0], batch_size):
            values = np.asarray(dataset[start : start + batch_size])
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains non-finite values")
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    return KSMinMaxStats(
        dataset=key,
        shape=shape,
        absolute_minimum=float(minimum),
        absolute_maximum=float(maximum),
    )


def write_ks_minmax_json(
    stats: KSMinMaxStats,
    path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write KS training extrema in the loader-compatible JSON schema."""

    target = _prepare_target(path, overwrite)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(stats.as_json(), file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        _commit_temporary_file(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def compute_dense_h5_channel_stats(
    h5_path: PathLike,
    *,
    dataset: str = "x",
    batch_size: int = 1,
) -> DenseChannelStats:
    """Stream per-channel min/max from ``[N, T, C, H, W]`` HDF5 data."""

    if not dataset:
        raise ValueError("dataset cannot be empty")
    _positive_integer(batch_size, "batch_size")
    with h5py.File(h5_path, "r") as file:
        if dataset not in file or not isinstance(file[dataset], h5py.Dataset):
            raise KeyError(f"dataset {dataset!r} was not found in {h5_path}")
        values = file[dataset]
        shape = tuple(int(size) for size in values.shape)
        if len(shape) != 5 or any(size <= 0 for size in shape):
            raise ValueError(
                f"{dataset} must have positive shape [N, T, C, H, W], got {shape}"
            )
        minimum = np.full(shape[2], np.inf, dtype=np.float64)
        maximum = np.full(shape[2], -np.inf, dtype=np.float64)
        for start in range(0, shape[0], batch_size):
            batch = np.asarray(values[start : start + batch_size])
            if not np.isfinite(batch).all():
                raise ValueError(f"{dataset} contains non-finite values")
            minimum = np.minimum(minimum, batch.min(axis=(0, 1, 3, 4)))
            maximum = np.maximum(maximum, batch.max(axis=(0, 1, 3, 4)))
    return DenseChannelStats(
        dataset=dataset,
        shape=shape,
        minimum=tuple(float(value) for value in minimum),
        maximum=tuple(float(value) for value in maximum),
    )


def write_dense_channel_stats(
    stats: DenseChannelStats,
    path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write loader-compatible ``{"min", "max"}`` PyTorch stats."""

    target = _prepare_target(path, overwrite)
    temporary = _temporary_path(target)
    payload = {
        "min": torch.tensor(stats.minimum, dtype=torch.float32),
        "max": torch.tensor(stats.maximum, dtype=torch.float32),
    }
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            not isinstance(loaded, Mapping)
            or "min" not in loaded
            or "max" not in loaded
            or tuple(loaded["min"].shape) != (stats.shape[2],)
            or tuple(loaded["max"].shape) != (stats.shape[2],)
        ):
            raise RuntimeError("saved dense channel statistics failed validation")
        _fsync_file(temporary)
        _commit_temporary_file(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def build_era5_cds_request(
    year: int, config: ERA5Config = ERA5Config()
) -> dict[str, Any]:
    """Build a CDS request payload without importing a client or downloading data.

    Submit the returned dictionary to the
    ``reanalysis-era5-pressure-levels`` dataset yourself.
    """

    if year not in config.years:
        raise ValueError(f"year must be one of {config.years}")
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": [str(config.pressure_level_hpa)],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": list(config.area),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _require_xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as error:
        raise ImportError(
            "ERA5 conversion requires the optional 'xarray' package"
        ) from error
    return xr


@contextmanager
def _open_era5_inputs(inputs: Any) -> Iterator[Any]:
    xr = _require_xarray()
    if isinstance(inputs, xr.Dataset):
        yield inputs
        return

    if isinstance(inputs, (str, os.PathLike)):
        paths = [Path(inputs)]
    elif isinstance(inputs, Sequence):
        paths = [Path(path) for path in inputs]
    else:
        raise TypeError("inputs must be an xarray.Dataset, path, or sequence of paths")
    if not paths:
        raise ValueError("at least one ERA5 input path is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ERA5 input files not found: {missing}")

    with ExitStack() as stack:
        datasets = []
        for path in paths:
            dataset = xr.open_dataset(path, decode_times=True)
            stack.callback(dataset.close)
            datasets.append(dataset)
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = xr.combine_by_coords(datasets, combine_attrs="override")
        yield combined


def _first_name(candidates: Sequence[str], available: Any, kind: str) -> str:
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"could not find {kind}; tried {tuple(candidates)}")


def _select_pressure_level(data: Any, level_hpa: int) -> Any:
    level_names = ("isobaricInhPa", "pressure_level", "level", "plev")
    for name in level_names:
        if name not in data.coords and name not in data.dims:
            continue
        coordinate = np.asarray(data[name].values, dtype=np.float64).reshape(-1)
        target = float(level_hpa)
        if coordinate.size and np.nanmax(np.abs(coordinate)) > 2_000:
            target *= 100.0
        matches = np.flatnonzero(np.isclose(coordinate, target))
        if matches.size != 1:
            raise ValueError(
                f"expected exactly one {level_hpa} hPa coordinate in {name!r}, "
                f"found {coordinate.tolist()}"
            )
        if name in data.dims:
            data = data.isel({name: int(matches[0])}, drop=True)
        return data
    return data


def _canonical_era5_dataset(dataset: Any, config: ERA5Config) -> Any:
    xr = _require_xarray()
    time_name = _first_name(("time", "valid_time"), dataset.dims, "time dimension")
    latitude_name = _first_name(
        ("latitude", "lat"), dataset.dims, "latitude dimension"
    )
    longitude_name = _first_name(
        ("longitude", "lon"), dataset.dims, "longitude dimension"
    )
    rename = {
        original: canonical
        for original, canonical in (
            (time_name, "time"),
            (latitude_name, "latitude"),
            (longitude_name, "longitude"),
        )
        if original != canonical
    }
    if rename:
        dataset = dataset.rename(rename)

    variable_names = {
        "temperature": _first_name(
            ("t", "temperature"), dataset.data_vars, "temperature variable"
        ),
        "u": _first_name(
            ("u", "u_component_of_wind"), dataset.data_vars, "zonal-wind variable"
        ),
        "v": _first_name(
            ("v", "v_component_of_wind"), dataset.data_vars, "meridional-wind variable"
        ),
    }
    variables = {}
    required_dims = {"time", "latitude", "longitude"}
    for canonical_name, source_name in variable_names.items():
        data = _select_pressure_level(dataset[source_name], config.pressure_level_hpa)
        for dimension in tuple(data.dims):
            if dimension not in required_dims:
                if data.sizes[dimension] != 1:
                    raise ValueError(
                        f"{source_name!r} has unsupported non-singleton dimension "
                        f"{dimension!r}"
                    )
                data = data.isel({dimension: 0}, drop=True)
        if set(data.dims) != required_dims:
            raise ValueError(
                f"{source_name!r} must resolve to dimensions "
                f"(time, latitude, longitude), got {data.dims}"
            )
        variables[canonical_name] = data.transpose("time", "latitude", "longitude")

    aligned = xr.align(*variables.values(), join="exact")
    canonical = xr.Dataset(dict(zip(variables.keys(), aligned, strict=True)))
    longitude = np.asarray(canonical.longitude.values, dtype=np.float64)
    if np.any(longitude > 180.0):
        longitude = (longitude + 180.0) % 360.0 - 180.0
        canonical = canonical.assign_coords(longitude=longitude)
    canonical = canonical.sortby(["time", "latitude", "longitude"])

    latitude = np.asarray(canonical.latitude.values, dtype=np.float64)
    longitude = np.asarray(canonical.longitude.values, dtype=np.float64)
    north, west, south, east = config.area
    latitude_indices = np.flatnonzero((latitude >= south) & (latitude <= north))
    longitude_indices = np.flatnonzero((longitude >= west) & (longitude <= east))
    if latitude_indices.size < 2 or longitude_indices.size < 2:
        raise ValueError(
            "ERA5 input does not cover the configured area with at least "
            "two grid points"
        )
    canonical = canonical.isel(
        latitude=latitude_indices, longitude=longitude_indices
    )

    for coordinate_name in ("time", "latitude", "longitude"):
        coordinate = np.asarray(canonical[coordinate_name].values)
        if np.unique(coordinate).size != coordinate.size:
            raise ValueError(f"ERA5 {coordinate_name} coordinate contains duplicates")
    return canonical


def _select_complete_years(
    dataset: Any, years: tuple[int, ...], interval_hours: int
) -> Any:
    start = np.datetime64(f"{years[0]:04d}-01-01T00:00:00", "ns")
    end = np.datetime64(f"{years[-1] + 1:04d}-01-01T00:00:00", "ns")
    times = np.asarray(dataset.time.values)
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("ERA5 time must decode to numpy datetime64 values")
    times = times.astype("datetime64[ns]")
    indices = np.flatnonzero((times >= start) & (times < end))
    selected = dataset.isel(time=indices)
    selected_times = np.asarray(selected.time.values).astype("datetime64[ns]")
    expected = np.arange(
        start,
        end,
        np.timedelta64(interval_hours, "h"),
        dtype="datetime64[ns]",
    )
    if not np.array_equal(selected_times, expected):
        raise ValueError(
            f"ERA5 years {years[0]}-{years[-1]} must contain every "
            f"{interval_hours}-hour timestamp exactly once; got "
            f"{selected_times.size}, expected {expected.size}"
        )
    return selected


def _era5_chunk(
    dataset: Any,
    start: int,
    stop: int,
    latitude_target: np.ndarray,
    longitude_target: np.ndarray,
) -> np.ndarray:
    _require_xarray()
    channels = []
    for name in ("temperature", "u", "v"):
        interpolated = dataset[name].isel(time=slice(start, stop)).interp(
            latitude=latitude_target,
            longitude=longitude_target,
        )
        values = np.asarray(
            interpolated.transpose("time", "latitude", "longitude").values,
            dtype=np.float32,
        )
        channels.append(values)
    result = np.stack(channels, axis=1)
    expected_shape = (
        stop - start,
        3,
        latitude_target.size,
        longitude_target.size,
    )
    if result.shape != expected_shape:
        raise RuntimeError(
            f"interpolated ERA5 chunk has shape {result.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("interpolated ERA5 data contains non-finite values")
    return result


def _write_era5_split(
    dataset: Any,
    temporary: Path,
    years: tuple[int, ...],
    config: ERA5Config,
    latitude_target: np.ndarray,
    longitude_target: np.ndarray,
    *,
    collect_stats: bool,
) -> tuple[tuple[int, int, int, int, int], int, np.ndarray, np.ndarray]:
    total_steps = int(dataset.sizes["time"])
    trajectories, dropped = divmod(total_steps, config.trajectory_steps)
    if trajectories == 0:
        raise ValueError(
            f"{total_steps} ERA5 timesteps cannot form a "
            f"{config.trajectory_steps}-step trajectory"
        )
    shape = (
        trajectories,
        config.trajectory_steps,
        3,
        config.target_height,
        config.target_width,
    )
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)

    with h5py.File(temporary, "w") as file:
        file.attrs["years"] = np.asarray(years, dtype=np.int32)
        file.attrs["interval_hours"] = config.interval_hours
        file.attrs["pressure_level_hpa"] = config.pressure_level_hpa
        file.attrs["channel_order"] = "temperature,u,v"
        values = file.create_dataset(
            "x",
            shape=shape,
            dtype="float32",
            chunks=(
                1,
                config.trajectory_steps,
                3,
                config.target_height,
                config.target_width,
            ),
        )
        for trajectory in range(trajectories):
            start = trajectory * config.trajectory_steps
            stop = start + config.trajectory_steps
            chunk = _era5_chunk(
                dataset, start, stop, latitude_target, longitude_target
            )
            values[trajectory] = chunk
            if collect_stats:
                minimum = np.minimum(minimum, chunk.min(axis=(0, 2, 3)))
                maximum = np.maximum(maximum, chunk.max(axis=(0, 2, 3)))

        if dropped:
            tail = _era5_chunk(
                dataset,
                trajectories * config.trajectory_steps,
                total_steps,
                latitude_target,
                longitude_target,
            )
            if collect_stats:
                minimum = np.minimum(minimum, tail.min(axis=(0, 2, 3)))
                maximum = np.maximum(maximum, tail.max(axis=(0, 2, 3)))
        file.flush()

    with h5py.File(temporary, "r") as file:
        if "x" not in file or tuple(file["x"].shape) != shape:
            raise RuntimeError("generated ERA5 HDF5 file failed shape validation")
        if not np.isfinite(file["x"][0]).all() or not np.isfinite(file["x"][-1]).all():
            raise RuntimeError("generated ERA5 HDF5 file contains non-finite values")
    return shape, dropped, minimum, maximum


def _write_stats_temporary(stats: DenseChannelStats, temporary: Path) -> None:
    torch.save(
        {
            "min": torch.tensor(stats.minimum, dtype=torch.float32),
            "max": torch.tensor(stats.maximum, dtype=torch.float32),
        },
        temporary,
    )
    loaded = torch.load(temporary, map_location="cpu", weights_only=True)
    if (
        not isinstance(loaded, Mapping)
        or tuple(loaded["min"].shape) != (3,)
        or tuple(loaded["max"].shape) != (3,)
        or not torch.isfinite(loaded["min"]).all()
        or not torch.isfinite(loaded["max"]).all()
    ):
        raise RuntimeError("generated ERA5 statistics failed validation")


def convert_era5_netcdf_to_hdf5(
    inputs: Any,
    train_path: PathLike,
    holdout_path: PathLike,
    stats_path: PathLike,
    config: ERA5Config = ERA5Config(),
    *,
    overwrite: bool = False,
) -> ERA5ConversionResult:
    """Convert downloaded ERA5 xarray/netCDF data to paper-protocol HDF5 files.

    ``inputs`` may be an open ``xarray.Dataset``, one netCDF path, or a sequence
    of netCDF paths.  No network client is imported or called.  The ``x``
    datasets are ``[N, 64, 3, 64, 64]`` in channel order temperature, u, v.
    Statistics are computed from all six-hourly training states before the
    incomplete final trajectory is dropped.
    """

    targets = [
        _prepare_target(train_path, overwrite),
        _prepare_target(holdout_path, overwrite),
        _prepare_target(stats_path, overwrite),
    ]
    if len({target.resolve(strict=False) for target in targets}) != len(targets):
        raise ValueError("train_path, holdout_path, and stats_path must be distinct")
    temporaries = [_temporary_path(target) for target in targets]

    try:
        with _open_era5_inputs(inputs) as source:
            canonical = _canonical_era5_dataset(source, config)
            train_data = _select_complete_years(
                canonical, config.train_years, config.interval_hours
            )
            holdout_data = _select_complete_years(
                canonical, (config.holdout_year,), config.interval_hours
            )
            latitude = np.asarray(canonical.latitude.values, dtype=np.float64)
            longitude = np.asarray(canonical.longitude.values, dtype=np.float64)
            latitude_target = np.linspace(
                float(latitude.min()),
                float(latitude.max()),
                config.target_height,
                dtype=np.float64,
            )
            longitude_target = np.linspace(
                float(longitude.min()),
                float(longitude.max()),
                config.target_width,
                dtype=np.float64,
            )

            train_shape, train_dropped, minimum, maximum = _write_era5_split(
                train_data,
                temporaries[0],
                config.train_years,
                config,
                latitude_target,
                longitude_target,
                collect_stats=True,
            )
            holdout_shape, holdout_dropped, _, _ = _write_era5_split(
                holdout_data,
                temporaries[1],
                (config.holdout_year,),
                config,
                latitude_target,
                longitude_target,
                collect_stats=False,
            )

        train_stats = DenseChannelStats(
            dataset="x",
            shape=train_shape,
            minimum=tuple(float(value) for value in minimum),
            maximum=tuple(float(value) for value in maximum),
        )
        _write_stats_temporary(train_stats, temporaries[2])
        for temporary in temporaries:
            _fsync_file(temporary)
        for temporary, target in zip(temporaries, targets, strict=True):
            _commit_temporary_file(temporary, target, overwrite=overwrite)
    finally:
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()

    return ERA5ConversionResult(
        train=ERA5SplitResult(
            path=targets[0],
            years=config.train_years,
            shape=train_shape,
            dropped_timesteps=train_dropped,
        ),
        holdout=ERA5SplitResult(
            path=targets[1],
            years=(config.holdout_year,),
            shape=holdout_shape,
            dropped_timesteps=holdout_dropped,
        ),
        stats_path=targets[2],
        train_stats=train_stats,
    )


__all__ = [
    "DenseChannelStats",
    "ERA5Config",
    "ERA5ConversionResult",
    "ERA5SplitResult",
    "ERA5_HOLDOUT_YEAR",
    "ERA5_PAPER_YEARS",
    "ERA5_TRAIN_YEARS",
    "KSConfig",
    "KSH5Result",
    "KSMinMaxStats",
    "KSSplit",
    "KSSplitSeeds",
    "KSTrajectory",
    "build_era5_cds_request",
    "compute_dense_h5_channel_stats",
    "compute_ks_training_minmax",
    "convert_era5_netcdf_to_hdf5",
    "derive_ks_split_seeds",
    "generate_ks_hdf5",
    "generate_ks_trajectory",
    "ks_pseudospectral_rhs",
    "make_ks_splits",
    "sample_ks_initial_condition",
    "simulate_ks",
    "write_dense_channel_stats",
    "write_ks_minmax_json",
]
