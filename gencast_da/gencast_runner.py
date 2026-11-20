"""Thin wrapper that loads a GenCast checkpoint and returns JIT'd predict /
loss functions.

Why this file exists
--------------------
Every downstream script needs the same boilerplate: read the checkpoint and
four statistics files, override attention
config so it runs on a GPU (splash_mha is TPU-only), wrap with
`InputsAndResiduals` + `NaNCleaner`, and `jax.jit` a `run_forward` closure
over the params.

We do that once here so the rest of the code is short.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray
from graphcast import (
    checkpoint,
    denoiser,
    gencast,
    nan_cleaning,
    normalization,
)

from gencast_da.runtime import sha256_file

DEFAULT_CHECKPOINT = "GenCast 1p0deg Mini <2019.npz"
DEFAULT_DATASET = "source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"


def default_data_root() -> Path:
    """Return GENCAST_DATA_ROOT, or $CADA_ROOT/data/gencast when it is unset."""

    if os.environ.get("GENCAST_DATA_ROOT"):
        return Path(os.environ["GENCAST_DATA_ROOT"])
    if os.environ.get("CADA_ROOT"):
        return Path(os.environ["CADA_ROOT"]) / "data" / "gencast"
    raise RuntimeError("GENCAST_DATA_ROOT and CADA_ROOT are both unset")


@dataclass
class GenCastBundle:
    """Everything a downstream script needs to call GenCast on a GPU."""

    params: dict
    state: dict
    task_config: object
    sampler_config: gencast.SamplerConfig
    noise_config: gencast.NoiseConfig
    noise_encoder_config: denoiser.NoiseEncoderConfig
    denoiser_architecture_config: denoiser.DenoiserArchitectureConfig
    diffs_stddev_by_level: xarray.Dataset
    mean_by_level: xarray.Dataset
    stddev_by_level: xarray.Dataset
    min_by_level: xarray.Dataset
    description: str
    data_root: Path
    checkpoint_path: Path
    statistic_paths: tuple[Path, ...]

    def construct_wrapped(self):
        """Recreate the wrapped predictor (matches the official mini demo)."""
        predictor = gencast.GenCast(
            sampler_config=self.sampler_config,
            task_config=self.task_config,
            denoiser_architecture_config=self.denoiser_architecture_config,
            noise_config=self.noise_config,
            noise_encoder_config=self.noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=self.diffs_stddev_by_level,
            mean_by_level=self.mean_by_level,
            stddev_by_level=self.stddev_by_level,
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=self.min_by_level,
            var_to_clean="sea_surface_temperature",
        )
        return predictor


def load_bundle(
    data_root: os.PathLike | None = None,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    use_gpu_attention: bool = True,
) -> GenCastBundle:
    """Read checkpoint + stats; override attention config if running on GPU.

    The override is documented in `docs/cloud_vm_setup.md` of the graphcast
    repo: splash attention is TPU-only, so on GPU we have to switch to
    `triblockdiag_mha` with `mask_type="full"`.
    """
    root = Path(data_root) if data_root else default_data_root()
    ckpt_path = root / "params" / checkpoint_name
    stats_dir = root / "stats"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Run gencast_da/download_assets.sh first."
        )

    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    denoiser_cfg = ckpt.denoiser_architecture_config
    if use_gpu_attention:
        denoiser_cfg.sparse_transformer_config.attention_type = "triblockdiag_mha"
        denoiser_cfg.sparse_transformer_config.mask_type = "full"

    def _load_stat(name: str) -> xarray.Dataset:
        with open(stats_dir / name, "rb") as f:
            return xarray.load_dataset(f).compute()

    statistic_names = (
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc",
        "min_by_level.nc",
    )
    return GenCastBundle(
        params=ckpt.params,
        state={},
        task_config=ckpt.task_config,
        sampler_config=ckpt.sampler_config,
        noise_config=ckpt.noise_config,
        noise_encoder_config=ckpt.noise_encoder_config,
        denoiser_architecture_config=denoiser_cfg,
        diffs_stddev_by_level=_load_stat(statistic_names[0]),
        mean_by_level=_load_stat(statistic_names[1]),
        stddev_by_level=_load_stat(statistic_names[2]),
        min_by_level=_load_stat(statistic_names[3]),
        description=ckpt.description,
        data_root=root.resolve(),
        checkpoint_path=ckpt_path.resolve(),
        statistic_paths=tuple((stats_dir / name).resolve() for name in statistic_names),
    )


def bundle_compatibility(
    bundle: GenCastBundle,
    *,
    controlled_variable: str,
    cada_variant: str,
) -> dict:
    """Return hard model/data compatibility fields for a controller."""

    source_digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for name in (
        "cada_controller.py",
        "cada_features.py",
        "cada_guided.py",
        "cada_train_config.py",
        "checkpointing.py",
        "gencast_runner.py",
        "obs_operator.py",
        "observation_arrays.py",
        "random_utils.py",
        "rollout_utils.py",
        "train_cada_controller.py",
    ):
        source_digest.update((source_root / name).read_bytes())
    distribution = importlib.metadata.distribution("graphcast")
    direct_url_text = distribution.read_text("direct_url.json")
    return {
        "schema_version": 1,
        "graphcast_version": importlib.metadata.version("graphcast"),
        "graphcast_source": (
            json.loads(direct_url_text) if direct_url_text else None
        ),
        "cada_variant": cada_variant,
        "cada_source_sha256": source_digest.hexdigest(),
        "controlled_variable": controlled_variable,
        "checkpoint_sha256": sha256_file(bundle.checkpoint_path),
        "statistics_sha256": {
            path.name: sha256_file(path) for path in bundle.statistic_paths
        },
        "task_config": dataclasses.asdict(bundle.task_config),
        "sampler_config": dataclasses.asdict(bundle.sampler_config),
        "noise_config": dataclasses.asdict(bundle.noise_config),
        "noise_encoder_config": dataclasses.asdict(bundle.noise_encoder_config),
        "denoiser_architecture_config": dataclasses.asdict(
            bundle.denoiser_architecture_config
        ),
        "pressure_levels": tuple(bundle.task_config.pressure_levels),
    }


def make_run_forward_jitted(bundle: GenCastBundle):
    """Return a JIT'd `run_forward(rng, inputs, targets_template, forcings)`."""

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = bundle.construct_wrapped()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    params, state = bundle.params, bundle.state

    def _apply(rng, inputs, targets_template, forcings):
        out, _ = run_forward.apply(
            params, state, rng, inputs, targets_template, forcings
        )
        return out

    return jax.jit(_apply)


def load_example_batch(
    data_root: os.PathLike | None = None,
    dataset_name: str = DEFAULT_DATASET,
    time_window: slice | None = None,
    load_into_memory: bool = True,
) -> xarray.Dataset:
    """Load an ERA5 example trajectory.

    If `time_window` is given (an integer slice along the `time` dim), the
    file is opened lazily and only that time slice is materialised. This
    is critical for large files (e.g. the 2022 Feb-Mar file, 118 steps,
    ~2.5 GB on disk): the comparison/eval scripts only need ~K+2 steps,
    so loading the full file blows up RAM.

    `load_into_memory=True` preserves the historical behavior (`ds.load()`).
    Set `False` to keep the dataset lazy and avoid upfront RAM spikes when
    many files are opened for training.
    """
    root = Path(data_root) if data_root else default_data_root()
    path = root / "dataset" / dataset_name
    if not path.exists():
        raise FileNotFoundError(
            f"Example dataset not found at {path}. "
            f"Run gencast_da/download_assets.sh first."
        )
    # Load by path (not file handle) so xarray can use the .nc extension to
    # pick an engine that supports both NetCDF3-classic (the 2019 demo file)
    # and HDF5/NetCDF4 (files written by `build_mini_era5_nc`). Try a few
    # engines explicitly because the env may not have netcdf4 installed.
    last_err: Exception | None = None
    for engine in (None, "h5netcdf", "netcdf4", "scipy"):
        try:
            ds = xarray.open_dataset(
                str(path),
                engine=engine,
                cache=bool(load_into_memory),
            )
            if time_window is not None:
                ds = ds.isel(time=time_window)
            if load_into_memory:
                return ds.load()
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"Failed to load {path} with any NetCDF engine; last error: {last_err}"
    )


def load_example_batches(
    dataset_names,
    data_root: os.PathLike | None = None,
    load_into_memory: bool = True,
):
    """Load multiple ERA5 example trajectories.

    Returns a list of (name, xarray.Dataset) preserving input order.
    By default, each dataset is fully loaded into memory via
    `load_example_batch(..., load_into_memory=True)` to preserve historical
    behavior. For large multi-file training runs, pass `load_into_memory=False`
    to keep files lazy and reduce host RAM usage.

    Sizes/coords (lat, lon, level, batch) must be identical across files;
    only `time` length may differ.
    """
    out = []
    reference = None
    for name in dataset_names:
        ds = load_example_batch(
            data_root=data_root,
            dataset_name=name,
            load_into_memory=load_into_memory,
        )
        if reference is None:
            reference = ds
        else:
            for coordinate in ("lat", "lon", "level", "batch"):
                if not np.array_equal(
                    ds.coords[coordinate].values,
                    reference.coords[coordinate].values,
                ):
                    raise ValueError(f"dataset {name!r} has incompatible {coordinate}")
        out.append((name, ds))
    return out
