"""Command-line orchestration for CADA data, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    ControllerExperimentConfig,
    DatasetConfig,
    EvaluationExperimentConfig,
    ModelConfig,
    PriorExperimentConfig,
    load_config,
    to_resolved_dict,
)

_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{12}$")
_KS_TRAIN_SAMPLES = 1_000
_KS_VALID_SAMPLES = 100
_KS_TEST_SAMPLES = 100
_CHECKPOINT_EVERY = 10_000
_DEFAULT_TTO_STEPS = 50
_DEFAULT_TTO_LR = 1e-2
_FIGURE_ENVIRONMENT_VARIABLE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


def build_trajectory_source(
    config: DatasetConfig,
    *,
    split: str,
) -> Any:
    """Build the configured lazy grouped-KS or dense-HDF5 source."""

    from .data import DenseH5TrajectoryDataset, KSH5TrajectoryDataset

    if split not in {"train", "holdout"}:
        raise ValueError("split must be 'train' or 'holdout'")
    path = (
        config.train_data_path
        if split == "train"
        else config.holdout_data_path
    )
    split_name = (
        config.train_split if split == "train" else config.holdout_split
    )
    if config.domain == "ks":
        return KSH5TrajectoryDataset(
            path,
            config.normalization.statistics_path,
            split=split_name,
        )
    return DenseH5TrajectoryDataset(
        path,
        config.normalization.statistics_path,
        key=config.hdf5_key or "x",
    )


def build_prior_model(config: ModelConfig) -> Any:
    """Build the configured 1D or 2D prior U-Net."""

    from .models import UNet1D, UNet2D

    model_type = UNet1D if config.spatial_dims == 1 else UNet2D
    return model_type(
        config.physical_channels,
        base_width=config.base_width,
        dim_multipliers=config.channel_multipliers,
        sinusoidal_embedding_dim=config.learned_sinusoidal_embedding_dim,
    )


def build_diffusion(config: ModelConfig, model: Any | None = None) -> Any:
    """Build the pred-v diffusion wrapper for a validated model config."""

    from .diffusion import GaussianDiffusion

    return GaussianDiffusion(
        model or build_prior_model(config),
        timesteps=config.train_diffusion_steps,
        sampling_steps=config.sampling_steps,
        auto_normalize=True,
    )


def build_controller(
    config: ControllerExperimentConfig | EvaluationExperimentConfig,
) -> Any:
    """Build the configured FiLM controller."""

    from .control import ControlNet1D, ControlNet2D

    spatial_dims = config.prior_model.spatial_dims
    controller_type = ControlNet1D if spatial_dims == 1 else ControlNet2D
    return controller_type(
        config.dataset.physical_channels,
        hidden_width=config.controller_model.hidden_width,
    )


def _validate_observation_runtime(config: Any) -> None:
    observation = config.observation
    if observation.observation_noise_std != 0:
        raise ValueError(
            "current CADA observation operators require "
            "observation_noise_std=0"
        )


def build_observation_components(config: Any) -> tuple[Any, Any]:
    """Build the spatial operator and independent temporal cadence."""

    from .observations import TemporalCadence, make_observation_operator

    _validate_observation_runtime(config)
    observation = config.observation
    operator = make_observation_operator(
        observation.family,
        spatial_dims=len(config.dataset.resolution),
        factor=observation.spatial_factor,
        mask_phase=observation.mask_phase or 0,
    )
    cadence = TemporalCadence(
        interval=observation.temporal_cadence_steps,
        start=observation.temporal_cadence_steps,
    )
    return operator, cadence


def build_dataloader(
    config: PriorExperimentConfig | ControllerExperimentConfig,
    source: Any,
    *,
    workers: int,
    device: str,
) -> Any:
    """Build a shuffled loader with explicit order and worker seed streams."""

    from torch.utils.data import DataLoader

    from .data import (
        OneStepTransitionDataset,
        PreviewWindowDataset,
        ResumableRandomSampler,
    )
    from .runtime import make_torch_generator, make_worker_init_fn

    if workers < 0:
        raise ValueError("workers must be non-negative")
    if isinstance(config, PriorExperimentConfig):
        dataset = OneStepTransitionDataset(source)
    else:
        cadence = config.observation.temporal_cadence_steps
        dataset = PreviewWindowDataset(
            source,
            horizon=config.preview_length,
            stride=cadence,
        )
    sampler = ResumableRandomSampler(
        dataset,
        make_torch_generator(config.seeds.data_order),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        generator=make_torch_generator(
            (config.seeds.data_order + 1) % (2**63)
        ),
        worker_init_fn=make_worker_init_fn(config.seeds.data_order),
    )


def load_prior_weights(
    diffusion: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a prior checkpoint, preferring its EMA model parameters."""

    from .checkpoint import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state = checkpoint["trainer_state"]
    if "ema_model" in state:
        diffusion.model.load_state_dict(state["ema_model"], strict=True)
    elif "diffusion" in state:
        diffusion.load_state_dict(state["diffusion"], strict=True)
    else:
        raise ValueError("prior checkpoint contains neither EMA nor diffusion state")
    return checkpoint


def load_controller_weights(
    diffusion: Any,
    controller: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
    restore_prior: bool = True,
) -> dict[str, Any]:
    """Load the controller and, by default, its frozen prior state."""

    from .checkpoint import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state = checkpoint["trainer_state"]
    if "controller" not in state:
        raise ValueError("controller checkpoint is missing controller state")
    if restore_prior:
        if "diffusion" not in state:
            raise ValueError("controller checkpoint is missing prior diffusion state")
        diffusion.load_state_dict(state["diffusion"], strict=True)
    controller.load_state_dict(state["controller"], strict=True)
    return checkpoint


def validate_controller_prior(
    controller_checkpoint: Mapping[str, Any],
    prior_checkpoint_path: str | os.PathLike[str],
) -> None:
    """Require evaluation to use the prior recorded during controller training."""

    from .artifacts import sha256_file

    expected = controller_checkpoint["metadata"].get("prior_checkpoint_sha256")
    if expected is None:
        raise ValueError("controller checkpoint does not identify its frozen prior")
    if expected != sha256_file(prior_checkpoint_path):
        raise ValueError("controller checkpoint was trained with a different prior")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML config or manifest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved action without side effects",
    )
    parser.add_argument("--root", help="override CADA_ROOT")
    parser.add_argument("--device", default="cuda", help="runtime torch device")
    parser.add_argument("--run-id", help="canonical run ID override")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the latest matching run",
    )
    parser.add_argument(
        "--tracking",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="experiment tracking mode",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cada")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="inspect or prepare data")
    _add_common_arguments(data)
    data.add_argument(
        "--action",
        choices=("inspect", "generate", "request", "convert", "stats"),
        default="inspect",
    )
    data.add_argument("--input", nargs="+", help="ERA5 netCDF input path(s)")
    data.add_argument("--year", type=int, action="append")
    data.add_argument("--overwrite", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run bounded synthetic plumbing")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--dry-run", action="store_true")

    prior = subparsers.add_parser("train-prior", help="train a diffusion prior")
    _add_common_arguments(prior)
    prior.add_argument("--max-steps", type=int)
    prior.add_argument("--checkpoint", help="final checkpoint path override")
    prior.add_argument("--resume-checkpoint", help="specific resume checkpoint")
    prior.add_argument("--workers", type=int, default=0)
    prior.add_argument("--overwrite", action="store_true")

    controller = subparsers.add_parser(
        "train-controller", help="train a CADA controller"
    )
    _add_common_arguments(controller)
    controller.add_argument("--max-steps", type=int)
    controller.add_argument("--prior-checkpoint")
    controller.add_argument("--checkpoint", help="final checkpoint path override")
    controller.add_argument("--resume-checkpoint", help="specific resume checkpoint")
    controller.add_argument("--workers", type=int, default=0)
    controller.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run CADA evaluation")
    _add_common_arguments(evaluate)
    evaluate.add_argument("--prior-checkpoint")
    evaluate.add_argument("--controller-checkpoint")
    evaluate.add_argument(
        "--method",
        choices=("cada", "tto-da"),
        default="cada",
    )
    evaluate.add_argument("--tto-steps", type=int, default=_DEFAULT_TTO_STEPS)
    evaluate.add_argument("--tto-lr", type=float, default=_DEFAULT_TTO_LR)
    evaluate.add_argument("--overwrite", action="store_true")

    figure = subparsers.add_parser("figure", help="render a saved artifact")
    _add_common_arguments(figure)
    figure.add_argument("--overwrite", action="store_true")
    return parser


def _root(args: argparse.Namespace) -> Path:
    value = args.root or os.environ.get("CADA_ROOT")
    if not value:
        raise ConfigError("CADA_ROOT is not set; export it or pass --root")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ConfigError("CADA_ROOT must resolve to an absolute path")
    return root.resolve(strict=False)


def _load_config(args: argparse.Namespace) -> Any:
    return load_config(args.config, env={"CADA_ROOT": str(_root(args))})


def _require_type(config: Any, expected: type[Any], command: str) -> None:
    if not isinstance(config, expected):
        raise ConfigError(
            f"{command} requires a {expected.__name__}, got "
            f"{type(config).__name__}"
        )


def _bounded_steps(requested: int | None, configured: int) -> int:
    if requested is None:
        return configured
    if requested < 1 or requested > configured:
        raise ValueError(
            f"--max-steps must be between 1 and configured limit {configured}"
        )
    return requested


def _run_path(
    root: Path,
    config: Any,
    run_id: str | None,
    *,
    resolved_identity: Any | None = None,
    experiment_id: str | None = None,
) -> Path:
    from .runtime import (
        new_run_id,
        resolved_config_sha256,
        sanitize_experiment_id,
    )

    digest = resolved_config_sha256(
        config if resolved_identity is None else resolved_identity
    )
    if run_id is not None:
        if (
            _RUN_ID_PATTERN.fullmatch(run_id) is None
            or not run_id.endswith(f"-{digest[:12]}")
        ):
            raise ValueError(
                "--run-id must be canonical and match the resolved config"
            )
        run_component = run_id
    else:
        run_component = new_run_id(digest)
    return (
        root
        / "runs"
        / sanitize_experiment_id(config.dataset.domain)
        / sanitize_experiment_id(config.kind)
        / sanitize_experiment_id(experiment_id or config.config_id)
        / run_component
    )


def _base_plan(
    args: argparse.Namespace,
    config: Any,
    *,
    action: str,
    paths: dict[str, Any],
) -> dict[str, Any]:
    plan = {
        "action": action,
        "config_path": str(Path(args.config).expanduser().resolve(strict=False)),
        "resolved_config": to_resolved_dict(config),
        "root": str(_root(args)),
        "device": args.device,
        "tracking": args.tracking,
        "resume": bool(args.resume),
        "run_id": args.run_id,
        "paths": paths,
    }
    return plan


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _json_dataclass(value: Any) -> dict[str, Any]:
    result = asdict(value)
    for key, item in tuple(result.items()):
        if isinstance(item, Path):
            result[key] = str(item)
    return result


def _dataset_paths(config: DatasetConfig) -> dict[str, str]:
    return {
        "train_data": config.train_data_path,
        "holdout_data": config.holdout_data_path,
        "normalization": config.normalization.statistics_path,
    }


def _training_paths(
    args: argparse.Namespace,
    config: PriorExperimentConfig | ControllerExperimentConfig,
    *,
    final_checkpoint: str,
) -> dict[str, Any]:
    run = _run_path(_root(args), config, args.run_id)
    paths: dict[str, Any] = {
        **_dataset_paths(config.dataset),
        "run": str(run),
        "checkpoints": str(run / "checkpoints"),
        "wandb": str(run / "wandb"),
        "logs": str(run / "logs"),
        "final_checkpoint": final_checkpoint,
        "resume_checkpoint": (
            args.resume_checkpoint
            if args.resume_checkpoint
            else str(run / "checkpoints" / "latest.pt")
        ),
    }
    if isinstance(config, ControllerExperimentConfig):
        paths["prior_checkpoint"] = (
            args.prior_checkpoint or config.prior_checkpoint_path
        )
    return paths


def _validate_data_action(
    args: argparse.Namespace,
    config: DatasetConfig,
) -> None:
    action = args.action
    if action == "generate" and config.domain != "ks":
        raise ValueError("data --action generate is supported only for KS")
    if action in {"request", "convert"} and config.domain != "era5_small":
        raise ValueError(f"data --action {action} is supported only for ERA5")
    if action == "stats" and config.domain == "ks":
        raise ValueError(
            "KS statistics are generated with data --action generate; "
            "--action stats is for dense data"
        )
    if action == "convert" and not args.input:
        raise ValueError("ERA5 conversion requires --input")
    if action != "convert" and args.input:
        raise ValueError("--input is valid only with data --action convert")
    if args.year and action != "request":
        raise ValueError("--year is valid only with data --action request")
    if args.resume or args.run_id or args.tracking != "disabled":
        raise ValueError("data does not use --resume, --run-id, or tracking")


def _data_plan(
    args: argparse.Namespace,
    config: DatasetConfig,
) -> dict[str, Any]:
    paths: dict[str, Any] = _dataset_paths(config)
    if args.input:
        paths["inputs"] = [
            str(Path(item).expanduser().resolve(strict=False))
            for item in args.input
        ]
    plan = _base_plan(
        args,
        config,
        action=f"data:{args.action}",
        paths=paths,
    )
    if config.domain == "ks" and args.action == "generate":
        plan["ks_split_sizes"] = {
            "train": _KS_TRAIN_SAMPLES,
            "valid": _KS_VALID_SAMPLES,
            "test": _KS_TEST_SAMPLES,
        }
    if args.year:
        plan["years"] = args.year
    plan["overwrite"] = bool(args.overwrite)
    return plan


def _inspect_data(config: DatasetConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "holdout"):
        source = build_trajectory_source(config, split=split)
        try:
            result[split] = {
                "path": str(source.file),
                "dataset": source.dataset_key,
                "shape": list(source.shape),
                "normalization_channels": source.normalizer.channels,
            }
        finally:
            source.close()
    return result


def _require_outputs_available(
    paths: Sequence[str | os.PathLike[str]],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "output already exists; pass --overwrite to replace: "
            + ", ".join(existing)
        )


def _run_data(args: argparse.Namespace) -> int:
    config = _load_config(args)
    _require_type(config, DatasetConfig, "data")
    _validate_data_action(args, config)
    if args.dry_run:
        _print_json(_data_plan(args, config))
        return 0

    from .preprocessing import (
        ERA5Config,
        build_era5_cds_request,
        compute_dense_h5_channel_stats,
        compute_ks_training_minmax,
        convert_era5_netcdf_to_hdf5,
        generate_ks_hdf5,
        make_ks_splits,
        write_dense_channel_stats,
        write_ks_minmax_json,
    )

    if args.action == "inspect":
        _print_json({"action": "data:inspect", "datasets": _inspect_data(config)})
        return 0
    if args.action == "request":
        era5 = ERA5Config()
        years = tuple(args.year or config.train_years + config.holdout_years)
        requests = {
            str(year): build_era5_cds_request(year, era5) for year in years
        }
        _print_json(requests)
        return 0
    if args.action == "generate":
        outputs = (
            config.train_data_path,
            config.holdout_data_path,
            config.normalization.statistics_path,
        )
        _require_outputs_available(outputs, overwrite=args.overwrite)
        train, _valid, test = make_ks_splits(
            _KS_TRAIN_SAMPLES,
            _KS_VALID_SAMPLES,
            _KS_TEST_SAMPLES,
        )
        train_result = generate_ks_hdf5(
            config.train_data_path, train, overwrite=args.overwrite
        )
        holdout_result = generate_ks_hdf5(
            config.holdout_data_path, test, overwrite=args.overwrite
        )
        stats = compute_ks_training_minmax(config.train_data_path)
        stats_path = write_ks_minmax_json(
            stats,
            config.normalization.statistics_path,
            overwrite=args.overwrite,
        )
        _print_json(
            {
                "action": "data:generate",
                "train": _json_dataclass(train_result),
                "holdout": _json_dataclass(holdout_result),
                "normalization": str(stats_path),
                "datasets": _inspect_data(config),
            }
        )
        return 0
    if args.action == "convert":
        outputs = (
            config.train_data_path,
            config.holdout_data_path,
            config.normalization.statistics_path,
        )
        _require_outputs_available(outputs, overwrite=args.overwrite)
        result = convert_era5_netcdf_to_hdf5(
            args.input,
            config.train_data_path,
            config.holdout_data_path,
            config.normalization.statistics_path,
            ERA5Config(),
            overwrite=args.overwrite,
        )
        _print_json(
            {
                "action": "data:convert",
                "train": _json_dataclass(result.train),
                "holdout": _json_dataclass(result.holdout),
                "normalization": str(result.stats_path),
                "datasets": _inspect_data(config),
            }
        )
        return 0
    if args.action == "stats":
        output = config.normalization.statistics_path
        _require_outputs_available((output,), overwrite=args.overwrite)
        stats = compute_dense_h5_channel_stats(
            config.train_data_path,
            dataset=config.hdf5_key or "x",
        )
        path = write_dense_channel_stats(
            stats, output, overwrite=args.overwrite
        )
        _print_json(
            {
                "action": "data:stats",
                "statistics": str(path),
                "minimum": list(stats.minimum),
                "maximum": list(stats.maximum),
                "datasets": _inspect_data(config),
            }
        )
        return 0
    raise AssertionError("unreachable data action")


def _run_smoke(args: argparse.Namespace) -> int:
    if args.dry_run:
        _print_json(
            {
                "action": "smoke",
                "device": args.device,
                "seed": args.seed,
            }
        )
        return 0
    from .smoke import run_smoke

    _print_json(run_smoke(_torch_device(args.device), seed=args.seed))
    return 0


def _torch_device(value: str) -> Any:
    import torch

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def _atomic_copy(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> Path:
    if source.resolve() == destination.expanduser().resolve(strict=False):
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("checkpoint parent may not be a symbolic link")
    if os.path.lexists(destination) and not overwrite:
        raise FileExistsError(
            f"{destination} exists; pass --overwrite to replace it"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _create_layout(args: argparse.Namespace, config: Any) -> Any:
    from .runtime import RunLayout

    return RunLayout.from_config(
        config,
        root=_root(args),
        resume_existing=args.resume,
        run_id=args.run_id,
        repository=Path(__file__).resolve().parents[2],
    )


def _resume_trainer(
    args: argparse.Namespace,
    trainer: Any,
    layout: Any,
    resolved_config: Any,
) -> Path | None:
    if not args.resume:
        if args.resume_checkpoint:
            raise ValueError("--resume-checkpoint requires --resume")
        return None
    from .checkpoint import (
        discover_latest_checkpoint,
        load_checkpoint,
        restore_trainer_checkpoint,
    )
    from .runtime import resolved_config_dict

    path = (
        Path(args.resume_checkpoint).expanduser()
        if args.resume_checkpoint
        else discover_latest_checkpoint(layout.checkpoints)
    )
    if path is None:
        raise FileNotFoundError(
            f"no checkpoint found in {layout.checkpoints}"
        )
    checkpoint = load_checkpoint(path, map_location=trainer.device)
    if checkpoint["resolved_config"] != resolved_config_dict(resolved_config):
        raise ValueError("resume checkpoint configuration does not match")
    trainer_step = checkpoint["trainer_state"].get("step")
    if trainer_step != checkpoint["step"]:
        raise ValueError("checkpoint step disagrees with trainer state")
    restore_trainer_checkpoint(
        trainer,
        checkpoint,
        restore_rng=True,
        restore_cuda_rng=trainer.device.type == "cuda",
    )
    return Path(path)


def _fit_with_checkpoints(
    trainer: Any,
    *,
    max_steps: int,
    adapter: Any,
) -> tuple[Any | None, bool]:
    from .checkpoint import CheckpointSignalHandler

    last_metrics = None
    signaled = False
    with CheckpointSignalHandler() as signal_handler:
        adapter.signal_handler = signal_handler
        while trainer.step < max_steps:
            last_metrics = trainer.train_step()
            if adapter.save_if_requested(
                trainer.step, trainer.state_dict(), last_metrics
            ):
                signaled = True
                break
    if not signaled:
        adapter(trainer.step, trainer.state_dict(), last_metrics)
    return last_metrics, signaled


def _run_train_prior(args: argparse.Namespace) -> int:
    config = _load_config(args)
    _require_type(config, PriorExperimentConfig, "train-prior")
    max_steps = _bounded_steps(args.max_steps, config.train_steps)
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.workers:
        raise ValueError("exact checkpointing requires --workers 0")
    if args.resume_checkpoint and not args.resume:
        raise ValueError("--resume-checkpoint requires --resume")
    final_checkpoint = args.checkpoint or config.checkpoint_path
    if args.dry_run:
        plan = _base_plan(
            args,
            config,
            action="train-prior",
            paths=_training_paths(
                args,
                config,
                final_checkpoint=final_checkpoint,
            ),
        )
        plan["max_steps"] = max_steps
        plan["workers"] = args.workers
        plan["overwrite"] = bool(args.overwrite)
        _print_json(plan)
        return 0

    from .checkpoint import CheckpointCallbackAdapter
    from .runtime import seed_everything
    from .tracking import init_tracker
    from .training import PriorTrainer

    _require_outputs_available(
        (final_checkpoint,), overwrite=args.overwrite or args.resume
    )
    device = _torch_device(args.device)
    layout = _create_layout(args, config)
    with ExitStack() as stack:
        source = build_trajectory_source(config.dataset, split="train")
        stack.callback(source.close)
        tracker = init_tracker(
            layout,
            config,
            mode=args.tracking,
            tags=("prior", config.dataset.domain),
        )
        stack.callback(tracker.finish)
        seed_everything(config.seeds.training)
        dataloader = build_dataloader(
            config,
            source,
            workers=args.workers,
            device=str(device),
        )
        diffusion = build_diffusion(config.model)
        adapter = CheckpointCallbackAdapter(
            layout.checkpoints,
            config,
            metadata={"stage": "prior"},
            wandb_run_id=tracker.run_id,
        )
        trainer = PriorTrainer(
            diffusion,
            dataloader,
            learning_rate=config.optimizer.learning_rate,
            device=device,
            precision=config.precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.optimizer.weight_decay,
            ema_decay=config.ema_decay,
            ema_update_every=config.ema_update_every,
            log_callback=tracker,
            checkpoint_callback=adapter,
            log_every=1,
            checkpoint_every=_CHECKPOINT_EVERY,
        )
        resumed_from = _resume_trainer(args, trainer, layout, config)
        _, signaled = _fit_with_checkpoints(
            trainer, max_steps=max_steps, adapter=adapter
        )
        if adapter.last_path is None:
            raise RuntimeError("training did not produce a checkpoint")
        alias = None
        if not signaled:
            alias = _atomic_copy(
                adapter.last_path,
                Path(final_checkpoint).expanduser(),
                overwrite=args.overwrite or args.resume,
            )
        _print_json(
            {
                "action": "train-prior",
                "run": str(layout.run_dir),
                "step": trainer.step,
                "checkpoint": str(adapter.last_path),
                "final_checkpoint": str(alias) if alias else None,
                "resumed_from": str(resumed_from) if resumed_from else None,
                "signal_exit": signaled,
            }
        )
    return 75 if signaled else 0


def _run_train_controller(args: argparse.Namespace) -> int:
    config = _load_config(args)
    _require_type(config, ControllerExperimentConfig, "train-controller")
    _validate_observation_runtime(config)
    max_steps = _bounded_steps(args.max_steps, config.train_steps)
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.workers:
        raise ValueError("exact checkpointing requires --workers 0")
    if args.resume_checkpoint and not args.resume:
        raise ValueError("--resume-checkpoint requires --resume")
    if args.resume and args.prior_checkpoint:
        raise ValueError(
            "--prior-checkpoint cannot override the prior in a resumed "
            "controller checkpoint"
        )
    final_checkpoint = args.checkpoint or config.checkpoint_path
    prior_checkpoint = args.prior_checkpoint or config.prior_checkpoint_path
    if args.dry_run:
        plan = _base_plan(
            args,
            config,
            action="train-controller",
            paths=_training_paths(
                args,
                config,
                final_checkpoint=final_checkpoint,
            ),
        )
        plan["paths"]["prior_checkpoint"] = prior_checkpoint
        plan["max_steps"] = max_steps
        plan["workers"] = args.workers
        plan["overwrite"] = bool(args.overwrite)
        _print_json(plan)
        return 0

    from .artifacts import sha256_file
    from .checkpoint import CheckpointCallbackAdapter
    from .runtime import seed_everything
    from .tracking import init_tracker
    from .training import ControllerTrainer

    _require_outputs_available(
        (final_checkpoint,), overwrite=args.overwrite or args.resume
    )
    device = _torch_device(args.device)
    layout = _create_layout(args, config)
    with ExitStack() as stack:
        source = build_trajectory_source(config.dataset, split="train")
        stack.callback(source.close)
        tracker = init_tracker(
            layout,
            config,
            mode=args.tracking,
            tags=("controller", config.dataset.domain),
        )
        stack.callback(tracker.finish)
        seed_everything(config.seeds.training)
        dataloader = build_dataloader(
            config,
            source,
            workers=args.workers,
            device=str(device),
        )
        diffusion = build_diffusion(config.prior_model)
        controller = build_controller(config)
        operator, cadence = build_observation_components(config)
        if not args.resume:
            load_prior_weights(
                diffusion, prior_checkpoint, map_location=device
            )
        adapter = CheckpointCallbackAdapter(
            layout.checkpoints,
            config,
            metadata={
                "stage": "controller",
                "prior_checkpoint": str(prior_checkpoint),
                "prior_checkpoint_sha256": sha256_file(prior_checkpoint),
            },
            wandb_run_id=tracker.run_id,
        )
        trainer = ControllerTrainer(
            diffusion,
            controller,
            dataloader,
            operator,
            cadence,
            preview_steps=config.preview_length,
            gamma=config.gamma,
            beta=config.beta,
            learning_rate=config.optimizer.learning_rate,
            intermediate_weight=config.intermediate_weight,
            sampling_steps=config.prior_model.sampling_steps,
            device=device,
            precision=config.precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            weight_decay=config.optimizer.weight_decay,
            log_callback=tracker,
            checkpoint_callback=adapter,
            log_every=1,
            checkpoint_every=_CHECKPOINT_EVERY,
        )
        resumed_from = _resume_trainer(args, trainer, layout, config)
        _, signaled = _fit_with_checkpoints(
            trainer, max_steps=max_steps, adapter=adapter
        )
        if adapter.last_path is None:
            raise RuntimeError("training did not produce a checkpoint")
        alias = None
        if not signaled:
            alias = _atomic_copy(
                adapter.last_path,
                Path(final_checkpoint).expanduser(),
                overwrite=args.overwrite or args.resume,
            )
        _print_json(
            {
                "action": "train-controller",
                "run": str(layout.run_dir),
                "step": trainer.step,
                "checkpoint": str(adapter.last_path),
                "final_checkpoint": str(alias) if alias else None,
                "resumed_from": str(resumed_from) if resumed_from else None,
                "signal_exit": signaled,
            }
        )
    return 75 if signaled else 0


def _evaluation_method_spec(
    args: argparse.Namespace,
    config: EvaluationExperimentConfig,
) -> dict[str, Any]:
    if args.method == "tto-da":
        if (
            isinstance(args.tto_steps, bool)
            or not isinstance(args.tto_steps, int)
            or args.tto_steps < 1
        ):
            raise ValueError("--tto-steps must be a positive integer")
        if not math.isfinite(args.tto_lr) or args.tto_lr <= 0:
            raise ValueError("--tto-lr must be finite and positive")
        return {
            "name": "tto-da",
            "iterations": args.tto_steps,
            "lr": args.tto_lr,
            "gamma": config.gamma,
            "beta": config.beta,
            "preview_steps": config.preview_length,
            "seed": config.seeds.diffusion,
        }
    return {
        "name": "cada",
        "gamma": config.gamma,
        "beta": config.beta,
        "preview_steps": config.preview_length,
        "seed": config.seeds.diffusion,
    }


def _evaluation_identity(
    config: EvaluationExperimentConfig,
    method_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "base_config": to_resolved_dict(config),
        "evaluation_method": method_spec,
    }


def _evaluation_experiment_id(
    config: EvaluationExperimentConfig,
    method_spec: dict[str, Any],
) -> str:
    return f"{config.config_id}--{method_spec['name']}"


def _evaluation_paths(
    args: argparse.Namespace,
    config: EvaluationExperimentConfig,
    method_spec: dict[str, Any],
) -> dict[str, Any]:
    identity = _evaluation_identity(config, method_spec)
    run = _run_path(
        _root(args),
        config,
        args.run_id,
        resolved_identity=identity,
        experiment_id=_evaluation_experiment_id(config, method_spec),
    )
    paths = {
        **_dataset_paths(config.dataset),
        "prior_checkpoint": (
            args.prior_checkpoint or config.prior_checkpoint_path
        ),
        "run": str(run),
        "artifact": str(run / "samples" / "evaluation.npz"),
        "artifact_checksum": str(
            run / "samples" / "evaluation.npz.sha256"
        ),
        "metrics": str(run / "metrics" / "evaluation.json"),
        "metrics_checksum": str(
            run / "metrics" / "evaluation.json.sha256"
        ),
        "wandb": str(run / "wandb"),
    }
    if method_spec["name"] == "cada":
        paths["controller_checkpoint"] = (
            args.controller_checkpoint or config.controller_checkpoint_path
        )
    return paths


def _create_evaluation_layout(
    args: argparse.Namespace,
    config: EvaluationExperimentConfig,
    method_spec: dict[str, Any],
) -> Any:
    from .runtime import RunLayout

    identity = _evaluation_identity(config, method_spec)
    return RunLayout.create(
        dataset=config.dataset.domain,
        stage=config.kind,
        experiment_id=_evaluation_experiment_id(config, method_spec),
        resolved_config=identity,
        root=_root(args),
        resume_existing=args.resume,
        run_id=args.run_id,
        seeds=to_resolved_dict(config).get("seeds"),
        repository=Path(__file__).resolve().parents[2],
        metadata={"evaluation_method": method_spec},
    )


def _run_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    _require_type(config, EvaluationExperimentConfig, "evaluate")
    _validate_observation_runtime(config)
    method_spec = _evaluation_method_spec(args, config)
    prior_checkpoint = args.prior_checkpoint or config.prior_checkpoint_path
    controller_checkpoint = None
    if args.method == "cada":
        controller_checkpoint = (
            args.controller_checkpoint or config.controller_checkpoint_path
        )
    if args.dry_run:
        plan = _base_plan(
            args,
            config,
            action="evaluate",
            paths=_evaluation_paths(args, config, method_spec),
        )
        plan["resolved_config"] = _evaluation_identity(config, method_spec)
        plan["method"] = method_spec
        plan["trajectories"] = config.evaluation_trajectories
        plan["horizon"] = config.horizon.forecast_steps
        plan["overwrite"] = bool(args.overwrite)
        _print_json(plan)
        return 0

    from .artifacts import sha256_file
    from .evaluation import run_evaluation
    from .runtime import seed_everything
    from .tracking import init_tracker

    device = _torch_device(args.device)
    layout = _create_evaluation_layout(args, config, method_spec)
    expected_outputs = (
        layout.samples / "evaluation.npz",
        layout.samples / "evaluation.npz.sha256",
        layout.metrics / "evaluation.json",
        layout.metrics / "evaluation.json.sha256",
    )
    _require_outputs_available(expected_outputs, overwrite=args.overwrite)
    with ExitStack() as stack:
        source = build_trajectory_source(config.dataset, split="holdout")
        stack.callback(source.close)
        tracker = init_tracker(
            layout,
            _evaluation_identity(config, method_spec),
            mode=args.tracking,
            tags=("evaluation", config.dataset.domain),
        )
        stack.callback(tracker.finish)
        seed_everything(config.seeds.diffusion)
        diffusion = build_diffusion(config.prior_model).to(device)
        controller = None
        if args.method == "cada":
            assert controller_checkpoint is not None
            controller = build_controller(config).to(device)
            controller_payload = load_controller_weights(
                diffusion,
                controller,
                controller_checkpoint,
                map_location=device,
                restore_prior=False,
            )
            validate_controller_prior(controller_payload, prior_checkpoint)
        load_prior_weights(
            diffusion, prior_checkpoint, map_location=device
        )
        operator, cadence = build_observation_components(config)
        evaluation_metadata = {
            "prior_checkpoint": str(
                Path(prior_checkpoint).expanduser().resolve()
            ),
            "prior_checkpoint_sha256": sha256_file(prior_checkpoint),
            "checkpoint_ids": [f"prior.{config.dataset.domain}.paper"],
        }
        if controller_checkpoint is not None:
            evaluation_metadata["checkpoint_ids"].append(
                config.controller_config_id
            )
            evaluation_metadata.update(
                {
                    "controller_checkpoint": str(
                        Path(controller_checkpoint).expanduser().resolve()
                    ),
                    "controller_checkpoint_sha256": sha256_file(
                        controller_checkpoint
                    ),
                }
            )
        outputs = run_evaluation(
            config,
            source=source,
            normalizer=source.normalizer,
            diffusion=diffusion,
            controller=controller,
            observation_operator=operator,
            cadence=cadence,
            layout=layout,
            device=device,
            method=args.method,
            tto_iterations=args.tto_steps,
            tto_learning_rate=args.tto_lr,
            metadata=evaluation_metadata,
        )
        _print_json(
            {
                "action": "evaluate",
                "method": method_spec,
                "run": str(layout.run_dir),
                "artifact": str(outputs.artifact),
                "metrics": str(outputs.metrics),
                "checksums": [
                    str(outputs.artifact_checksum),
                    str(outputs.metrics_checksum),
                ],
            }
        )
    return 0


def _figure_manifest_dict(manifest: Any) -> dict[str, Any]:
    result = {
        "figure_type": manifest.figure_type,
        "artifact": str(manifest.artifact),
        "baseline_artifacts": [
            {
                "path": str(reference.path),
                "label": reference.label,
                "field": reference.field,
            }
            for reference in manifest.baseline_artifacts
        ],
        "output": str(manifest.output),
        "trajectory": manifest.trajectory,
        "timesteps": list(manifest.timesteps),
        "config_ids": list(manifest.config_ids),
        "checkpoint_ids": list(manifest.checkpoint_ids),
        "seeds": list(manifest.seeds),
        "observation_details": dict(manifest.observation_details),
        "rendering": dict(manifest.rendering),
    }
    return result


def _dry_figure_manifest(
    path: str | os.PathLike[str],
    *,
    cada_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and resolve a figure manifest without reading its inputs."""

    from .figures import FigureManifest

    return _figure_manifest_dict(
        FigureManifest.from_yaml(path, cada_root=cada_root)
    )

def _run_figure(args: argparse.Namespace) -> int:
    if args.resume or args.run_id or args.tracking != "disabled":
        raise ValueError("figure does not use --resume, --run-id, or tracking")
    if args.dry_run:
        manifest = _dry_figure_manifest(
            args.config, cada_root=_root(args)
        )
        output = Path(manifest["output"])
        input_paths = {
            "artifact": manifest["artifact"],
            "baselines": [
                item["path"]
                for item in manifest["baseline_artifacts"]
            ],
        }
        plan = {
            "action": "figure",
            "config_path": str(
                Path(args.config).expanduser().resolve(strict=False)
            ),
            "resolved_config": manifest,
            "root": str(_root(args)),
            "device": args.device,
            "tracking": args.tracking,
            "resume": False,
            "run_id": None,
            "paths": {
                **input_paths,
                "output": str(output),
                "provenance": str(
                    output.with_suffix(output.suffix + ".provenance.json")
                ),
            },
            "overwrite": bool(args.overwrite),
        }
        _print_json(plan)
        return 0

    from .figures import (
        FigureManifest,
        provenance_sidecar_path,
        render_figure,
    )

    manifest = FigureManifest.from_yaml(
        args.config, cada_root=_root(args)
    )
    input_paths = [
        manifest.artifact,
        *[
            reference.path
            for reference in manifest.baseline_artifacts
        ],
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "manifest input does not exist: " + ", ".join(missing_inputs)
        )
    plan_inputs = {
        "artifact": str(manifest.artifact),
        "baselines": [
            str(reference.path)
            for reference in manifest.baseline_artifacts
        ],
    }
    plan = {
        "action": "figure",
        "config_path": str(
            Path(args.config).expanduser().resolve(strict=False)
        ),
        "resolved_config": _figure_manifest_dict(manifest),
        "root": str(_root(args)),
        "device": args.device,
        "tracking": args.tracking,
        "resume": False,
        "run_id": None,
        "paths": {
            **plan_inputs,
            "output": str(manifest.output),
            "provenance": str(provenance_sidecar_path(manifest.output)),
        },
        "overwrite": bool(args.overwrite),
    }
    _require_outputs_available(
        (
            manifest.output,
            provenance_sidecar_path(manifest.output),
        ),
        overwrite=args.overwrite,
    )
    output, provenance = render_figure(
        manifest,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "action": "figure",
            "output": str(output),
            "provenance": str(provenance),
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one CADA command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "data": _run_data,
        "smoke": _run_smoke,
        "train-prior": _run_train_prior,
        "train-controller": _run_train_controller,
        "evaluate": _run_evaluate,
        "figure": _run_figure,
    }
    try:
        return commands[args.command](args)
    except (
        ConfigError,
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"cada: error: {error}", file=sys.stderr)
        return 1


__all__ = [
    "build_controller",
    "build_dataloader",
    "build_diffusion",
    "build_observation_components",
    "build_parser",
    "build_prior_model",
    "build_trajectory_source",
    "load_controller_weights",
    "load_prior_weights",
    "main",
]
