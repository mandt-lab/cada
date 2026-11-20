"""Lazy optional experiment tracking with a no-op implementation."""

from __future__ import annotations

import math
import numbers
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .runtime import resolved_config_dict

TrackingMode = Literal["disabled", "offline", "online"]
_WANDB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _scalar(value: Any, location: str) -> str | bool | int | float | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} contains a non-finite scalar")
        return result
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(f"{location} tensor must be scalar")
        return _scalar(value.detach().cpu().item(), location)
    raise TypeError(
        f"{location} contains non-scalar value of type {type(value).__name__}"
    )


def flatten_resolved_config(config: Any) -> dict[str, Any]:
    """Flatten a resolved mapping/dataclass into JSON-safe scalar leaves."""

    clean = resolved_config_dict(config)
    flattened: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                if not prefix:
                    raise ValueError("resolved config may not be empty")
                flattened[prefix] = "{}"
                return
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else key
                visit(item, child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if not value:
                flattened[prefix] = "[]"
                return
            for index, item in enumerate(value):
                child = f"{prefix}.{index}" if prefix else str(index)
                visit(item, child)
            return
        if not prefix:
            raise ValueError("flat config values require a non-empty key")
        if prefix in flattened:
            raise ValueError(f"flat config key collision at {prefix!r}")
        flattened[prefix] = _scalar(value, f"config.{prefix}")

    visit(clean, "")
    return flattened


def flatten_scalars(values: Any) -> dict[str, Any]:
    """Flatten a metrics mapping/dataclass and reject non-scalar leaves."""

    flattened: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                child = f"{prefix}.{field.name}" if prefix else field.name
                visit(getattr(value, field.name), child)
            return
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"{prefix or 'metrics'} may not be empty")
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError("metric mapping keys must be non-empty strings")
                child = f"{prefix}.{key}" if prefix else key
                visit(item, child)
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            raise TypeError(f"metric {prefix!r} must be scalar")
        if not prefix:
            raise TypeError("metrics must be a mapping or dataclass")
        if prefix in flattened:
            raise ValueError(f"metric key collision at {prefix!r}")
        flattened[prefix] = _scalar(value, f"metrics.{prefix}")

    visit(values, "")
    return flattened


def _read_wandb_id(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("persisted W&B ID must be a regular file")
    run_id = path.read_text(encoding="utf-8").strip()
    if _WANDB_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("persisted W&B ID is invalid")
    return run_id


def load_or_create_wandb_id(
    wandb_dir: str | os.PathLike[str],
) -> str:
    """Load or atomically create the stable W&B ID for a run directory."""

    directory = Path(wandb_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("W&B directory may not be a symbolic link")
    directory = directory.resolve()
    id_path = directory / "run_id.txt"
    if id_path.exists() or id_path.is_symlink():
        return _read_wandb_id(id_path)

    candidate = uuid.uuid4().hex
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(id_path, flags, 0o644)
    except FileExistsError:
        return _read_wandb_id(id_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(candidate + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return candidate


class Tracker(Protocol):
    mode: TrackingMode
    run_id: str | None

    def log(self, metrics: Any, *, step: int | None = None) -> None: ...

    def finish(self) -> None: ...

    def __call__(self, metrics: Any) -> None: ...


@dataclass
class NoOpTracker:
    """Drop-in tracker used when tracking is disabled."""

    mode: TrackingMode = "disabled"
    run_id: str | None = None

    def log(self, metrics: Any, *, step: int | None = None) -> None:
        del metrics, step

    def __call__(self, metrics: Any) -> None:
        self.log(metrics)

    def finish(self) -> None:
        pass


class WandbTracker:
    """Small wrapper that exposes only scalar logging."""

    def __init__(
        self,
        run: Any,
        *,
        mode: Literal["offline", "online"],
        run_id: str,
    ) -> None:
        self._run = run
        self.mode: TrackingMode = mode
        self.run_id = run_id

    def log(self, metrics: Any, *, step: int | None = None) -> None:
        payload = flatten_scalars(metrics)
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("tracking step must be a non-negative integer")
        self._run.log(payload, step=step)

    def __call__(self, metrics: Any) -> None:
        payload = flatten_scalars(metrics)
        step_value = payload.get("step")
        step = (
            int(step_value)
            if isinstance(step_value, int) and not isinstance(step_value, bool)
            else None
        )
        self._run.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


def init_tracker(
    layout: Any,
    resolved_config: Any,
    *,
    mode: TrackingMode = "disabled",
    tags: Sequence[str] = (),
    run_name: str | None = None,
) -> Tracker:
    """Initialize W&B for a RunLayout, or return a lazy no-op tracker."""

    if mode not in ("disabled", "offline", "online"):
        raise ValueError("tracking mode must be disabled, offline, or online")
    if mode == "disabled":
        return NoOpTracker()

    required_attributes = (
        "wandb",
        "dataset",
        "stage",
        "experiment_id",
        "run_id",
    )
    missing = [
        name for name in required_attributes if not hasattr(layout, name)
    ]
    if missing:
        raise TypeError(f"layout is missing attributes: {', '.join(missing)}")
    if not all(isinstance(tag, str) and tag for tag in tags):
        raise ValueError("tracking tags must be non-empty strings")

    wandb_dir = Path(layout.wandb).expanduser()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = wandb_dir.resolve()
    os.environ["WANDB_DIR"] = str(wandb_dir)
    run_id = load_or_create_wandb_id(wandb_dir)
    flat_config = flatten_resolved_config(resolved_config)

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "tracking mode requires the optional 'wandb' package"
        ) from error

    dataset = str(layout.dataset)
    stage = str(layout.stage)
    effective_tags = list(dict.fromkeys((dataset, stage, *tags)))
    run = wandb.init(
        project="cada",
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(wandb_dir),
        group=f"{dataset}/{stage}",
        tags=effective_tags,
        name=run_name or f"{layout.experiment_id}/{layout.run_id}",
        config=flat_config,
    )
    if run is None:
        raise RuntimeError("wandb.init did not return a run")
    return WandbTracker(run, mode=mode, run_id=run_id)


__all__ = [
    "NoOpTracker",
    "Tracker",
    "TrackingMode",
    "WandbTracker",
    "flatten_resolved_config",
    "flatten_scalars",
    "init_tracker",
    "load_or_create_wandb_id",
]
