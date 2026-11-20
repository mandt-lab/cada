"""Atomic, resumable trainer checkpoints and signal checkpoint requests."""

from __future__ import annotations

import os
import random
import re
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from .runtime import resolved_config_dict

CHECKPOINT_SCHEMA_VERSION = 1
_STEP_PATTERN = re.compile(r"^step_(\d{9})\.pt$")
_WANDB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("checkpoint step must be an integer")
    if not 0 <= step <= 999_999_999:
        raise ValueError("checkpoint step must lie in [0, 999999999]")
    return step


def checkpoint_filename(step: int) -> str:
    return f"step_{_validate_step(step):09d}.pt"


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and Torch CUDA RNG state."""

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required to checkpoint RNG state") from error
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to checkpoint RNG state") from error

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_nested_tuple(item) for item in value)
    return value


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    restore_cuda: bool = True,
) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required to restore RNG state") from error
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to restore RNG state") from error

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("NumPy RNG state must be a mapping")
    numpy_keys = numpy_state.get("keys")
    if not isinstance(numpy_keys, Sequence) or isinstance(
        numpy_keys, (str, bytes, bytearray)
    ):
        raise TypeError("NumPy RNG keys must be a sequence")
    random.setstate(_nested_tuple(state["python"]))
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_keys, dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )

    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        raise TypeError("Torch CPU RNG state must be a tensor")
    torch.set_rng_state(cpu_state.cpu())
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, Sequence) or isinstance(
        cuda_states, (str, bytes, bytearray)
    ):
        raise TypeError("Torch CUDA RNG state must be a sequence")
    if restore_cuda and cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG device count does not match this runtime"
            )
        if not all(isinstance(item, torch.Tensor) for item in cuda_states):
            raise TypeError("Torch CUDA RNG states must be tensors")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def build_checkpoint(
    *,
    step: int,
    trainer_state: Mapping[str, Any],
    resolved_config: Any,
    metadata: Mapping[str, Any] | None = None,
    wandb_run_id: str | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete, weights-only-loadable checkpoint payload."""

    step = _validate_step(step)
    if not isinstance(trainer_state, Mapping):
        raise TypeError("trainer_state must be a mapping")
    if wandb_run_id is not None and (
        not isinstance(wandb_run_id, str)
        or _WANDB_ID_PATTERN.fullmatch(wandb_run_id) is None
    ):
        raise ValueError("wandb_run_id contains unsupported characters")
    clean_metadata = resolved_config_dict(dict(metadata or {}))
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "trainer_state": dict(trainer_state),
        "resolved_config": resolved_config_dict(resolved_config),
        "rng_state": dict(rng_state or capture_rng_state()),
        "metadata": clean_metadata,
        "wandb_run_id": wandb_run_id,
    }


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to write checkpoints") from error

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
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
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    directory: str | os.PathLike[str],
    *,
    step: int,
    trainer_state: Mapping[str, Any],
    resolved_config: Any,
    metadata: Mapping[str, Any] | None = None,
    wandb_run_id: str | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a numbered checkpoint and a full ``latest.pt`` copy."""

    checkpoint_dir = Path(directory).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.is_symlink():
        raise ValueError("checkpoint directory may not be a symbolic link")
    checkpoint_dir = checkpoint_dir.resolve()
    payload = build_checkpoint(
        step=step,
        trainer_state=trainer_state,
        resolved_config=resolved_config,
        metadata=metadata,
        wandb_run_id=wandb_run_id,
        rng_state=rng_state,
    )
    numbered = checkpoint_dir / checkpoint_filename(step)
    _atomic_torch_save(payload, numbered)
    _atomic_copy(numbered, checkpoint_dir / "latest.pt")
    return numbered


def discover_latest_checkpoint(
    directory: str | os.PathLike[str],
) -> Path | None:
    """Find ``latest.pt`` or the greatest valid numbered checkpoint."""

    checkpoint_dir = Path(directory).expanduser()
    if not checkpoint_dir.exists():
        return None
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise ValueError("checkpoint directory must be a real directory")
    checkpoint_dir = checkpoint_dir.resolve()
    candidates: list[tuple[int, Path]] = []
    for candidate in checkpoint_dir.iterdir():
        match = _STEP_PATTERN.fullmatch(candidate.name)
        if (
            match is not None
            and candidate.is_file()
            and not candidate.is_symlink()
        ):
            candidates.append((int(match.group(1)), candidate))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    latest = checkpoint_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or not latest.is_file():
            raise ValueError("latest.pt must be a regular file, not a symlink")
        return latest
    return None


def _validate_checkpoint(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    required = {
        "schema_version",
        "step",
        "trainer_state",
        "resolved_config",
        "rng_state",
        "metadata",
        "wandb_run_id",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {payload['schema_version']!r}"
        )
    _validate_step(payload["step"])
    for name in ("trainer_state", "resolved_config", "rng_state", "metadata"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"checkpoint {name} must be a mapping")
    wandb_run_id = payload["wandb_run_id"]
    if wandb_run_id is not None and (
        not isinstance(wandb_run_id, str)
        or _WANDB_ID_PATTERN.fullmatch(wandb_run_id) is None
    ):
        raise ValueError("checkpoint has an invalid W&B run ID")
    return dict(payload)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | Any = "cpu",
    restore_rng: bool = False,
    restore_cuda_rng: bool = True,
) -> dict[str, Any]:
    """Safely load and validate a checkpoint with weights-only unpickling."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to load checkpoints") from error

    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        discovered = discover_latest_checkpoint(checkpoint_path)
        if discovered is None:
            raise FileNotFoundError(
                f"no checkpoints found in {checkpoint_path}"
            )
        checkpoint_path = discovered
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("checkpoint path must be a regular file, not a symlink")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "safe checkpoint loading requires PyTorch weights_only support"
        ) from error
    checkpoint = _validate_checkpoint(payload)
    if restore_rng:
        restore_rng_state(
            checkpoint["rng_state"], restore_cuda=restore_cuda_rng
        )
    return checkpoint


def restore_trainer_checkpoint(
    trainer: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
    restore_rng: bool = True,
    restore_cuda_rng: bool = True,
) -> None:
    """Restore a CADA trainer and optionally all RNG streams."""

    validated = _validate_checkpoint(checkpoint)
    loader = getattr(trainer, "load_state_dict", None)
    if not callable(loader):
        raise TypeError("trainer must provide load_state_dict")
    try:
        loader(validated["trainer_state"], strict=strict)
    except TypeError:
        if strict is not True:
            raise
        loader(validated["trainer_state"])
    if restore_rng:
        restore_rng_state(
            validated["rng_state"], restore_cuda=restore_cuda_rng
        )


class CheckpointSignalHandler:
    """Turn SIGUSR1/SIGTERM into pollable checkpoint requests."""

    def __init__(
        self,
        signals: Sequence[int] | None = None,
    ) -> None:
        defaults = [
            value
            for value in (
                getattr(signal, "SIGUSR1", None),
                getattr(signal, "SIGTERM", None),
            )
            if value is not None
        ]
        selected = tuple(defaults if signals is None else signals)
        if not selected:
            raise ValueError("at least one checkpoint signal is required")
        if len(set(selected)) != len(selected):
            raise ValueError("checkpoint signals may not contain duplicates")
        self.signals = selected
        self._requested_signal: int | None = None
        self._previous: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self._requested_signal is not None

    @property
    def requested_signal(self) -> int | None:
        return self._requested_signal

    def request(self, signum: int | None = None) -> None:
        """Request a checkpoint directly, including from tests."""

        if signum is None:
            signum = self.signals[0]
        if signum not in self.signals:
            raise ValueError("signal is not configured for checkpoint requests")
        self._requested_signal = int(signum)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        self.request(signum)

    def consume(self) -> int | None:
        signum = self._requested_signal
        self._requested_signal = None
        return signum

    def install(self) -> CheckpointSignalHandler:
        if self._previous:
            return self
        for signum in self.signals:
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def uninstall(self) -> None:
        for signum, previous in self._previous.items():
            if signal.getsignal(signum) == self._handle:
                signal.signal(signum, previous)
        self._previous.clear()

    def __enter__(self) -> CheckpointSignalHandler:
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.uninstall()


@dataclass
class CheckpointCallbackAdapter:
    """Adapter matching ``training.CheckpointCallback`` without importing it."""

    directory: str | os.PathLike[str]
    resolved_config: Any
    metadata: Mapping[str, Any] | None = None
    wandb_run_id: str | None = None
    signal_handler: CheckpointSignalHandler | None = None
    last_path: Path | None = None

    def __call__(
        self,
        step: int,
        state: Mapping[str, Any],
        metrics: Any,
    ) -> None:
        checkpoint_metadata = dict(self.metadata or {})
        if metrics is not None:
            checkpoint_metadata["metrics"] = resolved_config_dict(metrics)
        self.last_path = save_checkpoint(
            self.directory,
            step=step,
            trainer_state=state,
            resolved_config=self.resolved_config,
            metadata=checkpoint_metadata,
            wandb_run_id=self.wandb_run_id,
        )

    def save_if_requested(
        self,
        step: int,
        state: Mapping[str, Any],
        metrics: Any = None,
    ) -> bool:
        """Save once when a configured signal request is pending."""

        if self.signal_handler is None or not self.signal_handler.requested:
            return False
        signum = self.signal_handler.consume()
        checkpoint_metadata = dict(self.metadata or {})
        checkpoint_metadata["requested_by_signal"] = signum
        if metrics is not None:
            checkpoint_metadata["metrics"] = resolved_config_dict(metrics)
        self.last_path = save_checkpoint(
            self.directory,
            step=step,
            trainer_state=state,
            resolved_config=self.resolved_config,
            metadata=checkpoint_metadata,
            wandb_run_id=self.wandb_run_id,
        )
        return True


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCallbackAdapter",
    "CheckpointSignalHandler",
    "build_checkpoint",
    "capture_rng_state",
    "checkpoint_filename",
    "discover_latest_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "restore_trainer_checkpoint",
    "save_checkpoint",
]
