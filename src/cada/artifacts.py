"""Strict, atomic evaluation artifact and metrics writers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runtime import atomic_write_json, atomic_write_text

EVALUATION_ARTIFACT_SCHEMA_VERSION = 2
METRICS_SCHEMA_VERSION = 1


def _json_safe(value: Any, location: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), location)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return _json_safe(value.detach().cpu().item(), location)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} mapping keys must be strings")
            result[key] = _json_safe(item, f"{location}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_safe(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value of type {type(value).__name__}"
    )


def _trajectory_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ValueError(f"{name} may not use an object dtype")
        if value.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError(f"{name} ndarray must use float16, float32, or float64")
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if tensor.ndim not in (4, 5) or tensor.shape[1] < 2:
        raise ValueError(
            f"{name} must have shape [B,T,C,X] or [B,T,C,H,W] with T >= 2"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    if any(size < 1 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    return tensor.contiguous()


@dataclass(frozen=True)
class EvaluationArtifactData:
    """Validated CPU data accepted by both artifact formats."""

    truth: torch.Tensor
    uncontrolled: torch.Tensor
    controlled: torch.Tensor
    metadata: Mapping[str, Any]


def build_evaluation_artifact(
    *,
    truth: Any,
    uncontrolled: Any,
    controlled: Any,
    metadata: Mapping[str, Any],
) -> EvaluationArtifactData:
    """Validate evaluation arrays and schema-versioned JSON metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("artifact metadata must be a mapping")
    truth_tensor = _trajectory_tensor(truth, "truth")
    uncontrolled_tensor = _trajectory_tensor(uncontrolled, "uncontrolled")
    controlled_tensor = _trajectory_tensor(controlled, "controlled")
    if not (
        truth_tensor.shape
        == uncontrolled_tensor.shape
        == controlled_tensor.shape
    ):
        raise ValueError(
            "truth, uncontrolled, and controlled must have identical shapes"
        )
    clean_metadata = _json_safe(metadata)
    existing_version = clean_metadata.get("schema_version")
    if existing_version not in (None, EVALUATION_ARTIFACT_SCHEMA_VERSION):
        raise ValueError(
            f"unsupported evaluation artifact schema {existing_version!r}"
        )
    clean_metadata["schema_version"] = EVALUATION_ARTIFACT_SCHEMA_VERSION
    if clean_metadata.get("artifact_space") != "physical_space":
        raise ValueError("schema 2 evaluation artifacts require physical_space")
    axes = (
        ["batch", "time", "channel", "x"]
        if truth_tensor.ndim == 4
        else ["batch", "time", "channel", "height", "width"]
    )
    if clean_metadata.get("axes") not in (None, axes):
        raise ValueError("artifact metadata axes do not match trajectory rank")
    clean_metadata["axes"] = axes
    resolved_config = clean_metadata.get("resolved_config")
    if isinstance(resolved_config, Mapping):
        dataset = resolved_config.get("dataset", {})
        horizon = resolved_config.get("horizon", {})
        expected_channels = dataset.get("physical_channels")
        expected_steps = horizon.get("forecast_steps")
        if (
            isinstance(expected_channels, int)
            and truth_tensor.shape[2] != expected_channels
        ):
            raise ValueError("artifact channels do not match resolved config")
        if (
            isinstance(expected_steps, int)
            and truth_tensor.shape[1] != expected_steps + 1
        ):
            raise ValueError("artifact horizon does not match resolved config")
    return EvaluationArtifactData(
        truth=truth_tensor,
        uncontrolled=uncontrolled_tensor,
        controlled=controlled_tensor,
        metadata=clean_metadata,
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_stream_write(
    destination: Path,
    writer: Callable[[Any], None],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("artifact directory may not be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            writer(handle)
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
    return destination


def write_evaluation_artifact_pt(
    path: str | os.PathLike[str],
    *,
    truth: Any,
    uncontrolled: Any,
    controlled: Any,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write a weights-only-loadable ``.pt`` artifact."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".pt":
        raise ValueError("Torch evaluation artifacts must use a .pt suffix")
    artifact = build_evaluation_artifact(
        truth=truth,
        uncontrolled=uncontrolled,
        controlled=controlled,
        metadata=metadata,
    )
    payload = {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "truth": artifact.truth,
        "uncontrolled": artifact.uncontrolled,
        "controlled": artifact.controlled,
        "metadata": dict(artifact.metadata),
    }
    written = _atomic_stream_write(
        destination, lambda handle: torch.save(payload, handle)
    )
    write_sha256_sidecar(written)
    return written


def _numpy_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    try:
        array = tensor.numpy()
    except (TypeError, RuntimeError) as error:
        raise TypeError(
            f"{name} dtype cannot be represented safely in NumPy"
        ) from error
    if array.dtype not in (np.float16, np.float32, np.float64):
        raise TypeError(
            f"{name} must use float16, float32, or float64 for .npz output"
        )
    return array


def write_evaluation_artifact_npz(
    path: str | os.PathLike[str],
    *,
    truth: Any,
    uncontrolled: Any,
    controlled: Any,
    metadata: Mapping[str, Any],
    compressed: bool = True,
) -> Path:
    """Atomically write a pickle-free ``.npz`` evaluation artifact."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        raise ValueError("NumPy evaluation artifacts must use a .npz suffix")
    artifact = build_evaluation_artifact(
        truth=truth,
        uncontrolled=uncontrolled,
        controlled=controlled,
        metadata=metadata,
    )
    metadata_json = json.dumps(
        artifact.metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload = {
        "schema_version": np.asarray(
            EVALUATION_ARTIFACT_SCHEMA_VERSION, dtype=np.int64
        ),
        "truth": _numpy_array(artifact.truth, "truth"),
        "uncontrolled": _numpy_array(
            artifact.uncontrolled, "uncontrolled"
        ),
        "controlled": _numpy_array(artifact.controlled, "controlled"),
        "metadata": np.frombuffer(metadata_json, dtype=np.uint8),
    }
    saver = np.savez_compressed if compressed else np.savez
    written = _atomic_stream_write(
        destination, lambda handle: saver(handle, **payload)
    )
    write_sha256_sidecar(written)
    return written


def write_evaluation_artifact(
    path: str | os.PathLike[str],
    *,
    truth: Any,
    uncontrolled: Any,
    controlled: Any,
    metadata: Mapping[str, Any],
    compressed: bool = True,
) -> Path:
    """Dispatch to the strict writer selected by the output suffix."""

    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        destination = write_evaluation_artifact_pt(
            path,
            truth=truth,
            uncontrolled=uncontrolled,
            controlled=controlled,
            metadata=metadata,
        )
    elif suffix == ".npz":
        destination = write_evaluation_artifact_npz(
            path,
            truth=truth,
            uncontrolled=uncontrolled,
            controlled=controlled,
            metadata=metadata,
            compressed=compressed,
        )
    else:
        raise ValueError("evaluation artifacts must use .pt or .npz")
    return destination


def write_metrics_json(
    path: str | os.PathLike[str],
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write versioned JSON-safe metrics."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".json":
        raise ValueError("metrics output must use a .json suffix")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metrics metadata must be a mapping")
    document = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "metrics": _json_safe(metrics, "metrics"),
        "metadata": _json_safe(metadata or {}, "metadata"),
    }
    written = atomic_write_json(destination, document)
    write_sha256_sidecar(written)
    return written


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a lowercase SHA-256 digest for any written artifact."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: str | os.PathLike[str]) -> Path:
    """Atomically write ``<name>.sha256`` beside an existing artifact."""

    artifact = Path(path).expanduser()
    digest = sha256_file(artifact)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    return atomic_write_text(sidecar, f"{digest}  {artifact.name}\n")


__all__ = [
    "EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "METRICS_SCHEMA_VERSION",
    "EvaluationArtifactData",
    "build_evaluation_artifact",
    "sha256_file",
    "write_evaluation_artifact",
    "write_evaluation_artifact_npz",
    "write_evaluation_artifact_pt",
    "write_metrics_json",
    "write_sha256_sidecar",
]
