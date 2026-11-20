"""Portable GenCast metrics and array artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gencast_da.runtime import (
    atomic_write_json,
    verify_sha256_sidecar,
    write_sha256_sidecar,
)

EVALUATION_SCHEMA_VERSION = 1
_REQUIRED_COORDINATES = (
    "latitude",
    "longitude",
    "pressure_level",
    "lead_hours",
    "observation_mask",
)
_RESERVED_KEYS = {
    "schema_version",
    "truth",
    "prediction",
    "observations",
    "metadata",
}


def _json_safe(value: Any, location: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite value")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), location)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings")
            result[key] = _json_safe(item, f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{location} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationArtifact:
    truth: np.ndarray
    prediction: np.ndarray
    observations: np.ndarray
    coordinates: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def _validated_arrays(
    truth: np.ndarray,
    prediction: np.ndarray,
    observations: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    observations = np.asarray(observations)
    if truth.ndim != 4 or truth.shape != prediction.shape:
        raise ValueError("truth and prediction must share shape [step,lat,lon,level]")
    if any(size < 1 for size in truth.shape):
        raise ValueError("evaluation dimensions must be non-empty")
    if observations.shape != truth.shape:
        raise ValueError("observations must be full-grid with the truth shape")
    if not all(
        np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all()
        for value in (truth, prediction, observations)
    ):
        raise ValueError("evaluation arrays must be finite floating-point values")
    missing = sorted(set(_REQUIRED_COORDINATES) - set(coordinates))
    if missing:
        raise ValueError(f"missing evaluation coordinates: {missing}")
    reserved = sorted(_RESERVED_KEYS & set(coordinates))
    if reserved:
        raise ValueError(f"coordinate keys are reserved: {reserved}")
    extra = sorted(set(coordinates) - set(_REQUIRED_COORDINATES))
    if extra:
        raise ValueError(f"unsupported evaluation coordinates: {extra}")
    clean = {key: np.asarray(value) for key, value in coordinates.items()}
    expected_lengths = {
        "lead_hours": truth.shape[0],
        "latitude": truth.shape[1],
        "longitude": truth.shape[2],
        "pressure_level": truth.shape[3],
    }
    for key, length in expected_lengths.items():
        if clean[key].ndim != 1 or len(clean[key]) != length:
            raise ValueError(f"{key} does not match the evaluation shape")
        if not np.issubdtype(clean[key].dtype, np.number):
            raise ValueError(f"{key} must be numeric")
        if not np.isfinite(clean[key]).all():
            raise ValueError(f"{key} must be finite")
        if len(clean[key]) > 1 and not np.all(np.diff(clean[key]) > 0):
            raise ValueError(f"{key} must be strictly increasing")
    observation_mask = clean["observation_mask"]
    if observation_mask.shape not in {
        truth.shape[1:3],
        truth.shape[:1] + truth.shape[1:3],
    }:
        raise ValueError("observation_mask must be [lat,lon] or [step,lat,lon]")
    if not np.isin(observation_mask, (0, 1)).all():
        raise ValueError("observation_mask must be binary")
    return truth, prediction, observations, clean


def write_metrics(
    path: str | os.PathLike[str],
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    destination = atomic_write_json(
        path,
        {
            "schema_version": 1,
            "metrics": _json_safe(metrics, "metrics"),
            "metadata": _json_safe(metadata, "metadata"),
        },
    )
    return destination, write_sha256_sidecar(destination)


def write_evaluation_npz(
    path: str | os.PathLike[str],
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    observations: np.ndarray,
    metadata: Mapping[str, Any],
    coordinates: Mapping[str, np.ndarray] | None = None,
) -> tuple[Path, Path]:
    if coordinates is None:
        raise ValueError("evaluation coordinates are required")
    truth, prediction, observations, clean_coordinates = _validated_arrays(
        truth,
        prediction,
        observations,
        coordinates,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    clean_metadata = _json_safe(metadata, "metadata")
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(EVALUATION_SCHEMA_VERSION, dtype=np.int64),
        "truth": truth,
        "prediction": prediction,
        "observations": observations,
        "metadata": np.frombuffer(
            json.dumps(
                clean_metadata,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    payload.update(clean_coordinates)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination, write_sha256_sidecar(destination)


def load_evaluation_npz(
    path: str | os.PathLike[str],
    *,
    verify_checksum: bool = True,
) -> EvaluationArtifact:
    source = Path(path)
    if verify_checksum:
        verify_sha256_sidecar(source)
    with np.load(source, allow_pickle=False) as payload:
        if "schema_version" not in payload:
            raise ValueError("evaluation artifact has no schema version")
        version = int(np.asarray(payload["schema_version"]))
        if version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema version: {version}")
        required = {
            "truth",
            "prediction",
            "observations",
            "metadata",
            *_REQUIRED_COORDINATES,
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"evaluation artifact is missing: {missing}")
        metadata = json.loads(
            np.asarray(payload["metadata"], dtype=np.uint8)
            .tobytes()
            .decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value}")
            ),
        )
        coordinates = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key
            not in {
                "schema_version",
                "truth",
                "prediction",
                "observations",
                "metadata",
            }
        }
        truth, prediction, observations, coordinates = _validated_arrays(
            payload["truth"],
            payload["prediction"],
            payload["observations"],
            coordinates,
        )
    if not isinstance(metadata, dict):
        raise ValueError("evaluation metadata must be a JSON object")
    return EvaluationArtifact(
        truth=truth,
        prediction=prediction,
        observations=observations,
        coordinates=coordinates,
        metadata=metadata,
    )
