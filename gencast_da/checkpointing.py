"""Trusted GenCast pickle checkpoints with checksum fallback."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gencast_da.runtime import (
    atomic_write_bytes,
    verify_sha256_sidecar,
    write_sha256_sidecar,
)

CHECKPOINT_SCHEMA_VERSION = 1


def fallback_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.previous{path.suffix}")


def save_trusted_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint payload has an unsupported schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            verify_sha256_sidecar(path)
        except ValueError:
            pass
        else:
            previous = fallback_path(path)
            atomic_write_bytes(previous, path.read_bytes())
            write_sha256_sidecar(previous)
    serialized = pickle.dumps(
        dict(payload),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    atomic_write_bytes(path, serialized)
    write_sha256_sidecar(path)


def load_trusted_checkpoint(path: Path) -> tuple[dict[str, Any], Path]:
    errors = []
    for candidate in (path, fallback_path(path)):
        if not candidate.is_file():
            continue
        try:
            verify_sha256_sidecar(candidate)
            payload = pickle.loads(candidate.read_bytes())
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload must be a dictionary")
            if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("unsupported GenCast checkpoint schema")
            return payload, candidate
        except Exception as error:  # noqa: BLE001
            errors.append(f"{candidate}: {error}")
    raise ValueError("no verified GenCast checkpoint: " + "; ".join(errors))


def validate_compatibility(
    saved: Mapping[str, Any] | None,
    active: Mapping[str, Any],
) -> None:
    if saved != active:
        raise ValueError(
            "checkpoint is incompatible with the active GenCast "
            "model, controller, sampler, or statistics"
        )


def truncate_history(records: Any, recovered_step: int) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("training history must be a list")
    return [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("step"), int)
        and record["step"] <= recovered_step
    ]
