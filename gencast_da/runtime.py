"""Backend-neutral run layout and atomic files for GenCast experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_ID = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{12}$")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identifier(value: str, name: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must contain only letters, digits, ., _, and -")
    return value


def make_run_id(config: Mapping[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{config_digest(config)[:12]}"


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
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
    return destination


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def write_sha256_sidecar(path: str | os.PathLike[str]) -> Path:
    artifact = Path(path)
    sidecar = artifact.with_name(f"{artifact.name}.sha256")
    return atomic_write_text(sidecar, f"{sha256_file(artifact)}  {artifact.name}\n")


def verify_sha256_sidecar(path: str | os.PathLike[str]) -> None:
    artifact = Path(path)
    sidecar = artifact.with_name(f"{artifact.name}.sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing checksum sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(artifact)
    if expected != actual:
        raise ValueError(f"checksum mismatch for {artifact}")


def _git_metadata(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--short")),
    }


def collect_metadata(repository: Path) -> dict[str, Any]:
    packages = {}
    for name in ("jax", "jaxlib", "dm-haiku", "optax", "graphcast"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "git": _git_metadata(repository),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_NODELIST",
                "SLURM_PROCID",
            )
            if os.environ.get(key) is not None
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_dir: Path
    run_id: str
    checkpoints: Path
    samples: Path
    metrics: Path
    figures: Path
    logs: Path
    wandb: Path

    @classmethod
    def plan(
        cls,
        *,
        root: str | os.PathLike[str],
        stage: str,
        config_id: str,
        resolved_config: Mapping[str, Any],
        run_id: str | None = None,
    ) -> RunLayout:
        root_candidate = Path(root).expanduser()
        if not root_candidate.is_absolute():
            raise ValueError("CADA_ROOT must be absolute")
        root_path = root_candidate.resolve(strict=False)
        stage = _validate_identifier(stage, "stage")
        config_id = _validate_identifier(config_id, "config_id")
        digest = config_digest(resolved_config)
        selected_run_id = run_id or make_run_id(resolved_config)
        if (
            _RUN_ID.fullmatch(selected_run_id) is None
            or not selected_run_id.endswith(f"-{digest[:12]}")
        ):
            raise ValueError(f"invalid GenCast run ID: {selected_run_id!r}")
        run_dir = root_path / "runs" / "gencast" / stage / config_id / selected_run_id
        return cls(
            root=root_path,
            run_dir=run_dir,
            run_id=selected_run_id,
            checkpoints=run_dir / "checkpoints",
            samples=run_dir / "samples",
            metrics=run_dir / "metrics",
            figures=run_dir / "figures",
            logs=run_dir / "logs",
            wandb=run_dir / "wandb",
        )

    def create(
        self,
        *,
        resolved_config: Mapping[str, Any],
        repository: Path,
    ) -> RunLayout:
        if self.run_dir.exists():
            raise FileExistsError(f"GenCast run already exists: {self.run_dir}")
        for directory in (
            self.checkpoints,
            self.samples,
            self.metrics,
            self.figures,
            self.logs,
            self.wandb,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        atomic_write_text(
            self.run_dir / "resolved.yaml",
            yaml.safe_dump(dict(resolved_config), sort_keys=True),
        )
        metadata = collect_metadata(repository)
        metadata.update(
            {
                "run_id": self.run_id,
                "config_sha256": config_digest(resolved_config),
            }
        )
        atomic_write_json(self.run_dir / "metadata.json", metadata)
        return self

    def open_existing(
        self,
        *,
        resolved_config: Mapping[str, Any],
    ) -> RunLayout:
        required = (
            self.checkpoints,
            self.samples,
            self.metrics,
            self.figures,
            self.logs,
            self.wandb,
        )
        if not self.run_dir.is_dir() or not all(path.is_dir() for path in required):
            raise FileNotFoundError(f"GenCast run is incomplete: {self.run_dir}")
        metadata_path = self.run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = config_digest(resolved_config)
        if metadata.get("config_sha256") != expected:
            raise ValueError("resume config does not match the existing GenCast run")
        return self

    def paths(self) -> dict[str, str]:
        return {
            "run": str(self.run_dir),
            "checkpoints": str(self.checkpoints),
            "samples": str(self.samples),
            "metrics": str(self.metrics),
            "figures": str(self.figures),
            "logs": str(self.logs),
            "wandb": str(self.wandb),
        }
