"""Run layout, provenance, and deterministic seeding utilities."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

_COMPONENT_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{12}$")
_SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_NTASKS",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NODELIST",
)


def _json_safe(value: Any, location: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(
                getattr(value, field.name), f"{location}.{field.name}"
            )
            for field in fields(value)
        }
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


def resolved_config_dict(config: Any) -> dict[str, Any]:
    """Return a strict JSON-safe resolved config without importing config.py."""

    result = _json_safe(config, "config")
    if not isinstance(result, dict):
        raise TypeError("resolved config must be a mapping or dataclass")
    return result


def resolved_config_sha256(config: Any) -> str:
    """Hash a resolved config using canonical JSON encoding."""

    encoded = json.dumps(
        resolved_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def sanitize_experiment_id(value: str, *, max_length: int = 96) -> str:
    """Create a deterministic, collision-resistant filesystem identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiment ID must be a non-empty string")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length < 18:
        raise ValueError("max_length must be at least 18")

    original = value.strip()
    ascii_value = (
        unicodedata.normalize("NFKD", original)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", ascii_value)
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("._-")
    if not sanitized:
        sanitized = "experiment"

    unchanged = (
        sanitized == original
        and len(sanitized) <= max_length
        and _COMPONENT_PATTERN.fullmatch(sanitized) is not None
    )
    if unchanged:
        return sanitized

    suffix = sha256(original.encode("utf-8")).hexdigest()[:8]
    base = sanitized[: max_length - len(suffix) - 1].rstrip("._-")
    if not base:
        base = "experiment"
    return f"{base}-{suffix}"


def cada_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit root or CADA_ROOT."""

    value: str | os.PathLike[str] | None = path
    if value is None:
        value = os.environ.get("CADA_ROOT")
    if value is None:
        raise RuntimeError("CADA_ROOT is not set")
    if not str(value).strip():
        raise RuntimeError("CADA_ROOT cannot be empty")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError("CADA_ROOT must be an absolute path")
    return root.resolve()


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    mode: int = 0o644,
) -> Path:
    """Atomically write bytes using a same-directory temporary file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
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


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    mode: int = 0o644,
) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    mode: int = 0o644,
) -> Path:
    clean = _json_safe(value)
    text = json.dumps(
        clean,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return atomic_write_text(path, text + "\n", mode=mode)


def _git_output(repository: Path, arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_runtime_metadata(
    *,
    seeds: Any = None,
    repository: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without invoking a shell."""

    repository_path = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else Path.cwd().resolve()
    )
    git_sha = _git_output(repository_path, ("rev-parse", "HEAD"))
    git_status = _git_output(
        repository_path, ("status", "--porcelain", "--untracked-files=normal")
    )

    torch_metadata: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_metadata = {
            "version": str(torch.__version__),
            "cuda_build_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_names": (
                [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
                if cuda_available
                else []
            ),
            "cudnn_version": (
                torch.backends.cudnn.version() if cuda_available else None
            ),
        }
    except ImportError:
        torch_metadata = {"available": False}

    clean_seeds = _json_safe(seeds or {}, "seeds")
    if not isinstance(clean_seeds, dict):
        raise TypeError("seeds must be a mapping or dataclass")
    slurm = {
        name: os.environ[name]
        for name in _SLURM_VARIABLES
        if name in os.environ
    }
    return {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "repository": str(repository_path),
            "sha": git_sha,
            "dirty": None if git_status is None else bool(git_status),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "torch": torch_metadata,
        "hostname": socket.gethostname(),
        "slurm": slurm,
        "seeds": clean_seeds,
    }


def new_run_id(config_digest: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{config_digest[:12]}"


@dataclass(frozen=True)
class RunLayout:
    """Canonical paths for one CADA run."""

    root: Path
    dataset: str
    stage: str
    experiment_id: str
    run_id: str

    def __post_init__(self) -> None:
        root = cada_root(self.root)
        object.__setattr__(self, "root", root)
        for name in ("dataset", "stage", "experiment_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _COMPONENT_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{name} is not a sanitized identifier")
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a canonical CADA run ID")

    @property
    def run_dir(self) -> Path:
        return (
            self.root
            / "runs"
            / self.dataset
            / self.stage
            / self.experiment_id
            / self.run_id
        )

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def wandb(self) -> Path:
        return self.run_dir / "wandb"

    @property
    def samples(self) -> Path:
        return self.run_dir / "samples"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def figures(self) -> Path:
        return self.run_dir / "figures"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "resolved.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.checkpoints,
            self.wandb,
            self.samples,
            self.metrics,
            self.figures,
            self.logs,
        )

    @classmethod
    def create(
        cls,
        *,
        dataset: str,
        stage: str,
        experiment_id: str,
        resolved_config: Any,
        root: str | os.PathLike[str] | None = None,
        resume_existing: bool = False,
        run_id: str | None = None,
        seeds: Any = None,
        repository: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunLayout:
        """Create a run or resume the newest matching resolved-config run."""

        root_path = cada_root(root)
        dataset_id = sanitize_experiment_id(dataset)
        stage_id = sanitize_experiment_id(stage)
        experiment = sanitize_experiment_id(experiment_id)
        clean_config = resolved_config_dict(resolved_config)
        config_digest = resolved_config_sha256(clean_config)
        experiment_dir = (
            root_path / "runs" / dataset_id / stage_id / experiment
        )

        if resume_existing:
            if run_id is not None:
                if (
                    _RUN_ID_PATTERN.fullmatch(run_id) is None
                    or not run_id.endswith(f"-{config_digest[:12]}")
                ):
                    raise ValueError("run_id is not a canonical CADA run ID")
                candidates = [experiment_dir / run_id]
            else:
                candidates = sorted(
                    (
                        path
                        for path in experiment_dir.glob(
                            f"*-{config_digest[:12]}"
                        )
                        if (
                            path.is_dir()
                            and not path.is_symlink()
                            and _RUN_ID_PATTERN.fullmatch(path.name) is not None
                        )
                    ),
                    reverse=True,
                )
            if not candidates or not candidates[0].is_dir():
                raise FileNotFoundError(
                    "no existing run matches the resolved config"
                )
            layout = cls(
                root=root_path,
                dataset=dataset_id,
                stage=stage_id,
                experiment_id=experiment,
                run_id=candidates[0].name,
            )
            if layout.run_dir.is_symlink():
                raise ValueError("existing run directory may not be a symlink")
            for required_path in (
                layout.resolved_config_path,
                layout.metadata_path,
            ):
                if required_path.is_symlink() or not required_path.is_file():
                    raise FileNotFoundError(
                        f"existing run is incomplete: {required_path}"
                    )
            existing_metadata = json.loads(
                layout.metadata_path.read_text(encoding="utf-8")
            )
            if existing_metadata.get("resolved_config_sha256") != config_digest:
                raise ValueError(
                    "existing run resolved config does not match requested config"
                )
            for directory in layout.directories:
                directory.mkdir(parents=True, exist_ok=True)
            return layout

        if run_id is not None and (
            _RUN_ID_PATTERN.fullmatch(run_id) is None
            or not run_id.endswith(f"-{config_digest[:12]}")
        ):
            raise ValueError(
                "run_id must be canonical and match the resolved config"
            )
        while True:
            candidate_id = run_id or new_run_id(config_digest)
            layout = cls(
                root=root_path,
                dataset=dataset_id,
                stage=stage_id,
                experiment_id=experiment,
                run_id=candidate_id,
            )
            try:
                layout.run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                if run_id is not None:
                    raise

        try:
            for directory in layout.directories:
                directory.mkdir()
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError(
                    "PyYAML is required to write resolved.yaml"
                ) from error
            resolved_yaml = yaml.safe_dump(
                clean_config,
                sort_keys=True,
                allow_unicode=True,
                default_flow_style=False,
            )
            atomic_write_text(layout.resolved_config_path, resolved_yaml)
            document = {
                "schema_version": 1,
                "run_id": layout.run_id,
                "dataset": layout.dataset,
                "stage": layout.stage,
                "experiment_id": layout.experiment_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "resolved_config_sha256": config_digest,
                "runtime": collect_runtime_metadata(
                    seeds=seeds, repository=repository
                ),
                "metadata": _json_safe(dict(metadata or {}), "metadata"),
            }
            atomic_write_json(layout.metadata_path, document)
        except BaseException:
            # Keep a partially initialized directory for diagnosis; resume rejects
            # it until metadata is complete.
            raise
        return layout

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        root: str | os.PathLike[str] | None = None,
        resume_existing: bool = False,
        run_id: str | None = None,
        repository: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunLayout:
        """Create a layout directly from a validated CADA config dataclass."""

        resolved = resolved_config_dict(config)
        kind = resolved.get("kind")
        config_id = resolved.get("config_id")
        dataset_value = resolved.get("dataset", resolved)
        if not isinstance(kind, str) or not isinstance(config_id, str):
            raise ValueError("config requires string kind and config_id fields")
        if not isinstance(dataset_value, Mapping):
            raise ValueError("config dataset must be a mapping")
        dataset = dataset_value.get("domain")
        if not isinstance(dataset, str):
            raise ValueError("config requires a dataset domain")
        return cls.create(
            dataset=dataset,
            stage=kind,
            experiment_id=config_id,
            resolved_config=resolved,
            root=root,
            resume_existing=resume_existing,
            run_id=run_id,
            seeds=resolved.get("seeds"),
            repository=repository,
            metadata=metadata,
        )


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must lie in [0, 2**63)")
    return seed


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> int:
    """Seed Python, NumPy, Torch CPU, and all visible CUDA devices."""

    seed = _validate_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for deterministic seeding") from error
    np.random.seed(seed % 2**32)

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for deterministic seeding") from error
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return seed


def seed_dataloader_worker(worker_id: int, *, base_seed: int | None = None) -> None:
    """Seed Python and NumPy inside a DataLoader worker process."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for worker seeding") from error
    worker_seed = (
        int(torch.initial_seed())
        if base_seed is None
        else _validate_seed(base_seed) + worker_id
    )
    worker_seed %= 2**32
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for worker seeding") from error
    np.random.seed(worker_seed)


def make_worker_init_fn(seed: int) -> Any:
    """Return a picklable DataLoader worker_init_fn."""

    return partial(seed_dataloader_worker, base_seed=_validate_seed(seed))


def make_torch_generator(seed: int) -> Any:
    """Return a seeded CPU generator suitable for a DataLoader."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for a seeded generator") from error
    generator = torch.Generator()
    generator.manual_seed(_validate_seed(seed))
    return generator


__all__ = [
    "RunLayout",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "cada_root",
    "collect_runtime_metadata",
    "make_torch_generator",
    "make_worker_init_fn",
    "new_run_id",
    "resolved_config_dict",
    "resolved_config_sha256",
    "sanitize_experiment_id",
    "seed_dataloader_worker",
    "seed_everything",
]
