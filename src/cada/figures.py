"""Manifest-driven plotting of saved CADA evaluation artifacts.

This module intentionally has no model or sampler imports. Plotting only reads
the artifact paths declared by a :class:`FigureManifest`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .metrics import (
    era5_vorticity,
    vorticity_2d_finite_difference,
    vorticity_2d_spectral,
)
from .runtime import atomic_write_text

matplotlib.rcParams["svg.hashsalt"] = "cada"

ArtifactField = Literal["truth", "uncontrolled", "controlled"]
FigureType = Literal["spacetime_1d", "snapshot_grid_2d"]
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SUPPORTED_ARTIFACT_SCHEMA_VERSION = 2


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class EvaluationArtifact:
    """Validated evaluation data loaded on CPU."""

    path: Path
    truth: torch.Tensor
    uncontrolled: torch.Tensor
    controlled: torch.Tensor
    metadata: Mapping[str, Any]

    def get(self, name: ArtifactField) -> torch.Tensor:
        return getattr(self, name)


@dataclass(frozen=True)
class ArtifactReference:
    """A labeled optional artifact input."""

    path: Path
    label: str
    field: ArtifactField | None = "controlled"


@dataclass(frozen=True)
class FigureManifest:
    """Declarative input, selection, provenance, and rendering information.

    YAML/dict keys are:

    ``figure_type``
        ``spacetime_1d`` or ``snapshot_grid_2d``.
    ``artifact`` / ``input_artifact``
        Primary evaluation artifact.
    ``output``
        A ``.png``, ``.pdf``, or ``.svg`` path.
    External ``baselines`` and ``baseline_artifacts`` are rejected because
    external methods are outside this repository scope.
    ``trajectory`` and ``timesteps``
        Zero-based selections. Snapshot grids require at least one timestep.
    ``config_ids``, ``checkpoint_ids``, ``seeds``, ``observation_details``
        Values copied verbatim into the provenance sidecar.
    ``rendering``
        Plot-specific rendering values; effective defaults are also recorded.
    """

    figure_type: FigureType
    artifact: Path
    output: Path
    baseline_artifacts: tuple[ArtifactReference, ...] = ()
    trajectory: int = 0
    timesteps: tuple[int, ...] = ()
    config_ids: tuple[str, ...] = ()
    checkpoint_ids: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    observation_details: Mapping[str, Any] = field(default_factory=dict)
    rendering: Mapping[str, Any] = field(default_factory=dict)
    primary_label: str = "CADA"

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        cada_root: str | Path | None = None,
    ) -> FigureManifest:
        """Build a manifest from the mapping returned by ``yaml.safe_load``."""

        if not isinstance(data, Mapping):
            raise TypeError("figure manifest must be a mapping")
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        path_root = _cada_root(cada_root)

        kind_value = data.get("figure_type", data.get("kind"))
        aliases = {
            "1d_spacetime": "spacetime_1d",
            "spacetime_1d": "spacetime_1d",
            "2d_snapshot_grid": "snapshot_grid_2d",
            "snapshot_grid_2d": "snapshot_grid_2d",
        }
        if kind_value not in aliases:
            raise ValueError(
                "figure_type must be spacetime_1d or snapshot_grid_2d"
            )
        figure_type = aliases[str(kind_value)]

        artifact_value = data.get(
            "artifact", data.get("input_artifact", data.get("input"))
        )
        if artifact_value is None or "output" not in data:
            raise ValueError("manifest requires artifact and output paths")

        def resolve(value: Any, location: str) -> Path:
            expanded = _expand_manifest_path(value, path_root, location)
            path = Path(expanded).expanduser()
            return path if path.is_absolute() else root / path

        baseline_value = data.get(
            "baseline_artifacts", data.get("baselines", ())
        )
        baselines: list[ArtifactReference] = []
        if isinstance(baseline_value, Mapping):
            if "path" in baseline_value:
                baseline_items: Sequence[Any] = [baseline_value]
            else:
                baseline_items = [
                    {"label": label, "path": path}
                    for label, path in baseline_value.items()
                ]
        elif isinstance(baseline_value, Sequence) and not isinstance(
            baseline_value, (str, bytes)
        ):
            baseline_items = baseline_value
        else:
            raise TypeError("baselines must be a list or label-to-path mapping")
        for index, item in enumerate(baseline_items):
            if isinstance(item, (str, Path)):
                path = resolve(item, f"baselines[{index}].path")
                baselines.append(
                    ArtifactReference(path=path, label=path.stem)
                )
                continue
            if not isinstance(item, Mapping) or "path" not in item:
                raise ValueError(f"baseline {index} requires a path")
            path = resolve(item["path"], f"baselines[{index}].path")
            requested_field = str(item.get("field", "controlled"))
            if requested_field not in {
                "truth",
                "uncontrolled",
                "controlled",
            }:
                raise ValueError(
                    f"baseline {index} field must be truth, "
                    "uncontrolled, or controlled"
                )
            artifact_field: ArtifactField = requested_field  # type: ignore[assignment]
            label = str(item.get("label", path.stem))
            if not label:
                raise ValueError(f"baseline {index} label must be non-empty")
            baselines.append(
                ArtifactReference(
                    path=path,
                    label=label,
                    field=artifact_field,
                )
            )
        if baselines:
            raise ValueError(
                "external method artifacts are outside this repository scope"
            )
        primary_label = str(data.get("primary_label", "CADA"))
        if not primary_label:
            raise ValueError("primary_label must be non-empty")
        labels = [primary_label, *[reference.label for reference in baselines]]
        if len(set(labels)) != len(labels):
            raise ValueError("primary and method labels must be unique")

        trajectory_value = data.get("trajectory", 0)
        if isinstance(trajectory_value, bool) or not isinstance(
            trajectory_value,
            (int, np.integer),
        ):
            raise TypeError("trajectory must be an integer")
        trajectory = int(trajectory_value)
        if trajectory < 0:
            raise ValueError("trajectory must be zero-based and non-negative")
        timesteps = _integer_tuple(data.get("timesteps", ()), "timesteps")
        if any(timestep < 0 for timestep in timesteps):
            raise ValueError("timesteps must be zero-based and non-negative")

        observation_details = data.get(
            "observation_details", data.get("observations", {})
        )
        rendering = data.get("rendering", {})
        if not isinstance(observation_details, Mapping):
            raise TypeError("observation_details must be a mapping")
        if not isinstance(rendering, Mapping):
            raise TypeError("rendering must be a mapping")
        for integer_field in ("channel", "u_channel", "v_channel", "dpi"):
            if integer_field not in rendering:
                continue
            value = rendering[integer_field]
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"rendering.{integer_field} must be an integer")
            if integer_field == "dpi" and value < 1:
                raise ValueError("rendering.dpi must be positive")

        artifact = resolve(artifact_value, "artifact")
        output = resolve(data["output"], "output")
        if artifact.suffix.lower() not in {".npz", ".pt"}:
            raise ValueError("evaluation artifacts must use .npz or .pt")
        if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
            raise ValueError("figure output must use .png, .pdf, or .svg")
        if figure_type == "snapshot_grid_2d" and not timesteps:
            raise ValueError(
                "snapshot_grid_2d requires explicit timesteps"
            )

        return cls(
            figure_type=figure_type,  # type: ignore[arg-type]
            artifact=artifact,
            output=output,
            baseline_artifacts=tuple(baselines),
            trajectory=trajectory,
            timesteps=timesteps,
            config_ids=_string_tuple(data.get("config_ids", ())),
            checkpoint_ids=_string_tuple(data.get("checkpoint_ids", ())),
            seeds=_integer_tuple(data.get("seeds", ()), "seeds"),
            observation_details=dict(observation_details),
            rendering=dict(rendering),
            primary_label=primary_label,
        )

    @classmethod
    def from_yaml_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        cada_root: str | Path | None = None,
    ) -> FigureManifest:
        """Alias with an explicit name for an already parsed YAML mapping."""

        return cls.from_dict(
            data, base_dir=base_dir, cada_root=cada_root
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        cada_root: str | Path | None = None,
    ) -> FigureManifest:
        """Safely load one YAML manifest, resolving relative paths beside it."""

        manifest_path = Path(path).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
        return cls.from_dict(
            data,
            base_dir=manifest_path.parent,
            cada_root=cada_root,
        )


def _cada_root(value: str | Path | None) -> str:
    candidate = str(value) if value is not None else os.environ.get("CADA_ROOT")
    if candidate is None:
        raise ValueError("CADA_ROOT is not set")
    if not candidate or not Path(candidate).is_absolute():
        raise ValueError("CADA_ROOT must be a non-empty absolute path")
    if "$" in candidate:
        raise ValueError("CADA_ROOT cannot contain placeholders")
    return candidate


def _expand_manifest_path(value: Any, cada_root: str, location: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{location} must be a path string")
    text = os.fspath(value)
    if not isinstance(text, str):
        raise TypeError(f"{location} must be a text path")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name != "CADA_ROOT":
            raise ValueError(
                f"{location} references unsupported placeholder {name!r}"
            )
        return cada_root

    expanded = _ENVIRONMENT_VARIABLE.sub(replace, text)
    if "${" in expanded or "$" in expanded:
        raise ValueError(f"{location} contains unsupported placeholder syntax")
    return expanded


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (str(value),)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError("identifier fields must be scalars or sequences")
    return tuple(str(item) for item in value)


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers, not booleans")
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer or sequence of integers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in value
    ):
        raise TypeError(f"{name} must contain only integers")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} may not contain duplicates")
    if name == "timesteps" and tuple(sorted(result)) != result:
        raise ValueError("timesteps must be strictly increasing")
    return result


def _json_value(value: Any, location: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), location)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return _json_value(value.item(), location)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} mapping keys must be strings")
            result[key] = _json_value(item, f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported value of type {type(value).__name__}"
    )


def _decode_npz_metadata(value: np.ndarray) -> Mapping[str, Any]:
    if value.dtype == object:
        raise ValueError("npz metadata must not use an object/pickle dtype")
    if value.dtype == np.uint8 and value.ndim == 1:
        encoded: str | bytes = value.tobytes()
    elif value.size == 1 and value.dtype.kind in {"U", "S"}:
        encoded = value.reshape(()).item()
    else:
        raise ValueError(
            "npz metadata must be a scalar JSON string or a uint8 JSON buffer"
        )
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("npz metadata is not valid JSON") from error
    if not isinstance(decoded, Mapping):
        raise ValueError("artifact metadata must decode to a mapping")
    return _json_value(decoded)


def _trajectory_tensor(value: Any, key: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ValueError(f"artifact field {key!r} has unsafe object dtype")
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        raise TypeError(f"artifact field {key!r} must be a tensor or ndarray")
    if not tensor.is_floating_point():
        raise TypeError(f"artifact field {key!r} must be floating point")
    if tensor.ndim not in (4, 5) or tensor.shape[1] < 2:
        raise ValueError(
            f"artifact field {key!r} must be [B,T,C,X] or [B,T,C,H,W]"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"artifact field {key!r} must be finite")
    if any(size < 1 for size in tensor.shape):
        raise ValueError(f"artifact field {key!r} has an empty dimension")
    return tensor


def load_evaluation_artifact(path: str | Path) -> EvaluationArtifact:
    """Safely load a strict evaluation artifact from ``.npz`` or ``.pt``.

    Both formats must be dictionaries with exactly the required semantic keys
    ``truth``, ``uncontrolled``, ``controlled``, and ``metadata`` (additional
    keys are ignored). The three arrays must be floating point with identical
    non-empty shapes. ``.npz`` is opened with ``allow_pickle=False`` and stores
    metadata as JSON text. ``.pt`` uses ``torch.load(weights_only=True)`` and
    never falls back to unsafe pickle loading.
    """

    artifact_path = Path(path).expanduser().resolve()
    checksum_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise ValueError(f"artifact checksum sidecar is missing: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="utf-8").split()[0]
    if expected_checksum != sha256_file(artifact_path):
        raise ValueError(f"artifact checksum mismatch: {artifact_path}")
    required = {
        "schema_version",
        "truth",
        "uncontrolled",
        "controlled",
        "metadata",
    }
    suffix = artifact_path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(artifact_path, allow_pickle=False) as payload:
                missing = required.difference(payload.files)
                if missing:
                    raise ValueError(
                        f"artifact is missing required keys: {sorted(missing)}"
                    )
                raw: Mapping[str, Any] = {
                    "schema_version": int(np.asarray(payload["schema_version"])),
                    "truth": np.array(payload["truth"], copy=True),
                    "uncontrolled": np.array(
                        payload["uncontrolled"], copy=True
                    ),
                    "controlled": np.array(payload["controlled"], copy=True),
                    "metadata": _decode_npz_metadata(payload["metadata"]),
                }
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "artifact is missing"
            ):
                raise
            raise ValueError(
                f"could not safely load npz artifact {artifact_path}"
            ) from error
    elif suffix == ".pt":
        try:
            raw_object = torch.load(
                artifact_path, map_location="cpu", weights_only=True
            )
        except TypeError as error:
            raise RuntimeError(
                "safe .pt loading requires a PyTorch version with weights_only"
            ) from error
        except Exception as error:
            raise ValueError(
                f"could not safely load torch artifact {artifact_path}"
            ) from error
        if not isinstance(raw_object, Mapping):
            raise ValueError(".pt evaluation artifact must contain a dictionary")
        missing = required.difference(raw_object.keys())
        if missing:
            raise ValueError(
                f"artifact is missing required keys: {sorted(missing)}"
            )
        raw = raw_object
    else:
        raise ValueError("evaluation artifacts must use .npz or .pt")

    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("artifact metadata must be a mapping")
    clean_metadata = _json_value(metadata)
    version = raw["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _SUPPORTED_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported evaluation artifact schema {version!r}")
    if clean_metadata.get("schema_version") != version:
        raise ValueError("artifact metadata schema_version does not match payload")
    if clean_metadata.get("artifact_space") != "physical_space":
        raise ValueError("schema 2 artifacts must declare physical_space")
    truth = _trajectory_tensor(raw["truth"], "truth")
    uncontrolled = _trajectory_tensor(raw["uncontrolled"], "uncontrolled")
    controlled = _trajectory_tensor(raw["controlled"], "controlled")
    if not (truth.shape == uncontrolled.shape == controlled.shape):
        raise ValueError(
            "truth, uncontrolled, and controlled must have identical shapes; "
            f"got {tuple(truth.shape)}, {tuple(uncontrolled.shape)}, "
            f"{tuple(controlled.shape)}"
        )
    expected_axes = (
        ["batch", "time", "channel", "x"]
        if truth.ndim == 4
        else ["batch", "time", "channel", "height", "width"]
    )
    if clean_metadata.get("axes") != expected_axes:
        raise ValueError("artifact axes metadata does not match trajectory shape")
    resolved_config = clean_metadata.get("resolved_config")
    if isinstance(resolved_config, Mapping):
        dataset = resolved_config.get("dataset", {})
        horizon = resolved_config.get("horizon", {})
        expected_channels = dataset.get("physical_channels")
        expected_steps = horizon.get("forecast_steps")
        if isinstance(expected_channels, int) and truth.shape[2] != expected_channels:
            raise ValueError("artifact channels do not match resolved config")
        if isinstance(expected_steps, int) and truth.shape[1] != expected_steps + 1:
            raise ValueError("artifact horizon does not match resolved config")
    return EvaluationArtifact(
        path=artifact_path,
        truth=truth,
        uncontrolled=uncontrolled,
        controlled=controlled,
        metadata=clean_metadata,
    )


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA256 digest of a file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_sidecar_path(output: str | Path) -> Path:
    """Return ``<output-suffix>.provenance.json`` beside an output figure."""

    output_path = Path(output)
    return output_path.with_suffix(output_path.suffix + ".provenance.json")


def _validate_manifest_identity(
    manifest: FigureManifest,
    artifacts: Sequence[EvaluationArtifact],
) -> None:
    if not artifacts:
        return
    primary_metadata = artifacts[0].metadata
    if manifest.config_ids and manifest.config_ids != (
        primary_metadata.get("config_id"),
    ):
        raise ValueError("manifest config_ids do not match artifact metadata")
    if manifest.checkpoint_ids and set(
        primary_metadata.get("checkpoint_ids", ())
    ) != set(manifest.checkpoint_ids):
        raise ValueError("manifest checkpoint_ids do not match artifact metadata")
    if manifest.seeds and list(manifest.seeds) != [
        primary_metadata.get("initial_noise_seed")
    ]:
        raise ValueError("manifest seeds do not match artifact metadata")


def write_provenance_sidecar(
    manifest: FigureManifest,
    *,
    artifacts: Sequence[EvaluationArtifact] | None = None,
    rendering: Mapping[str, Any] | None = None,
) -> Path:
    """Write deterministic JSON provenance and return its path."""

    references = [
        ("primary", manifest.primary_label, manifest.artifact, None),
        *[
            (
                "baseline",
                reference.label,
                reference.path,
                reference.field,
            )
            for reference in manifest.baseline_artifacts
        ],
    ]
    metadata_by_path = (
        {artifact.path.resolve(): artifact.metadata for artifact in artifacts}
        if artifacts is not None
        else {}
    )
    if artifacts:
        _validate_manifest_identity(manifest, artifacts)
    inputs = []
    for role, label, path, artifact_field in references:
        resolved = path.expanduser().resolve()
        item: dict[str, Any] = {
            "role": role,
            "label": label,
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
        if artifact_field is not None:
            item["field"] = artifact_field
        if resolved in metadata_by_path:
            item["metadata"] = metadata_by_path[resolved]
        inputs.append(item)

    sidecar = provenance_sidecar_path(manifest.output)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    output_path = manifest.output.expanduser().resolve()
    effective_rendering = _json_value(
        rendering if rendering is not None else manifest.rendering
    )
    manifest_payload = {
        "figure_type": manifest.figure_type,
        "artifact": str(manifest.artifact),
        "output": str(manifest.output),
        "config_ids": list(manifest.config_ids),
        "checkpoint_ids": list(manifest.checkpoint_ids),
        "seeds": list(manifest.seeds),
        "trajectory": manifest.trajectory,
        "timesteps": list(manifest.timesteps),
        "observation_details": _json_value(manifest.observation_details),
        "rendering": effective_rendering,
        "primary_label": manifest.primary_label,
    }
    manifest_sha256 = sha256(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    document = {
        "schema_version": 1,
        "figure_type": manifest.figure_type,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "manifest_sha256": manifest_sha256,
        "inputs": inputs,
        "config_ids": list(manifest.config_ids),
        "checkpoint_ids": list(manifest.checkpoint_ids),
        "seeds": list(manifest.seeds),
        "trajectory": manifest.trajectory,
        "timesteps": list(manifest.timesteps),
        "observation_details": _json_value(manifest.observation_details),
        "rendering": effective_rendering,
        "libraries": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_write_text(
        sidecar,
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return sidecar


def _load_manifest_artifacts(
    manifest: FigureManifest,
) -> tuple[EvaluationArtifact, list[tuple[ArtifactReference, EvaluationArtifact]]]:
    primary = load_evaluation_artifact(manifest.artifact)
    baselines = [
        (reference, load_evaluation_artifact(reference.path))
        for reference in manifest.baseline_artifacts
    ]
    return primary, baselines


def _rows(
    primary: EvaluationArtifact,
    baselines: Sequence[tuple[ArtifactReference, EvaluationArtifact]],
    labels: Mapping[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    rows = [
        (str(labels.get("truth", "Truth")), primary.truth),
        (
            str(labels.get("uncontrolled", "Uncontrolled")),
            primary.uncontrolled,
        ),
        (str(labels.get("controlled", "Controlled")), primary.controlled),
    ]
    rows.extend(
        (reference.label, artifact.get(_artifact_field(reference)))
        for reference, artifact in baselines
    )
    return rows


def _artifact_field(reference: ArtifactReference) -> ArtifactField:
    if reference.field is None:
        raise ValueError("plot baseline requires an artifact field")
    return reference.field


def _color_limits(
    arrays: Sequence[np.ndarray], rendering: Mapping[str, Any]
) -> tuple[float, float]:
    requested_vmin = rendering.get("vmin")
    requested_vmax = rendering.get("vmax")
    if (requested_vmin is None) != (requested_vmax is None):
        raise ValueError("rendering vmin and vmax must be specified together")
    if requested_vmin is not None:
        vmin, vmax = float(requested_vmin), float(requested_vmax)
    elif bool(rendering.get("symmetric", True)):
        absolute_max = max(float(np.nanmax(np.abs(array))) for array in arrays)
        absolute_max = absolute_max if absolute_max > 0 else 1.0
        vmin, vmax = -absolute_max, absolute_max
    else:
        vmin = min(float(np.nanmin(array)) for array in arrays)
        vmax = max(float(np.nanmax(array)) for array in arrays)
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    if not np.isfinite([vmin, vmax]).all() or not vmin < vmax:
        raise ValueError("rendering color limits must be finite with vmin < vmax")
    return vmin, vmax


def _save_figure(figure: Any, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".png", ".pdf", ".svg"}:
        raise ValueError("figure output must use .png, .pdf, or .svg")
    if suffix == ".pdf":
        metadata = {
            "Creator": "cada.figures",
            "CreationDate": None,
            "ModDate": None,
        }
    elif suffix == ".svg":
        metadata = {"Creator": "cada.figures", "Date": None}
    else:
        metadata = {"Software": "cada.figures"}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _one_dimensional_matrix(
    tensor: torch.Tensor,
    trajectory: int,
    channel: int,
    timesteps: tuple[int, ...],
) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError("1D plots require [B,T,C,W] artifacts")
    if channel < 0 or channel >= tensor.shape[2]:
        raise IndexError(
            f"channel {channel} is out of range for shape {tuple(tensor.shape)}"
        )
    selected = tensor[trajectory, :, channel, :]
    if timesteps:
        if max(timesteps) >= selected.shape[0]:
            raise IndexError(
                f"timestep selection {timesteps} exceeds T={selected.shape[0]}"
            )
        selected = selected[list(timesteps)]
    return selected.numpy()


def plot_spacetime_1d(manifest: FigureManifest) -> tuple[Path, Path]:
    """Render a 1D spacetime comparison and its provenance sidecar."""

    if manifest.figure_type != "spacetime_1d":
        raise ValueError("manifest figure_type is not spacetime_1d")
    primary, baselines = _load_manifest_artifacts(manifest)
    _validate_manifest_identity(
        manifest,
        [primary, *[artifact for _, artifact in baselines]],
    )
    if manifest.trajectory >= primary.truth.shape[0]:
        raise IndexError(
            f"trajectory {manifest.trajectory} exceeds B={primary.truth.shape[0]}"
        )
    labels_value = manifest.rendering.get("labels", {})
    if not isinstance(labels_value, Mapping):
        raise TypeError("rendering.labels must be a mapping")
    row_data = _rows(primary, baselines, labels_value)
    channel = int(manifest.rendering.get("channel", 0))
    matrices = [
        _one_dimensional_matrix(
            tensor, manifest.trajectory, channel, manifest.timesteps
        )
        for _, tensor in row_data
    ]
    reference_shape = matrices[0].shape
    if any(matrix.shape != reference_shape for matrix in matrices[1:]):
        raise ValueError("all displayed 1D arrays must have identical T,W shapes")

    dpi = int(manifest.rendering.get("dpi", 300))
    if dpi <= 0:
        raise ValueError("rendering.dpi must be positive")
    figsize_value = manifest.rendering.get(
        "figsize", (7.0, max(1.5 * len(matrices), 2.0))
    )
    figsize = tuple(float(value) for value in figsize_value)
    if len(figsize) != 2 or min(figsize) <= 0:
        raise ValueError("rendering.figsize must contain two positive values")
    vmin, vmax = _color_limits(matrices, manifest.rendering)
    effective = {
        "channel": channel,
        "cmap": str(manifest.rendering.get("cmap", "coolwarm")),
        "origin": str(manifest.rendering.get("origin", "lower")),
        "symmetric": bool(manifest.rendering.get("symmetric", True)),
        "vmin": vmin,
        "vmax": vmax,
        "colorbar": bool(manifest.rendering.get("colorbar", True)),
        "dpi": dpi,
        "figsize": list(figsize),
        "labels": dict(labels_value),
        "title": manifest.rendering.get("title"),
        "xlabel": str(manifest.rendering.get("xlabel", "Space index")),
        "ylabel": str(manifest.rendering.get("ylabel", "Timestep")),
    }

    figure, axes = plt.subplots(
        len(matrices), 1, figsize=figsize, squeeze=False, constrained_layout=True
    )
    image = None
    try:
        for axis, (label, _), matrix in zip(axes[:, 0], row_data, matrices):
            image = axis.imshow(
                matrix,
                aspect="auto",
                origin=effective["origin"],
                cmap=effective["cmap"],
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_ylabel(label)
            if manifest.timesteps:
                tick_count = min(6, len(manifest.timesteps))
                positions = np.linspace(
                    0, len(manifest.timesteps) - 1, tick_count, dtype=int
                )
                axis.set_yticks(
                    positions, [manifest.timesteps[index] for index in positions]
                )
            if axis is not axes[-1, 0]:
                axis.set_xticklabels([])
        axes[-1, 0].set_xlabel(
            str(manifest.rendering.get("xlabel", "Space index"))
        )
        figure.supylabel(
            str(manifest.rendering.get("ylabel", "Timestep"))
        )
        title = manifest.rendering.get("title")
        if title:
            figure.suptitle(str(title))
        if effective["colorbar"] and image is not None:
            figure.colorbar(image, ax=axes[:, 0].tolist(), label="Value")
        _save_figure(figure, manifest.output, dpi)
    finally:
        plt.close(figure)

    loaded = [primary, *[artifact for _, artifact in baselines]]
    sidecar = write_provenance_sidecar(
        manifest, artifacts=loaded, rendering=effective
    )
    return manifest.output, sidecar


def _two_dimensional_series(
    tensor: torch.Tensor,
    trajectory: int,
    rendering: Mapping[str, Any],
) -> torch.Tensor:
    field_name = str(rendering.get("field", "channel"))
    channel = int(rendering.get("channel", 0))
    if trajectory >= tensor.shape[0]:
        raise IndexError(
            f"trajectory {trajectory} exceeds B={tensor.shape[0]}"
        )
    trajectory_data = tensor[trajectory]
    if field_name in {"channel", "temperature"}:
        if trajectory_data.ndim != 4:
            raise ValueError(
                "2D channel plots require [B,T,C,H,W]"
            )
        if channel < 0 or channel >= trajectory_data.shape[1]:
            raise IndexError(
                f"channel {channel} exceeds C={trajectory_data.shape[1]}"
            )
        return trajectory_data[:, channel]

    if trajectory_data.ndim != 4:
        raise ValueError(
            "vorticity plots require artifacts shaped [B,T,C,H,W]"
        )
    if "dx" not in rendering or "dy" not in rendering:
        raise ValueError("vorticity rendering requires explicit dx and dy")
    dx, dy = float(rendering["dx"]), float(rendering["dy"])
    if field_name == "vorticity_fd":
        return vorticity_2d_finite_difference(
            trajectory_data,
            dx=dx,
            dy=dy,
            u_channel=int(rendering.get("u_channel", 0)),
            v_channel=int(rendering.get("v_channel", 1)),
        )
    if field_name == "vorticity_spectral":
        return vorticity_2d_spectral(
            trajectory_data,
            dx=dx,
            dy=dy,
            u_channel=int(rendering.get("u_channel", 0)),
            v_channel=int(rendering.get("v_channel", 1)),
        )
    if field_name == "era5_vorticity":
        return era5_vorticity(
            trajectory_data,
            dx=dx,
            dy=dy,
            u_channel=int(rendering.get("u_channel", 1)),
            v_channel=int(rendering.get("v_channel", 2)),
            periodic_longitude=bool(
                rendering.get("periodic_longitude", True)
            ),
            periodic_latitude=bool(
                rendering.get("periodic_latitude", False)
            ),
        )
    raise ValueError(
        "rendering.field must be channel, temperature, vorticity_fd, "
        "vorticity_spectral, or era5_vorticity"
    )


def plot_snapshot_grid_2d(manifest: FigureManifest) -> tuple[Path, Path]:
    """Render selected 2D snapshots and their provenance sidecar."""

    if manifest.figure_type != "snapshot_grid_2d":
        raise ValueError("manifest figure_type is not snapshot_grid_2d")
    if not manifest.timesteps:
        raise ValueError("snapshot_grid_2d requires explicit timesteps")
    primary, baselines = _load_manifest_artifacts(manifest)
    _validate_manifest_identity(
        manifest,
        [primary, *[artifact for _, artifact in baselines]],
    )
    labels_value = manifest.rendering.get("labels", {})
    if not isinstance(labels_value, Mapping):
        raise TypeError("rendering.labels must be a mapping")
    row_data = _rows(primary, baselines, labels_value)
    series = [
        _two_dimensional_series(
            tensor, manifest.trajectory, manifest.rendering
        )
        for _, tensor in row_data
    ]
    if any(max(manifest.timesteps) >= item.shape[0] for item in series):
        raise IndexError("a selected timestep exceeds an artifact time horizon")
    panels = [
        item[list(manifest.timesteps)].numpy()
        for item in series
    ]
    reference_shape = panels[0].shape
    if any(panel.shape != reference_shape for panel in panels[1:]):
        raise ValueError(
            "all displayed 2D arrays must have identical timestep,H,W shapes"
        )

    layout = str(manifest.rendering.get("layout", "methods_as_rows"))
    if layout not in {"methods_as_rows", "methods_as_columns"}:
        raise ValueError(
            "rendering.layout must be methods_as_rows or methods_as_columns"
        )
    if layout == "methods_as_rows":
        nrows, ncols = len(panels), len(manifest.timesteps)
    else:
        nrows, ncols = len(manifest.timesteps), len(panels)
    dpi = int(manifest.rendering.get("dpi", 300))
    if dpi <= 0:
        raise ValueError("rendering.dpi must be positive")
    figsize_value = manifest.rendering.get(
        "figsize", (2.0 * ncols, 2.0 * nrows)
    )
    figsize = tuple(float(value) for value in figsize_value)
    if len(figsize) != 2 or min(figsize) <= 0:
        raise ValueError("rendering.figsize must contain two positive values")
    vmin, vmax = _color_limits(panels, manifest.rendering)
    field_name = str(manifest.rendering.get("field", "channel"))
    default_u_channel = 1 if field_name == "era5_vorticity" else 0
    default_v_channel = 2 if field_name == "era5_vorticity" else 1
    effective = {
        "field": field_name,
        "channel": int(manifest.rendering.get("channel", 0)),
        "u_channel": int(
            manifest.rendering.get("u_channel", default_u_channel)
        ),
        "v_channel": int(
            manifest.rendering.get("v_channel", default_v_channel)
        ),
        "dx": manifest.rendering.get("dx"),
        "dy": manifest.rendering.get("dy"),
        "periodic_longitude": bool(
            manifest.rendering.get("periodic_longitude", True)
        ),
        "periodic_latitude": bool(
            manifest.rendering.get("periodic_latitude", False)
        ),
        "layout": layout,
        "cmap": str(manifest.rendering.get("cmap", "coolwarm")),
        "origin": str(manifest.rendering.get("origin", "lower")),
        "symmetric": bool(manifest.rendering.get("symmetric", True)),
        "vmin": vmin,
        "vmax": vmax,
        "colorbar": bool(manifest.rendering.get("colorbar", True)),
        "dpi": dpi,
        "figsize": list(figsize),
        "labels": dict(labels_value),
        "title": manifest.rendering.get("title"),
    }

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    try:
        for method_index, ((label, _), method_panels) in enumerate(
            zip(row_data, panels)
        ):
            for timestep_index, timestep in enumerate(manifest.timesteps):
                row, column = (
                    (method_index, timestep_index)
                    if layout == "methods_as_rows"
                    else (timestep_index, method_index)
                )
                axis = axes[row, column]
                image = axis.imshow(
                    method_panels[timestep_index],
                    origin=effective["origin"],
                    cmap=effective["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if layout == "methods_as_rows":
                    if method_index == 0:
                        axis.set_title(f"t={timestep}")
                    if timestep_index == 0:
                        axis.set_ylabel(label)
                else:
                    if timestep_index == 0:
                        axis.set_title(label)
                    if method_index == 0:
                        axis.set_ylabel(f"t={timestep}", rotation=0, labelpad=20)
        title = manifest.rendering.get("title")
        if title:
            figure.suptitle(str(title))
        if effective["colorbar"] and image is not None:
            figure.colorbar(
                image, ax=axes.ravel().tolist(), label=effective["field"]
            )
        _save_figure(figure, manifest.output, dpi)
    finally:
        plt.close(figure)

    loaded = [primary, *[artifact for _, artifact in baselines]]
    sidecar = write_provenance_sidecar(
        manifest, artifacts=loaded, rendering=effective
    )
    return manifest.output, sidecar


def render_figure(
    manifest: FigureManifest,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Dispatch a validated manifest to its artifact-only plot function."""

    provenance = provenance_sidecar_path(manifest.output)
    if not overwrite:
        existing = [
            str(path)
            for path in (manifest.output, provenance)
            if os.path.lexists(path)
        ]
        if existing:
            raise FileExistsError(
                "render output exists; pass overwrite=True: "
                + ", ".join(existing)
            )
    lock_path = manifest.output.with_suffix(manifest.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as error:
            try:
                owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
                os.kill(owner_pid, 0)
            except (ProcessLookupError, ValueError):
                lock_path.unlink(missing_ok=True)
                if attempt == 0:
                    continue
            except PermissionError:
                pass
            raise RuntimeError(
                f"render already in progress: {manifest.output}"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        break
    try:
        if manifest.figure_type == "spacetime_1d":
            return plot_spacetime_1d(manifest)
        if manifest.figure_type == "snapshot_grid_2d":
            return plot_snapshot_grid_2d(manifest)
        raise ValueError(f"unsupported figure type {manifest.figure_type!r}")
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = [
    "ArtifactReference",
    "EvaluationArtifact",
    "FigureManifest",
    "load_evaluation_artifact",
    "plot_snapshot_grid_2d",
    "plot_spacetime_1d",
    "provenance_sidecar_path",
    "render_figure",
    "sha256_file",
    "write_provenance_sidecar",
]
