"""Strict configuration loading for the isolated GenCast runtime."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ConfigKind = Literal["data", "smoke", "controller", "evaluation", "figure"]
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_PLACEHOLDER = re.compile(r"\$\{(CADA_ROOT|GENCAST_DATA_ROOT)\}")
_MODULES: dict[str, tuple[str, ...]] = {
    "data": (
        "gencast_da.data_contract",
        "gencast_da.build_mini_era5_nc",
    ),
    "smoke": (
        "gencast_da.smoke_load",
        "gencast_da.smoke_cada",
    ),
    "controller": ("gencast_da.train_cada_controller",),
    "evaluation": ("gencast_da.run_cada_gencast",),
    "figure": ("gencast_da.plot_cada_temperature",),
}


class ConfigError(ValueError):
    """Raised when a GenCast YAML configuration is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class ExperimentConfig:
    kind: ConfigKind
    config_id: str
    stage: str
    module: str
    data_root: str
    parameters: Mapping[str, Any]
    seeds: Mapping[str, int]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _MODULES:
            raise ConfigError(f"unsupported GenCast config kind: {self.kind!r}")
        if _IDENTIFIER.fullmatch(self.config_id) is None:
            raise ConfigError(f"invalid config_id: {self.config_id!r}")
        if _IDENTIFIER.fullmatch(self.stage) is None:
            raise ConfigError(f"invalid stage: {self.stage!r}")
        if self.module not in _MODULES[self.kind]:
            raise ConfigError(
                f"module {self.module!r} is not allowed for kind {self.kind!r}"
            )
        if not Path(self.data_root).is_absolute():
            raise ConfigError("data_root must resolve to an absolute path")
        if not isinstance(self.parameters, Mapping):
            raise ConfigError("parameters must be a mapping")
        for key in self.parameters:
            if not isinstance(key, str) or not key or "_" in key:
                raise ConfigError(
                    "parameter names must be non-empty CLI names using hyphens"
                )
        if not isinstance(self.seeds, Mapping) or not self.seeds:
            raise ConfigError("seeds must be a non-empty mapping")
        for key, value in self.seeds.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ConfigError("seeds must map names to non-negative integers")
        executable_seed = self.parameters.get("seed")
        if executable_seed is not None:
            if isinstance(executable_seed, bool) or not isinstance(
                executable_seed, int
            ):
                raise ConfigError("parameters.seed must be a non-negative integer")
            if self.kind == "controller":
                expected = {
                    "training": executable_seed,
                    "observation": executable_seed * 13 + 11,
                    "diffusion": executable_seed,
                }
            elif self.kind == "evaluation":
                expected = {
                    "observation": executable_seed * 17 + 5,
                    "diffusion": executable_seed,
                }
            elif self.kind == "smoke":
                expected = {
                    "model": executable_seed,
                    "observation": executable_seed,
                }
            else:
                expected = dict(self.seeds)
            if dict(self.seeds) != expected:
                raise ConfigError(
                    "declarative seeds do not match executable seed derivation"
                )
        if self.kind == "evaluation":
            if self.parameters.get("steps", 0) < 1:
                raise ConfigError("evaluation steps must be positive")
            if self.parameters.get("num-noise-levels", 0) < 2:
                raise ConfigError("evaluation requires at least two noise levels")
        if self.kind == "smoke" and self.parameters.get("num-noise-levels") != 4:
            raise ConfigError("bounded smoke requires exactly four noise levels")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda match: environment[match.group(1)], value)
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, environment) for key, item in value.items()}
    return value


def load_config(
    path: str | os.PathLike[str],
    *,
    cada_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(loaded, dict):
        raise ConfigError("GenCast config must contain a YAML mapping")

    resolved_cada_root = Path(cada_root).expanduser().resolve(strict=False)
    resolved_data_root = Path(
        data_root if data_root is not None else resolved_cada_root / "data" / "gencast"
    ).expanduser()
    if not resolved_data_root.is_absolute():
        raise ConfigError("GENCAST_DATA_ROOT must be absolute")
    resolved = _expand(
        loaded,
        {
            "CADA_ROOT": str(resolved_cada_root),
            "GENCAST_DATA_ROOT": str(resolved_data_root),
        },
    )
    required = {
        "kind",
        "config_id",
        "stage",
        "module",
        "data_root",
        "parameters",
        "seeds",
    }
    missing = sorted(required - set(resolved))
    extra = sorted(set(resolved) - required - {"notes"})
    if missing:
        raise ConfigError(f"missing GenCast config fields: {missing}")
    if extra:
        raise ConfigError(f"unknown GenCast config fields: {extra}")
    return ExperimentConfig(
        kind=resolved["kind"],
        config_id=resolved["config_id"],
        stage=resolved["stage"],
        module=resolved["module"],
        data_root=resolved["data_root"],
        parameters=dict(resolved["parameters"]),
        seeds=dict(resolved["seeds"]),
        notes=tuple(resolved.get("notes", ())),
    )
