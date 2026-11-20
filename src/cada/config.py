"""Strict, dependency-light configuration loading for paper experiments."""

from __future__ import annotations

import json
import math
import os
import re
import types
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

_CONFIG_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_FIELD_REFERENCE = re.compile(
    r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*(?:\.\*)?$"
)


class ConfigError(ValueError):
    """Raised when a configuration is unsafe, incomplete, or inconsistent."""


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _CONFIG_ID.fullmatch(value):
        raise ConfigError(
            f"{name} must contain lowercase alphanumerics separated by '.', '-', or '_'"
        )


def _require_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")


def _require_positive_float(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a positive number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ConfigError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class SourceProvenance:
    """Separate paper values, executable defaults, and unresolved metadata."""

    paper_reference: str
    paper_fields: tuple[str, ...]
    repository_default_fields: tuple[str, ...]
    unresolved_fields: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_string(self.paper_reference, "paper_reference")
        for name, values in (
            ("paper_fields", self.paper_fields),
            ("repository_default_fields", self.repository_default_fields),
            ("unresolved_fields", self.unresolved_fields),
        ):
            if len(values) != len(set(values)):
                raise ConfigError(f"{name} contains duplicate field paths")
            if any(
                not isinstance(value, str) or not _FIELD_REFERENCE.fullmatch(value)
                for value in values
            ):
                raise ConfigError(f"{name} contains an invalid field path")
        groups = (
            ("paper_fields", set(self.paper_fields)),
            ("repository_default_fields", set(self.repository_default_fields)),
            ("unresolved_fields", set(self.unresolved_fields)),
        )
        for index, (left_name, left) in enumerate(groups):
            for right_name, right in groups[index + 1 :]:
                overlap = left.intersection(right)
                if overlap:
                    raise ConfigError(
                        f"{left_name} and {right_name} overlap: "
                        + ", ".join(sorted(overlap))
                    )
        if any(not isinstance(note, str) or not note for note in self.notes):
            raise ConfigError("provenance notes must be non-empty strings")


def _validate_provenance_paths(owner: object, provenance: SourceProvenance) -> None:
    for group_name in (
        "paper_fields",
        "repository_default_fields",
        "unresolved_fields",
    ):
        for field_path in getattr(provenance, group_name):
            current = owner
            for index, part in enumerate(field_path.split(".")):
                if part == "*":
                    if index != len(field_path.split(".")) - 1 or not is_dataclass(
                        current
                    ):
                        raise ConfigError(
                            f"{group_name} contains invalid wildcard path "
                            f"{field_path!r}"
                        )
                    break
                if not is_dataclass(current) or not hasattr(current, part):
                    raise ConfigError(
                        f"{group_name} references unknown field {field_path!r}"
                    )
                current = getattr(current, part)


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization is independent of metric reporting domains."""

    method: Literal["channelwise_minmax"]
    fit_split: str
    statistics_path: str
    applied_domain: Literal["model_input"]

    def __post_init__(self) -> None:
        if self.method != "channelwise_minmax":
            raise ConfigError("normalization.method must be 'channelwise_minmax'")
        _require_string(self.fit_split, "normalization.fit_split")
        _require_string(self.statistics_path, "normalization.statistics_path")
        if self.applied_domain != "model_input":
            raise ConfigError("normalization.applied_domain must be 'model_input'")


_CANONICAL_DATASETS: dict[
    str, tuple[int, tuple[str, ...], tuple[int, ...], float, str, int, int]
] = {
    "kolmogorov": (
        2,
        ("qx", "qy"),
        (64, 64),
        0.2,
        "simulator_time",
        64,
        180,
    ),
    "ks": (1, ("u",), (256,), 0.2, "simulator_time", 140, 640),
    "era5_small": (
        3,
        ("temperature", "u_wind", "v_wind"),
        (64, 64),
        6.0,
        "hours",
        64,
        60,
    ),
}


@dataclass(frozen=True)
class DatasetConfig:
    """A standalone data manifest, also embedded verbatim in experiments."""

    kind: Literal["dataset"]
    config_id: str
    domain: Literal["kolmogorov", "ks", "era5_small"]
    physical_channels: int
    channel_names: tuple[str, ...]
    resolution: tuple[int, ...]
    physical_dt: float
    physical_dt_unit: Literal["simulator_time", "hours"]
    prior_sequence_length: int
    maximum_evaluation_horizon: int
    train_data_path: str
    holdout_data_path: str
    hdf5_key: str | None
    data_sha256: str | None
    train_split: str
    holdout_split: str
    train_years: tuple[int, ...]
    holdout_years: tuple[int, ...]
    normalization: NormalizationConfig
    source_citation: str
    source_version: str | None
    source_provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.kind != "dataset":
            raise ConfigError("DatasetConfig.kind must be 'dataset'")
        _require_identifier(self.config_id, "config_id")
        if self.domain not in _CANONICAL_DATASETS:
            raise ConfigError(f"unsupported dataset domain {self.domain!r}")
        if self.config_id != f"dataset.{self.domain}":
            raise ConfigError("dataset config_id must identify its domain")
        expected = _CANONICAL_DATASETS[self.domain]
        actual = (
            self.physical_channels,
            self.channel_names,
            self.resolution,
            float(self.physical_dt),
            self.physical_dt_unit,
            self.prior_sequence_length,
            self.maximum_evaluation_horizon,
        )
        if actual != expected:
            raise ConfigError(
                f"dataset {self.domain!r} does not match the paper-canonical "
                "channels, resolution, cadence, or sequence lengths"
            )
        if len(self.channel_names) != self.physical_channels:
            raise ConfigError("channel_names must match physical_channels")
        if len(self.resolution) not in (1, 2):
            raise ConfigError("resolution must have one or two spatial dimensions")
        if any(size < 1 for size in self.resolution):
            raise ConfigError("resolution entries must be positive")
        _require_string(self.train_data_path, "train_data_path")
        _require_string(self.holdout_data_path, "holdout_data_path")
        if self.train_data_path == self.holdout_data_path:
            raise ConfigError("training and holdout data paths must be distinct")
        if (
            Path(self.train_data_path).expanduser().resolve(strict=False)
            == Path(self.holdout_data_path).expanduser().resolve(strict=False)
        ):
            raise ConfigError(
                "training and holdout data paths resolve to the same source"
            )
        expected_path_suffixes = (
            f"/data/{self.domain}/train.h5",
            f"/data/{self.domain}/holdout.h5",
        )
        if not self.train_data_path.endswith(expected_path_suffixes[0]):
            raise ConfigError(
                f"train_data_path must end with {expected_path_suffixes[0]!r}"
            )
        if not self.holdout_data_path.endswith(expected_path_suffixes[1]):
            raise ConfigError(
                f"holdout_data_path must end with {expected_path_suffixes[1]!r}"
            )
        if self.data_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.data_sha256
        ):
            raise ConfigError("data_sha256 must be null or a lowercase SHA-256 digest")
        _require_string(self.train_split, "train_split")
        _require_string(self.holdout_split, "holdout_split")
        _require_string(self.source_citation, "source_citation")
        if self.source_version is not None:
            _require_string(self.source_version, "source_version")
        expected_statistics_suffix = {
            "ks": "/data/ks/normalization.json",
            "kolmogorov": "/data/kolmogorov/normalization.pt",
            "era5_small": "/data/era5_small/normalization.pt",
        }[self.domain]
        if not self.normalization.statistics_path.endswith(
            expected_statistics_suffix
        ):
            raise ConfigError(
                "normalization statistics path does not match the dataset domain"
            )
        expected_unresolved = {
            field_name
            for field_name, value in (
                ("data_sha256", self.data_sha256),
                ("source_version", self.source_version),
            )
            if value is None
        }
        if set(self.source_provenance.unresolved_fields) != expected_unresolved:
            raise ConfigError(
                "dataset unresolved_fields must exactly identify null metadata"
            )
        if self.domain == "era5_small":
            if self.train_years != tuple(range(2006, 2016)):
                raise ConfigError("ERA5 training years must be 2006 through 2015")
            if self.holdout_years != (2016,):
                raise ConfigError("ERA5 holdout year must be 2016")
            if self.normalization.method != "channelwise_minmax":
                raise ConfigError("ERA5 must use training-set channelwise min/max")
        elif self.train_years or self.holdout_years:
            raise ConfigError("PDE datasets use split IDs, not calendar years")
        if self.domain == "ks":
            if (
                self.train_split != "train"
                or self.holdout_split != "test"
                or self.hdf5_key is not None
            ):
                raise ConfigError(
                    "KS must use train/test groups and no fixed HDF5 dataset key"
                )
        elif (
            self.train_split != "train"
            or self.holdout_split != "holdout"
            or self.hdf5_key != "x"
        ):
            raise ConfigError(
                "dense datasets must use train/holdout labels and HDF5 key 'x'"
            )
        required_path_defaults = {"train_data_path", "holdout_data_path"}
        if not required_path_defaults.issubset(
            self.source_provenance.repository_default_fields
        ):
            raise ConfigError(
                "dataset provenance must mark both data paths as repository defaults"
            )
        _validate_provenance_paths(self, self.source_provenance)


@dataclass(frozen=True)
class ModelConfig:
    """Paper-canonical autoregressive diffusion prior."""

    role: Literal["prior"]
    architecture: Literal["unet"]
    spatial_dims: int
    physical_channels: int
    base_width: int
    channel_multipliers: tuple[int, ...]
    learned_sinusoidal_embedding_dim: int
    objective: Literal["pred_v"]
    beta_schedule: Literal["sigmoid"]
    train_diffusion_steps: int
    sampler: Literal["ddim"]
    sampling_steps: int

    def __post_init__(self) -> None:
        if self.role != "prior" or self.architecture != "unet":
            raise ConfigError("prior model must use the U-Net architecture")
        if self.spatial_dims not in (1, 2):
            raise ConfigError("model.spatial_dims must be 1 or 2")
        _require_positive_int(self.physical_channels, "model.physical_channels")
        if (
            self.base_width != 64
            or self.channel_multipliers != (1, 2, 4, 8)
            or self.learned_sinusoidal_embedding_dim != 128
            or self.objective != "pred_v"
            or self.beta_schedule != "sigmoid"
            or self.train_diffusion_steps != 1000
            or self.sampler != "ddim"
            or self.sampling_steps != 3
        ):
            raise ConfigError("prior model does not match the paper-canonical ARDM")


@dataclass(frozen=True)
class ControllerModelConfig:
    """Paper-reported Appendix H controller architecture."""

    architecture: Literal["film_controlnet"]
    hidden_width: int
    group_norm_groups: int

    def __post_init__(self) -> None:
        if (
            self.architecture != "film_controlnet"
            or self.hidden_width != 768
            or self.group_norm_groups != 8
        ):
            raise ConfigError(
                "controller model must match the paper Appendix H architecture"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Executable optimizer settings."""

    name: Literal["adam"]
    learning_rate: float
    weight_decay: float
    schedule: Literal["constant"]

    def __post_init__(self) -> None:
        if self.name != "adam":
            raise ConfigError("optimizer.name must be 'adam'")
        _require_positive_float(self.learning_rate, "optimizer.learning_rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError("optimizer.weight_decay must be finite and non-negative")
        if self.schedule != "constant":
            raise ConfigError("optimizer.schedule must be 'constant'")


@dataclass(frozen=True)
class SeedConfig:
    """Independent reproducibility streams."""

    training: int
    data_order: int
    diffusion: int
    observation: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative_int(value, f"seeds.{name}")
            if value >= 2**63:
                raise ConfigError(f"seeds.{name} must be smaller than 2**63")


@dataclass(frozen=True)
class ObservationConfig:
    """Spatial operator and physical-time arrival schedule are independent."""

    family: Literal["ds", "ms"]
    spatial_operator: Literal["average_pool_lift", "regular_grid_mask"]
    spatial_factor: int
    temporal_cadence_steps: int
    arrival_offsets: tuple[int, ...]
    mask_phase: tuple[int, ...]
    alignment: Literal["lower_index_origin"]
    pooling: Literal["average", "not_applicable"]
    lifting: Literal["nearest", "not_applicable"]
    observation_noise_std: float

    def __post_init__(self) -> None:
        if self.spatial_factor not in (2, 4, 8):
            raise ConfigError("observation.spatial_factor must be 2, 4, or 8")
        _require_positive_int(
            self.temporal_cadence_steps, "observation.temporal_cadence_steps"
        )
        if not self.arrival_offsets:
            raise ConfigError(
                "observation.arrival_offsets must be explicit and non-empty"
            )
        if any(
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 1
            for offset in self.arrival_offsets
        ):
            raise ConfigError("arrival offsets must be positive integers")
        if tuple(sorted(set(self.arrival_offsets))) != self.arrival_offsets:
            raise ConfigError("arrival_offsets must be strictly increasing")
        if any(
            isinstance(phase, bool)
            or not isinstance(phase, int)
            or not 0 <= phase < self.spatial_factor
            for phase in self.mask_phase
        ):
            raise ConfigError("mask_phase entries must lie within the spatial factor")
        if not math.isfinite(float(self.observation_noise_std)):
            raise ConfigError("observation_noise_std must be finite")
        if self.observation_noise_std < 0:
            raise ConfigError("observation_noise_std must be non-negative")
        if self.alignment != "lower_index_origin":
            raise ConfigError("observation alignment must be explicit")
        if self.family == "ds":
            if (
                self.spatial_operator != "average_pool_lift"
                or self.temporal_cadence_steps != 1
                or self.mask_phase
                or self.pooling != "average"
                or self.lifting != "nearest"
            ):
                raise ConfigError("DS must use average pooling/lifting every step")
        elif (
            self.spatial_operator != "regular_grid_mask"
            or self.temporal_cadence_steps != 4
            or not self.mask_phase
            or self.pooling != "not_applicable"
            or self.lifting != "not_applicable"
        ):
            raise ConfigError(
                "MS must use a phased regular-grid mask every fourth step"
            )

    def validate_for(self, dataset: DatasetConfig, preview_length: int) -> None:
        if any(size % self.spatial_factor for size in dataset.resolution):
            raise ConfigError("spatial factor must divide every dataset resolution")
        if self.family == "ms" and len(self.mask_phase) != len(dataset.resolution):
            raise ConfigError("MS mask_phase must have one entry per spatial dimension")
        expected = tuple(
            range(
                self.temporal_cadence_steps,
                preview_length + 1,
                self.temporal_cadence_steps,
            )
        )
        if self.arrival_offsets != expected:
            raise ConfigError(
                "arrival_offsets must explicitly match cadence within preview_length; "
                f"expected {expected}"
            )


@dataclass(frozen=True)
class HorizonConfig:
    """Forecast length semantics cannot be inferred from trajectory shape."""

    forecast_steps: int
    semantics: Literal["predicted_states_excluding_initial_state"]
    final_chunk: Literal["shorten_to_remaining_steps"]

    def __post_init__(self) -> None:
        _require_positive_int(self.forecast_steps, "horizon.forecast_steps")
        if self.semantics != "predicted_states_excluding_initial_state":
            raise ConfigError("horizon semantics must exclude the initial state")
        if self.final_chunk != "shorten_to_remaining_steps":
            raise ConfigError("final preview chunk semantics must be explicit")


@dataclass(frozen=True)
class MetricConfig:
    """Evaluation variables and reporting domains remain explicit."""

    rmse_variable: Literal["vorticity", "state", "all_channels"]
    rmse_domain: Literal["normalized_model_space", "physical_space"]
    physics_diagnostic: Literal[
        "dissipation_rate", "total_variation", "derived_vorticity"
    ]
    physics_domain: Literal["normalized_model_space", "physical_space"]
    high_correlation_time: bool
    correlation_threshold: float | None
    aggregation: str

    def __post_init__(self) -> None:
        if self.high_correlation_time:
            if self.correlation_threshold != 0.9:
                raise ConfigError("paper HCT threshold must be 0.9")
        elif self.correlation_threshold is not None:
            raise ConfigError("correlation_threshold requires high_correlation_time")
        _require_string(self.aggregation, "metrics.aggregation")


def _validate_model_dataset(model: ModelConfig, dataset: DatasetConfig) -> None:
    if model.spatial_dims != len(dataset.resolution):
        raise ConfigError("model spatial dimensions do not match dataset resolution")
    if model.physical_channels != dataset.physical_channels:
        raise ConfigError("model physical channels do not match dataset channels")


def _validate_preview(preview_length: int, dataset: DatasetConfig) -> None:
    canonical = {"kolmogorov": 16, "ks": 54, "era5_small": 16}[dataset.domain]
    if preview_length != canonical:
        raise ConfigError(
            f"{dataset.domain} preview_length must be the paper value {canonical}"
        )


@dataclass(frozen=True)
class PriorExperimentConfig:
    kind: Literal["prior"]
    config_id: str
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    batch_size: int
    train_steps: int
    gradient_accumulation_steps: int
    precision: Literal["fp16"]
    ema_decay: float
    ema_update_every: int
    seeds: SeedConfig
    checkpoint_path: str
    source_provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.kind != "prior":
            raise ConfigError("PriorExperimentConfig.kind must be 'prior'")
        _require_identifier(self.config_id, "config_id")
        if self.config_id != f"prior.{self.dataset.domain}":
            raise ConfigError("prior config_id must identify its dataset domain")
        _validate_model_dataset(self.model, self.dataset)
        if self.optimizer != OptimizerConfig(
            name="adam",
            learning_rate=3.2e-4,
            weight_decay=0.0,
            schedule="constant",
        ):
            raise ConfigError(
                "prior optimizer must use Adam with the paper learning rate"
            )
        if (
            self.batch_size != 32
            or self.train_steps != 1_000_000
            or self.gradient_accumulation_steps != 1
            or self.precision != "fp16"
            or self.ema_decay != 0.995
            or self.ema_update_every != 10
        ):
            raise ConfigError("prior training does not match paper-canonical values")
        _require_string(self.checkpoint_path, "checkpoint_path")
        required_defaults = {
            "optimizer.name",
            "optimizer.weight_decay",
            "optimizer.schedule",
            "seeds.*",
            "checkpoint_path",
        }
        if not required_defaults.issubset(
            self.source_provenance.repository_default_fields
        ):
            raise ConfigError(
                "prior provenance must identify executable repository defaults"
            )
        if "optimizer.learning_rate" not in self.source_provenance.paper_fields:
            raise ConfigError(
                "prior provenance must mark its learning rate as paper-reported"
            )
        if self.source_provenance.unresolved_fields:
            raise ConfigError(
                "prior provenance cannot contain unresolved runtime fields"
            )
        _validate_provenance_paths(self, self.source_provenance)


@dataclass(frozen=True)
class ControllerExperimentConfig:
    kind: Literal["controller"]
    config_id: str
    dataset: DatasetConfig
    prior_config_id: str
    prior_checkpoint_path: str
    prior_model: ModelConfig
    controller_model: ControllerModelConfig
    optimizer: OptimizerConfig
    batch_size: int
    train_steps: int
    gradient_accumulation_steps: int
    precision: Literal["fp16"]
    max_grad_norm: float
    intermediate_weight: float
    preview_length: int
    gamma: float
    beta: float
    observation: ObservationConfig
    seeds: SeedConfig
    checkpoint_path: str
    source_provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.kind != "controller":
            raise ConfigError("ControllerExperimentConfig.kind must be 'controller'")
        _require_identifier(self.config_id, "config_id")
        _require_identifier(self.prior_config_id, "prior_config_id")
        expected_id = (
            f"controller.{self.dataset.domain}."
            f"{self.observation.family}_{self.observation.spatial_factor}"
        )
        if self.config_id != expected_id:
            raise ConfigError(
                "controller config_id must identify its domain and observation regime"
            )
        if self.prior_config_id != f"prior.{self.dataset.domain}":
            raise ConfigError("prior_config_id must identify the dataset prior")
        _validate_model_dataset(self.prior_model, self.dataset)
        _validate_preview(self.preview_length, self.dataset)
        self.observation.validate_for(self.dataset, self.preview_length)
        if self.gamma != 0.1 or self.beta != 1e-2:
            raise ConfigError("controller gamma and beta must be 0.1 and 1e-2")
        if self.optimizer != OptimizerConfig(
            name="adam",
            learning_rate=5e-5,
            weight_decay=0.0,
            schedule="constant",
        ):
            raise ConfigError(
                "controller optimizer must use the repository Adam defaults"
            )
        expected_train_steps = {
            "kolmogorov": 200_000,
            "ks": 100_000,
            "era5_small": 150_000,
        }[self.dataset.domain]
        if (
            self.batch_size != 8
            or self.train_steps != expected_train_steps
            or self.gradient_accumulation_steps != 1
            or self.precision != "fp16"
            or self.max_grad_norm != 1.0
            or self.intermediate_weight != 1.0
        ):
            raise ConfigError(
                "controller training settings do not match repository defaults"
            )
        for name, value in (
            ("prior_checkpoint_path", self.prior_checkpoint_path),
            ("checkpoint_path", self.checkpoint_path),
        ):
            _require_string(value, name)
        required_defaults = {
            "optimizer.*",
            "batch_size",
            "train_steps",
            "gradient_accumulation_steps",
            "precision",
            "max_grad_norm",
            "intermediate_weight",
            "beta",
        }
        if not required_defaults.issubset(
            set(self.source_provenance.repository_default_fields)
        ):
            raise ConfigError(
                "controller provenance must mark optimizer, batch, and "
                "training loss settings as repository defaults"
            )
        if "controller_model.*" not in self.source_provenance.paper_fields:
            raise ConfigError(
                "controller provenance must mark its Appendix H model as paper-reported"
            )
        if self.source_provenance.unresolved_fields:
            raise ConfigError(
                "controller provenance cannot contain unresolved runtime fields"
            )
        _validate_provenance_paths(self, self.source_provenance)


@dataclass(frozen=True)
class EvaluationExperimentConfig:
    kind: Literal["evaluation"]
    config_id: str
    dataset: DatasetConfig
    prior_model: ModelConfig
    prior_checkpoint_path: str
    controller_config_id: str
    controller_model: ControllerModelConfig
    controller_checkpoint_path: str
    gamma: float
    beta: float
    intermediate_weight: float
    evaluation_trajectories: int
    observation: ObservationConfig
    preview_length: int
    horizon: HorizonConfig
    seeds: SeedConfig
    metrics: MetricConfig
    source_provenance: SourceProvenance

    def __post_init__(self) -> None:
        if self.kind != "evaluation":
            raise ConfigError("EvaluationExperimentConfig.kind must be 'evaluation'")
        _require_identifier(self.config_id, "config_id")
        _require_identifier(self.controller_config_id, "controller_config_id")
        expected_controller_id = (
            f"controller.{self.dataset.domain}."
            f"{self.observation.family}_{self.observation.spatial_factor}"
        )
        if self.controller_config_id != expected_controller_id:
            raise ConfigError(
                "controller_config_id must identify the evaluation observation regime"
            )
        expected_config_id = (
            f"evaluation.{self.dataset.domain}."
            f"{self.observation.family}_{self.observation.spatial_factor}."
            f"h{self.horizon.forecast_steps}"
        )
        if self.config_id != expected_config_id:
            raise ConfigError(
                "evaluation config_id must identify its domain, regime, and horizon"
            )
        _validate_model_dataset(self.prior_model, self.dataset)
        if self.gamma != 0.1 or self.beta != 1e-2:
            raise ConfigError("evaluation gamma and beta must be 0.1 and 1e-2")
        if self.intermediate_weight != 1.0:
            raise ConfigError("evaluation intermediate_weight must be 1.0")
        if self.evaluation_trajectories != 8:
            raise ConfigError("evaluation_trajectories must be 8")
        _validate_preview(self.preview_length, self.dataset)
        self.observation.validate_for(self.dataset, self.preview_length)
        expected_horizons = {
            "kolmogorov": (60, 180),
            "ks": (140, 640),
            "era5_small": (60,),
        }[self.dataset.domain]
        if self.horizon.forecast_steps not in expected_horizons:
            raise ConfigError(
                f"{self.dataset.domain} evaluation horizon must be one of "
                f"{expected_horizons}"
            )
        expected_metric = {
            "kolmogorov": ("vorticity", "dissipation_rate", True),
            "ks": ("state", "total_variation", True),
            "era5_small": ("all_channels", "derived_vorticity", False),
        }[self.dataset.domain]
        actual_metric = (
            self.metrics.rmse_variable,
            self.metrics.physics_diagnostic,
            self.metrics.high_correlation_time,
        )
        if actual_metric != expected_metric:
            raise ConfigError("metrics do not match the dataset's paper evaluation")
        _require_string(self.prior_checkpoint_path, "prior_checkpoint_path")
        _require_string(self.controller_checkpoint_path, "controller_checkpoint_path")
        required_paper_fields = {"controller_model.*", "gamma", "beta"}
        if not required_paper_fields.issubset(
            self.source_provenance.paper_fields
        ):
            raise ConfigError(
                "evaluation provenance must identify paper controller settings"
            )
        required_defaults = {
            "prior_checkpoint_path",
            "intermediate_weight",
            "evaluation_trajectories",
        }
        if not required_defaults.issubset(
            self.source_provenance.repository_default_fields
        ):
            raise ConfigError(
                "evaluation provenance must identify executable repository defaults"
            )
        if self.source_provenance.unresolved_fields:
            raise ConfigError(
                "evaluation provenance cannot contain unresolved runtime fields"
            )
        _validate_provenance_paths(self, self.source_provenance)


ExperimentConfig: TypeAlias = (
    DatasetConfig
    | PriorExperimentConfig
    | ControllerExperimentConfig
    | EvaluationExperimentConfig
)

_KIND_TYPES: dict[str, type[ExperimentConfig]] = {
    "dataset": DatasetConfig,
    "prior": PriorExperimentConfig,
    "controller": ControllerExperimentConfig,
    "evaluation": EvaluationExperimentConfig,
}


def _convert_value(value: object, annotation: object, path: str) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Literal:
        if value not in arguments:
            raise ConfigError(f"{path} must be one of {arguments}, got {value!r}")
        return value

    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a YAML sequence")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _convert_value(item, arguments[0], f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ConfigError(f"{path} must contain exactly {len(arguments)} entries")
        return tuple(
            _convert_value(item, item_type, f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments))
        )

    if origin in (Union, types.UnionType):
        errors: list[str] = []
        for option in arguments:
            try:
                return _convert_value(value, option, path)
            except ConfigError as error:
                errors.append(str(error))
        raise ConfigError(f"{path} has invalid type: {'; '.join(errors)}")

    if annotation is type(None):
        if value is not None:
            raise ConfigError(f"{path} must be null")
        return None

    if isinstance(annotation, type) and is_dataclass(annotation):
        return _build_dataclass(annotation, value, path)

    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{path} must be an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"{path} must be a number")
        return float(value)
    if annotation is str:
        if type(value) is not str:
            raise ConfigError(f"{path} must be a string")
        return value

    raise TypeError(f"unsupported schema annotation {annotation!r} at {path}")


def _build_dataclass(
    config_type: type[Any], value: object, path: str
) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a YAML mapping")
    schema_fields = {field.name: field for field in fields(config_type)}
    unknown = sorted(set(value).difference(schema_fields))
    if unknown:
        raise ConfigError(f"{path} has unknown key(s): {', '.join(unknown)}")
    missing = sorted(
        name
        for name, field in schema_fields.items()
        if name not in value
        and field.default is MISSING
        and field.default_factory is MISSING
    )
    if missing:
        raise ConfigError(f"{path} is missing key(s): {', '.join(missing)}")
    type_hints = get_type_hints(config_type)
    converted = {
        name: _convert_value(value[name], type_hints[name], f"{path}.{name}")
        for name in schema_fields
        if name in value
    }
    try:
        return config_type(**converted)
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from error


def _expand_environment(value: object, cada_root: str, path: str) -> object:
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, cada_root, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_environment(item, cada_root, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name != "CADA_ROOT":
                raise ConfigError(
                    f"{path} references unsupported environment variable {name!r}"
                )
            return cada_root

        expanded = _ENVIRONMENT_VARIABLE.sub(replace, value)
        if "${" in expanded:
            raise ConfigError(f"{path} contains unsupported environment syntax")
        return expanded
    return value


def _validate_path_templates(value: object, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key.endswith("_path"):
                if not isinstance(item, str) or not item.startswith("${CADA_ROOT}/"):
                    raise ConfigError(
                        f"{item_path} must be rooted at '${{CADA_ROOT}}/'"
                    )
            _validate_path_templates(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_path_templates(item, f"{path}[{index}]")


def _validate_resolved_paths(config: ExperimentConfig, cada_root: str) -> None:
    root = os.path.normpath(os.path.expanduser(cada_root))
    if not os.path.isabs(root):
        raise ConfigError("CADA_ROOT must resolve to an absolute path")

    def visit(value: object, path: str = "config") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}"
                if (
                    key.endswith("_path")
                    and isinstance(item, str)
                ):
                    candidate = os.path.normpath(os.path.expanduser(item))
                    if not os.path.isabs(candidate):
                        raise ConfigError(
                            f"{item_path} must resolve to an absolute path"
                        )
                    try:
                        inside_root = os.path.commonpath((root, candidate)) == root
                    except ValueError as error:
                        raise ConfigError(
                            f"{item_path} resolves outside CADA_ROOT"
                        ) from error
                    if not inside_root:
                        raise ConfigError(
                            f"{item_path} resolves outside CADA_ROOT"
                        )
                visit(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(to_resolved_dict(config))


def _safe_yaml_load(text: str, source: str) -> object:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to load CADA YAML configurations"
        ) from error

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: Any, node: Any, deep: bool = False
    ) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError(f"{source} contains a non-string mapping key")
            if key in mapping:
                raise ConfigError(f"{source} contains duplicate key {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        for token in yaml.scan(text):
            if isinstance(
                token,
                (
                    yaml.tokens.AnchorToken,
                    yaml.tokens.AliasToken,
                    yaml.tokens.TagToken,
                ),
            ):
                raise ConfigError(
                    f"{source} uses a YAML anchor, alias, or explicit tag"
                )
        return yaml.load(text, Loader=UniqueKeySafeLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {source}: {error}") from error


def load_config_text(
    text: str,
    *,
    source: str = "<string>",
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    """Safely parse, expand, dispatch, and validate one standalone YAML config."""

    raw = _safe_yaml_load(text, source)
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must contain one top-level YAML mapping")
    _validate_path_templates(raw)
    kind = raw.get("kind")
    if kind not in _KIND_TYPES:
        expected = ", ".join(sorted(_KIND_TYPES))
        raise ConfigError(
            f"{source} has unknown config kind {kind!r}; expected {expected}"
        )
    environment: dict[str, str] = {}
    if env is not None:
        environment.update(env)
    cada_root = environment.get("CADA_ROOT", os.environ.get("CADA_ROOT"))
    if cada_root is None:
        raise ConfigError("CADA_ROOT is not set")
    if not isinstance(cada_root, str) or not cada_root:
        raise ConfigError("CADA_ROOT must be a non-empty string")
    cada_root = os.path.normpath(os.path.expanduser(cada_root))
    if not os.path.isabs(cada_root):
        raise ConfigError("CADA_ROOT must resolve to an absolute path")
    if "${" in cada_root:
        raise ConfigError("CADA_ROOT cannot contain environment placeholders")
    expanded = _expand_environment(raw, cada_root, "config")
    config = _build_dataclass(_KIND_TYPES[kind], expanded, "config")
    _validate_resolved_paths(config, cada_root)
    return config


def load_config(
    path: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    """Load one standalone YAML file with strict schema validation."""

    config_path = Path(path)
    return load_config_text(
        config_path.read_text(encoding="utf-8"),
        source=str(config_path),
        env=env,
    )


def to_resolved_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize a validated config to a resolved JSON-compatible dictionary."""

    if not is_dataclass(config):
        raise TypeError("config must be a validated CADA configuration dataclass")

    def convert(value: object) -> object:
        if is_dataclass(value):
            return {
                field.name: convert(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(f"configuration contains non-JSON value {value!r}")

    result = convert(config)
    if not isinstance(result, dict):
        raise TypeError("top-level configuration must serialize to a dictionary")
    json.dumps(result, allow_nan=False)
    return result


__all__ = [
    "ConfigError",
    "ControllerExperimentConfig",
    "ControllerModelConfig",
    "DatasetConfig",
    "EvaluationExperimentConfig",
    "ExperimentConfig",
    "HorizonConfig",
    "MetricConfig",
    "ModelConfig",
    "NormalizationConfig",
    "ObservationConfig",
    "OptimizerConfig",
    "PriorExperimentConfig",
    "SeedConfig",
    "SourceProvenance",
    "load_config",
    "load_config_text",
    "to_resolved_dict",
]
