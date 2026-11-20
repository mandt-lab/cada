"""Deterministic CADA evaluation orchestration and domain metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .ablations import tto_da_rollout
from .artifacts import (
    sha256_file,
    write_evaluation_artifact,
    write_metrics_json,
    write_sha256_sidecar,
)
from .assimilation import preview_rollout, uncontrolled_rollout
from .config import EvaluationExperimentConfig, to_resolved_dict
from .data import MinMaxNormalizer, TrajectorySource
from .diffusion import GaussianDiffusion
from .metrics import (
    era5_vorticity,
    high_correlation_time,
    kolmogorov_dissipation,
    pearson_correlation,
    periodic_total_variation_1d,
    rmse,
    synchronized_cuda_time,
    vorticity_2d_spectral,
)
from .observations import ObservationOperator, TemporalCadence


@dataclass(frozen=True)
class EvaluationOutputs:
    """Files and checksums produced by one evaluation run."""

    artifact: Path
    metrics: Path
    artifact_checksum: Path
    metrics_checksum: Path


def _trajectory_domain(
    trajectory: Tensor,
    domain: str,
    normalizer: MinMaxNormalizer,
) -> Tensor:
    if domain == "normalized_model_space":
        return trajectory * 2 - 1
    if domain == "physical_space":
        return normalizer.denormalize(trajectory, channel_dim=2)
    raise ValueError(f"unsupported metric domain {domain!r}")


def _rmse_field(
    config: EvaluationExperimentConfig,
    trajectory: Tensor,
) -> Tensor:
    variable = config.metrics.rmse_variable
    if variable in {"state", "all_channels"}:
        return trajectory
    if variable == "vorticity":
        height, width = config.dataset.resolution
        return vorticity_2d_spectral(
            trajectory,
            dx=2 * pi / width,
            dy=2 * pi / height,
            u_channel=0,
            v_channel=1,
        )
    raise ValueError(f"unsupported RMSE variable {variable!r}")


def _physics_field(
    config: EvaluationExperimentConfig,
    trajectory: Tensor,
) -> Tensor:
    diagnostic = config.metrics.physics_diagnostic
    if diagnostic == "total_variation":
        variation = periodic_total_variation_1d(
            trajectory, normalize=False, reduction="none"
        )
        return variation.mean(dim=-1) if variation.ndim == 3 else variation
    if diagnostic == "dissipation_rate":
        height, width = config.dataset.resolution
        return kolmogorov_dissipation(
            trajectory,
            viscosity=1e-3,
            dx=2 * pi / width,
            dy=2 * pi / height,
            reduction="none",
        )
    if diagnostic == "derived_vorticity":
        height, width = config.dataset.resolution
        return era5_vorticity(
            trajectory,
            dx=60.0 / (width - 1),
            dy=30.0 / (height - 1),
            u_channel=1,
            v_channel=2,
            periodic_longitude=False,
            periodic_latitude=False,
        )
    raise ValueError(f"unsupported physics diagnostic {diagnostic!r}")


def _json_tensor(value: Tensor) -> Any:
    value = value.detach().cpu()
    if value.ndim == 0:
        scalar = float(value)
        return scalar if torch.isfinite(value) else None
    return [_json_tensor(item) for item in value]


def _finite_mean(value: Tensor) -> float | None:
    finite = value[torch.isfinite(value)]
    return float(finite.mean()) if finite.numel() else None


def _method_metrics(
    config: EvaluationExperimentConfig,
    prediction: Tensor,
    truth: Tensor,
    normalizer: MinMaxNormalizer,
) -> dict[str, Any]:
    rmse_prediction = _rmse_field(
        config,
        _trajectory_domain(
            prediction, config.metrics.rmse_domain, normalizer
        ),
    )
    rmse_truth = _rmse_field(
        config,
        _trajectory_domain(truth, config.metrics.rmse_domain, normalizer),
    )
    per_item_rmse = rmse(rmse_prediction, rmse_truth, reduction="none")
    result: dict[str, Any] = {
        "rmse_mean": float(per_item_rmse.mean()),
        "rmse_by_time": _json_tensor(per_item_rmse.mean(dim=0)),
    }

    if config.metrics.high_correlation_time:
        correlation = pearson_correlation(
            rmse_prediction, rmse_truth, reduction="none"
        )
        threshold = config.metrics.correlation_threshold
        assert threshold is not None
        hct = high_correlation_time(
            correlation,
            threshold=threshold,
            dt=1.0,
        )
        hct_physical = hct * config.dataset.physical_dt
        result.update(
            {
                "correlation_by_time": _json_tensor(
                    torch.nanmean(correlation, dim=0)
                ),
                "high_correlation_time_mean": _finite_mean(hct),
                "high_correlation_time_per_trajectory": _json_tensor(hct),
                "high_correlation_time_physical_mean": _finite_mean(hct_physical),
                "high_correlation_time_physical_per_trajectory": _json_tensor(
                    hct_physical
                ),
            }
        )

    physics_prediction = _physics_field(
        config,
        _trajectory_domain(
            prediction, config.metrics.physics_domain, normalizer
        ),
    )
    physics_truth = _physics_field(
        config,
        _trajectory_domain(
            truth, config.metrics.physics_domain, normalizer
        ),
    )
    if physics_prediction.ndim > 2:
        event_dims = tuple(range(2, physics_prediction.ndim))
        physics_error = (
            physics_prediction - physics_truth
        ).abs().mean(dim=event_dims)
        prediction_curve = physics_prediction.mean(dim=event_dims).mean(dim=0)
        truth_curve = physics_truth.mean(dim=event_dims).mean(dim=0)
    else:
        physics_error = (physics_prediction - physics_truth).abs()
        prediction_curve = physics_prediction.mean(dim=0)
        truth_curve = physics_truth.mean(dim=0)
    result.update(
        {
            "physics_mae_mean": float(physics_error.mean()),
            "physics_mae_by_time": _json_tensor(physics_error.mean(dim=0)),
            "physics_prediction_by_time": _json_tensor(prediction_curve),
            "physics_truth_by_time": _json_tensor(truth_curve),
        }
    )
    return result


def compute_core_metrics(
    config: EvaluationExperimentConfig,
    *,
    truth: Tensor,
    uncontrolled: Tensor,
    controlled: Tensor,
    normalizer: MinMaxNormalizer,
) -> dict[str, Any]:
    """Compute configured RMSE, correlation, and physics diagnostics.

    Inputs are normalized data-space trajectories shaped ``[B,T,C,...]``.
    The supplied trajectories include their initial state; metrics intentionally
    cover only forecast states.
    """

    if not isinstance(config, EvaluationExperimentConfig):
        raise TypeError("config must be an EvaluationExperimentConfig")
    if not (truth.shape == uncontrolled.shape == controlled.shape):
        raise ValueError("evaluation trajectories must have identical shapes")
    if truth.ndim not in (4, 5) or truth.shape[1] < 2:
        raise ValueError(
            "evaluation trajectories must be [B,T,C,X] or [B,T,C,H,W]"
        )
    if truth.shape[2] != config.dataset.physical_channels:
        raise ValueError("trajectory channels do not match the evaluation config")

    forecast_truth = truth[:, 1:]
    return {
        "domain": config.dataset.domain,
        "forecast_steps": config.horizon.forecast_steps,
        "trajectories": int(truth.shape[0]),
        "rmse_variable": config.metrics.rmse_variable,
        "rmse_domain": config.metrics.rmse_domain,
        "physics_diagnostic": config.metrics.physics_diagnostic,
        "physics_domain": config.metrics.physics_domain,
        "aggregation": config.metrics.aggregation,
        "uncontrolled": _method_metrics(
            config, uncontrolled[:, 1:], forecast_truth, normalizer
        ),
        "controlled": _method_metrics(
            config, controlled[:, 1:], forecast_truth, normalizer
        ),
    }


def physical_artifact_trajectories(
    normalizer: MinMaxNormalizer,
    *,
    truth: Tensor,
    uncontrolled: Tensor,
    controlled: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert normalized data-space trajectories to physical artifact values."""

    if not (truth.shape == uncontrolled.shape == controlled.shape):
        raise ValueError("artifact trajectories must have identical shapes")
    return tuple(
        normalizer.denormalize(value, channel_dim=2)
        for value in (truth, uncontrolled, controlled)
    )


def _holdout_batch(
    source: TrajectorySource,
    *,
    trajectories: int,
    horizon: int,
) -> Tensor:
    if source.n_trajectories < trajectories:
        raise ValueError(
            f"holdout contains {source.n_trajectories} trajectories, "
            f"but evaluation requests {trajectories}"
        )
    if source.trajectory_length < horizon + 1:
        raise ValueError(
            f"holdout trajectories contain {source.trajectory_length} states, "
            f"but horizon {horizon} requires {horizon + 1}"
        )
    return torch.stack(
        tuple(
            source.read_window(index, 0, horizon + 1)
            for index in range(trajectories)
        )
    )


def _deterministic_noises(
    trajectory: Tensor,
    *,
    horizon: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (horizon, trajectory.shape[0], *trajectory.shape[2:])
    return torch.randn(
        shape,
        generator=generator,
        dtype=trajectory.dtype,
        device="cpu",
    ).to(device)


def run_evaluation(
    config: EvaluationExperimentConfig,
    *,
    source: TrajectorySource,
    normalizer: MinMaxNormalizer,
    diffusion: GaussianDiffusion,
    controller: nn.Module | None,
    observation_operator: ObservationOperator,
    cadence: TemporalCadence,
    layout: Any,
    device: torch.device | str,
    method: Literal["cada", "tto-da"] = "cada",
    tto_iterations: int = 50,
    tto_learning_rate: float = 1e-2,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationOutputs:
    """Run paired rollouts and atomically write strict run-layout outputs."""

    if not isinstance(config, EvaluationExperimentConfig):
        raise TypeError("config must be an EvaluationExperimentConfig")
    if method not in {"cada", "tto-da"}:
        raise ValueError("method must be 'cada' or 'tto-da'")
    if method == "cada" and controller is None:
        raise ValueError("CADA evaluation requires a controller")
    for attribute in ("samples", "metrics", "run_id"):
        if not hasattr(layout, attribute):
            raise TypeError(f"layout is missing required attribute {attribute!r}")
    device = torch.device(device)
    horizon = config.horizon.forecast_steps
    truth = _holdout_batch(
        source,
        trajectories=config.evaluation_trajectories,
        horizon=horizon,
    ).to(device)
    noises = _deterministic_noises(
        truth,
        horizon=horizon,
        seed=config.seeds.diffusion,
        device=device,
    )
    diffusion.eval()
    if controller is not None:
        controller.eval()

    with torch.no_grad():
        uncontrolled_result, uncontrolled_seconds = synchronized_cuda_time(
            uncontrolled_rollout,
            diffusion,
            truth[:, 0],
            horizon,
            sampling_steps=config.prior_model.sampling_steps,
            initial_noises=noises,
            device=device,
        )
    if method == "cada":
        assert controller is not None
        with torch.no_grad():
            method_result, method_seconds = synchronized_cuda_time(
                preview_rollout,
                diffusion,
                controller,
                truth,
                observation_operator,
                cadence,
                preview_steps=config.preview_length,
                gamma=config.gamma,
                beta=config.beta,
                intermediate_weight=config.intermediate_weight,
                sampling_steps=config.prior_model.sampling_steps,
                initial_noises=noises,
                device=device,
            )
        method_parameters: dict[str, Any] = {
            "name": "cada",
            "gamma": config.gamma,
            "beta": config.beta,
            "preview_steps": config.preview_length,
            "seed": config.seeds.diffusion,
        }
        rollout_metadata: dict[str, Any] = {
            "observation_indices": list(method_result.observation_indices),
            "chunk_boundaries": list(method_result.chunk_boundaries),
        }
    else:
        method_result, method_seconds = synchronized_cuda_time(
            tto_da_rollout,
            diffusion,
            truth,
            observation_operator,
            cadence,
            preview_steps=config.preview_length,
            iterations=tto_iterations,
            lr=tto_learning_rate,
            gamma=config.gamma,
            beta=config.beta,
            seed=config.seeds.diffusion,
            intermediate_weight=config.intermediate_weight,
            sampling_steps=config.prior_model.sampling_steps,
            device=device,
        )
        method_parameters = {
            "name": "tto-da",
            "iterations": tto_iterations,
            "lr": tto_learning_rate,
            "gamma": config.gamma,
            "beta": config.beta,
            "preview_steps": config.preview_length,
            "seed": config.seeds.diffusion,
        }
        rollout_metadata = {
            "observation_indices": list(method_result.observation_indices),
            "chunk_boundaries": list(method_result.chunk_boundaries),
            "initial_objective": float(method_result.initial_objective),
            "final_objective": float(method_result.final_objective),
            "gradient_norms": list(method_result.gradient_norms),
        }

    truth_cpu = truth.detach().cpu()
    uncontrolled_cpu = uncontrolled_result.states.detach().cpu()
    controlled_cpu = method_result.states.detach().cpu()
    metrics = compute_core_metrics(
        config,
        truth=truth_cpu,
        uncontrolled=uncontrolled_cpu,
        controlled=controlled_cpu,
        normalizer=normalizer,
    )
    metrics["timing_seconds"] = {
        "uncontrolled": uncontrolled_seconds,
        "controlled": method_seconds,
    }
    metrics["evaluation_method"] = method_parameters
    metrics["controlled_rollout"] = rollout_metadata

    artifact_path = Path(layout.samples) / "evaluation.npz"
    metrics_path = Path(layout.metrics) / "evaluation.json"
    clean_metadata = dict(metadata or {})
    clean_metadata.update(
        {
            "run_id": str(layout.run_id),
            "config_id": config.config_id,
            "resolved_config": to_resolved_dict(config),
            "resolved_evaluation_identity": {
                "base_config": to_resolved_dict(config),
                "evaluation_method": method_parameters,
            },
            "initial_noise_seed": config.seeds.diffusion,
            "evaluation_method": method_parameters,
            "artifact_space": "physical_space",
            "metric_parameters": {
                "kolmogorov_domain_length": 2 * pi,
                "kolmogorov_viscosity": 1e-3,
                "era5_longitude_span_degrees": 60.0,
                "era5_latitude_span_degrees": 30.0,
            },
        }
    )
    artifact_truth, artifact_uncontrolled, artifact_controlled = (
        physical_artifact_trajectories(
            normalizer,
            truth=truth_cpu,
            uncontrolled=uncontrolled_cpu,
            controlled=controlled_cpu,
        )
    )
    write_evaluation_artifact(
        artifact_path,
        truth=artifact_truth,
        uncontrolled=artifact_uncontrolled,
        controlled=artifact_controlled,
        metadata=clean_metadata,
    )
    clean_metadata["artifact_sha256"] = sha256_file(artifact_path)
    write_metrics_json(metrics_path, metrics, metadata=clean_metadata)
    return EvaluationOutputs(
        artifact=artifact_path,
        metrics=metrics_path,
        artifact_checksum=write_sha256_sidecar(artifact_path),
        metrics_checksum=write_sha256_sidecar(metrics_path),
    )


__all__ = [
    "EvaluationOutputs",
    "compute_core_metrics",
    "physical_artifact_trajectories",
    "run_evaluation",
]
