"""Core CADA sampling and preview assimilation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .control import ControlContext
from .diffusion import GaussianDiffusion
from .observations import (
    BernoulliMask,
    Observation,
    ObservationOperator,
    RegularGridMask,
    TemporalCadence,
    apply_observation,
)


@dataclass(frozen=True)
class StepMetrics:
    """Differentiable losses and diagnostics for one physical transition."""

    controlled: bool
    observation_available: bool
    target_index: int | None
    total_loss: Tensor
    observation_loss: Tensor
    final_observation_mse: Tensor
    intermediate_observation_mse: Tensor
    final_full_state_rmse: Tensor
    intermediate_full_state_rmse: Tensor
    weighted_full_state_rmse: Tensor
    control_rms: Tensor
    denoising_steps: int


@dataclass(frozen=True)
class RolloutResult:
    """A trajectory and transition-aligned CADA metrics.

    ``states`` has shape ``[B, T + 1, C, ...]``. ``step_metrics`` has one
    entry for each of the ``T`` transitions, including uncontrolled ones.
    """

    states: Tensor
    step_metrics: tuple[StepMetrics, ...]
    observation_indices: tuple[int, ...]
    chunk_boundaries: tuple[int, ...]

    @property
    def controlled_metrics(self) -> tuple[StepMetrics, ...]:
        return tuple(metric for metric in self.step_metrics if metric.controlled)

    @property
    def loss(self) -> Tensor:
        """Arrival-normalized observation loss plus step-normalized dense loss."""

        controlled = self.controlled_metrics
        if not controlled:
            return self.states.sum() * 0
        dense_loss = torch.stack(
            tuple(metric.weighted_full_state_rmse for metric in controlled)
        ).mean()
        observed = tuple(
            metric for metric in controlled if metric.observation_available
        )
        if not observed:
            return dense_loss
        observation_loss = torch.stack(
            tuple(metric.observation_loss for metric in observed)
        ).mean()
        return observation_loss + dense_loss


def _validate_state(diffusion: GaussianDiffusion, state: Tensor, name: str) -> None:
    if not isinstance(state, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if state.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape [B,C,X] or [B,C,H,W]")
    if state.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if state.shape[1] != diffusion.physical_channels:
        raise ValueError(
            f"{name} must have {diffusion.physical_channels} physical channels"
        )
    if not state.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if state.device != diffusion.device:
        raise ValueError(f"{name} and diffusion must be on the same device")


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _normalized_context(
    value: float | Tensor,
    *,
    batch: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    context = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if context.ndim == 0 or context.numel() == 1:
        context = context.reshape(1).expand(batch)
    elif context.shape == (batch, 1):
        context = context[:, 0]
    elif context.shape != (batch,):
        raise ValueError(f"{name} must be scalar or contain one value per batch")
    if not bool(torch.isfinite(context).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((context < 0) | (context > 1)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return context


def _ddim_time_pairs(
    diffusion: GaussianDiffusion, sampling_steps: int | None
) -> tuple[tuple[int, int], ...]:
    steps = diffusion.sampling_steps if sampling_steps is None else sampling_steps
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("sampling_steps must be an integer")
    if not 1 <= steps <= diffusion.num_timesteps:
        raise ValueError("sampling_steps must be between 1 and timesteps")
    times = (
        torch.linspace(
            -1,
            diffusion.num_timesteps - 1,
            steps=steps + 1,
            device=diffusion.device,
        )
        .long()
        .flip(0)
    )
    return tuple(
        (int(time.item()), int(next_time.item()))
        for time, next_time in zip(times[:-1], times[1:])
    )


def _validate_active_observation(
    target_state: Tensor, observation: Observation
) -> None:
    if not isinstance(observation, Observation):
        raise TypeError("active_observation must be an Observation")
    if observation.value.shape != target_state.shape:
        raise ValueError(
            "active observation and target state must have identical shapes"
        )
    if observation.mask.shape != target_state.shape:
        raise ValueError("active mask and target state must have identical shapes")
    if observation.value.device != target_state.device:
        raise ValueError("active observation and target state must share a device")
    if observation.mask.device != target_state.device:
        raise ValueError("active mask and target state must share a device")
    if (
        not observation.value.is_floating_point()
        or not observation.mask.is_floating_point()
    ):
        raise TypeError("active observation value and mask must be floating point")
    if not bool(torch.isfinite(observation.value).all()):
        raise ValueError("active observation must be finite")
    if not bool(torch.isfinite(observation.mask).all()):
        raise ValueError("active mask must be finite")
    if bool(((observation.mask < 0) | (observation.mask > 1)).any()):
        raise ValueError("active mask values must lie in [0, 1]")


def _observe_prediction(
    prediction: Tensor,
    active_observation: Observation,
    operator: ObservationOperator,
) -> Tensor:
    # A materialized mask is the actual realization for regular/Bernoulli
    # point observations. Recalling a stateful Bernoulli operator would draw
    # a different mask.
    if isinstance(operator, (BernoulliMask, RegularGridMask)):
        return prediction * active_observation.mask

    predicted = apply_observation(prediction, operator)
    if not torch.allclose(
        predicted.mask,
        active_observation.mask,
        rtol=0,
        atol=0,
    ):
        raise ValueError("observation operator changed the active observation mask")
    return predicted.value


def _observed_mse(
    prediction: Tensor,
    active_observation: Observation,
    operator: ObservationOperator,
) -> Tensor:
    predicted_observation = _observe_prediction(
        prediction, active_observation, operator
    )
    mask = active_observation.mask.to(device=prediction.device, dtype=prediction.dtype)
    observed = mask.flatten(1).sum(1)
    if bool((observed <= 0).any()):
        raise ValueError("every sample must contain at least one observed point")
    squared_error = (predicted_observation - active_observation.value).square() * mask
    return (squared_error.flatten(1).sum(1) / observed).mean()


def _full_state_rmse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and full-state target must have identical shapes")
    error = (prediction - target).flatten(1)
    return (torch.linalg.vector_norm(error, dim=1) / (error.shape[1] ** 0.5)).mean()


def _aggregate_substep_losses(
    losses: list[Tensor],
    intermediate_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not losses:
        raise ValueError("at least one denoising substep loss is required")
    final = losses[-1]
    intermediate = (
        torch.stack(losses[:-1]).mean()
        if len(losses) > 1
        else final * 0
    )
    return final, intermediate, final + intermediate_weight * intermediate


def _zero_metrics(reference: Tensor) -> StepMetrics:
    zero = reference.sum() * 0
    return StepMetrics(
        controlled=False,
        observation_available=False,
        target_index=None,
        total_loss=zero,
        observation_loss=zero,
        final_observation_mse=zero,
        intermediate_observation_mse=zero,
        final_full_state_rmse=zero,
        intermediate_full_state_rmse=zero,
        weighted_full_state_rmse=zero,
        control_rms=zero,
        denoising_steps=0,
    )


def _noise_at(
    initial_noises: Tensor | None,
    index: int,
    steps: int,
    state: Tensor,
) -> Tensor | None:
    if initial_noises is None:
        return None
    expected = (steps, *state.shape)
    if initial_noises.shape != expected:
        raise ValueError(
            f"initial_noises must have shape {expected}, got "
            f"{tuple(initial_noises.shape)}"
        )
    if initial_noises.device != state.device:
        raise ValueError("initial_noises and state must share a device")
    return initial_noises[index]


def uncontrolled_rollout(
    diffusion: GaussianDiffusion,
    initial_state: Tensor,
    steps: int,
    *,
    sampling_steps: int | None = None,
    initial_noises: Tensor | None = None,
    clip_x_start: bool = True,
) -> RolloutResult:
    """Autoregressively sample an uncontrolled physical trajectory."""

    _validate_state(diffusion, initial_state, "initial_state")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    _ddim_time_pairs(diffusion, sampling_steps)

    current = initial_state
    states = [current]
    metrics: list[StepMetrics] = []
    for index in range(steps):
        current = diffusion.ddim_sample(
            current,
            initial_noise=_noise_at(initial_noises, index, steps, current),
            sampling_steps=sampling_steps,
            clip_x_start=clip_x_start,
        )
        states.append(current)
        metrics.append(_zero_metrics(current))
    return RolloutResult(
        states=torch.stack(states, dim=1),
        step_metrics=tuple(metrics),
        observation_indices=(),
        chunk_boundaries=(),
    )


def controlled_ddim_next_state(
    diffusion: GaussianDiffusion,
    controller: nn.Module,
    previous_state: Tensor,
    target_state: Tensor,
    observation_operator: ObservationOperator,
    *,
    observation_lead: float | Tensor,
    rollout_position: float | Tensor,
    gamma: float,
    beta: float,
    intermediate_weight: float = 1.0,
    target_index: int | None = None,
    active_observation: Observation | None = None,
    ground_truth_state: Tensor,
    observation_available: bool,
    initial_noise: Tensor | None = None,
    sampling_steps: int | None = None,
    clip_x_start: bool = True,
) -> tuple[Tensor, StepMetrics]:
    """Sample one controlled transition with deterministic eta-zero DDIM.

    The sampler is deterministic for a supplied ``initial_noise``. Prior
    parameters may be frozen, but prior operations remain in the autograd
    graph so observation or dense full-state losses train the controller.
    The active observation may lie later in the preview than the supervised
    physical transition.
    """

    _validate_state(diffusion, previous_state, "previous_state")
    _validate_state(diffusion, target_state, "target_state")
    _validate_state(diffusion, ground_truth_state, "ground_truth_state")
    if previous_state.shape != target_state.shape:
        raise ValueError("previous_state and target_state must have identical shapes")
    if previous_state.shape != ground_truth_state.shape:
        raise ValueError(
            "previous_state and ground_truth_state must have identical shapes"
        )
    if not isinstance(observation_available, bool):
        raise TypeError("observation_available must be a boolean")
    if getattr(controller, "physical_channels", None) != diffusion.physical_channels:
        raise ValueError("controller and diffusion physical channels must match")
    gamma = _validate_nonnegative(gamma, "gamma")
    beta = _validate_nonnegative(beta, "beta")
    intermediate_weight = _validate_nonnegative(
        intermediate_weight, "intermediate_weight"
    )
    if target_index is not None and (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or target_index < 1
    ):
        raise ValueError("target_index must be a positive integer or None")

    time_pairs = _ddim_time_pairs(diffusion, sampling_steps)
    batch = previous_state.shape[0]
    lead = _normalized_context(
        observation_lead,
        batch=batch,
        reference=previous_state,
        name="observation_lead",
    )
    position = _normalized_context(
        rollout_position,
        batch=batch,
        reference=previous_state,
        name="rollout_position",
    )
    if active_observation is None:
        active_observation = apply_observation(target_state, observation_operator)
    _validate_active_observation(target_state, active_observation)

    condition = diffusion.normalize(previous_state)
    observation_value = (
        diffusion.normalize(active_observation.value) * active_observation.mask
    )
    mask = active_observation.mask.to(device=condition.device, dtype=condition.dtype)
    if initial_noise is None:
        latent = torch.randn_like(condition)
    else:
        if initial_noise.shape != condition.shape:
            raise ValueError(
                "initial_noise and previous_state must have identical shapes"
            )
        if initial_noise.device != condition.device:
            raise ValueError("initial_noise and previous_state must share a device")
        latent = initial_noise

    previous_control = torch.zeros_like(latent)
    observation_mses: list[Tensor] = []
    full_state_rmses: list[Tensor] = []
    control_squares: list[Tensor] = []

    for time, next_time in time_pairs:
        times = torch.full((batch,), time, device=condition.device, dtype=torch.long)
        alpha = diffusion.alphas_cumprod[time].to(dtype=condition.dtype)
        log_snr = torch.log(alpha.clamp_min(1e-12)) - torch.log1p(
            -alpha.clamp(max=1 - 1e-12)
        )
        context = ControlContext(
            dt=lead,
            tau=position,
            snr=log_snr.expand(batch),
        )
        control = controller(
            latent,
            condition,
            observation_value,
            mask,
            previous_control.detach(),
            context,
        )
        if control.shape != latent.shape:
            raise ValueError("controller output must match one physical state")
        controlled_latent = latent + gamma * control
        prediction = diffusion.model_predictions(
            controlled_latent,
            condition,
            times,
            clip_x_start=clip_x_start,
        )
        tweedie_state = diffusion.unnormalize(prediction.pred_x_start)
        zero = tweedie_state.sum() * 0
        full_state_rmses.append(_full_state_rmse(tweedie_state, ground_truth_state))
        if observation_available:
            observation_mses.append(
                _observed_mse(
                    tweedie_state,
                    active_observation,
                    observation_operator,
                )
            )
        else:
            observation_mses.append(zero)
        control_square = (gamma * control).square().flatten(1).mean(1).mean()
        control_squares.append(control_square)
        previous_control = control

        if next_time < 0:
            latent = prediction.pred_x_start
        else:
            alpha_next = diffusion.alphas_cumprod[next_time].to(
                dtype=controlled_latent.dtype
            )
            latent = (
                alpha_next.sqrt() * prediction.pred_x_start
                + (1 - alpha_next).clamp_min(0).sqrt() * prediction.pred_noise
            )

    (
        final_observation_mse,
        intermediate_observation_mse,
        observation_loss,
    ) = _aggregate_substep_losses(
        observation_mses,
        intermediate_weight,
    )
    (
        final_full_state_rmse,
        intermediate_full_state_rmse,
        full_state_loss,
    ) = _aggregate_substep_losses(
        full_state_rmses,
        intermediate_weight,
    )
    weighted_full_state_rmse = beta * full_state_loss
    total_loss = observation_loss + weighted_full_state_rmse
    control_rms = torch.stack(control_squares).mean().sqrt()
    next_state = diffusion.unnormalize(latent)
    return next_state, StepMetrics(
        controlled=True,
        observation_available=observation_available,
        target_index=target_index,
        total_loss=total_loss,
        observation_loss=observation_loss,
        final_observation_mse=final_observation_mse,
        intermediate_observation_mse=intermediate_observation_mse,
        final_full_state_rmse=final_full_state_rmse,
        intermediate_full_state_rmse=intermediate_full_state_rmse,
        weighted_full_state_rmse=weighted_full_state_rmse,
        control_rms=control_rms,
        denoising_steps=len(time_pairs),
    )


def preview_rollout(
    diffusion: GaussianDiffusion,
    controller: nn.Module,
    trajectory: Tensor,
    observation_operator: ObservationOperator,
    cadence: TemporalCadence,
    *,
    preview_steps: int,
    gamma: float,
    beta: float,
    intermediate_weight: float = 1.0,
    start_index: int = 0,
    sampling_steps: int | None = None,
    initial_noises: Tensor | None = None,
    clip_x_start: bool = True,
) -> RolloutResult:
    """Run fixed, anchored preview chunks with autoregressive handoff."""

    if not isinstance(trajectory, Tensor):
        raise TypeError("trajectory must be a torch.Tensor")
    if trajectory.ndim not in (4, 5):
        raise ValueError("trajectory must have shape [B,T,C,X] or [B,T,C,H,W]")
    if trajectory.shape[1] < 2:
        raise ValueError("trajectory must contain at least two states")
    _validate_state(diffusion, trajectory[:, 0], "trajectory states")
    if isinstance(preview_steps, bool) or not isinstance(preview_steps, int):
        raise TypeError("preview_steps must be an integer")
    if preview_steps < 1:
        raise ValueError("preview_steps must be positive")
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    _ddim_time_pairs(diffusion, sampling_steps)

    horizon = trajectory.shape[1] - 1
    current = trajectory[:, 0]
    states = [current]
    metrics: list[StepMetrics] = []
    observation_indices = tuple(
        local_index
        for local_index in range(1, horizon + 1)
        if cadence.is_observation_time(start_index + local_index)
    )
    chunk_boundaries = tuple(
        min(chunk_start + preview_steps, horizon)
        for chunk_start in range(0, horizon, preview_steps)
    )

    for chunk_start in range(0, horizon, preview_steps):
        chunk_end = min(chunk_start + preview_steps, horizon)
        chunk_length = chunk_end - chunk_start
        arrivals = tuple(
            index
            for index in range(chunk_start + 1, chunk_end + 1)
            if cadence.is_observation_time(start_index + index)
        )
        active_observations = {
            index: apply_observation(
                trajectory[:, index],
                observation_operator,
            )
            for index in arrivals
        }

        for offset, next_index in enumerate(range(chunk_start + 1, chunk_end + 1)):
            target_index = next(
                (index for index in arrivals if index >= next_index),
                None,
            )
            if target_index is None:
                target_state = trajectory[:, next_index]
                zeros = torch.zeros_like(target_state)
                active_observation = Observation(value=zeros, mask=zeros)
                lead = 0.0
            else:
                target_state = trajectory[:, target_index]
                active_observation = active_observations[target_index]
                lead = (target_index - next_index + 1) / chunk_length
            position = offset / max(chunk_length - 1, 1)
            current, step_metric = controlled_ddim_next_state(
                diffusion,
                controller,
                current,
                target_state,
                observation_operator,
                observation_lead=lead,
                rollout_position=position,
                gamma=gamma,
                beta=beta,
                intermediate_weight=intermediate_weight,
                target_index=target_index,
                active_observation=active_observation,
                ground_truth_state=trajectory[:, next_index],
                observation_available=target_index == next_index,
                initial_noise=_noise_at(
                    initial_noises, next_index - 1, horizon, current
                ),
                sampling_steps=sampling_steps,
                clip_x_start=clip_x_start,
            )
            states.append(current)
            metrics.append(step_metric)

    return RolloutResult(
        states=torch.stack(states, dim=1),
        step_metrics=tuple(metrics),
        observation_indices=observation_indices,
        chunk_boundaries=chunk_boundaries,
    )


__all__ = [
    "RolloutResult",
    "StepMetrics",
    "controlled_ddim_next_state",
    "preview_rollout",
    "uncontrolled_rollout",
]
