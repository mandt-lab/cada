"""Evaluation-only ablations for a frozen autoregressive diffusion prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .diffusion import GaussianDiffusion
from .observations import (
    Observation,
    ObservationOperator,
    TemporalCadence,
    apply_observation,
)


@dataclass(frozen=True)
class TTODAResult:
    """A rollout optimized through a frozen prior at evaluation time."""

    states: Tensor
    controls: Tensor
    initial_objective: Tensor
    final_objective: Tensor
    loss_history: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    observation_indices: tuple[int, ...]
    chunk_boundaries: tuple[int, ...]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if not 0 <= value < 2**63:
        raise ValueError("seed must lie in [0, 2**63)")
    return value


def _validate_trajectory(
    diffusion: GaussianDiffusion,
    trajectory: Tensor,
) -> int:
    if not isinstance(trajectory, Tensor):
        raise TypeError("trajectory must be a torch.Tensor")
    if trajectory.ndim not in (4, 5):
        raise ValueError("trajectory must have shape [B,T,C,X] or [B,T,C,H,W]")
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 2:
        raise ValueError("trajectory must contain a batch and a forecast state")
    if trajectory.shape[2] != diffusion.physical_channels:
        raise ValueError("trajectory channels do not match the diffusion prior")
    if trajectory.device != diffusion.device:
        raise ValueError("trajectory and diffusion must share a device")
    if not trajectory.is_floating_point():
        raise TypeError("trajectory must be floating point")
    return trajectory.shape[1] - 1


def _observation_indices(
    cadence: TemporalCadence,
    horizon: int,
    start_index: int,
) -> tuple[int, ...]:
    indices = tuple(
        index
        for index in range(1, horizon + 1)
        if cadence.is_observation_time(start_index + index)
    )
    if not indices:
        raise ValueError("the rollout contains no scheduled observations")
    return indices


def _time_pairs(
    diffusion: GaussianDiffusion,
    sampling_steps: int | None,
) -> tuple[tuple[int, int], ...]:
    steps = diffusion.sampling_steps if sampling_steps is None else sampling_steps
    _positive_integer(steps, "sampling_steps")
    if steps > diffusion.num_timesteps:
        raise ValueError("sampling_steps cannot exceed diffusion timesteps")
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


def _deterministic_noises(
    state: Tensor,
    *,
    horizon: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(seed))
    return torch.randn(
        (horizon, *state.shape),
        generator=generator,
        dtype=state.dtype,
        device="cpu",
    ).to(state.device)


def _per_item_observation_mse(
    prediction: Tensor,
    active_observation: Observation,
    operator: ObservationOperator,
) -> Tensor:
    predicted = apply_observation(prediction, operator)
    if not torch.allclose(
        predicted.mask,
        active_observation.mask,
        rtol=0,
        atol=0,
    ):
        raise ValueError("observation operator changed the scheduled mask")
    mask = active_observation.mask.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    observed = mask.flatten(1).sum(1)
    if bool((observed <= 0).any()):
        raise ValueError("every sample must contain at least one observed point")
    error = (predicted.value - active_observation.value).square() * mask
    return error.flatten(1).sum(1) / observed


def _free_control_step(
    diffusion: GaussianDiffusion,
    previous_state: Tensor,
    active_observation: Observation | None,
    observation_operator: ObservationOperator,
    controls: Tensor,
    initial_noise: Tensor,
    time_pairs: tuple[tuple[int, int], ...],
    *,
    gamma: float,
    intermediate_weight: float,
    clip_x_start: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    condition = diffusion.normalize(previous_state)
    latent = initial_noise
    observation_mses = []
    control_energies = []
    for control, (time, next_time) in zip(controls, time_pairs):
        controlled_latent = latent + gamma * control
        times = torch.full(
            (condition.shape[0],),
            time,
            device=condition.device,
            dtype=torch.long,
        )
        prediction = diffusion.model_predictions(
            controlled_latent,
            condition,
            times,
            clip_x_start=clip_x_start,
        )
        if active_observation is not None:
            observation_mses.append(
                _per_item_observation_mse(
                    diffusion.unnormalize(prediction.pred_x_start),
                    active_observation,
                    observation_operator,
                )
            )
        control_energies.append(0.5 * (gamma * control).square().flatten(1).mean(1))
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

    if not observation_mses:
        observation_loss = previous_state.flatten(1).sum(1) * 0
    else:
        final_mse = observation_mses[-1]
        if len(observation_mses) == 1:
            intermediate_mse = final_mse * 0
        else:
            intermediate_mse = torch.stack(observation_mses[:-1]).mean(dim=0)
        observation_loss = final_mse + intermediate_weight * intermediate_mse
    control_energy = torch.stack(control_energies).mean(dim=0)
    return diffusion.unnormalize(latent), observation_loss, control_energy


def _controlled_chunk(
    diffusion: GaussianDiffusion,
    initial_state: Tensor,
    active_observations: dict[int, Observation],
    observation_operator: ObservationOperator,
    controls: Tensor,
    noises: Tensor,
    time_pairs: tuple[tuple[int, int], ...],
    *,
    gamma: float,
    beta: float,
    intermediate_weight: float,
    clip_x_start: bool,
) -> tuple[tuple[Tensor, ...], Tensor]:
    current = initial_state
    states = []
    observation_losses = []
    control_energies = []
    for local_index, (step_controls, noise) in enumerate(
        zip(controls, noises),
        start=1,
    ):
        current, observation_loss, control_energy = _free_control_step(
            diffusion,
            current,
            active_observations.get(local_index),
            observation_operator,
            step_controls,
            noise,
            time_pairs,
            gamma=gamma,
            intermediate_weight=intermediate_weight,
            clip_x_start=clip_x_start,
        )
        states.append(current)
        if local_index in active_observations:
            observation_losses.append(observation_loss)
        control_energies.append(control_energy)
    mean_observation = (
        torch.stack(observation_losses).mean()
        if observation_losses
        else initial_state.sum() * 0
    )
    mean_energy = torch.stack(control_energies).mean()
    return tuple(states), mean_observation + beta * mean_energy


def tto_da_rollout(
    diffusion: GaussianDiffusion,
    trajectory: Tensor,
    observation_operator: ObservationOperator,
    cadence: TemporalCadence,
    *,
    preview_steps: int,
    iterations: int,
    lr: float,
    gamma: float,
    beta: float,
    seed: int,
    intermediate_weight: float = 1.0,
    start_index: int = 0,
    sampling_steps: int | None = None,
    clip_x_start: bool = True,
) -> TTODAResult:
    """Optimize free latent controls independently in each preview chunk.

    Every controlled physical transition owns one free control tensor per DDIM
    step. Adam updates only those tensors; gradients are requested only with
    respect to the controls, so prior parameters are neither optimized nor
    assigned gradients.
    """

    horizon = _validate_trajectory(diffusion, trajectory)
    preview_steps = _positive_integer(preview_steps, "preview_steps")
    iterations = _positive_integer(iterations, "iterations")
    lr = _positive_float(lr, "lr")
    gamma = _nonnegative_float(gamma, "gamma")
    beta = _nonnegative_float(beta, "beta")
    intermediate_weight = _nonnegative_float(
        intermediate_weight,
        "intermediate_weight",
    )
    _seed(seed)
    time_pairs = _time_pairs(diffusion, sampling_steps)
    indices = _observation_indices(cadence, horizon, start_index)
    noises = _deterministic_noises(
        trajectory[:, 0],
        horizon=horizon,
        seed=seed,
    )
    all_controls = torch.zeros(
        (horizon, len(time_pairs), *trajectory[:, 0].shape),
        device=trajectory.device,
        dtype=trajectory.dtype,
    )

    current = trajectory[:, 0].detach()
    states = [current]
    chunk_boundaries = []
    initial_objectives = []
    final_objectives = []
    objective_weights = []
    loss_history: list[float] = []
    gradient_norms: list[float] = []
    was_training = diffusion.training
    diffusion.eval()
    try:
        for chunk_start in range(0, horizon, preview_steps):
            chunk_end = min(chunk_start + preview_steps, horizon)
            chunk_length = chunk_end - chunk_start
            arrivals = tuple(
                index
                for index in range(chunk_start + 1, chunk_end + 1)
                if cadence.is_observation_time(start_index + index)
            )
            chunk_boundaries.append(chunk_end)
            if not arrivals:
                with torch.no_grad():
                    for next_index in range(chunk_start + 1, chunk_end + 1):
                        current = diffusion.ddim_sample(
                            current,
                            initial_noise=noises[next_index - 1],
                            sampling_steps=sampling_steps,
                            clip_x_start=clip_x_start,
                        ).detach()
                        states.append(current)
                continue

            active_observations = {
                index - chunk_start: apply_observation(
                    trajectory[:, index],
                    observation_operator,
                )
                for index in arrivals
            }
            controls = torch.zeros(
                (chunk_length, len(time_pairs), *current.shape),
                device=current.device,
                dtype=current.dtype,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam((controls,), lr=lr)
            chunk_noises = noises[chunk_start:chunk_end]
            chunk_states, objective = _controlled_chunk(
                diffusion,
                current,
                active_observations,
                observation_operator,
                controls,
                chunk_noises,
                time_pairs,
                gamma=gamma,
                beta=beta,
                intermediate_weight=intermediate_weight,
                clip_x_start=clip_x_start,
            )
            del chunk_states
            if not bool(torch.isfinite(objective.detach())):
                raise RuntimeError("TTO-DA produced a non-finite objective")
            initial_objectives.append(objective.detach())
            objective_weights.append(chunk_length)
            best_value = float(objective.detach())
            best_controls = controls.detach().clone()
            loss_history.append(best_value)

            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                (gradient,) = torch.autograd.grad(objective, (controls,))
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("TTO-DA produced a non-finite control gradient")
                gradient_norm = gradient.detach().double().norm()
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("TTO-DA produced a non-finite gradient norm")
                gradient_norms.append(float(gradient_norm))
                controls.grad = gradient
                optimizer.step()
                _, objective = _controlled_chunk(
                    diffusion,
                    current,
                    active_observations,
                    observation_operator,
                    controls,
                    chunk_noises,
                    time_pairs,
                    gamma=gamma,
                    beta=beta,
                    intermediate_weight=intermediate_weight,
                    clip_x_start=clip_x_start,
                )
                value = float(objective.detach())
                if not torch.isfinite(objective.detach()):
                    raise RuntimeError("TTO-DA produced a non-finite objective")
                loss_history.append(value)
                if value < best_value:
                    best_value = value
                    best_controls = controls.detach().clone()

            controls = best_controls.requires_grad_(False)
            with torch.no_grad():
                optimized_states, optimized_objective = _controlled_chunk(
                    diffusion,
                    current,
                    active_observations,
                    observation_operator,
                    controls,
                    chunk_noises,
                    time_pairs,
                    gamma=gamma,
                    beta=beta,
                    intermediate_weight=intermediate_weight,
                    clip_x_start=clip_x_start,
                )
                all_controls[chunk_start:chunk_end].copy_(controls)
                states.extend(state.detach() for state in optimized_states)
                current = states[-1]
            final_objectives.append(optimized_objective.detach())
    finally:
        diffusion.train(was_training)

    total_weight = sum(objective_weights)
    initial_objective = (
        sum(
            value * weight
            for value, weight in zip(initial_objectives, objective_weights)
        )
        / total_weight
    )
    final_objective = (
        sum(
            value * weight for value, weight in zip(final_objectives, objective_weights)
        )
        / total_weight
    )
    return TTODAResult(
        states=torch.stack(states, dim=1),
        controls=all_controls,
        initial_objective=initial_objective,
        final_objective=final_objective,
        loss_history=tuple(loss_history),
        gradient_norms=tuple(gradient_norms),
        observation_indices=indices,
        chunk_boundaries=tuple(chunk_boundaries),
    )


__all__ = [
    "TTODAResult",
    "tto_da_rollout",
]
