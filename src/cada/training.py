"""Focused trainers for the CADA diffusion prior and controller."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import torch
from torch import Tensor, nn

from .assimilation import preview_rollout
from .diffusion import GaussianDiffusion
from .observations import ObservationOperator, TemporalCadence

Precision: TypeAlias = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class PriorStepMetrics:
    """Scalar diagnostics from one prior optimizer step."""

    step: int
    loss: float
    grad_norm: float | None
    learning_rate: float


@dataclass(frozen=True)
class ControllerStepMetrics:
    """Scalar diagnostics from one controller optimizer step."""

    step: int
    total_loss: float
    observation_loss: float
    final_observation_mse: float
    intermediate_observation_mse: float
    final_full_state_rmse: float
    intermediate_full_state_rmse: float
    weighted_full_state_rmse: float
    control_rms: float
    grad_norm: float | None
    learning_rate: float


TrainingMetrics: TypeAlias = PriorStepMetrics | ControllerStepMetrics


class LogCallback(Protocol):
    def __call__(self, metrics: TrainingMetrics) -> None: ...


class CheckpointCallback(Protocol):
    def __call__(
        self,
        step: int,
        state: Mapping[str, Any],
        metrics: TrainingMetrics,
    ) -> None: ...


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not bool(torch.isfinite(torch.tensor(value))) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if not bool(torch.isfinite(torch.tensor(value))) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_precision(precision: Precision, device: torch.device) -> Precision:
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 autocast requires a CUDA device")
    if precision != "fp32" and device.type not in ("cpu", "cuda"):
        raise ValueError("autocast is supported only on CPU and CUDA devices")
    return precision


def _autocast(
    precision: Precision, device: torch.device
) -> torch.autocast:
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=precision != "fp32",
    )


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("module has no trainable parameters")
    return parameters


def _gradient_norm(
    parameters: Sequence[nn.Parameter], max_grad_norm: float | None
) -> float | None:
    if max_grad_norm is not None:
        norm = float(nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu())
    else:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0
        norm = float(torch.stack(norms).norm(2).cpu())
    # fp16 overflow steps have non-finite gradients; GradScaler skips them.
    return norm if math.isfinite(norm) else None


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    def move(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            state[key] = move(value)


class _BatchStream:
    def __init__(self, dataloader: Iterable[Any]) -> None:
        self.dataloader = dataloader
        generator = getattr(dataloader, "generator", None)
        self._generator = generator
        self._initial_generator_state = (
            generator.get_state().clone() if generator is not None else None
        )
        self.iterator = iter(dataloader)

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise ValueError("dataloader must yield at least one batch") from error

    def seek(self, batch_count: int) -> None:
        if batch_count < 0:
            raise ValueError("batch_count must be non-negative")
        if self._initial_generator_state is not None:
            self._generator.set_state(self._initial_generator_state)
        self.iterator = iter(self.dataloader)
        for _ in range(batch_count):
            self.next()

    def state_dict(self) -> dict[str, Any] | None:
        sampler = getattr(self.dataloader, "sampler", None)
        sampler_state = getattr(sampler, "state_dict", None)
        if not callable(sampler_state):
            return None
        return sampler_state()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        sampler = getattr(self.dataloader, "sampler", None)
        load_sampler_state = getattr(sampler, "load_state_dict", None)
        if not callable(load_sampler_state):
            raise ValueError("dataloader sampler is not resumable")
        load_sampler_state(state)
        self.iterator = iter(self.dataloader)


def _single_tensor_batch(batch: Any, name: str) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if (
        isinstance(batch, Sequence)
        and not isinstance(batch, (str, bytes))
        and len(batch) == 1
        and isinstance(batch[0], Tensor)
    ):
        return batch[0]
    raise TypeError(f"{name} dataloader must yield a tensor")


def _prior_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Tensor):
        if batch.ndim not in (4, 5) or batch.shape[1] != 2:
            raise ValueError(
                "prior tensor batches must have shape [B,2,C,X] or [B,2,C,H,W]"
            )
        return batch[:, 0], batch[:, 1]
    if (
        isinstance(batch, Sequence)
        and not isinstance(batch, (str, bytes))
        and len(batch) == 1
    ):
        return _prior_batch(batch[0])
    if (
        isinstance(batch, Sequence)
        and not isinstance(batch, (str, bytes))
        and len(batch) == 2
        and isinstance(batch[0], Tensor)
        and isinstance(batch[1], Tensor)
    ):
        return batch[0], batch[1]
    raise TypeError(
        "prior dataloader must yield [B,2,C,...] or (previous_state, next_state)"
    )


def _validate_callbacks(
    log_callback: LogCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    log_every: int | None,
    checkpoint_every: int | None,
) -> tuple[int | None, int | None]:
    if log_callback is not None and not callable(log_callback):
        raise TypeError("log_callback must be callable")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable")
    if log_every is not None:
        log_every = _positive_int(log_every, "log_every")
    if checkpoint_every is not None:
        checkpoint_every = _positive_int(checkpoint_every, "checkpoint_every")
    return log_every, checkpoint_every


def _emit_callbacks(
    owner: Any,
    metrics: TrainingMetrics,
    *,
    log_callback: LogCallback | None,
    checkpoint_callback: CheckpointCallback | None,
    log_every: int | None,
    checkpoint_every: int | None,
) -> None:
    if (
        log_callback is not None
        and log_every is not None
        and metrics.step % log_every == 0
    ):
        log_callback(metrics)
    if (
        checkpoint_callback is not None
        and checkpoint_every is not None
        and metrics.step % checkpoint_every == 0
    ):
        checkpoint_callback(metrics.step, owner.state_dict(), metrics)


class PriorTrainer:
    """Train a conditional pred-v prior from one-step transition batches."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        dataloader: Iterable[Any],
        *,
        learning_rate: float,
        device: torch.device | str = "cpu",
        precision: Precision = "fp32",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        weight_decay: float = 0.0,
        ema_decay: float = 0.995,
        ema_update_every: int = 1,
        step: int = 0,
        log_callback: LogCallback | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        log_every: int | None = 1,
        checkpoint_every: int | None = None,
    ) -> None:
        if not isinstance(diffusion, GaussianDiffusion):
            raise TypeError("diffusion must be a GaussianDiffusion")
        self.device = torch.device(device)
        self.precision = _validate_precision(precision, self.device)
        self.gradient_accumulation_steps = _positive_int(
            gradient_accumulation_steps, "gradient_accumulation_steps"
        )
        self.max_grad_norm = (
            None
            if max_grad_norm is None
            else _positive_float(max_grad_norm, "max_grad_norm")
        )
        learning_rate = _positive_float(learning_rate, "learning_rate")
        weight_decay = _nonnegative_float(weight_decay, "weight_decay")
        ema_decay = float(ema_decay)
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.ema_decay = ema_decay
        self.ema_update_every = _positive_int(
            ema_update_every, "ema_update_every"
        )
        self.step = _nonnegative_int(step, "step")
        self.log_callback = log_callback
        self.checkpoint_callback = checkpoint_callback
        self.log_every, self.checkpoint_every = _validate_callbacks(
            log_callback,
            checkpoint_callback,
            log_every,
            checkpoint_every,
        )

        self.diffusion = diffusion.to(self.device)
        self.parameters = _trainable_parameters(self.diffusion.model)
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.precision == "fp16",
        )
        self.ema_model = deepcopy(self.diffusion.model).to(self.device).eval()
        self.ema_model.requires_grad_(False)
        self._batches = _BatchStream(dataloader)

    @torch.no_grad()
    def _update_ema(self) -> None:
        model_state = self.diffusion.model.state_dict()
        for name, ema_value in self.ema_model.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.lerp_(model_value, 1 - self.ema_decay)
            else:
                ema_value.copy_(model_value)

    def train_step(self) -> PriorStepMetrics:
        """Run one optimizer step, including configured accumulation."""

        self.diffusion.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(self.gradient_accumulation_steps):
            previous_state, next_state = _prior_batch(self._batches.next())
            previous_state = previous_state.to(self.device, non_blocking=True)
            next_state = next_state.to(self.device, non_blocking=True)
            with _autocast(self.precision, self.device):
                loss = self.diffusion.training_loss(previous_state, next_state)
                scaled_loss = loss / self.gradient_accumulation_steps
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("prior loss is not finite")
            self.scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().cpu())

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = _gradient_norm(self.parameters, self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        if self.step % self.ema_update_every == 0:
            self._update_ema()

        metrics = PriorStepMetrics(
            step=self.step,
            loss=loss_sum / self.gradient_accumulation_steps,
            grad_norm=grad_norm,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )
        _emit_callbacks(
            self,
            metrics,
            log_callback=self.log_callback,
            checkpoint_callback=self.checkpoint_callback,
            log_every=self.log_every,
            checkpoint_every=self.checkpoint_every,
        )
        return metrics

    def fit(self, max_steps: int) -> list[PriorStepMetrics]:
        """Train until the absolute optimizer step ``max_steps``."""

        max_steps = _nonnegative_int(max_steps, "max_steps")
        history: list[PriorStepMetrics] = []
        while self.step < max_steps:
            history.append(self.train_step())
        return history

    def state_dict(self) -> dict[str, Any]:
        state = {
            "step": self.step,
            "diffusion": self.diffusion.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        batch_state = self._batches.state_dict()
        if batch_state is not None:
            state["batch_stream"] = batch_state
        return state

    def load_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        self.diffusion.load_state_dict(state["diffusion"], strict=strict)
        self.ema_model.load_state_dict(state["ema_model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        _optimizer_to(self.optimizer, self.device)
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.step = _nonnegative_int(int(state["step"]), "checkpoint step")
        if "batch_stream" in state:
            self._batches.load_state_dict(state["batch_stream"])
        else:
            self._batches.seek(self.step * self.gradient_accumulation_steps)


class ControllerTrainer:
    """Train ControlNet through a frozen prior and preview CADA rollout."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        controller: nn.Module,
        dataloader: Iterable[Any],
        observation_operator: ObservationOperator,
        cadence: TemporalCadence,
        *,
        preview_steps: int,
        gamma: float,
        beta: float,
        learning_rate: float,
        intermediate_weight: float = 1.0,
        start_index: int = 0,
        sampling_steps: int | None = None,
        clip_x_start: bool = True,
        device: torch.device | str = "cpu",
        precision: Precision = "fp32",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        weight_decay: float = 0.0,
        step: int = 0,
        log_callback: LogCallback | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        log_every: int | None = 1,
        checkpoint_every: int | None = None,
    ) -> None:
        if not isinstance(diffusion, GaussianDiffusion):
            raise TypeError("diffusion must be a GaussianDiffusion")
        if not isinstance(controller, nn.Module):
            raise TypeError("controller must be a torch.nn.Module")
        if not isinstance(cadence, TemporalCadence):
            raise TypeError("cadence must be a TemporalCadence")
        self.device = torch.device(device)
        self.precision = _validate_precision(precision, self.device)
        self.preview_steps = _positive_int(preview_steps, "preview_steps")
        self.gamma = _nonnegative_float(gamma, "gamma")
        self.beta = _nonnegative_float(beta, "beta")
        self.intermediate_weight = _nonnegative_float(
            intermediate_weight, "intermediate_weight"
        )
        self.start_index = _nonnegative_int(start_index, "start_index")
        if sampling_steps is not None:
            sampling_steps = _positive_int(sampling_steps, "sampling_steps")
            if sampling_steps > diffusion.num_timesteps:
                raise ValueError("sampling_steps cannot exceed diffusion timesteps")
        self.sampling_steps = sampling_steps
        self.clip_x_start = bool(clip_x_start)
        self.gradient_accumulation_steps = _positive_int(
            gradient_accumulation_steps, "gradient_accumulation_steps"
        )
        self.max_grad_norm = (
            None
            if max_grad_norm is None
            else _positive_float(max_grad_norm, "max_grad_norm")
        )
        learning_rate = _positive_float(learning_rate, "learning_rate")
        weight_decay = _nonnegative_float(weight_decay, "weight_decay")
        self.step = _nonnegative_int(step, "step")
        self.log_callback = log_callback
        self.checkpoint_callback = checkpoint_callback
        self.log_every, self.checkpoint_every = _validate_callbacks(
            log_callback,
            checkpoint_callback,
            log_every,
            checkpoint_every,
        )

        self.diffusion = diffusion.to(self.device)
        self.diffusion.requires_grad_(False)
        self.diffusion.eval()
        self.controller = controller.to(self.device)
        if (
            getattr(controller, "physical_channels", None)
            != diffusion.physical_channels
        ):
            raise ValueError("controller and diffusion physical channels must match")
        self.parameters = _trainable_parameters(self.controller)
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.precision == "fp16",
        )
        self.observation_operator = observation_operator
        self.cadence = cadence
        dataset = getattr(dataloader, "dataset", None)
        stride = getattr(dataset, "stride", None)
        if isinstance(stride, int) and stride % cadence.interval != 0:
            raise ValueError(
                "controller dataset stride must preserve temporal cadence phase"
            )
        self._batches = _BatchStream(dataloader)

    def train_step(self) -> ControllerStepMetrics:
        """Run one controller optimizer step through the frozen prior."""

        self.diffusion.eval()
        self.controller.train()
        self.optimizer.zero_grad(set_to_none=True)
        sums = {
            "total_loss": 0.0,
            "observation_loss": 0.0,
            "final_observation_mse": 0.0,
            "intermediate_observation_mse": 0.0,
            "final_full_state_rmse": 0.0,
            "intermediate_full_state_rmse": 0.0,
            "weighted_full_state_rmse": 0.0,
            "control_rms": 0.0,
        }

        for _ in range(self.gradient_accumulation_steps):
            trajectory = _single_tensor_batch(
                self._batches.next(), "controller"
            ).to(self.device, non_blocking=True)
            with _autocast(self.precision, self.device):
                rollout = preview_rollout(
                    self.diffusion,
                    self.controller,
                    trajectory,
                    self.observation_operator,
                    self.cadence,
                    preview_steps=self.preview_steps,
                    gamma=self.gamma,
                    beta=self.beta,
                    intermediate_weight=self.intermediate_weight,
                    start_index=self.start_index,
                    sampling_steps=self.sampling_steps,
                    clip_x_start=self.clip_x_start,
                )
                controlled = rollout.controlled_metrics
                loss = rollout.loss
                scaled_loss = loss / self.gradient_accumulation_steps
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("controller loss is not finite")
            self.scaler.scale(scaled_loss).backward()

            sums["total_loss"] += float(loss.detach().cpu())
            observed = tuple(
                metric for metric in controlled if metric.observation_available
            )
            for name in (
                "observation_loss",
                "final_observation_mse",
                "intermediate_observation_mse",
            ):
                if not observed:
                    continue
                values = torch.stack(
                    tuple(getattr(metric, name) for metric in observed)
                )
                sums[name] += float(values.mean().detach().cpu())
            for name in (
                "final_full_state_rmse",
                "intermediate_full_state_rmse",
                "weighted_full_state_rmse",
                "control_rms",
            ):
                values = torch.stack(
                    tuple(getattr(metric, name) for metric in controlled)
                )
                sums[name] += float(values.mean().detach().cpu())

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = _gradient_norm(self.parameters, self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        divisor = self.gradient_accumulation_steps
        metrics = ControllerStepMetrics(
            step=self.step,
            total_loss=sums["total_loss"] / divisor,
            observation_loss=sums["observation_loss"] / divisor,
            final_observation_mse=sums["final_observation_mse"] / divisor,
            intermediate_observation_mse=(
                sums["intermediate_observation_mse"] / divisor
            ),
            final_full_state_rmse=(
                sums["final_full_state_rmse"] / divisor
            ),
            intermediate_full_state_rmse=(
                sums["intermediate_full_state_rmse"] / divisor
            ),
            weighted_full_state_rmse=(
                sums["weighted_full_state_rmse"] / divisor
            ),
            control_rms=sums["control_rms"] / divisor,
            grad_norm=grad_norm,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )
        _emit_callbacks(
            self,
            metrics,
            log_callback=self.log_callback,
            checkpoint_callback=self.checkpoint_callback,
            log_every=self.log_every,
            checkpoint_every=self.checkpoint_every,
        )
        return metrics

    def fit(self, max_steps: int) -> list[ControllerStepMetrics]:
        """Train until the absolute optimizer step ``max_steps``."""

        max_steps = _nonnegative_int(max_steps, "max_steps")
        history: list[ControllerStepMetrics] = []
        while self.step < max_steps:
            history.append(self.train_step())
        return history

    def state_dict(self) -> dict[str, Any]:
        state = {
            "step": self.step,
            "diffusion": self.diffusion.state_dict(),
            "controller": self.controller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        operator_state = getattr(self.observation_operator, "state_dict", None)
        if callable(operator_state):
            state["observation_operator"] = operator_state()
        batch_state = self._batches.state_dict()
        if batch_state is not None:
            state["batch_stream"] = batch_state
        return state

    def load_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        self.diffusion.load_state_dict(state["diffusion"], strict=strict)
        self.controller.load_state_dict(state["controller"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        _optimizer_to(self.optimizer, self.device)
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.step = _nonnegative_int(int(state["step"]), "checkpoint step")
        if "batch_stream" in state:
            self._batches.load_state_dict(state["batch_stream"])
        else:
            self._batches.seek(self.step * self.gradient_accumulation_steps)
        if "observation_operator" in state:
            load_operator_state = getattr(
                self.observation_operator,
                "load_state_dict",
                None,
            )
            if not callable(load_operator_state):
                raise ValueError(
                    "checkpoint contains observation state for a stateless operator"
                )
            load_operator_state(state["observation_operator"])
        self.diffusion.requires_grad_(False)
        self.diffusion.eval()


__all__ = [
    "CheckpointCallback",
    "ControllerStepMetrics",
    "ControllerTrainer",
    "LogCallback",
    "Precision",
    "PriorStepMetrics",
    "PriorTrainer",
    "TrainingMetrics",
]
