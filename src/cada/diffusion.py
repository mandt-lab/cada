"""Pred-v Gaussian diffusion for conditional 1D and 2D CADA priors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionPrediction:
    """A prior prediction in the three equivalent diffusion parameterizations."""

    pred_noise: Tensor
    pred_x_start: Tensor
    pred_v: Tensor


def sigmoid_beta_schedule(
    timesteps: int,
    *,
    start: float = -3.0,
    end: float = 3.0,
    tau: float = 1.0,
) -> Tensor:
    """Return the sigmoid DDPM schedule used by the maintained CADA models."""
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if tau <= 0:
        raise ValueError("tau must be positive")
    time = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    start_value = torch.tensor(start / tau, dtype=torch.float64).sigmoid()
    end_value = torch.tensor(end / tau, dtype=torch.float64).sigmoid()
    alphas_cumprod = (
        -((time * (end - start) + start) / tau).sigmoid() + end_value
    ) / (end_value - start_value)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999)


def _extract(values: Tensor, times: Tensor, shape: torch.Size) -> Tensor:
    extracted = values.gather(0, times)
    return extracted.reshape(times.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Conditional pred-v diffusion shared by 1D and 2D physical states.

    The wrapped model must expose ``physical_channels`` and implement
    ``model(cat(previous_state, noisy_next_state), diffusion_time)``.
    Public training and sampling methods accept data-space tensors; conversion
    helpers and :meth:`model_predictions` operate in normalized model space.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        timesteps: int = 1000,
        sampling_steps: int = 3,
        auto_normalize: bool = True,
    ) -> None:
        super().__init__()
        physical_channels = getattr(model, "physical_channels", None)
        if not isinstance(physical_channels, int) or physical_channels < 1:
            raise TypeError("model must expose a positive integer physical_channels")
        if not 1 <= sampling_steps <= timesteps:
            raise ValueError("sampling_steps must be between 1 and timesteps")

        self.model = model
        self.physical_channels = physical_channels
        self.num_timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.auto_normalize = auto_normalize

        betas = sigmoid_beta_schedule(timesteps)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        def register(name: str, value: Tensor) -> None:
            self.register_buffer(name, value.to(dtype=torch.float32))

        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        register(
            "sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt()
        )
        register("sqrt_recip_alphas_cumprod", (1 / alphas_cumprod).sqrt())
        register(
            "sqrt_recipm1_alphas_cumprod", (1 / alphas_cumprod - 1).sqrt()
        )
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register("loss_weight", snr / (snr + 1))

    @property
    def device(self) -> torch.device:
        """Device holding the diffusion schedule."""
        return self.betas.device

    def normalize(self, value: Tensor) -> Tensor:
        """Map data from ``[0, 1]`` to model space when enabled."""
        return value * 2 - 1 if self.auto_normalize else value

    def unnormalize(self, value: Tensor) -> Tensor:
        """Map model-space data back to ``[0, 1]`` when enabled."""
        return (value + 1) * 0.5 if self.auto_normalize else value

    def _validate_state(self, state: Tensor, name: str) -> None:
        if state.ndim not in (3, 4):
            raise ValueError(f"{name} must have shape [B,C,X] or [B,C,H,W]")
        if state.shape[1] != self.physical_channels:
            raise ValueError(
                f"{name} must have {self.physical_channels} physical channels"
            )

    def _validate_times(self, times: Tensor, batch: int) -> None:
        if times.dtype != torch.long or times.shape != (batch,):
            raise ValueError("times must be a long tensor with shape [batch]")
        if bool(((times < 0) | (times >= self.num_timesteps)).any()):
            raise ValueError("times must lie within the diffusion schedule")

    def predict_start_from_noise(
        self, noisy_state: Tensor, times: Tensor, noise: Tensor
    ) -> Tensor:
        """Convert an epsilon prediction into an x0 prediction."""
        return (
            _extract(self.sqrt_recip_alphas_cumprod, times, noisy_state.shape)
            * noisy_state
            - _extract(self.sqrt_recipm1_alphas_cumprod, times, noisy_state.shape)
            * noise
        )

    def predict_noise_from_start(
        self, noisy_state: Tensor, times: Tensor, x_start: Tensor
    ) -> Tensor:
        """Convert an x0 prediction into an epsilon prediction."""
        return (
            _extract(self.sqrt_recip_alphas_cumprod, times, noisy_state.shape)
            * noisy_state
            - x_start
        ) / _extract(
            self.sqrt_recipm1_alphas_cumprod, times, noisy_state.shape
        )

    def predict_v(self, x_start: Tensor, times: Tensor, noise: Tensor) -> Tensor:
        """Convert x0 and epsilon into the v parameterization."""
        return (
            _extract(self.sqrt_alphas_cumprod, times, x_start.shape) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, times, x_start.shape)
            * x_start
        )

    def predict_start_from_v(
        self, noisy_state: Tensor, times: Tensor, v: Tensor
    ) -> Tensor:
        """Convert a v prediction into an x0 prediction."""
        return (
            _extract(self.sqrt_alphas_cumprod, times, noisy_state.shape)
            * noisy_state
            - _extract(
                self.sqrt_one_minus_alphas_cumprod, times, noisy_state.shape
            )
            * v
        )

    def predict_noise_from_v(
        self, noisy_state: Tensor, times: Tensor, v: Tensor
    ) -> Tensor:
        """Convert a v prediction into an epsilon prediction."""
        return (
            _extract(
                self.sqrt_one_minus_alphas_cumprod, times, noisy_state.shape
            )
            * noisy_state
            + _extract(self.sqrt_alphas_cumprod, times, noisy_state.shape) * v
        )

    def q_sample(
        self, x_start: Tensor, times: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        """Sample q(x_t | x_0) in normalized model space."""
        self._validate_times(times, x_start.shape[0])
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("noise and x_start must have identical shapes")
        return (
            _extract(self.sqrt_alphas_cumprod, times, x_start.shape) * x_start
            + _extract(
                self.sqrt_one_minus_alphas_cumprod, times, x_start.shape
            )
            * noise
        )

    def model_predictions(
        self,
        noisy_state: Tensor,
        previous_state: Tensor,
        times: Tensor,
        *,
        clip_x_start: bool = False,
    ) -> DiffusionPrediction:
        """Run the prior and return equivalent v, x0, and noise predictions."""
        self._validate_state(noisy_state, "noisy_state")
        self._validate_state(previous_state, "previous_state")
        if noisy_state.shape != previous_state.shape:
            raise ValueError(
                "noisy_state and previous_state must have identical shapes"
            )
        self._validate_times(times, noisy_state.shape[0])

        pred_v = self.model(torch.cat((previous_state, noisy_state), dim=1), times)
        if pred_v.shape != noisy_state.shape:
            raise ValueError("model output must match one physical state")
        pred_x_start = self.predict_start_from_v(noisy_state, times, pred_v)
        if clip_x_start:
            pred_x_start = pred_x_start.clamp(-1, 1)
            pred_noise = self.predict_noise_from_start(
                noisy_state, times, pred_x_start
            )
        else:
            pred_noise = self.predict_noise_from_v(noisy_state, times, pred_v)
        return DiffusionPrediction(pred_noise, pred_x_start, pred_v)

    def tweedie(
        self,
        noisy_state: Tensor,
        previous_state: Tensor,
        times: Tensor,
        *,
        clip: bool = False,
    ) -> Tensor:
        """Return the model's Tweedie/x0 estimate in normalized model space."""
        return self.model_predictions(
            noisy_state, previous_state, times, clip_x_start=clip
        ).pred_x_start

    def training_loss(
        self,
        previous_state: Tensor,
        next_state: Tensor,
        *,
        times: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Return the weighted pred-v MSE for a batch of state transitions."""
        self._validate_state(previous_state, "previous_state")
        self._validate_state(next_state, "next_state")
        if previous_state.shape != next_state.shape:
            raise ValueError("previous_state and next_state must have identical shapes")
        batch = previous_state.shape[0]
        if times is None:
            times = torch.randint(
                self.num_timesteps, (batch,), device=previous_state.device
            )
        self._validate_times(times, batch)
        if noise is None:
            noise = torch.randn_like(next_state)
        if noise.shape != next_state.shape:
            raise ValueError("noise and next_state must have identical shapes")

        previous_model = self.normalize(previous_state)
        next_model = self.normalize(next_state)
        noisy_state = self.q_sample(next_model, times, noise)
        pred_v = self.model(
            torch.cat((previous_model, noisy_state), dim=1), times
        )
        if pred_v.shape != next_state.shape:
            raise ValueError("model output must match one physical state")
        target_v = self.predict_v(next_model, times, noise)
        loss = F.mse_loss(pred_v, target_v, reduction="none").flatten(1).mean(1)
        return (
            loss * _extract(self.loss_weight, times, loss.shape)
        ).mean()

    def forward(
        self,
        previous_state: Tensor,
        next_state: Tensor,
        *,
        times: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Alias for :meth:`training_loss` for normal ``nn.Module`` use."""
        return self.training_loss(
            previous_state, next_state, times=times, noise=noise
        )

    def ddim_sample(
        self,
        previous_state: Tensor,
        *,
        initial_noise: Tensor | None = None,
        sampling_steps: int | None = None,
        clip_x_start: bool = True,
    ) -> Tensor:
        """Generate the next state with deterministic (eta=0) DDIM.

        The mapping is deterministic for a supplied ``initial_noise``. Without
        it, one standard-normal latent is drawn before the deterministic steps.
        Gradients are retained, which is required by adjoint/control training.
        """
        self._validate_state(previous_state, "previous_state")
        steps = self.sampling_steps if sampling_steps is None else sampling_steps
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("sampling_steps must be between 1 and timesteps")

        condition = self.normalize(previous_state)
        if initial_noise is None:
            state = torch.randn_like(condition)
        else:
            if initial_noise.shape != condition.shape:
                raise ValueError(
                    "initial_noise and previous_state must have identical shapes"
                )
            state = initial_noise

        times = torch.linspace(
            -1,
            self.num_timesteps - 1,
            steps=steps + 1,
            device=condition.device,
        )
        time_pairs = tuple(zip(times.long().flip(0)[:-1], times.long().flip(0)[1:]))

        for time, next_time in time_pairs:
            batch_times = torch.full(
                (condition.shape[0],),
                int(time.item()),
                device=condition.device,
                dtype=torch.long,
            )
            prediction = self.model_predictions(
                state,
                condition,
                batch_times,
                clip_x_start=clip_x_start,
            )
            if next_time < 0:
                state = prediction.pred_x_start
                continue
            alpha_next = self.alphas_cumprod[next_time]
            state = (
                alpha_next.sqrt() * prediction.pred_x_start
                + (1 - alpha_next).clamp_min(0).sqrt()
                * prediction.pred_noise
            )

        return self.unnormalize(state)


__all__ = [
    "DiffusionPrediction",
    "GaussianDiffusion",
    "sigmoid_beta_schedule",
]
