"""FiLM-conditioned 1D and 2D controllers for CADA rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ControlContext:
    """Per-sample scalar controller context.

    ``dt`` and ``tau`` are normalized observation-gap and rollout-position
    scalars. ``snr`` is the diffusion signal-to-noise ratio. Each value may
    have shape ``[]``, ``[1]``, ``[B]``, or ``[B, 1]``.
    """

    dt: Tensor
    tau: Tensor
    snr: Tensor


def _group_count(channels: int) -> int:
    return next(
        groups for groups in range(min(8, channels), 0, -1) if channels % groups == 0
    )


def _scalar_column(value: Tensor, batch: int, reference: Tensor, name: str) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1, 1).expand(batch, 1)
    if value.shape == (batch,):
        return value[:, None]
    if value.shape == (batch, 1):
        return value
    raise ValueError(f"context.{name} must be scalar or have one value per batch")


class _ControlNet(nn.Module):
    def __init__(
        self, physical_channels: int, hidden_width: int, spatial_dims: int
    ) -> None:
        super().__init__()
        if physical_channels < 1 or hidden_width < 1:
            raise ValueError("physical_channels and hidden_width must be positive")
        self.physical_channels = physical_channels
        self.spatial_dims = spatial_dims
        conv = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        groups = _group_count(hidden_width)

        self.encoder = nn.Sequential(
            conv(physical_channels * 5, hidden_width, 3, padding=1),
            nn.ReLU(),
            conv(hidden_width, hidden_width, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder_norm = nn.GroupNorm(groups, hidden_width)
        self.down = nn.Sequential(
            conv(hidden_width, hidden_width, 3, stride=2, padding=1),
            nn.ReLU(),
            conv(hidden_width, hidden_width, 3, padding=1),
            nn.ReLU(),
        )
        self.up_projection = nn.Sequential(
            nn.ReLU(), conv(hidden_width, hidden_width, 3, padding=1)
        )
        self.fuse = conv(hidden_width, hidden_width, 1)
        self.fused_norm = nn.GroupNorm(groups, hidden_width)

        def scalar_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(1, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, hidden_width),
            )

        self.dt_mlp = scalar_mlp()
        self.tau_mlp = scalar_mlp()
        self.snr_mlp = scalar_mlp()
        self.film = nn.Linear(hidden_width * 3, hidden_width * 2)
        self.delta = nn.Sequential(
            conv(hidden_width, hidden_width, 3, padding=1),
            nn.ReLU(),
            conv(hidden_width, physical_channels, 3, padding=1),
        )
        self.previous_control_norm = nn.GroupNorm(1, physical_channels)

    def forward(
        self,
        noisy_state: Tensor,
        previous_state: Tensor,
        observation: Tensor,
        mask: Tensor,
        previous_control: Tensor,
        context: ControlContext,
    ) -> Tensor:
        expected_dims = self.spatial_dims + 2
        if noisy_state.ndim != expected_dims:
            shape = "[B,C,X]" if self.spatial_dims == 1 else "[B,C,H,W]"
            raise ValueError(f"controller inputs must have shape {shape}")
        if noisy_state.shape[1] != self.physical_channels:
            raise ValueError(
                f"controller inputs must have {self.physical_channels} channels"
            )
        inputs = (
            previous_state,
            observation,
            mask,
            previous_control,
        )
        if any(value.shape != noisy_state.shape for value in inputs):
            raise ValueError("all controller state inputs must have identical shapes")

        features = self.encoder(
            torch.cat(
                (
                    noisy_state,
                    previous_state,
                    observation,
                    mask,
                    previous_control,
                ),
                dim=1,
            )
        )
        features = self.encoder_norm(features)
        coarse = self.down(features)
        coarse = F.interpolate(
            coarse, size=features.shape[2:], mode="nearest"
        )
        fused = self.fused_norm(
            self.fuse(features + self.up_projection(coarse))
        )

        batch = noisy_state.shape[0]
        dt = _scalar_column(context.dt, batch, noisy_state, "dt")
        tau = _scalar_column(context.tau, batch, noisy_state, "tau")
        snr = _scalar_column(context.snr, batch, noisy_state, "snr")
        gamma, beta = self.film(
            torch.cat(
                (self.dt_mlp(dt), self.tau_mlp(tau), self.snr_mlp(snr)), dim=1
            )
        ).chunk(2, dim=1)
        broadcast_shape = (batch, gamma.shape[1], *([1] * self.spatial_dims))
        gamma = gamma.reshape(broadcast_shape)
        beta = beta.reshape(broadcast_shape)
        delta = self.delta(fused * (1 + gamma) + beta)
        return self.previous_control_norm(previous_control) + delta


class ControlNet1D(_ControlNet):
    """1D control map with scalar FiLM and residual previous-control input."""

    def __init__(
        self, physical_channels: int, *, hidden_width: int = 64
    ) -> None:
        super().__init__(
            physical_channels=physical_channels,
            hidden_width=hidden_width,
            spatial_dims=1,
        )


class ControlNet2D(_ControlNet):
    """2D control map with scalar FiLM and residual previous-control input."""

    def __init__(
        self, physical_channels: int, *, hidden_width: int = 64
    ) -> None:
        super().__init__(
            physical_channels=physical_channels,
            hidden_width=hidden_width,
            spatial_dims=2,
        )


__all__ = ["ControlContext", "ControlNet1D", "ControlNet2D"]
