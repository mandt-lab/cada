from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WANDB_MODE", "disabled")


class TinyPrior(nn.Module):
    physical_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.projection.weight, 0.1)

    def forward(
        self, states: torch.Tensor, diffusion_time: torch.Tensor
    ) -> torch.Tensor:
        time = diffusion_time.to(states).reshape(-1, 1, 1)
        return self.projection(states) + 0.001 * time


class TinyController(nn.Module):
    physical_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        noisy_state: torch.Tensor,
        previous_state: torch.Tensor,
        observation: torch.Tensor,
        mask: torch.Tensor,
        previous_control: torch.Tensor,
        context: object,
    ) -> torch.Tensor:
        del previous_state, context
        return self.gain * (observation - noisy_state) * mask + 0.01 * previous_control


@pytest.fixture(autouse=True)
def cpu_offline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_connection(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("tests may not access the network")

    monkeypatch.setattr(socket, "create_connection", blocked_connection)
    monkeypatch.setattr(socket.socket, "connect", blocked_connection)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture
def tiny_diffusion() -> object:
    from cada.diffusion import GaussianDiffusion

    return GaussianDiffusion(
        TinyPrior(), timesteps=4, sampling_steps=2, auto_normalize=True
    )
