"""Bounded synthetic CPU/CUDA smoke test for the PyTorch CADA backend."""

from __future__ import annotations

import argparse
import json

import torch

from .assimilation import preview_rollout
from .control import ControlNet1D
from .diffusion import GaussianDiffusion
from .models import UNet1D
from .observations import RegularGridMask, TemporalCadence


def run_smoke(
    device: torch.device,
    *,
    seed: int = 0,
) -> dict[str, float | str | bool]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda is unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = UNet1D(
        1,
        base_width=4,
        dim_multipliers=(1, 2),
        sinusoidal_embedding_dim=4,
        attention_heads=1,
        attention_dim_head=4,
    ).to(device)
    diffusion = GaussianDiffusion(
        model,
        timesteps=4,
        sampling_steps=2,
    ).to(device)
    previous = torch.linspace(0.1, 0.8, 16, device=device).reshape(2, 1, 8)
    following = previous.flip(-1)
    prior_loss = diffusion.training_loss(
        previous,
        following,
        times=torch.tensor([1, 3], device=device),
        noise=torch.zeros_like(following),
    )
    prior_loss.backward()
    if not bool(torch.isfinite(prior_loss)):
        raise RuntimeError("CADA prior smoke produced a non-finite loss")
    if not any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise RuntimeError("CADA prior smoke produced no finite gradients")

    diffusion.zero_grad(set_to_none=True)
    diffusion.requires_grad_(False)
    controller = ControlNet1D(1, hidden_width=4).to(device)
    trajectory = torch.stack((previous, following, previous.roll(1, -1)), dim=1)
    rollout = preview_rollout(
        diffusion,
        controller,
        trajectory,
        RegularGridMask(1, 2, 0),
        TemporalCadence(interval=2, start=2),
        preview_steps=2,
        gamma=0.2,
        beta=0.01,
        initial_noises=torch.zeros(2, *previous.shape, device=device),
        sampling_steps=2,
        clip_x_start=False,
    )
    rollout.loss.backward()
    if not bool(torch.isfinite(rollout.loss)):
        raise RuntimeError("CADA controller smoke produced a non-finite loss")
    if not any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in controller.parameters()
    ):
        raise RuntimeError("CADA controller smoke produced no finite gradients")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / 2**20
        cuda_state = torch.cuda.get_rng_state_all()
        expected_random = torch.rand(4, device=device)
        torch.cuda.set_rng_state_all(cuda_state)
        actual_random = torch.rand(4, device=device)
        cuda_rng_replay = bool(torch.equal(expected_random, actual_random))
        if not cuda_rng_replay:
            raise RuntimeError("CUDA RNG state replay failed")
    else:
        peak_memory = 0.0
        cuda_rng_replay = False
    return {
        "device": str(device),
        "prior_loss": float(prior_loss.detach().cpu()),
        "controller_loss": float(rollout.loss.detach().cpu()),
        "peak_memory_mib": peak_memory,
        "cuda_rng_replay": cuda_rng_replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(run_smoke(torch.device(args.device), seed=args.seed), sort_keys=True)
    )


if __name__ == "__main__":
    main()
