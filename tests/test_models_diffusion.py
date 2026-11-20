from __future__ import annotations

import pytest
import torch

from cada.diffusion import GaussianDiffusion
from cada.models import UNet1D, UNet2D


@pytest.mark.parametrize(
    ("model", "shape"),
    [
        (
            UNet1D(
                1,
                base_width=4,
                dim_multipliers=(1, 2),
                sinusoidal_embedding_dim=4,
                attention_heads=1,
                attention_dim_head=4,
            ),
            (2, 2, 8),
        ),
        (
            UNet2D(
                1,
                base_width=4,
                dim_multipliers=(1, 2),
                sinusoidal_embedding_dim=4,
                attention_heads=1,
                attention_dim_head=4,
            ),
            (1, 2, 4, 4),
        ),
    ],
)
def test_tiny_unet_forward_and_backward(
    model: torch.nn.Module, shape: tuple[int, ...]
) -> None:
    states = torch.randn(shape, requires_grad=True)
    times = torch.arange(shape[0])

    output = model(states, times)
    output.square().mean().backward()

    assert output.shape == (shape[0], 1, *shape[2:])
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pred_v_training_and_deterministic_ddim(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    previous = torch.linspace(0.1, 0.8, 16).reshape(2, 1, 8)
    following = previous.flip(-1)
    times = torch.tensor([1, 3], dtype=torch.long)
    noise = torch.zeros_like(following)

    loss = tiny_diffusion.training_loss(previous, following, times=times, noise=noise)
    previous_model = tiny_diffusion.normalize(previous)
    following_model = tiny_diffusion.normalize(following)
    noisy = tiny_diffusion.q_sample(following_model, times, noise)
    predicted_v = tiny_diffusion.model(torch.cat((previous_model, noisy), dim=1), times)
    target_v = tiny_diffusion.predict_v(following_model, times, noise)
    expected = (
        (predicted_v - target_v).square().flatten(1).mean(1)
        * tiny_diffusion.loss_weight[times]
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert tiny_diffusion.model.projection.weight.grad is not None

    initial_noise = torch.linspace(-1, 1, previous.numel()).reshape_as(previous)
    first = tiny_diffusion.ddim_sample(
        previous,
        initial_noise=initial_noise,
        sampling_steps=2,
        clip_x_start=False,
    )
    second = tiny_diffusion.ddim_sample(
        previous,
        initial_noise=initial_noise.clone(),
        sampling_steps=2,
        clip_x_start=False,
    )
    different = tiny_diffusion.ddim_sample(
        previous,
        initial_noise=-initial_noise,
        sampling_steps=2,
        clip_x_start=False,
    )

    assert first.shape == previous.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
