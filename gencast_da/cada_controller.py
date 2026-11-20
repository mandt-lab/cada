"""Haiku CADA controller for GenCast.

Variant label: `cada_single_pass_latent_control` (see `CADA_VARIANT` below).

The controller takes the same inputs as the PDE CADA `ControlNet`
(`denoising_diffusion_pytorch_kolm_adjoint_modes_aware_random.py`):

  - noisy latent (`x_sigma` for `temperature`),
  - last-input frame conditioning (`cond_T`),
  - observation context (`obs_like` = `mask*y + (1-mask)*Tweedie`),
  - observation mask,
  - last sub-step's control (`last_u`),
  - noise level `sigma_t` (per batch).

It outputs a `u` of the same shape as `noisy_T` with a final zero-init
projection so the controlled sampler initially reproduces unconditional
GenCast bit-for-bit.

The wrapper in `cada_guided.py` is responsible for:
  * running an auxiliary `stop_gradient`-wrapped frozen denoiser pass to
    fill `obs_like` and provide `pred_noise_aux` for the score-loss
    diagnostic,
  * threading `last_u` through the DPM sampler chain via Haiku state,
  * injecting `gamma * u` into `noisy_targets["temperature"]` before the
    productive denoiser pass,
  * accumulating per-substep score and control loss diagnostics into
    Haiku state.

All tensors here live in **normalized residual space** (the same space
GenCast's preconditioned denoiser operates in).
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp

CADA_VARIANT = "cada_single_pass_latent_control"


def _sinusoidal_embedding(log_sigma: jnp.ndarray, dim: int = 32) -> jnp.ndarray:
    """Standard log-sigma sinusoidal embedding.

    Args:
        log_sigma: (B,) float scalar per batch.
        dim:       output embedding size (must be even).
    """
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, half).astype(log_sigma.dtype))
    angles = log_sigma[:, None] * freqs[None, :]  # (B, half)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class CADAController(hk.Module):
    """Small residual CNN controller mirroring the PDE CADA ControlNet contract.

    Operates on `(B, L, H, W)` tensors with `L` pressure levels treated as
    channels. Concatenates 5 input variables along the channel axis (5*L
    input channels), passes through a few Conv+SiLU+FiLM blocks, then a
    zero-initialized 1x1 projection back to L channels.

    Output `u` has shape `(B, L, H, W)`.
    """

    def __init__(
        self,
        num_levels: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        sigma_emb_dim: int = 64,
        name: str | None = "cada_controller",
    ):
        super().__init__(name=name)
        self.num_levels = num_levels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.sigma_emb_dim = sigma_emb_dim

    def _sigma_emb(self, sigma_t: jnp.ndarray) -> jnp.ndarray:
        log_sigma = jnp.log(sigma_t.astype(jnp.float32) + 1e-8)  # (B,)
        emb = _sinusoidal_embedding(log_sigma, dim=32)
        emb = hk.Linear(self.sigma_emb_dim, name="sigma_lin1")(emb)
        emb = jax.nn.silu(emb)
        emb = hk.Linear(self.sigma_emb_dim, name="sigma_lin2")(emb)
        return emb  # (B, sigma_emb_dim)

    def _res_block(
        self, x: jnp.ndarray, sigma_emb: jnp.ndarray, name: str
    ) -> jnp.ndarray:
        """Pre-act residual block with FiLM conditioning on sigma_emb.

        x:         (B, H, W, C)
        sigma_emb: (B, sigma_emb_dim)
        """
        film = hk.Linear(2 * self.hidden_channels, name=f"{name}_film")(sigma_emb)
        scale, shift = jnp.split(film, 2, axis=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]

        h = hk.Conv2D(
            self.hidden_channels,
            kernel_shape=3,
            padding="SAME",
            name=f"{name}_c1",
        )(x)
        h = h * (1.0 + scale) + shift
        h = jax.nn.silu(h)
        h = hk.Conv2D(
            self.hidden_channels,
            kernel_shape=3,
            padding="SAME",
            name=f"{name}_c2",
        )(h)
        return x + h

    def __call__(
        self,
        noisy_T: jnp.ndarray,
        cond_T: jnp.ndarray,
        obs_like: jnp.ndarray,
        mask: jnp.ndarray,
        last_u: jnp.ndarray,
        sigma_t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the per-substep control map u.

        Args:
            noisy_T:  (B, L, H, W) - normalized residual noisy temperature.
            cond_T:   (B, L, H, W) - last input frame's normalized temperature.
            obs_like: (B, L, H, W) - mask*y + (1-mask)*x0_hat_aux.
            mask:     (B, L, H, W) - 1.0 at observed cells.
            last_u:   (B, L, H, W) - u from the previous sampler substep.
            sigma_t:  (B,)         - current noise level.

        Returns:
            u: (B, L, H, W) - control map; zero at init.
        """

        # NCHW -> NHWC for haiku Conv2D (expects channels-last).
        def to_nhwc(z):
            return jnp.transpose(z, (0, 2, 3, 1))

        x = jnp.concatenate(
            [
                to_nhwc(noisy_T),
                to_nhwc(cond_T),
                to_nhwc(obs_like),
                to_nhwc(mask),
                to_nhwc(last_u),
            ],
            axis=-1,
        )  # (B, H, W, 5*L)

        sigma_emb = self._sigma_emb(sigma_t)

        h = hk.Conv2D(
            self.hidden_channels, kernel_shape=3, padding="SAME", name="stem"
        )(x)
        h = jax.nn.silu(h)
        for i in range(self.num_blocks):
            h = self._res_block(h, sigma_emb, name=f"block_{i}")
        h = jax.nn.silu(h)
        # Zero-init final projection -> u starts at identically zero.
        u = hk.Conv2D(
            self.num_levels,
            kernel_shape=1,
            padding="SAME",
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="proj_out",
        )(h)
        # NHWC -> NCHW.
        return jnp.transpose(u, (0, 3, 1, 2))
