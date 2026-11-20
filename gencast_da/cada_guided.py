"""CADA-guided GenCast sampler — single-pass latent-control wrapper.

Variant label: `cada_single_pass_latent_control` (re-exported from
`cada_controller.CADA_VARIANT`).

This file is the GenCast analogue of the PDE CADA sampling loop in
`denoising_diffusion_pytorch_kolm_adjoint_modes_aware_random.py`. Each
sampler substep runs:

  1. **Auxiliary** frozen `_preconditioned_denoiser` call on the *raw*
     `noisy_targets`, wrapped in `jax.lax.stop_gradient`. The output
     does double duty: it fills `obs_like` at unobserved cells (Tweedie
     fill) and provides `pred_noise_aux` for the score-loss diagnostic.
  2. **Controller** call producing `u(noisy_T, cond_T, obs_like, mask,
     last_u, sigma_t)`.
  3. **Productive** frozen `_preconditioned_denoiser` call on
     `noisy_targets` after injecting `gamma * u` into the `temperature`
     variable. This is the only pass whose output drives the sampler.

`last_u` is held in Haiku state and threaded through the unrolled
DPM-Solver++ 2S loop. Per-substep score, control, and masked
observation losses are accumulated into Haiku state too, so the train
loop can read them out of the final state without an extra backprop.

The masked observation loss is the GenCast analogue of `loss_mid` in
`denoising_diffusion_pytorch_kolm_adjoint_modes_aware_random.py`
(`loss_mid = w_t * sum(mask*(x_start_c - gt)^2) / sum(mask)`), but
applied in physical Kelvin: GenCast's denoiser predicts normalized
residuals, so we de-normalize `x0_T_ctrl` via
`T_pred = last_input + r_pred * diffs_stddev_T` before comparing to
the physical-K observation. This makes the substep obs loss directly
comparable to the evaluation observation-innovation MSE (mirroring how Kolm CADA's
loss space *is* its data space).

Score and control losses (`(pred_noise_aux - pred_noise_c)^2` and
`(gamma*u)^2`) follow `losses['terminal']` semantics from
`train_controlnet` (line 2373 of the PDE CADA file): they are computed
and exposed but contribute to the gradient only if the train script
weights them non-zero.

Mirrors the structure of `mmps_guided.py` so most of the wiring
(`InputsAndResiduals`, `NaNCleaner`, `make_obs_arrays`) is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from graphcast import (
    casting,
    gencast,
    nan_cleaning,
    normalization,
    samplers_utils,
    xarray_jax,
    xarray_tree,
)

from gencast_da.cada_controller import CADA_VARIANT, CADAController
from gencast_da.gencast_runner import GenCastBundle

__all__ = [
    "CADA_VARIANT",
    "CADAControllerConfig",
    "make_cada_run_forward",
    "make_scan_unconditional_run_forward",
    "split_params",
    "merge_params",
    "init_cada",
]


@dataclass
class CADAControllerConfig:
    """Hyper-params for the controller; passed straight to `CADAController`.

    `apply_remat=True` wraps only the productive GenCast denoiser pass
    in `hk.remat`. The auxiliary pass stays outside remat so graphcast's
    lazy transformer mask is initialised in the outer trace; the
    productive pass then captures that mask safely and drops its large
    activations until backward recompute.

    Default is `True` because the productive denoiser activations are the
    main memory pressure during controller training. Turn it off only for
    forward-only diagnostics where the recompute cost matters.
    """

    hidden_channels: int = 32
    num_blocks: int = 3
    sigma_emb_dim: int = 64
    gamma: float = 1.0
    apply_remat: bool = True

    def kwargs(self) -> dict[str, Any]:
        """Args forwarded to `CADAController.__init__` (excludes gamma)."""
        return dict(
            hidden_channels=self.hidden_channels,
            num_blocks=self.num_blocks,
            sigma_emb_dim=self.sigma_emb_dim,
        )


# ---------------------------------------------------------------------------- #
#  Differentiable GenCast: short Python-loop DPM-Solver++ 2S
# ---------------------------------------------------------------------------- #
#
# The default `gencast.GenCast.__call__` uses `dpm_solver_plus_plus_2s.Sampler`
# which uses `hk.fori_loop`. That lowers to `jax.lax.while_loop` and is NOT
# reverse-mode differentiable. CADA training requires backprop through the
# full sampler chain.
#
# `hk.scan` keeps the forward graph compact, but its reverse-mode transpose
# allocates a large saved-residual buffer (~9.7 GiB even at N=4 here). Since
# the pipeline now caps the shared train/eval schedule at CADA_NOISE_LEVELS=4,
# the lower-memory gradient path is a short Python loop plus fine-grained
# remat around the productive denoiser pass inside `controlled_pcd`. Avoid
# raising N without revisiting this tradeoff: at N=8 the Python-unrolled graph
# was large enough for the smoke test to be OS-killed during Test 1.


class _CADAScanGenCast(gencast.GenCast):
    """GenCast that runs DPM-Solver++ 2S as a short differentiable loop.

    `apply_remat` is accepted for API compatibility but is currently a
    no-op at this level: rematerialisation is applied at a finer grain
    inside `controlled_pcd` (around the productive `original_pcd` call).
    See the comment near the Python-loop call site, and
    `_wrap_with_cada_guidance`'s `original_pcd_remat` comment, for the
    rationale.
    """

    def __init__(self, *args, apply_remat: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_remat = apply_remat

    def __call__(self, inputs, targets_template, forcings, **kwargs):
        if self._sampler_config is None:
            raise ValueError(
                "Sampler config must be specified to run inference on GenCast."
            )
        cfg = self._sampler_config
        noise_levels_np = samplers_utils.noise_schedule(
            cfg.max_noise_level,
            cfg.min_noise_level,
            cfg.num_noise_levels,
            cfg.rho,
        )
        churn_rates_np = samplers_utils.stochastic_churn_rate_schedule(
            noise_levels_np,
            cfg.stochastic_churn_rate,
            cfg.churn_min_noise_level,
            cfg.churn_max_noise_level,
        )
        dtype = casting.infer_floating_dtype(targets_template)
        noise_levels = jnp.array(noise_levels_np).astype(dtype)
        churn_rates = jnp.array(churn_rates_np).astype(dtype)
        stochastic = cfg.stochastic_churn_rate > 0

        def _denoiser_call(x, noise_level):
            bcast_noise_level = xarray_jax.DataArray(
                jnp.tile(noise_level, x.sizes["batch"]), dims=("batch",)
            )
            return self._preconditioned_denoiser(
                inputs=inputs,
                noisy_targets=x,
                noise_levels=bcast_noise_level,
                forcings=forcings,
                **kwargs,
            )

        # Initialise from white noise at the highest noise level. RNG is
        # consumed once here; stochastic churn inside the Python loop uses
        # Haiku RNG in the same way as a normal unrolled Haiku forward.
        x_init = xarray.zeros_like(targets_template)
        x_init = x_init + noise_levels[0] * samplers_utils.spherical_white_noise_like(
            targets_template
        )

        def scan_body(x_in, scan_xs):
            ni, nn, churn, do_second_denoise = scan_xs
            if stochastic:
                x_use, sigma_used = samplers_utils.apply_stochastic_churn(
                    x_in,
                    ni,
                    stochastic_churn_rate=churn,
                    noise_level_inflation_factor=cfg.noise_level_inflation_factor,
                )
            else:
                x_use, sigma_used = x_in, ni

            mid_noise_level = jnp.sqrt(sigma_used * nn)
            mid_over_current = mid_noise_level / sigma_used
            next_over_current = nn / sigma_used

            x_denoised = _denoiser_call(x_use, sigma_used)
            x_mid = mid_over_current * x_use + (1 - mid_over_current) * x_denoised
            if do_second_denoise:
                x_mid_denoised = _denoiser_call(x_mid, mid_noise_level)
                x_next = (
                    next_over_current * x_use + (1 - next_over_current) * x_mid_denoised
                )
                x_out = x_next
            else:
                # Final substep: Euler update straight to noise level 0.
                # The official sampler issues a second denoiser call at
                # sigma==0 and discards it. Here we skip that discarded call:
                # GenCast can emit NaNs at noise_level=0, and even inactive
                # `where` branches can poison reverse-mode with NaN
                # cotangents. The physical update is identical because the
                # discarded second call never contributes to x_out.
                x_out = x_denoised
            return x_out, None

        # noise_levels has length N+1 (includes the terminal 0); churn_rates
        # has length N already (samplers_utils.stochastic_churn_rate_schedule
        # returns per-substep values via its own [:-1] slice). All scan_xs
        # entries below must therefore have length N (or N-1 after the
        # warm-start iter below splits one off).

        # Warm-start: run iter 0 in the OUTER trace so graphcast's lazy
        # initialisation (MeshTransformer._maybe_init_batch_first_transformer
        # -> SparseTransformer with `self.mask = mask_block_diags(...)`)
        # populates `self.mask` here. If we let iter 0 run inside hk.remat,
        # `self.mask` would get bound to the jax.checkpoint sub-trace and
        # subsequent iters would see a stale tracer (UnexpectedTracerError on
        # `int32[1,3,2,1,1297,1297]`). Iters 1..N-1 then access self.mask via
        # closure from the outer scope, which is valid even under
        # remat/scan.
        x_after_iter0, _ = scan_body(
            x_init,
            (
                noise_levels[0],
                noise_levels[1],
                churn_rates[0],
                bool(noise_levels_np[1] > 0),
            ),
        )

        # Do NOT use hk.scan on the gradient path. Its transpose still
        # stacks a large residual buffer at N=4 in this model. The shared
        # pipeline schedule is capped at N=4, so a short Python loop avoids
        # the scan-transpose allocation while keeping the graph small enough
        # for the smoke/train path.
        x_final = x_after_iter0
        for i in range(1, noise_levels.shape[0] - 1):
            x_final, _ = scan_body(
                x_final,
                (
                    noise_levels[i],
                    noise_levels[i + 1],
                    churn_rates[i],
                    bool(noise_levels_np[i + 1] > 0),
                ),
            )
        return x_final


# ---------------------------------------------------------------------------- #
#  Wrapper around `_preconditioned_denoiser`
# ---------------------------------------------------------------------------- #


def _wrap_with_cada_guidance(
    gencast_predictor: gencast.GenCast,
    var: str,
    obs_full_jax: jnp.ndarray,
    obs_mask_jax: jnp.ndarray,
    obs_phys_full_jax: jnp.ndarray,
    last_input_T_phys: jnp.ndarray,
    diffs_stddev_T: jnp.ndarray,
    cfg: CADAControllerConfig,
    num_levels: int,
) -> None:
    """Replace `gencast_predictor._preconditioned_denoiser` in place.

    The replacement runs an auxiliary stop-gradient pass to build
    `obs_like` and `pred_noise_aux`, calls the controller, injects
    `gamma * u` into the noisy temperature, then runs the productive
    pass on the controlled latent. Per-substep score, control, and
    masked-observation diagnostics are accumulated into Haiku state.

    The controller is instantiated *once* in this scope (which is itself
    inside the `hk.transform_with_state` context) and captured by
    closure, so every DPM-Solver++ 2S substep reuses the *same* params.
    Without this, `controlled_pcd` would create a fresh `CADAController`
    per call, and Haiku's auto-suffixed names (`cada_controller_1`,
    `cada_controller_2`, ...) would silently make every substep its own
    independent controller — breaking the `last_u_t = controller_θ(...,
    last_u_{t-1})` recurrence the design depends on.

    Args:
        gencast_predictor: instance with `_preconditioned_denoiser`
            attribute (will be monkey-patched).
        var: variable name to control. For paper-idealized scenario this
            is `"temperature"`.
        obs_full_jax, obs_mask_jax: shape `(B, 1, L, H, W)`. `obs_full_jax`
            is in *normalized residual space* and is what the controller
            sees on its input channel (Tweedie fill is applied here, then
            the controller reasons about residuals like the rest of the
            denoiser does). Same convention as `mmps_guided.make_obs_arrays`.
        obs_phys_full_jax: shape `(B, 1, L, H, W)`, physical-Kelvin obs
            zero-filled at unobserved cells. This is the supervision
            target for the substep observation loss; see comment at
            substep_obs_substep below.
        last_input_T_phys: shape `(B, L, H, W)`, physical Kelvin at
            `inputs[time=-1]` (the most recent input frame). Used to
            de-normalize `x0_T_ctrl` back to physical K for the obs loss.
        diffs_stddev_T: shape `(L,)` per-level diff stddev for `var`.
            Same constant the bundle's `InputsAndResiduals` uses.
        cfg: controller hyper-params + gamma.
        num_levels: number of pressure levels (controller output channels).
    """
    original_pcd = gencast_predictor._preconditioned_denoiser
    gamma = cfg.gamma

    # `original_pcd_remat` is used ONLY for the productive pass below.
    # The aux pass intentionally calls the un-rematted `original_pcd`:
    # it is the FIRST MeshTransformer call per `cada_xform.apply` and
    # is what populates graphcast's lazy `self.mask` attribute. If the
    # aux call were rematted, `self.mask` would get bound to the
    # `jax.checkpoint` sub-trace and the productive call (which reads
    # `self.mask` via Python closure) would see a stale tracer
    # (`UnexpectedTracerError` on `int32[1,3,2,1,1297,1297]`). With aux
    # outside remat, `self.mask` lives in the outer trace and the
    # productive remat captures it as a legal outer→inner closure
    # constant. Productive's activations — the only ones whose
    # cotangents we need — are then dropped after forward and recomputed
    # in backward, bounding per-substep memory.
    original_pcd_remat = hk.remat(original_pcd) if cfg.apply_remat else original_pcd

    # One controller, shared across every _preconditioned_denoiser call
    # inside the DPM-Solver++ 2S unrolled loop. See docstring above.
    controller = CADAController(num_levels=num_levels, **cfg.kwargs())

    def controlled_pcd(inputs, noisy_targets, noise_levels, forcings, **kwargs):
        # ---------- 0. Pull primary tensors and broadcasts. ----------
        # noise_levels is xarray_jax.DataArray of shape (B,)
        sigma_t = xarray_jax.unwrap_data(noise_levels).astype(jnp.float32)  # (B,)
        sigma_b = sigma_t[:, None, None, None, None]  # (B, 1, 1, 1, 1)
        valid = sigma_b.reshape(()) > 0
        valid_f = valid.astype(jnp.float32)

        temp_da = noisy_targets[var].transpose("batch", "time", "level", "lat", "lon")
        noisy_T = xarray_jax.unwrap_data(temp_da)  # (B, 1, L, H, W) jax array

        cond_da = inputs[var].isel(time=-1).transpose("batch", "level", "lat", "lon")
        cond_T = xarray_jax.unwrap_data(cond_da).astype(jnp.float32)  # (B, L, H, W)

        # ---------- 1. Auxiliary frozen pass (no-grad). -------------
        x0_hat_aux_full = original_pcd(
            inputs=inputs,
            noisy_targets=noisy_targets,
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs,
        )
        x0_T_aux_da = x0_hat_aux_full[var].transpose(
            "batch", "time", "level", "lat", "lon"
        )
        x0_T_aux_raw = jax.lax.stop_gradient(
            xarray_jax.unwrap_data(x0_T_aux_da).astype(jnp.float32)
        )  # (B, 1, L, H, W)
        # At sigma==0 this denoiser call is structurally executed but its
        # result is discarded by the DPM-Solver update. GenCast may emit NaNs
        # there; do not let them poison obs_like/controller diagnostics.
        x0_T_aux = jnp.where(valid, x0_T_aux_raw, noisy_T.astype(jnp.float32))

        # Tweedie fill at unobserved cells: obs_like = mask*y + (1-mask)*x0_aux
        obs_like = obs_mask_jax * obs_full_jax + (1.0 - obs_mask_jax) * x0_T_aux

        # pred_noise_aux for score-loss diagnostic (also no-grad).
        # The DPM-Solver++ 2S body calls _preconditioned_denoiser at
        # mid_noise_level == 0 on its final substep (the result gets
        # discarded by the tree_where in scan_body). At that call
        # `sigma_b == 0`, which would NaN the 1/sigma noise estimate. We
        # use a safe divisor and gate the diagnostic accumulation by
        # `sigma_b > 0` below so the discarded substep contributes 0.
        safe_sigma_b = jnp.where(sigma_b > 0, sigma_b, 1.0)
        pred_noise_aux = jax.lax.stop_gradient(
            (noisy_T.astype(jnp.float32) - x0_T_aux) / safe_sigma_b
        )

        # ---------- 2. Controller. ----------
        # Drop the singleton time dim for controller -> (B, L, H, W).
        noisy_T_b = noisy_T[:, 0]
        obs_like_b = obs_like[:, 0]
        mask_b = obs_mask_jax[:, 0]

        last_u = hk.get_state(
            "last_u",
            shape=noisy_T_b.shape,
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )

        u_raw = controller(
            noisy_T=noisy_T_b.astype(jnp.float32),
            cond_T=cond_T,
            obs_like=obs_like_b.astype(jnp.float32),
            mask=mask_b.astype(jnp.float32),
            last_u=last_u,
            sigma_t=sigma_t,
        )  # (B, L, H, W) float32
        # The sigma==0 call is discarded. Keep the executed graph finite and
        # preserve the previous recurrent control state for the next real
        # substep (there usually is none, but this avoids state NaNs).
        u = jnp.where(valid, u_raw, jnp.zeros_like(u_raw))

        hk.set_state("last_u", jnp.where(valid, u, last_u))

        # ---------- 3. Inject control into noisy temperature. ----------
        u_b1 = u[:, None]  # (B, 1, L, H, W)
        new_T = (noisy_T.astype(jnp.float32) + gamma * u_b1).astype(noisy_T.dtype)
        new_T_da = xarray_jax.DataArray(
            new_T,
            dims=temp_da.dims,
            coords=temp_da.coords,
            name=var,
        )

        def _replace_var(da):
            return new_T_da if da.name == var else da

        noisy_targets_ctrl = xarray_tree.map_structure(_replace_var, noisy_targets)

        # ---------- 4. Productive frozen pass (with grad). ----------
        # Rematted: drops activations after forward, recomputed in
        # backward. See `original_pcd_remat` comment above.
        x0_hat_ctrl_full = original_pcd_remat(
            inputs=inputs,
            noisy_targets=noisy_targets_ctrl,
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs,
        )
        x0_T_ctrl_da = x0_hat_ctrl_full[var].transpose(
            "batch", "time", "level", "lat", "lon"
        )
        x0_T_ctrl_raw = xarray_jax.unwrap_data(x0_T_ctrl_da).astype(jnp.float32)
        x0_T_ctrl = jnp.where(valid, x0_T_ctrl_raw, new_T.astype(jnp.float32))

        pred_noise_c = (new_T.astype(jnp.float32) - x0_T_ctrl) / safe_sigma_b

        # ---------- 5. Diagnostics (always computed; weight in train loop). ----------
        # The discarded sigma==0 call can still produce NaNs inside
        # `x0_T_ctrl`; use `where`, not multiplication by 0, because
        # `0 * nan` is still nan.
        score_substep = jnp.where(
            valid, jnp.mean((pred_noise_aux - pred_noise_c) ** 2), 0.0
        )
        control_substep = jnp.where(valid, jnp.mean((gamma * u) ** 2), 0.0)
        u_l2_substep = jnp.where(valid, jnp.mean(u**2), 0.0)

        # Kolm-style masked substep observation loss on the controlled
        # denoiser's clean estimate, in PHYSICAL KELVIN space. In Kolm
        # CADA the per-substep observation term is
        #     loss_mid = w_t * sum( mask * (x_start_c - gt)^2 ) / sum(mask)
        # and Kolm's model lives in the same units as the supervision, so
        # the loss is unambiguously in "model space == data space". For
        # GenCast the inner `_preconditioned_denoiser` returns normalized
        # residuals `r = (T - T_last_input) / diffs_stddev_T`, so we have
        # to first map back to physical Kelvin before comparing to obs:
        #     T_pred_phys = T_last_input_phys + r * diffs_stddev_T
        # This makes the substep loss the *exact* physical-K masked obs
        # MSE — i.e. `sqrt(loss)` at convergence equals the rollout-step
        # observation-innovation RMSE reported in eval. It also drops the implicit
        # `1/diffs_stddev_T^2` per-level reweighting the normalized-
        # residual version was applying (which had no analogue in Kolm
        # CADA and over-weighted stratospheric T where 12 h diffs are
        # tiny). Gated by `valid` to skip the discarded sigma==0 call.
        x0_T_ctrl_phys = (
            last_input_T_phys[:, None]
            + x0_T_ctrl * diffs_stddev_T[None, None, :, None, None]
        )
        diff_obs_phys = (x0_T_ctrl_phys - obs_phys_full_jax) * obs_mask_jax
        n_obs = jnp.maximum(jnp.sum(obs_mask_jax), 1.0)
        substep_obs_substep = jnp.where(valid, jnp.sum(diff_obs_phys**2) / n_obs, 0.0)

        score_sum = hk.get_state(
            "score_sum",
            shape=(),
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )
        control_sum = hk.get_state(
            "control_sum",
            shape=(),
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )
        u_l2_sum = hk.get_state(
            "u_l2_sum",
            shape=(),
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )
        substep_obs_sum = hk.get_state(
            "substep_obs_sum",
            shape=(),
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )
        substep_count = hk.get_state(
            "substep_count",
            shape=(),
            dtype=jnp.float32,
            init=lambda s, d: jnp.zeros(s, d),
        )
        hk.set_state("score_sum", score_sum + score_substep)
        hk.set_state("control_sum", control_sum + control_substep)
        hk.set_state("u_l2_sum", u_l2_sum + u_l2_substep)
        hk.set_state("substep_obs_sum", substep_obs_sum + substep_obs_substep)
        hk.set_state("substep_count", substep_count + valid_f)

        return x0_hat_ctrl_full

    gencast_predictor._preconditioned_denoiser = controlled_pcd


def _construct_cada_wrapped(
    bundle: GenCastBundle,
    obs_full_jax: jnp.ndarray,
    obs_mask_jax: jnp.ndarray,
    obs_phys_full_jax: jnp.ndarray,
    last_input_T_phys: jnp.ndarray,
    diffs_stddev_T: jnp.ndarray,
    cfg: CADAControllerConfig,
    var: str,
):
    """Same as `bundle.construct_wrapped()` but with CADA injected and a
    differentiable DPM-Solver++ 2S sampler (so backprop works under
    bounded activation memory; see _CADAScanGenCast docstring).

    Extra args vs. unconditional path are needed by the substep obs loss:
    `obs_phys_full_jax` is the supervision target (physical K),
    `last_input_T_phys` and `diffs_stddev_T` are needed to undo
    `InputsAndResiduals`' normalization of `x0_T_ctrl` back to physical K.
    See `_wrap_with_cada_guidance` docstring.
    """
    gencast_predictor = _CADAScanGenCast(
        sampler_config=bundle.sampler_config,
        task_config=bundle.task_config,
        denoiser_architecture_config=bundle.denoiser_architecture_config,
        noise_config=bundle.noise_config,
        noise_encoder_config=bundle.noise_encoder_config,
        apply_remat=cfg.apply_remat,
    )
    num_levels = len(bundle.task_config.pressure_levels)
    _wrap_with_cada_guidance(
        gencast_predictor,
        var,
        obs_full_jax,
        obs_mask_jax,
        obs_phys_full_jax,
        last_input_T_phys,
        diffs_stddev_T,
        cfg,
        num_levels,
    )
    predictor = normalization.InputsAndResiduals(
        gencast_predictor,
        diffs_stddev_by_level=bundle.diffs_stddev_by_level,
        mean_by_level=bundle.mean_by_level,
        stddev_by_level=bundle.stddev_by_level,
    )
    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=bundle.min_by_level,
        var_to_clean="sea_surface_temperature",
    )
    return predictor


# ---------------------------------------------------------------------------- #
#  Param plumbing: separate frozen GenCast params from trainable controller
# ---------------------------------------------------------------------------- #


def _is_controller_key(scope: str) -> bool:
    """Heuristic: any Haiku module path containing `cada_controller` is ours."""
    return "cada_controller" in scope


def split_params(params: hk.Params) -> tuple[hk.Params, hk.Params]:
    """Split a flat Haiku params dict into (gencast_params, controller_params)."""
    g, c = {}, {}
    for k, v in params.items():
        if _is_controller_key(k):
            c[k] = v
        else:
            g[k] = v
    return g, c


def merge_params(gencast_params: hk.Params, controller_params: hk.Params) -> hk.Params:
    """Merge the two param dicts back together for `apply`."""
    out = dict(gencast_params)
    out.update(controller_params)
    return out


# ---------------------------------------------------------------------------- #
#  Init + apply
# ---------------------------------------------------------------------------- #


def make_scan_unconditional_run_forward(
    bundle: GenCastBundle,
    apply_remat: bool = True,
):
    """Unconditional GenCast sampler that uses `_CADAScanGenCast`.

    Same DPM-Solver++ 2S substep math as the official sampler, but driven by
    the differentiable Python-loop path instead of `hk.fori_loop`. The two
    control-flow forms reserve and split RNG differently (the official sampler
    calls `spherical_white_noise_like` *inside* `body_fn` at `i==0`; our path
    samples the initial noise *outside* the loop), so even with a noiseless
    sampler the two produce different but equally valid samples that disagree
    by O(0.05 K). Use this loop-based reference for the smoke parity check
    so a zero-init CADA controller is bit-equivalent to unconditional.
    """

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        gencast_predictor = _CADAScanGenCast(
            sampler_config=bundle.sampler_config,
            task_config=bundle.task_config,
            denoiser_architecture_config=bundle.denoiser_architecture_config,
            noise_config=bundle.noise_config,
            noise_encoder_config=bundle.noise_encoder_config,
            apply_remat=apply_remat,
        )
        predictor = normalization.InputsAndResiduals(
            gencast_predictor,
            diffs_stddev_by_level=bundle.diffs_stddev_by_level,
            mean_by_level=bundle.mean_by_level,
            stddev_by_level=bundle.stddev_by_level,
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=bundle.min_by_level,
            var_to_clean="sea_surface_temperature",
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    params, state = bundle.params, bundle.state

    def _apply(rng, inputs, targets_template, forcings):
        out, _ = run_forward.apply(
            params, state, rng, inputs, targets_template, forcings
        )
        return out

    return jax.jit(_apply)

def _make_run_forward_transform(
    bundle: GenCastBundle,
    cfg: CADAControllerConfig,
    var: str,
):
    """Build the `hk.transform_with_state` for the CADA-wrapped sampler.

    The signature is shared by `init` and `apply`:
        cada_run(inputs, targets_template, forcings,
                 obs_full_jax, obs_mask_jax, obs_phys_full_jax)

    `inputs` is in physical Kelvin (it gets normalized inside
    `InputsAndResiduals`); we extract `last_input_T_phys` from it here,
    before the normalization wrapper sees it. `diffs_stddev_T` is a
    bundle constant pulled out once so it's hoisted as a closure-time
    constant rather than a per-step JAX input.
    """
    diffs_stddev_T = jnp.asarray(
        bundle.diffs_stddev_by_level[var].values.astype(np.float32)
    )

    @hk.transform_with_state
    def cada_run(
        inputs,
        targets_template,
        forcings,
        obs_full_jax,
        obs_mask_jax,
        obs_phys_full_jax,
    ):
        last_input_T_phys = xarray_jax.unwrap_data(
            inputs[var].isel(time=-1).transpose("batch", "level", "lat", "lon")
        ).astype(jnp.float32)
        predictor = _construct_cada_wrapped(
            bundle,
            obs_full_jax,
            obs_mask_jax,
            obs_phys_full_jax,
            last_input_T_phys,
            diffs_stddev_T,
            cfg,
            var,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return cada_run


def init_cada(
    bundle: GenCastBundle,
    rng: jnp.ndarray,
    inputs,
    targets_template,
    forcings,
    obs_full_jax: jnp.ndarray,
    obs_mask_jax: jnp.ndarray,
    obs_phys_full_jax: jnp.ndarray,
    cfg: CADAControllerConfig | None = None,
    var: str = "temperature",
) -> tuple[hk.Params, hk.State, CADAControllerConfig]:
    """Initialise the CADA controller while reusing the bundle's GenCast params.

    `obs_phys_full_jax` is shape-only here (only the structure is needed
    to trace the substep obs-loss path during init); use any
    correctly-shaped array — `cada_features.build_step` returns one.

    Returns:
        params:     full param dict (frozen GenCast + freshly-initialised
                    controller). Use `split_params` to separate.
        state:      empty Haiku state (last_u, score_sum, etc. are created
                    on first apply, with the init callbacks above).
        cfg:        the resolved CADAControllerConfig (echo for caller).
    """
    if cfg is None:
        cfg = CADAControllerConfig()
    cada_run = _make_run_forward_transform(bundle, cfg, var)

    # We only need to initialise the CONTROLLER params; for the GenCast
    # params we discard the random init from `init` and merge in the
    # bundle's pretrained values.
    full_params, full_state = cada_run.init(
        rng,
        inputs,
        targets_template,
        forcings,
        obs_full_jax,
        obs_mask_jax,
        obs_phys_full_jax,
    )
    _, controller_params = split_params(full_params)
    final_params = merge_params(bundle.params, controller_params)
    return final_params, full_state, cfg


def make_cada_run_forward(
    bundle: GenCastBundle,
    cfg: CADAControllerConfig | None = None,
    var: str = "temperature",
):
    """JIT'd CADA-guided sampler.

    Returns a function with signature:
        cada_apply(params, state, rng,
                   inputs, targets_template, forcings,
                   obs_full_jax, obs_mask_jax, obs_phys_full_jax)
        -> (predictions, final_state)

    `obs_phys_full_jax` is the physical-Kelvin observation buffer
    (zero-filled at unobserved cells) used by the substep obs loss.
    `final_state` carries the per-sample diagnostics (`score_sum`,
    `control_sum`, `u_l2_sum`, `substep_obs_sum`, `substep_count`,
    `last_u`) that the train loop divides by `substep_count` to compute
    mean diagnostics.
    """
    if cfg is None:
        cfg = CADAControllerConfig()
    cada_run = _make_run_forward_transform(bundle, cfg, var)

    def _apply(
        params,
        state,
        rng,
        inputs,
        targets_template,
        forcings,
        obs_full_jax,
        obs_mask_jax,
        obs_phys_full_jax,
    ):
        out, final_state = cada_run.apply(
            params,
            state,
            rng,
            inputs,
            targets_template,
            forcings,
            obs_full_jax,
            obs_mask_jax,
            obs_phys_full_jax,
        )
        return out, final_state

    return jax.jit(_apply)
