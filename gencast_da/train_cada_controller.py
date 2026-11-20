"""Train the CADA controller on top of frozen GenCast Mini.

`--rollout-K K`: each training step rolls out K consecutive 12 h
physical steps autoregressively. Gradient is detached at every physical-
step boundary (via `value_and_grad` per step + `_update_memory` on the
concrete prediction), so per-step memory matches the K=1 case and per-
train-step wall time grows linearly in K. Per-step losses are averaged
(or summed; see `--rollout-loss-mode`) and the controller receives one
optimizer update with the accumulated gradient.

K=1 (default) reproduces the original single-step path bit-for-bit.
K>1 closes the train/eval distribution shift: at eval time the
controller sees its own (drifted) predictions in the input window from
physical step 2 onward, and rollout training exposes it to the same
distribution. Mirrors the standard scheduled-sampling/detached-BPTT
pattern used for autoregressive models.

GenCast params (~57 M) are frozen; only controller params (~30 k)
receive gradients.

Loss assembly mirrors PDE CADA's `train_controlnet`:

    terminal = w_full * cos_lat_full_mse        (physical K, final sample,
                                                 cos(lat)-weighted, dense)
               + w_obs * substep_obs_mse        (physical K, masked, accumulated
                                                 per DPM-Solver++ 2S substep on
                                                 x0_T_ctrl after de-normalising
                                                 from residual to Kelvin)
    total    = w_t * terminal
               + w_s * score_substep_mean       (diagnostic by default)
               + w_c * control_substep_mean     (diagnostic by default)
               + w_reg * u_l2_substep_mean

The masked substep observation term is the analogue of `loss_mid` in
`denoising_diffusion_pytorch_kolm_adjoint_modes_aware_random.py`. Kolm
CADA accumulates a masked obs loss on `x_start_c` (the controlled
denoiser's clean estimate) at every denoising substep, in the model's
own data space. For Kolm that data space *is* physical, so there's no
units gap between training loss and reported RMSE. We mirror that by
de-normalising `x0_T_ctrl` (a normalized residual) back to physical K
before computing the masked obs MSE — the result has the property that
`sqrt(substep_obs)` is directly comparable to the observation-innovation
RMSE reported at evaluation time.

`w_s = w_c = 0` by default — same as the production trainer at line 2373
of `denoising_diffusion_pytorch_kolm_adjoint_modes_aware_random.py`.

Run:
    # K_phys=1 (current default; same as before):
    CUDA_VISIBLE_DEVICES=0 python -m gencast_da.train_cada_controller \\
        --num-steps 40 --gamma 1.0

    # K_phys=4 rollout training (matches the K=4 eval scenario):
    CUDA_VISIBLE_DEVICES=0 python -m gencast_da.train_cada_controller \\
        --num-steps 40 --rollout-K 4 --gamma 1.0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import time
from pathlib import Path

import jax
import numpy as np
import optax
from graphcast import xarray_jax
from tqdm.auto import tqdm

from gencast_da.cada_features import (
    CADAStepArrays,
    build_step,
    build_step_with_memory,
    cos_lat_full_mse,
    masked_obs_mse,
)
from gencast_da.cada_guided import (
    CADA_VARIANT,
    CADAControllerConfig,
    _make_run_forward_transform,
    init_cada,
    merge_params,
    split_params,
)
from gencast_da.cada_train_config import CADALossConfig, CADAOptConfig
from gencast_da.checkpointing import (
    load_trusted_checkpoint,
    save_trusted_checkpoint,
    truncate_history,
    validate_compatibility,
)
from gencast_da.diagnostics import diagnostic_means
from gencast_da.gencast_runner import (
    bundle_compatibility,
    load_bundle,
    load_example_batches,
)
from gencast_da.obs_operator import ObsConfig
from gencast_da.random_utils import validation_key
from gencast_da.rollout_utils import (
    evolving_input_variables as _evolving_input_vars,
)
from gencast_da.rollout_utils import (
    initial_memory as _initial_memory,
)
from gencast_da.rollout_utils import (
    update_memory as _update_memory,
)
from gencast_da.runtime import (
    atomic_write_text,
    write_sha256_sidecar,
)

# Default time-index split for the 30-step example trajectory (legacy
# single-dataset constants kept for back-compat with --fixed-train-index
# documentation; the multi-dataset code paths below use TRAIN_FRAC).
DEFAULT_TRAIN_INDICES = list(range(2, 19))  # 17 transitions
DEFAULT_VAL_INDICES = list(range(19, 22))  # 3 transitions

# Multi-dataset training: list of demo trajectories. The training loop
# samples (file_id, target_idx) pairs uniformly from across all listed
# files, and each file is split independently along time by TRAIN_FRAC.
#
# IMPORTANT: every file in this list MUST be packaged for the same
# GenCast variant as the loaded checkpoint (here: GenCast 1p0deg Mini
# <2019), which means it must contain the same data_vars set
# (sea_surface_temperature, total_precipitation_12hr, day/year_progress_*,
# etc.). The other 1deg ERA5 demo files in `gs://dm_graphcast/` are for
# the bigger GenCast variants and use a *different* schema (TISR +
# 6hr precip, no SST); they will fail _build_step_args with KeyError.
# Only the 2019 file below is Mini-compatible in the public bucket. The
# 2022 Feb-Mar file (118 frames) is built locally via
# `python -m gencast_da.build_mini_era5_nc --validate`; we auto-include
# it when present so a fresh checkout still trains on just the 2019 file.
DEFAULT_DATASETS = [
    "source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc",
]
# _MINI_2022_FEBMAR = "source-era5_date-2022-01-31_res-1.0_levels-13_steps-118.nc"
# if (DEFAULT_DATA_ROOT / "dataset" / _MINI_2022_FEBMAR).exists():
#     DEFAULT_DATASETS.append(_MINI_2022_FEBMAR)
# 0.85 (instead of 0.9) so val = [28..31] in the 32-frame demo file and
# K=4 has at least one feasible val start (idx=28 -> targets 28..31).
# At 0.9 val = [29..31], all of which would run past the file's end.
TRAIN_FRAC = 0.85


def _split_indices_for_dataset(n_steps: int, train_frac: float = TRAIN_FRAC):
    """Split the valid target-time indices of one trajectory `train_frac` / rest.

    A "valid target index" is `t in [2, n_steps-1]` (we need 2 prior
    frames for the GenCast input window). The first
    `round(train_frac * len(valid))` go to train, the rest to val. Both
    lists are returned in increasing time order.
    """
    valid = list(range(2, n_steps))
    cutoff = int(round(len(valid) * train_frac))
    cutoff = max(1, min(cutoff, len(valid) - 1))
    return valid[:cutoff], valid[cutoff:]


def _build_train_val_pairs(examples, rollout_K: int):
    """Return (train_pairs, val_pairs) of `(file_id, target_idx)` tuples.

    For K_phys > 1 we drop any starting index whose K-step rollout would
    run past the trajectory's last time step. Val pairs are also filtered
    so we only validate on rollouts that fit inside their own file.
    """
    train_pairs = []
    val_pairs = []
    for fid, (_, ds) in enumerate(examples):
        n_steps = ds.sizes["time"]
        tr, va = _split_indices_for_dataset(n_steps)
        max_target = n_steps - 1
        for idx in tr:
            if idx + rollout_K - 1 <= max_target:
                train_pairs.append((fid, idx))
        for idx in va:
            if idx + rollout_K - 1 <= max_target:
                val_pairs.append((fid, idx))
    return train_pairs, val_pairs


def _pred_temperature_jax(pred):
    """Pull the predicted temperature out of a wrapped CADA prediction.

    `pred["temperature"]` is an `xarray_jax.DataArray`; we transpose to
    canonical `(batch, time, level, lat, lon)` and unwrap to a JAX array.
    """
    return xarray_jax.unwrap_data(
        pred["temperature"].transpose("batch", "time", "level", "lat", "lon")
    )


def _build_loss_fn(cada_xform, loss_cfg: CADALossConfig):
    """Closure over (loss_cfg, cada_xform) -> loss_fn(c_params, g_params, ...).

    `c_params` is the *only* differentiation target (controller weights).
    `g_params` is the frozen GenCast trunk and is fed through
    `jax.lax.stop_gradient` upstream so XLA can drop GenCast's cotangents
    from the backward graph entirely (huge memory win — see comment at
    `_train_step_unjit`).
    """

    def loss_fn(c_params, g_params, state, rng, step_arr: CADAStepArrays):
        merged = merge_params(g_params, c_params)
        pred, st = cada_xform.apply(
            merged,
            state,
            rng,
            step_arr.inputs,
            step_arr.targets_template,
            step_arr.forcings,
            step_arr.obs_full_jax,
            step_arr.obs_mask_jax,
            step_arr.obs_phys_full_jax,
        )
        pred_T = _pred_temperature_jax(pred)
        # Final-sample diagnostics in physical Kelvin space. Kept for
        # logging/eval interpretability — they make per-step loss values
        # comparable to the K-step rollout RMSE numbers.
        full_mse = cos_lat_full_mse(pred_T, step_arr.truth_T_jax, step_arr.lat_w_jax)
        obs_mse = masked_obs_mse(
            pred_T, step_arr.obs_phys_full_jax, step_arr.obs_mask_jax
        )

        diags = diagnostic_means(st)
        # Kolm-style training observation term: masked physical-K obs MSE
        # accumulated on x0_T_ctrl_phys at every DPM-Solver++ 2S substep
        # (see cada_guided._wrap_with_cada_guidance). Same units and
        # weighting as the final-sample `obs_mse` above, just averaged
        # over substeps; in particular `sqrt(substep_obs)` ≈ rollout
        # observation-innovation RMSE at convergence.
        substep_obs = diags["substep_obs"]
        terminal = loss_cfg.w_full * full_mse + loss_cfg.w_obs * substep_obs

        total = (
            loss_cfg.w_t * terminal
            + loss_cfg.w_s * diags["score"]
            + loss_cfg.w_c * diags["control"]
            + loss_cfg.w_reg * diags["u_l2"]
        )
        aux = {
            "total": total,
            "terminal": terminal,
            "full_mse": full_mse,
            "obs_mse": obs_mse,
            "substep_obs": substep_obs,
            "score_loss": diags["score"],
            "control_loss": diags["control"],
            "u_l2": diags["u_l2"],
            "substep_count": diags["substep_count"],
            # Returned for K_phys > 1 rollout: the next physical step
            # uses this prediction (detached from the graph by being a
            # value_and_grad aux output) to roll its 2-frame memory
            # forward. xarray.Dataset is pytree-registered by xarray_jax,
            # so JAX preserves its structure and returns concrete arrays.
            "pred": pred,
        }
        return total, aux

    return loss_fn


def _train_step_unjit(
    loss_fn,
    optimizer,
    params,
    opt_state,
    state,
    rng,
    step_arr: CADAStepArrays,
):
    """One unjitted train step.

    We don't JIT this whole function because it consumes an
    `xarray.Dataset` whose static coords change between target indices,
    which would invalidate the JIT cache every step. The inner forward
    `cada_xform.apply` and gradient computation can still be efficient
    via XLA caching keyed on the static structure (graphcast does this
    internally).

    GenCast (~57M params) is frozen; only the controller (~30k params)
    is trained. We split the params and feed GenCast's frozen half
    through `jax.lax.stop_gradient` *outside* `loss_fn`, then take the
    grad only w.r.t. the controller half (`argnums=0`). XLA then prunes
    GenCast's cotangent path from the backward graph entirely, which is
    critical: building cotangents for the frozen 57M-param trunk just to
    discard them OOMs even at small `num_noise_levels`.
    """
    g_params, c_params = split_params(params)
    g_params_frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, g_params)
    (loss, aux), c_grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        c_params, g_params_frozen, state, rng, step_arr
    )
    updates, new_opt_state = optimizer.update(c_grads, opt_state, c_params)
    new_c_params = optax.apply_updates(c_params, updates)
    new_params = merge_params(g_params, new_c_params)
    return new_params, new_opt_state, loss, aux


def _rollout_train_step_unjit(
    loss_fn,
    optimizer,
    params,
    opt_state,
    state,
    bundle,
    example,
    obs_cfg: ObsConfig,
    starting_idx: int,
    rollout_K: int,
    rng_root,
    obs_seeds,
    rollout_loss_mode: str,
):
    """K_phys-step rollout train step with detached gradient between physical steps.

    Per physical step k = 0..K-1:
      1. Build `step_arr_k` from the *current* 2-frame memory (initial
         memory = ground-truth frames `[starting_idx-2, starting_idx-1]`
         exactly as eval/run_cada_gencast does).
      2. Compute `(loss_k, aux_k), grads_k` via `value_and_grad` on this
         step's diffusion chain only. Memory came in detached (concrete
         arrays, not tracers), so the gradient stops at the boundary
         between physical steps — equivalent to a `stop_gradient` on the
         predicted state before it becomes the next step's input.
      3. Accumulate `grads_k` into the running grad sum.
      4. Roll memory forward: `_update_memory(memory, pred_k, evolving)`,
         where `pred_k = aux_k["pred"]` is the post-`value_and_grad`
         concrete xarray Dataset.

    The rationale for *not* backpropagating across physical steps:
      - Per-step memory is the same as K_phys=1 (each step's compute
        graph is freed after backprop before the next one starts).
      - We have a full per-step loss anyway (substep_obs + full_mse on
        physical-K target), so the controller still gets a direct
        gradient signal at every physical step. What we lose is the
        ability of step k+1's loss to retro-instruct step k's controller
        action — but that signal would otherwise need to live in
        activations across K diffusion chains, which doesn't fit at
        K_phys=4 even with `apply_remat`.
      - Mirrors scheduled-sampling / detached-BPTT for autoregressive
        models: the controller is exposed to its own (drifting)
        predictions at training time, closing the train/eval distribution
        shift, while the optimization stays per-step local.

    Args:
        rollout_loss_mode: "mean" averages losses + grads across K (so
            effective LR is comparable to K_phys=1); "sum" leaves them
            summed (useful if you want longer-horizon errors to weigh
            more, but you'll likely need a smaller LR).
        obs_seeds: length-K_phys list of np-RNG seeds. In fixed-index
            mode these are deterministic across training steps so the
            overfit curve stays interpretable.
    """
    g_params, c_params = split_params(params)
    g_params_frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, g_params)

    evolving = _evolving_input_vars(bundle.task_config)
    if starting_idx == 2:
        memory = _initial_memory(example, bundle.task_config).copy(deep=True)
    else:
        memory = example.isel(time=[starting_idx - 2, starting_idx - 1])[evolving].copy(
            deep=True
        )

    accum_grads = None
    aux_per_step = []
    total_loss = 0.0
    for k in range(rollout_K):
        target_k = starting_idx + k
        obs_rng_k = np.random.default_rng(int(obs_seeds[k]))
        diff_rng_k = jax.random.fold_in(rng_root, k)
        step_arr_k = build_step_with_memory(
            bundle,
            example,
            memory,
            target_k,
            obs_cfg,
            obs_rng_k,
        )
        (loss_k, aux_k), grads_k = jax.value_and_grad(
            loss_fn,
            argnums=0,
            has_aux=True,
        )(c_params, g_params_frozen, state, diff_rng_k, step_arr_k)
        total_loss = total_loss + float(loss_k)
        aux_per_step.append(aux_k)
        if accum_grads is None:
            accum_grads = grads_k
        else:
            accum_grads = jax.tree_util.tree_map(
                lambda a, b: a + b,
                accum_grads,
                grads_k,
            )
        # Roll memory using the (concrete, post-value_and_grad) pred.
        # The next iteration's `build_step_with_memory` will turn this
        # back into a tracer in a fresh trace, so there is no gradient
        # path back to step k's diffusion chain.
        memory = _update_memory(memory, aux_k["pred"], evolving)

    if rollout_loss_mode == "mean":
        accum_grads = jax.tree_util.tree_map(
            lambda x: x / float(rollout_K),
            accum_grads,
        )
        mean_loss = total_loss / float(rollout_K)
    else:
        mean_loss = total_loss

    updates, new_opt_state = optimizer.update(accum_grads, opt_state, c_params)
    new_c_params = optax.apply_updates(c_params, updates)
    new_params = merge_params(g_params, new_c_params)
    return new_params, new_opt_state, mean_loss, aux_per_step


def _save_checkpoint(
    path: Path,
    params,
    opt_state,
    step: int,
    cfg,
    num_noise_levels: int,
    *,
    training_config,
    compatibility,
    ema_total,
    ema_terminal,
    numpy_train_state,
    numpy_observation_state,
) -> None:
    g_params, c_params = split_params(params)
    payload = {
        "schema_version": 1,
        "controller_params": c_params,
        "opt_state": opt_state,
        "step": step,
        "cfg": dataclasses.asdict(cfg),
        "training_config": training_config,
        "compatibility": compatibility,
        "ema_total": ema_total,
        "ema_terminal": ema_terminal,
        # Persist the diffusion schedule N the controller was trained on
        # so inference can refuse to load a controller against a mismatched
        # sigma grid (controller is conditioned on sigma_t).
        "num_noise_levels": int(num_noise_levels),
        "numpy_train_state": numpy_train_state,
        "numpy_observation_state": numpy_observation_state,
    }
    save_trusted_checkpoint(path, payload)
    print(f"[train_cada] saved checkpoint -> {path}  (step={step})")


def _effective_training_config(args) -> dict:
    names = (
        "num_noise_levels",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "seed",
        "gamma",
        "hidden_channels",
        "num_blocks",
        "sigma_emb_dim",
        "apply_remat",
        "obs_lat_stride_deg",
        "obs_lon_stride_deg",
        "obs_mask_mode",
        "random_mask_fraction",
        "random_mask_seed",
        "w_t",
        "w_full",
        "w_obs",
        "w_s",
        "w_c",
        "w_reg",
        "ema_decay",
        "fixed_train_index",
        "rollout_K",
        "rollout_loss_mode",
    )
    return {name: getattr(args, name) for name in names}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--num-noise-levels",
        type=int,
        default=4,
        help="DPM-Solver++ 2S chain length used during training. The "
        "controller is conditioned on sigma_t and the rho-spaced "
        "sigma grid changes with N, so run_cada_gencast MUST use "
        "this same value. The trained "
        "value is persisted into the checkpoint and validated on "
        "load. Default 4 is the current 24 GB-safe setting for the "
        "full forward+backward.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument(
        "--sigma-emb-dim",
        type=int,
        default=64,
        help="Width of the sigma_t embedding fed into the controller. "
        "Bigger = more capacity to condition behavior on noise level.",
    )
    parser.add_argument(
        "--no-apply-remat",
        dest="apply_remat",
        action="store_false",
        help="Disable hk.remat on the hk.scan body. Default ON: each "
        "DPM-Solver++ 2S substep's activations are recomputed in the "
        "backward pass, bounding memory to one substep regardless of "
        "num_noise_levels. Turn this off only for forward-only "
        "diagnostics where the recompute cost matters.",
    )
    parser.set_defaults(apply_remat=True)
    parser.add_argument(
        "--obs-lat-stride-deg",
        type=int,
        default=4,
        help="Latitude stride (in degrees) of the observation mask used "
        "during training. Default 4 matches the original Savary "
        "paper-idealized scenario (every 4 deg). Increase for "
        "sparser observations (e.g. 8 -> 4x fewer obs cells).",
    )
    parser.add_argument(
        "--obs-lon-stride-deg",
        type=int,
        default=4,
        help="Longitude stride (in degrees) of the observation mask used "
        "during training. Default 4 matches the original Savary "
        "paper-idealized scenario.",
    )
    parser.add_argument(
        "--obs-mask-mode",
        type=str,
        default="stride",
        choices=("stride", "random"),
        help="Observation mask geometry. 'stride' preserves the original "
        "regular lat/lon grid. 'random' samples an irregular spatial "
        "mask with the same number of observed cells as the configured "
        "stride mask unless --random-mask-fraction is set.",
    )
    parser.add_argument(
        "--random-mask-fraction",
        type=float,
        default=None,
        help="Optional spatial-cell fraction for --obs-mask-mode=random. "
        "If unset, the random mask matches the observation count of "
        "--obs-lat-stride-deg x --obs-lon-stride-deg.",
    )
    parser.add_argument(
        "--random-mask-seed",
        type=int,
        default=0,
        help="Base seed for random observation masks. During random-index "
        "training a fresh mask seed is sampled per rollout; in fixed-"
        "index mode this seed is reused for deterministic overfit runs.",
    )
    parser.add_argument("--w-t", type=float, default=1.0)
    parser.add_argument("--w-full", type=float, default=1.0)
    parser.add_argument("--w-obs", type=float, default=1.0)
    parser.add_argument("--w-s", type=float, default=0.0)
    parser.add_argument("--w-c", type=float, default=0.0)
    parser.add_argument("--w-reg", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="EMA decay for loss values printed in the progress bar. Raw "
        "per-step losses sample different target_idx values, so they "
        "are noisy and not expected to be monotone.",
    )
    parser.add_argument(
        "--fixed-train-index",
        type=int,
        default=None,
        help="If set, train every step on this single target_idx. This is "
        "an overfit/debug mode for quickly verifying that terminal "
        "loss can decrease on a fixed transition; leave unset for the "
        "normal random-index training loop. With --rollout-K > 1 the "
        "fixed index is the *starting* idx and the K consecutive "
        "transitions are run autoregressively.",
    )
    parser.add_argument(
        "--rollout-K",
        type=int,
        default=1,
        help="K_phys: number of physical 12 h steps to roll out per "
        "training step. K=1 (default) reproduces the original "
        "single-step training. K>1 enables autoregressive rollout: "
        "each physical step builds inputs from the controller's own "
        "(detached) prediction at the previous physical step, so the "
        "controller is exposed to the same input distribution it "
        "sees at eval time. Gradients are detached at every physical-"
        "step boundary (we already have a per-step loss), so per-"
        "step memory is the same as K=1 — only wall-clock per train "
        "step grows linearly in K.",
    )
    parser.add_argument(
        "--rollout-loss-mode",
        type=str,
        default="mean",
        choices=("mean", "sum"),
        help="How to combine per-physical-step losses (and grads): 'mean' "
        "(default) keeps total-loss / grad-norm scale comparable to "
        "K=1; 'sum' weighs longer horizons more (you'll typically "
        "want a smaller --learning-rate to match).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=str(
            Path(
                os.environ.get(
                    "CADA_RUN_DIR",
                    os.path.join(
                        os.environ.get("CADA_ROOT", ""), "runs/gencast/manual"
                    ),
                )
            )
            / "checkpoints"
        ),
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="cada_controller_latest.pkl",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Training history JSON path; defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previously-saved controller checkpoint. If set, "
        "restore controller_params + opt_state + step counter and "
        "continue training. --num-steps is interpreted as the TOTAL "
        "target step count (e.g. resuming from step 100 with "
        "--num-steps 500 runs steps 101..500). The saved ctrl_cfg + "
        "num_noise_levels must match the current args; pass "
        "--no-resume-strict to relax this (controller arch must "
        "still match for the opt_state shapes to align).",
    )
    parser.add_argument(
        "--no-resume-strict",
        dest="resume_strict",
        action="store_false",
        help="When resuming, only require controller arch + noise levels "
        "to match (allow gamma / loss-weight / lr changes).",
    )
    parser.set_defaults(resume_strict=True)
    args = parser.parse_args()
    for name in (
        "num_steps",
        "num_noise_levels",
        "log_every",
        "val_every",
        "save_every",
        "rollout_K",
        "hidden_channels",
        "num_blocks",
        "sigma_emb_dim",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "grad_clip"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "weight_decay",
        "gamma",
        "w_t",
        "w_full",
        "w_obs",
        "w_s",
        "w_c",
        "w_reg",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must lie in [0, 1)")
    requested_signal: dict[str, int | None] = {"value": None}

    def request_checkpoint(signum, _frame) -> None:
        requested_signal["value"] = int(signum)
        print(
            f"[train_cada] signal {signum} received; "
            "checkpointing after the active operation"
        )

    signal.signal(signal.SIGUSR1, request_checkpoint)
    signal.signal(signal.SIGTERM, request_checkpoint)
    training_config = _effective_training_config(args)

    loss_cfg = CADALossConfig(
        w_t=args.w_t,
        w_full=args.w_full,
        w_obs=args.w_obs,
        w_s=args.w_s,
        w_c=args.w_c,
        w_reg=args.w_reg,
    )
    opt_cfg = CADAOptConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        num_train_steps=args.num_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        seed=args.seed,
    )
    ctrl_cfg = CADAControllerConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        sigma_emb_dim=args.sigma_emb_dim,
        gamma=args.gamma,
        apply_remat=args.apply_remat,
    )
    print(f"[train_cada] loss_cfg = {loss_cfg}")
    print(f"[train_cada] opt_cfg  = {opt_cfg}")
    print(f"[train_cada] ctrl_cfg = {ctrl_cfg}")

    bundle = load_bundle()

    # The chosen schedule is baked into the checkpoint (see
    # _save_checkpoint) so eval/comparison can refuse to load against a
    # different N. See --num-noise-levels help.
    bundle = dataclasses.replace(
        bundle,
        sampler_config=dataclasses.replace(
            bundle.sampler_config,
            num_noise_levels=args.num_noise_levels,
        ),
    )
    print(
        f"[train_cada] sampler num_noise_levels = "
        f"{bundle.sampler_config.num_noise_levels}"
    )

    examples = load_example_batches(DEFAULT_DATASETS, load_into_memory=False)
    print("[train_cada] dataset loading mode: lazy (no upfront ds.load())")
    print("[train_cada] datasets:")
    for fid, (name, ds) in enumerate(examples):
        print(f"   [{fid}] {name}  (time={ds.sizes['time']})")
    # Legacy alias: shape inference + single-dataset code paths use the
    # first file. All multi-file logic uses `examples[fid]`.
    example = examples[0][1]
    obs_cfg = ObsConfig(
        seed=opt_cfg.seed,
        mask_mode=args.obs_mask_mode,
        lat_stride_deg=args.obs_lat_stride_deg,
        lon_stride_deg=args.obs_lon_stride_deg,
        random_mask_seed=args.random_mask_seed,
        random_mask_fraction=args.random_mask_fraction,
    )
    print(
        f"[train_cada] obs_cfg: var={obs_cfg.var}, "
        f"mask_mode={obs_cfg.mask_mode}, "
        f"lat_stride={obs_cfg.lat_stride_deg} deg, "
        f"lon_stride={obs_cfg.lon_stride_deg} deg, "
        f"noise_std={obs_cfg.noise_std} K, "
        f"random_mask_fraction={obs_cfg.random_mask_fraction}, "
        f"random_mask_seed={obs_cfg.random_mask_seed}"
    )
    compatibility = bundle_compatibility(
        bundle,
        controlled_variable=obs_cfg.var,
        cada_variant=CADA_VARIANT,
    )

    print("[train_cada] init CADA params (compiles GenCast init pass)...")
    rng = jax.random.PRNGKey(opt_cfg.seed)

    # Build a dummy CADAStepArrays just to drive init.
    init_arr = build_step(
        bundle,
        example,
        target_time_idx=DEFAULT_TRAIN_INDICES[0],
        obs_cfg=obs_cfg,
        rng_np=np.random.default_rng(opt_cfg.seed),
    )
    t0 = time.time()
    params, state, ctrl_cfg = init_cada(
        bundle,
        jax.random.fold_in(rng, 1),
        init_arr.inputs,
        init_arr.targets_template,
        init_arr.forcings,
        init_arr.obs_full_jax,
        init_arr.obs_mask_jax,
        init_arr.obs_phys_full_jax,
        ctrl_cfg,
    )
    print(f"[train_cada] init done in {time.time() - t0:.1f}s")

    g_params, c_params = split_params(params)
    n_g = sum(p.size for d in g_params.values() for p in d.values())
    n_c = sum(p.size for d in c_params.values() for p in d.values())
    print(f"[train_cada] frozen GenCast params: {n_g:,} | controller params: {n_c:,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(opt_cfg.grad_clip_norm),
        optax.adamw(opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay),
    )
    opt_state = optimizer.init(c_params)

    cada_xform = _make_run_forward_transform(bundle, ctrl_cfg, obs_cfg.var)
    loss_fn = _build_loss_fn(cada_xform, loss_cfg)

    rng_np_train = np.random.default_rng(opt_cfg.seed * 2 + 7)
    rng_np_obs = np.random.default_rng(opt_cfg.seed * 13 + 11)

    ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
    history_path = (
        Path(args.history_path)
        if args.history_path is not None
        else ckpt_path.with_suffix(".history.json")
    )
    history = []
    start_step = 1
    restored_ema_total = None
    restored_ema_terminal = None

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        payload, loaded_path = load_trusted_checkpoint(resume_path)
        if loaded_path != resume_path:
            print(f"[train_cada] primary checkpoint invalid; using {loaded_path}")
        validate_compatibility(payload.get("compatibility"), compatibility)
        saved_n = int(payload["num_noise_levels"])
        if saved_n != args.num_noise_levels:
            raise ValueError(
                f"--resume: checkpoint num_noise_levels={saved_n} but "
                f"--num-noise-levels={args.num_noise_levels}. The "
                f"controller is conditioned on sigma_t; you must train "
                f"on the same DPM-Solver++ 2S grid."
            )
        saved_cfg = payload["cfg"]
        # Architecture knobs MUST match (otherwise opt_state shapes break).
        arch_keys = ("hidden_channels", "num_blocks", "sigma_emb_dim")
        for k in arch_keys:
            if saved_cfg.get(k) != getattr(ctrl_cfg, k):
                raise ValueError(
                    f"--resume: checkpoint {k}={saved_cfg.get(k)} but "
                    f"current arg {k}={getattr(ctrl_cfg, k)}. Controller "
                    f"architecture must match for opt_state to align."
                )
        saved_training_config = payload.get("training_config")
        if args.resume_strict and saved_training_config is not None:
            if saved_training_config != training_config:
                changed = sorted(
                    key
                    for key in set(saved_training_config) | set(training_config)
                    if saved_training_config.get(key) != training_config.get(key)
                )
                raise ValueError(
                    "--resume (strict): training configuration changed for "
                    + ", ".join(changed)
                )
        elif args.resume_strict:
            for k in ("gamma",):
                if saved_cfg.get(k) != getattr(ctrl_cfg, k):
                    raise ValueError(
                        f"--resume (strict): checkpoint {k}={saved_cfg.get(k)} "
                        f"but current arg {k}={getattr(ctrl_cfg, k)}. Pass "
                        f"--no-resume-strict to allow this change."
                    )
        params = merge_params(g_params, payload["controller_params"])
        opt_state = payload["opt_state"]
        if "numpy_train_state" in payload:
            rng_np_train.bit_generator.state = payload["numpy_train_state"]
        if "numpy_observation_state" in payload:
            rng_np_obs.bit_generator.state = payload["numpy_observation_state"]
        restored_ema_total = payload.get("ema_total")
        restored_ema_terminal = payload.get("ema_terminal")
        start_step = int(payload["step"]) + 1
        # Try to load and prepend prior history so the JSON stays
        # cumulative across resumes.
        prior_hist_path = (
            history_path
            if history_path.exists()
            else resume_path.with_suffix(".history.json")
        )
        if prior_hist_path.exists():
            try:
                loaded_history = json.loads(prior_hist_path.read_text())
                history = truncate_history(
                    loaded_history,
                    int(payload["step"]),
                )
                print(
                    f"[train_cada] resume: loaded prior history "
                    f"({len(history)} entries) from {prior_hist_path}"
                )
            except Exception as e:
                print(
                    f"[train_cada] resume: WARN could not parse prior "
                    f"history {prior_hist_path}: {e}; starting empty."
                )
        print(
            f"[train_cada] resumed from {resume_path} at step "
            f"{payload['step']}; will continue at step {start_step} "
            f"with --num-steps {opt_cfg.num_train_steps} (TOTAL)"
        )
        if start_step > opt_cfg.num_train_steps:
            print(
                f"[train_cada] resume step {start_step - 1} >= "
                f"--num-steps {opt_cfg.num_train_steps}; nothing to do. "
                f"Increase --num-steps to extend training."
            )
            return

    print("[train_cada] starting training loop...")

    # Build (file_id, target_idx) pairs from the 90/10 split of each
    # dataset, dropping any starting idx whose K-step rollout would run
    # past its trajectory's last frame.
    train_pairs, val_pairs = _build_train_val_pairs(examples, args.rollout_K)
    if args.fixed_train_index is not None:
        # Fixed-index overfit mode picks the first occurrence in any
        # file. Keep this behaviour for backward compat with debug runs.
        match = [
            (fid, idx) for (fid, idx) in train_pairs if idx == args.fixed_train_index
        ]
        if not match:
            raise ValueError(
                f"--fixed-train-index={args.fixed_train_index} not present "
                f"in any dataset's training portion (after K-rollout filter)."
            )
        train_pairs = [match[0]]
        print(f"[train_cada] fixed-train-index → {train_pairs[0]} (file_id, idx)")
    if not train_pairs:
        raise ValueError(
            f"--rollout-K={args.rollout_K} leaves no valid training "
            f"starts across {len(examples)} files; reduce --rollout-K."
        )

    print(
        f"[train_cada] train pairs: {len(train_pairs)} "
        f"(per-file: "
        + ", ".join(
            f"f{fid}={sum(1 for f, _ in train_pairs if f == fid)}"
            for fid in range(len(examples))
        )
        + f")  val pairs: {len(val_pairs)} "
        f"(per-file: "
        + ", ".join(
            f"f{fid}={sum(1 for f, _ in val_pairs if f == fid)}"
            for fid in range(len(examples))
        )
        + f")  rollout_K = {args.rollout_K} ({args.rollout_loss_mode})"
    )
    if not val_pairs:
        print(
            f"[train_cada] WARN: no val starts fit within rollout_K="
            f"{args.rollout_K} in any file; validation loop will be skipped."
        )
    if args.fixed_train_index is not None:
        print(
            "[train_cada] fixed-index overfit/debug mode enabled: "
            "obs noise and diffusion RNG are also fixed across steps, so "
            "raw terminal loss is a function of controller params only "
            "and should drop monotonically (modulo Adam noise) on a "
            "single 12 h transition."
        )
    else:
        print(
            "[train_cada] random-index mode: raw per-step terminal loss is "
            "noisy across different target_idx values; watch terminal_ema "
            "and validation instead of expecting monotone step losses."
        )
    ema_total = restored_ema_total
    ema_terminal = restored_ema_terminal
    pbar = tqdm(
        range(start_step, opt_cfg.num_train_steps + 1),
        desc="train",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.5,
        initial=start_step - 1,
        total=opt_cfg.num_train_steps,
    )
    # In fixed-index mode reuse one obs RNG seed and one JAX sampling
    # key for every step so the only thing that changes between steps is
    # controller params. Without this:
    #   * `rng_np_obs` advances each step, so obs noise drifts and the
    #     masked obs target moves under the controller.
    #   * `jax.random.fold_in(rng, step)` reseeds the diffusion init
    #     noise/stochastic-churn each step, so two consecutive forwards
    #     with identical params would still disagree.
    # Either alone makes the overfit curve look "stuck" even when the
    # gradient is correct.
    fixed_obs_seed = (
        (opt_cfg.seed * 13 + 11)
        if args.fixed_train_index is None
        else (opt_cfg.seed * 13 + 11 + args.fixed_train_index)
    )
    fixed_rng_step = jax.random.fold_in(rng, 1)
    if requested_signal["value"] is not None:
        _save_checkpoint(
            ckpt_path,
            params,
            opt_state,
            start_step - 1,
            ctrl_cfg,
            num_noise_levels=bundle.sampler_config.num_noise_levels,
            training_config=training_config,
            compatibility=compatibility,
            ema_total=ema_total,
            ema_terminal=ema_terminal,
            numpy_train_state=rng_np_train.bit_generator.state,
            numpy_observation_state=rng_np_obs.bit_generator.state,
        )
        atomic_write_text(history_path, json.dumps(history, indent=2) + "\n")
        write_sha256_sidecar(history_path)
        raise SystemExit(75)

    for step in pbar:
        # Sample a (file_id, target_idx) pair. Uniform over pairs gives
        # each transition equal weight regardless of which file it lives
        # in; the longer 2022 file therefore contributes proportionally
        # more starts.
        pair_choice = int(rng_np_train.integers(0, len(train_pairs)))
        file_id, target_idx = train_pairs[pair_choice]
        cur_example = examples[file_id][1]

        t0 = time.time()
        if obs_cfg.mask_mode == "random":
            if args.fixed_train_index is not None:
                rollout_mask_seed = args.random_mask_seed
            else:
                rollout_mask_seed = int(rng_np_obs.integers(0, 2**31 - 1))
            step_obs_cfg = dataclasses.replace(
                obs_cfg,
                random_mask_seed=rollout_mask_seed,
            )
        else:
            step_obs_cfg = obs_cfg

        if args.rollout_K == 1:
            # Single-step path: bit-identical to the pre-rollout code.
            if args.fixed_train_index is not None:
                step_obs_rng = np.random.default_rng(fixed_obs_seed)
                rng_step = fixed_rng_step
            else:
                step_obs_rng = rng_np_obs
                rng_step = jax.random.fold_in(rng, step)
            step_arr = build_step(
                bundle, cur_example, target_idx, step_obs_cfg, step_obs_rng
            )
            params, opt_state, loss, aux = _train_step_unjit(
                loss_fn,
                optimizer,
                params,
                opt_state,
                state,
                rng_step,
                step_arr,
            )
            aux_per_step = [aux]
        else:
            # K_phys > 1 rollout: per-physical-step value_and_grad with
            # detached memory between steps.
            if args.fixed_train_index is not None:
                # Deterministic: same obs noise + same diffusion init per
                # physical step, every training step. Only thing changing
                # across training steps is `params`.
                obs_seeds = [fixed_obs_seed + k * 11_117 for k in range(args.rollout_K)]
                rollout_rng_root = fixed_rng_step
            else:
                obs_seeds = [
                    int(rng_np_obs.integers(0, 2**31 - 1))
                    for _ in range(args.rollout_K)
                ]
                rollout_rng_root = jax.random.fold_in(rng, step)
            params, opt_state, loss, aux_per_step = _rollout_train_step_unjit(
                loss_fn,
                optimizer,
                params,
                opt_state,
                state,
                bundle,
                cur_example,
                step_obs_cfg,
                starting_idx=target_idx,
                rollout_K=args.rollout_K,
                rng_root=rollout_rng_root,
                obs_seeds=obs_seeds,
                rollout_loss_mode=args.rollout_loss_mode,
            )
            # Use the *last* physical step's aux for the existing single-
            # value progress bar fields, and report per-step substep_obs
            # in the verbose log line below.
            aux = aux_per_step[-1]

        loss_val = float(loss)
        terminal_val = float(aux["terminal"])
        # Means across physical steps, for compact logging.
        substep_obs_mean = float(
            sum(float(a["substep_obs"]) for a in aux_per_step) / len(aux_per_step)
        )
        u_l2_mean = float(
            sum(float(a["u_l2"]) for a in aux_per_step) / len(aux_per_step)
        )
        elapsed = time.time() - t0
        if ema_total is None:
            ema_total = loss_val
            ema_terminal = terminal_val
        else:
            ema_total = args.ema_decay * ema_total + (1.0 - args.ema_decay) * loss_val
            ema_terminal = (
                args.ema_decay * ema_terminal + (1.0 - args.ema_decay) * terminal_val
            )

        pbar.set_postfix(
            total=f"{loss_val:.4f}",
            term=f"{terminal_val:.4f}",
            term_ema=f"{ema_terminal:.4f}",
            sub_obs=f"{substep_obs_mean:.4f}",
            u_l2=f"{u_l2_mean:.2e}",
            s_per_step=f"{elapsed:.1f}",
            refresh=False,
        )

        if step % args.log_every == 0 or step == 1:
            if args.rollout_K == 1:
                tqdm.write(
                    f"[train] step {step:4d}/{opt_cfg.num_train_steps}  "
                    f"file={file_id} target_idx={target_idx}  "
                    f"total={loss_val:.4f}  "
                    f"terminal={terminal_val:.4f}  "
                    f"terminal_ema={ema_terminal:.4f}  "
                    f"substep_obs={float(aux['substep_obs']):.4f}  "
                    f"full={float(aux['full_mse']):.4f}  "
                    f"obs={float(aux['obs_mse']):.4f}  "
                    f"score={float(aux['score_loss']):.3e}  "
                    f"control={float(aux['control_loss']):.3e}  "
                    f"u_l2={float(aux['u_l2']):.3e}  "
                    f"({elapsed:.1f}s)"
                )
            else:
                per_step_obs = ",".join(
                    f"{float(a['substep_obs']):.3f}" for a in aux_per_step
                )
                per_step_full = ",".join(
                    f"{float(a['full_mse']):.3f}" for a in aux_per_step
                )
                per_step_u = ",".join(f"{float(a['u_l2']):.2e}" for a in aux_per_step)
                tqdm.write(
                    f"[train] step {step:4d}/{opt_cfg.num_train_steps}  "
                    f"file={file_id} start_idx={target_idx}  K={args.rollout_K}  "
                    f"total_mean={loss_val:.4f}  "
                    f"terminal_mean_ema={ema_terminal:.4f}  "
                    f"substep_obs[k]=[{per_step_obs}]  "
                    f"full[k]=[{per_step_full}]  "
                    f"u_l2[k]=[{per_step_u}]  "
                    f"({elapsed:.1f}s, {elapsed / args.rollout_K:.1f}s/phys-step)"
                )

        history.append(
            {
                "step": step,
                "target_idx": target_idx,
                "file_id": file_id,
                "rollout_K": args.rollout_K,
                "total": loss_val,
                "terminal": terminal_val,
                "total_ema": ema_total,
                "terminal_ema": ema_terminal,
                "substep_obs_mean": substep_obs_mean,
                "u_l2_mean": u_l2_mean,
                "substep_obs": float(aux["substep_obs"]),
                "full_mse": float(aux["full_mse"]),
                "obs_mse": float(aux["obs_mse"]),
                "score_loss": float(aux["score_loss"]),
                "control_loss": float(aux["control_loss"]),
                "u_l2": float(aux["u_l2"]),
                "per_step_substep_obs": [float(a["substep_obs"]) for a in aux_per_step],
                "per_step_full_mse": [float(a["full_mse"]) for a in aux_per_step],
                "per_step_u_l2": [float(a["u_l2"]) for a in aux_per_step],
                "elapsed_s": elapsed,
            }
        )

        if requested_signal["value"] is not None:
            _save_checkpoint(
                ckpt_path,
                params,
                opt_state,
                step,
                ctrl_cfg,
                num_noise_levels=bundle.sampler_config.num_noise_levels,
                training_config=training_config,
                compatibility=compatibility,
                ema_total=ema_total,
                ema_terminal=ema_terminal,
                numpy_train_state=rng_np_train.bit_generator.state,
                numpy_observation_state=rng_np_obs.bit_generator.state,
            )
            break

        if step % args.val_every == 0 and val_pairs:
            val_loss_total = 0.0
            val_n = 0
            v_g_params, v_c_params = split_params(params)
            v_evolving = _evolving_input_vars(bundle.task_config)
            for vi, (v_fid, vidx) in enumerate(val_pairs):
                v_example = examples[v_fid][1]
                if args.rollout_K == 1:
                    v_arr = build_step(
                        bundle,
                        v_example,
                        vidx,
                        obs_cfg,
                        np.random.default_rng(opt_cfg.seed + 100 + vi),
                    )
                    v_rng = validation_key(rng, vi)
                    v_loss, _ = loss_fn(v_c_params, v_g_params, state, v_rng, v_arr)
                    val_loss_total += float(v_loss)
                    val_n += 1
                else:
                    # Rollout val: starting at vidx, K_phys forward
                    # steps; loss_mode-averaged the same way as train so
                    # train/val numbers are unit-comparable.
                    if vidx == 2:
                        v_memory = _initial_memory(v_example, bundle.task_config).copy(
                            deep=True
                        )
                    else:
                        v_memory = v_example.isel(time=[vidx - 2, vidx - 1])[
                            v_evolving
                        ].copy(deep=True)
                    v_total = 0.0
                    for vk in range(args.rollout_K):
                        v_target = vidx + vk
                        v_obs_rng = np.random.default_rng(
                            opt_cfg.seed + 100 + vi * 17 + vk * 31
                        )
                        v_arr_k = build_step_with_memory(
                            bundle,
                            v_example,
                            v_memory,
                            v_target,
                            obs_cfg,
                            v_obs_rng,
                        )
                        v_rng_k = validation_key(rng, vi, vk)
                        v_loss_k, v_aux_k = loss_fn(
                            v_c_params,
                            v_g_params,
                            state,
                            v_rng_k,
                            v_arr_k,
                        )
                        v_total += float(v_loss_k)
                        v_memory = _update_memory(
                            v_memory,
                            v_aux_k["pred"],
                            v_evolving,
                        )
                    if args.rollout_loss_mode == "mean":
                        v_total /= float(args.rollout_K)
                    val_loss_total += v_total
                    val_n += 1
            val_avg = val_loss_total / max(val_n, 1)
            tqdm.write(
                f"[val]   step {step:4d}  val_total_avg={val_avg:.4f}  "
                f"(over {val_n} val pairs, K={args.rollout_K})"
            )

        if (
            step % args.save_every == 0
            or step == opt_cfg.num_train_steps
            or requested_signal["value"] is not None
        ):
            _save_checkpoint(
                ckpt_path,
                params,
                opt_state,
                step,
                ctrl_cfg,
                num_noise_levels=bundle.sampler_config.num_noise_levels,
                training_config=training_config,
                compatibility=compatibility,
                ema_total=ema_total,
                ema_terminal=ema_terminal,
                numpy_train_state=rng_np_train.bit_generator.state,
                numpy_observation_state=rng_np_obs.bit_generator.state,
            )
        if requested_signal["value"] is not None:
            break
    pbar.close()

    atomic_write_text(history_path, json.dumps(history, indent=2) + "\n")
    write_sha256_sidecar(history_path)
    print(f"[train_cada] wrote training history -> {history_path}")
    if requested_signal["value"] is not None:
        raise SystemExit(75)
    print("[train_cada] DONE")


if __name__ == "__main__":
    main()
