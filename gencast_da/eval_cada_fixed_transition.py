"""One-step CADA eval on the exact fixed-index training transition.

This is a diagnostic for fixed-index overfit runs from
`train_cada_controller.py --fixed-train-index K`: it rebuilds the same
target transition, same observation noise, and same diffusion key used
during training, then compares a zero-init controller against the loaded
checkpoint on final-sample physical RMSE and CADA substep diagnostics.

Run:
    CUDA_VISIBLE_DEVICES=0 python -m gencast_da.eval_cada_fixed_transition \\
        --ckpt /path/to/controller.pkl \\
        --target-idx 10 --num-noise-levels 4 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gencast_da.cada_features import build_step, cos_lat_full_mse, masked_obs_mse
from gencast_da.cada_guided import (
    CADA_VARIANT,
    init_cada,
    make_cada_run_forward,
    merge_params,
    split_params,
)
from gencast_da.checkpointing import validate_compatibility
from gencast_da.diagnostics import diagnostic_means
from gencast_da.gencast_runner import (
    bundle_compatibility,
    load_bundle,
    load_example_batch,
)
from gencast_da.obs_operator import ObsConfig
from gencast_da.rollout_utils import partition_rmse
from gencast_da.run_cada_gencast import (
    _check_noise_levels_match,
    _load_controller_checkpoint,
)


def _pred_temperature_b1lhw(pred) -> jnp.ndarray:
    """Return physical temperature as `(B, 1, L, lat, lon)` JAX array."""
    return jnp.asarray(
        pred["temperature"].transpose("batch", "time", "level", "lat", "lon").values
    )


def _pred_temperature_tll(pred) -> np.ndarray:
    """Return physical temperature as `(time, lat, lon, level)` numpy array."""
    return (
        pred["temperature"]
        .isel(batch=0)
        .transpose("time", "lat", "lon", "level")
        .values
    )


def _truth_temperature_tll(step_arr) -> np.ndarray:
    """Return truth physical temperature as `(time, lat, lon, level)`."""
    return np.asarray(step_arr.truth_T_jax)[0].transpose(0, 2, 3, 1)


def _rmse_row(method: str, pred, step_arr, runtime_s: float) -> dict[str, float | str]:
    pred_T = _pred_temperature_b1lhw(pred)
    full_mse = cos_lat_full_mse(pred_T, step_arr.truth_T_jax, step_arr.lat_w_jax)
    obs_mse = masked_obs_mse(pred_T, step_arr.obs_phys_full_jax, step_arr.obs_mask_jax)

    pred_tll = _pred_temperature_tll(pred)
    truth_tll = _truth_temperature_tll(step_arr)
    rmse_unobserved_truth = partition_rmse(
        pred_tll,
        truth_tll,
        step_arr.observation_mask,
        observed=False,
    )

    return {
        "method": method,
        "rmse_full": float(jnp.sqrt(full_mse)),
        "rmse_observation_innovation": float(jnp.sqrt(obs_mse)),
        "rmse_unobserved_truth": rmse_unobserved_truth,
        "runtime_s": runtime_s,
    }


def _apply_and_measure(
    label: str,
    cada_apply,
    params,
    state,
    rng_step,
    step_arr,
    expected_substeps: int,
) -> tuple[dict[str, float | str], dict[str, float]]:
    t0 = time.time()
    pred, final_state = cada_apply(
        params,
        state,
        rng_step,
        step_arr.inputs,
        step_arr.targets_template,
        step_arr.forcings,
        step_arr.obs_full_jax,
        step_arr.obs_mask_jax,
        step_arr.obs_phys_full_jax,
    )
    jax.block_until_ready(pred)
    runtime_s = time.time() - t0
    row = _rmse_row(label, pred, step_arr, runtime_s)
    diags_jax = diagnostic_means(final_state)
    diags = {k: float(v) for k, v in diags_jax.items()}
    if diags["substep_count"] != expected_substeps:
        raise ValueError(
            f"expected {expected_substeps} denoising substeps, "
            f"got {diags['substep_count']}"
        )
    return row, diags


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--target-idx", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-noise-levels", type=int, default=4)
    args = parser.parse_args()
    if args.target_idx < 2:
        raise ValueError("--target-idx must be at least 2")
    if args.num_noise_levels < 2:
        raise ValueError("--num-noise-levels must be at least 2")

    ckpt_path = Path(args.ckpt)
    bundle = load_bundle()
    bundle = dataclasses.replace(
        bundle,
        sampler_config=dataclasses.replace(
            bundle.sampler_config,
            num_noise_levels=args.num_noise_levels,
        ),
    )
    print(
        f"[fixed_eval] target_idx={args.target_idx}  seed={args.seed}  "
        f"num_noise_levels={bundle.sampler_config.num_noise_levels}"
    )

    example = load_example_batch()
    if args.target_idx >= example.sizes["time"]:
        raise ValueError("--target-idx exceeds the dataset trajectory")
    obs_cfg = ObsConfig(seed=args.seed)
    obs_seed = args.seed * 13 + 11 + args.target_idx
    step_arr = build_step(
        bundle,
        example,
        target_time_idx=args.target_idx,
        obs_cfg=obs_cfg,
        rng_np=np.random.default_rng(obs_seed),
    )
    print(
        f"[fixed_eval] obs_seed={obs_seed}  "
        f"obs_mask_sum={float(step_arr.obs_mask_jax.sum())}"
    )

    rng_root = jax.random.PRNGKey(args.seed)
    rng_init = jax.random.fold_in(rng_root, 1)
    rng_step = jax.random.fold_in(rng_root, 1)

    (
        controller_params_loaded,
        ctrl_cfg,
        train_n,
        training_config,
        saved_compatibility,
    ) = _load_controller_checkpoint(ckpt_path)
    _check_noise_levels_match(train_n, args.num_noise_levels, ckpt_path)
    active_compatibility = bundle_compatibility(
        bundle,
        controlled_variable=obs_cfg.var,
        cada_variant=CADA_VARIANT,
    )
    validate_compatibility(saved_compatibility, active_compatibility)
    if training_config is None:
        raise ValueError("checkpoint does not record its training configuration")
    expected_training = {
        "fixed_train_index": args.target_idx,
        "seed": args.seed,
        "rollout_K": 1,
        "obs_mask_mode": obs_cfg.mask_mode,
        "obs_lat_stride_deg": obs_cfg.lat_stride_deg,
        "obs_lon_stride_deg": obs_cfg.lon_stride_deg,
        "random_mask_fraction": obs_cfg.random_mask_fraction,
        "random_mask_seed": obs_cfg.random_mask_seed,
    }
    mismatched = [
        key
        for key, expected in expected_training.items()
        if training_config.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "fixed-transition arguments do not match training: "
            + ", ".join(mismatched)
        )
    print(f"[fixed_eval] loaded checkpoint={ckpt_path}")
    print(f"[fixed_eval] ctrl_cfg={ctrl_cfg}")

    print("[fixed_eval] init CADA params/state...")
    t0 = time.time()
    params_zero, state, ctrl_cfg = init_cada(
        bundle,
        rng_init,
        step_arr.inputs,
        step_arr.targets_template,
        step_arr.forcings,
        step_arr.obs_full_jax,
        step_arr.obs_mask_jax,
        step_arr.obs_phys_full_jax,
        ctrl_cfg,
    )
    print(f"[fixed_eval] init done in {time.time() - t0:.1f}s")

    g_params, _ = split_params(params_zero)
    params_trained = merge_params(g_params, controller_params_loaded)
    cada_apply = make_cada_run_forward(bundle, ctrl_cfg, obs_cfg.var)

    rows = []
    diag_rows = {}
    for label, params in (("zero_init", params_zero), ("trained", params_trained)):
        print(f"[fixed_eval] running {label}...")
        row, diags = _apply_and_measure(
            label,
            cada_apply,
            params,
            state,
            rng_step,
            step_arr,
            2 * args.num_noise_levels - 1,
        )
        rows.append(row)
        diag_rows[label] = diags

    print("\n" + "=" * 96)
    print(
        "Fixed-transition one-step eval "
        f"(target_idx={args.target_idx}, seed={args.seed}, obs_seed={obs_seed})"
    )
    print("=" * 96)
    print(
        f"{'method':<12} {'rmse_full':<12} {'innovation':<12} "
        f"{'unobs_truth':<12} {'time_s':<10}"
    )
    print("-" * 64)
    for row in rows:
        print(
            f"{row['method']:<12} {row['rmse_full']:<12.4f} "
            f"{row['rmse_observation_innovation']:<12.4f} "
            f"{row['rmse_unobserved_truth']:<12.4f} "
            f"{row['runtime_s']:<10.2f}"
        )

    print(
        "\nCADA substep diagnostics "
        "(substep_obs in K^2; sqrt tracks observation innovation):"
    )
    print(
        f"{'method':<12} {'substep_obs':<14} {'score':<12} "
        f"{'control':<12} {'u_l2':<12} {'count':<8}"
    )
    print("-" * 78)
    for label in ("zero_init", "trained"):
        diags = diag_rows[label]
        print(
            f"{label:<12} {diags['substep_obs']:<14.6f} "
            f"{diags['score']:<12.3e} {diags['control']:<12.3e} "
            f"{diags['u_l2']:<12.3e} {diags['substep_count']:<8.1f}"
        )

    z, t = rows
    print("\nDelta trained - zero_init:")
    print(f"  rmse_full : {t['rmse_full'] - z['rmse_full']:+.4f}")
    print(
        "  innovation: "
        f"{t['rmse_observation_innovation'] - z['rmse_observation_innovation']:+.4f}"
    )
    print(
        f"  unobs truth: {t['rmse_unobserved_truth'] - z['rmse_unobserved_truth']:+.4f}"
    )
    substep_delta = (
        diag_rows["trained"]["substep_obs"] - diag_rows["zero_init"]["substep_obs"]
    )
    print(f"  substep_obs: {substep_delta:+.6f}")


if __name__ == "__main__":
    main()
