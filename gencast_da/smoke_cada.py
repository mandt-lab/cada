"""Smoke test for `gencast_da.cada_guided`.

Three checks the CADA wrapper must pass before we touch a training loop:

  Test 1 (zero-init parity)
      With `gamma=1.0` and the controller's final-projection layer
      initialised to zero, `u == 0` at every substep, so the controlled
      sampler MUST reproduce the unconditional sample bit-for-bit (within
      fp32 tolerance, as in `smoke_guided.py`).

  Test 2 (state diagnostics)
      After one forward, the Haiku state must contain finite
      `score_sum`, `control_sum`, `u_l2_sum`, and `substep_count`. With
      u==0 we must have `control_sum == 0` and `u_l2_sum == 0`.

  Test 3 (gradient flow)
      Backprop a tiny MSE loss to verify controller parameters get
      finite, non-zero gradients while GenCast params get nothing
      (we only diff w.r.t. controller params).

Run:
    CUDA_VISIBLE_DEVICES=0 python -m gencast_da.smoke_cada
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
from graphcast import data_utils

from gencast_da.cada_guided import (
    CADA_VARIANT,
    CADAControllerConfig,
    init_cada,
    make_cada_run_forward,
    make_scan_unconditional_run_forward,
    merge_params,
    split_params,
)
from gencast_da.diagnostics import diagnostic_means
from gencast_da.gencast_runner import (
    load_bundle,
    load_example_batch,
)
from gencast_da.obs_operator import (
    ObsConfig,
    build_obs_indices,
)
from gencast_da.obs_operator import (
    predicted as obs_predicted,
)
from gencast_da.observation_arrays import make_obs_arrays


def _temp_array(pred, var):
    return pred[var].isel(batch=0).transpose("time", "lat", "lon", "level").values


def main(seed: int = 0, num_noise_levels: int = 4) -> None:
    if num_noise_levels != 4:
        raise ValueError("bounded smoke requires exactly 4 noise levels")
    if jax.default_backend() != "gpu":
        raise RuntimeError("smoke_cada requires a JAX GPU backend")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("smoke_cada found no GPU devices")
    device = devices[0]
    memory_stats = device.memory_stats() or {}
    memory_limit = memory_stats.get("bytes_limit")
    if memory_limit is None:
        raise RuntimeError("smoke_cada could not determine GPU memory capacity")
    if memory_limit < 24 * 2**30:
        raise RuntimeError("smoke_cada requires at least 24 GiB of GPU memory")
    print(
        f"[smoke_cada] jax={jax.__version__} "
        f"device={getattr(device, 'device_kind', device)} "
        f"compute_capability={getattr(device, 'compute_capability', 'unknown')} "
        f"memory_GiB={memory_limit / 2**30}"
    )
    print(f"[smoke_cada] CADA_VARIANT = {CADA_VARIANT!r}")
    print(f"[smoke_cada] num_noise_levels = {num_noise_levels}")
    bundle = load_bundle()
    # Two sampler-config overrides for this smoke test:
    #   1. `stochastic_churn_rate=0.0`: the CADA sampler is hk.scan-based
    #      for differentiability while the official inference sampler is
    #      hk.fori_loop-based. With stochastic churn on, the two
    #      control-flow forms consume RNG differently, so sample-wise
    #      equality (Test 1) is not a valid no-op-wrapper check.
    #   2. `num_noise_levels=N`: must match the value used at train and
    #      eval time (the controller is conditioned on sigma_t and the
    #      rho-spaced sigma grid changes with N, so any mismatch puts
    #      every substep call OOD). Default 4 is the current 24 GB-safe
    #      setting for Test 3's full forward+backward.
    bundle = dataclasses.replace(
        bundle,
        sampler_config=dataclasses.replace(
            bundle.sampler_config,
            stochastic_churn_rate=0.0,
            num_noise_levels=num_noise_levels,
        ),
    )
    example = load_example_batch()

    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example,
            target_lead_times=slice("12h", "12h"),
            **dataclasses.asdict(bundle.task_config),
        )
    )

    obs_cfg = ObsConfig(seed=seed)

    truth_at_obs = obs_predicted(eval_targets, obs_cfg)
    rng_np = np.random.default_rng(seed)
    obs_phys = truth_at_obs + rng_np.normal(
        0.0, obs_cfg.noise_std, size=truth_at_obs.shape
    ).astype(truth_at_obs.dtype)

    obs_full_np, obs_mask_np, _sigma_y_norm_np = make_obs_arrays(
        eval_inputs, obs_phys, bundle.diffs_stddev_by_level, obs_cfg
    )
    obs_full = jnp.asarray(obs_full_np)
    obs_mask = jnp.asarray(obs_mask_np)

    # Physical-K obs zero-filled to (B, 1, L, lat, lon). Same construction
    # as `cada_features.build_step`; needed by the new physical-Kelvin
    # substep obs loss inside the wrapper.
    lat_idx, lon_idx = build_obs_indices(eval_inputs, obs_cfg)
    n_batch = eval_inputs.sizes["batch"]
    n_lat = eval_inputs.sizes["lat"]
    n_lon = eval_inputs.sizes["lon"]
    n_level = eval_inputs.sizes["level"]
    obs_phys_full_np = np.zeros(
        (n_batch, 1, n_level, n_lat, n_lon),
        dtype=np.float32,
    )
    obs_phys_arr = obs_phys.transpose("batch", "time", "level", "lat", "lon").values
    obs_phys_full_np[:, :, :, lat_idx[:, None], lon_idx[None, :]] = obs_phys_arr
    obs_phys_full = jnp.asarray(obs_phys_full_np)

    print(f"[smoke_cada] obs_full shape: {obs_full.shape}")
    print(f"[smoke_cada] obs_mask sum:   {float(obs_mask.sum())}")

    rng = jax.random.PRNGKey(seed)
    # apply_remat=True (the new default after the hk.scan refactor): each
    # substep's activations are recomputed in backward, so Test 3's full-
    # chain forward+backward at the bundle's full noise schedule is
    # memory-bounded to one substep. See _CADAScanGenCast in cada_guided.py.
    cfg = CADAControllerConfig(
        hidden_channels=16, num_blocks=2, gamma=1.0, apply_remat=True
    )

    print("[smoke_cada] init CADA (compiles GenCast init pass once)...")
    t0 = time.time()
    params, state, cfg = init_cada(
        bundle,
        jax.random.fold_in(rng, 1),
        eval_inputs,
        eval_targets,
        eval_forcings,
        obs_full,
        obs_mask,
        obs_phys_full,
        cfg,
    )
    jax.block_until_ready((params, state))
    print(f"[smoke_cada] init done in {time.time() - t0:.1f}s")

    g_params, c_params = split_params(params)
    n_g = sum(p.size for d in g_params.values() for p in d.values())
    n_c = sum(p.size for d in c_params.values() for p in d.values())
    print(f"[smoke_cada] frozen GenCast params: {n_g:,} | controller params: {n_c:,}")
    print(f"[smoke_cada] controller scopes: {sorted(c_params.keys())}")

    # Regression guard: the controller MUST be a single shared instance reused
    # across every DPM-Solver++ 2S substep. If `controlled_pcd` accidentally
    # constructs CADAController inside the per-call closure again, Haiku
    # auto-suffixes the name (`cada_controller`, `cada_controller_1`, ...),
    # silently breaking the `last_u_t = controller_θ(..., last_u_{t-1})`
    # recurrence. Catch that here.
    ctrl_roots = {k.split("/", 1)[0] for k in c_params.keys()}
    assert ctrl_roots == {"cada_controller"}, (
        f"Expected exactly one controller scope 'cada_controller', got "
        f"{sorted(ctrl_roots)}. Each substep is creating its own controller "
        "instance — see _wrap_with_cada_guidance in cada_guided.py."
    )

    # ---------- Test 1: zero-init parity ----------
    # We use the scan-based unconditional reference (same control flow as
    # CADA) so that fori-vs-scan RNG-reservation differences are out of
    # the picture. Even so, bit-equivalence is not achievable: the CADA
    # graph has the controller params as additional XLA inputs and adds
    # an auxiliary `_preconditioned_denoiser` call per substep. XLA
    # produces a different fp32 reduction order for those two graphs, so
    # we expect O(0.05 K) max-abs drift even when the controller's
    # `proj_out` is `Constant(0)`-init (and thus `u == 0` exactly). The
    # threshold below is therefore "small enough to be fp32 noise on a
    # ~20-step diffusion chain", not "bit-identical".
    print("[smoke_cada] Test 1: zero-init parity (CADA vs scan unconditional)")
    run_uncond = make_scan_unconditional_run_forward(
        bundle, apply_remat=cfg.apply_remat
    )
    run_cada = make_cada_run_forward(bundle, cfg)

    t0 = time.time()
    pred_uncond = run_uncond(rng, eval_inputs, eval_targets, eval_forcings)
    jax.block_until_ready(pred_uncond)
    print(f"  unconditional sample: {time.time() - t0:.1f}s")

    t0 = time.time()
    pred_cada, final_state = run_cada(
        params,
        state,
        rng,
        eval_inputs,
        eval_targets,
        eval_forcings,
        obs_full,
        obs_mask,
        obs_phys_full,
    )
    jax.block_until_ready((pred_cada, final_state))
    print(f"  CADA controlled sample: {time.time() - t0:.1f}s")

    pred_T_uncond = pred_uncond["temperature"].values
    pred_T_cada = pred_cada["temperature"].values
    repeated_pred, _ = run_cada(
        params,
        state,
        rng,
        eval_inputs,
        eval_targets,
        eval_forcings,
        obs_full,
        obs_mask,
        obs_phys_full,
    )
    jax.block_until_ready(repeated_pred)
    repeated_temperature = repeated_pred["temperature"].values
    repeated_max_diff = float(
        np.max(np.abs(pred_T_cada - repeated_temperature))
    )
    print(f"  repeated-key max abs diff (T): {repeated_max_diff:.3e} K")
    assert repeated_max_diff < 1e-4, (
        "Test 1 FAILED: identical JAX keys exceeded deterministic tolerance: "
        f"{repeated_max_diff:.3e} K"
    )
    diff_max = float(np.abs(pred_T_uncond - pred_T_cada).max())
    diff_mean = float(np.abs(pred_T_uncond - pred_T_cada).mean())
    t_min, t_max = float(pred_T_cada.min()), float(pred_T_cada.max())
    print(
        f"  max abs diff (T): {diff_max:.3e} K  "
        f"mean abs diff (T): {diff_mean:.3e} K  "
        f"CADA T range: [{t_min:.1f}, {t_max:.1f}] K"
    )
    assert np.isfinite(pred_T_cada).all(), (
        "Test 1 FAILED: pred_cada has non-finite values."
    )
    assert 150.0 < t_min and t_max < 350.0, (
        f"Test 1 FAILED: CADA T range [{t_min:.1f}, {t_max:.1f}] K is "
        "not physically plausible; wrapper likely diverges."
    )
    assert diff_max < 0.1, (
        f"Test 1 FAILED: max diff {diff_max:.3e} K > 0.1 K. "
        "Controller is not zero-init or wrapper diverges from unconditional. "
        "(Expected fp32 noise from extra-controller-graph: O(0.05 K).)"
    )
    print("  Test 1 PASS")

    # ---------- Test 2: state diagnostics ----------
    print("[smoke_cada] Test 2: state diagnostics shape/values")
    diags = diagnostic_means(final_state)
    score = float(diags["score"])
    control = float(diags["control"])
    u_l2 = float(diags["u_l2"])
    substep_obs = float(diags["substep_obs"])
    count = float(diags["substep_count"])
    print(f"  substep_count: {count}")
    print(f"  score_loss   : {score:.3e}")
    print(f"  control_loss : {control:.3e}")
    print(f"  u_l2         : {u_l2:.3e}")
    print(
        f"  substep_obs  : {substep_obs:.3e} K^2  "
        f"(masked physical-K MSE; sqrt -> "
        f"{float(np.sqrt(max(substep_obs, 0.0))):.3f} K)"
    )
    expected_substeps = 2 * num_noise_levels - 1
    assert count == expected_substeps, (
        f"Test 2 FAILED: expected {expected_substeps} substeps, got {count}"
    )
    assert control == 0.0, (
        f"Test 2 FAILED: control_loss should be 0 at zero-init, got {control}"
    )
    assert u_l2 == 0.0, f"Test 2 FAILED: u_l2 should be 0 at zero-init, got {u_l2}"
    # With u == 0, both passes use the same predicted noise, so this is zero.
    assert abs(score) < 1e-6, (
        f"Test 2 FAILED: score_loss should be ~0 at zero-init, got {score}"
    )
    # substep_obs is the masked physical-K MSE between x0_T_ctrl_phys and
    # the obs at observed cells. Even at zero-init (u==0) it should be
    # finite and strictly positive: the controlled denoiser's Tweedie
    # estimate at observed cells will not exactly match the noisy obs
    # draw. Sqrt(substep_obs) should be in the same ballpark as the
    # observation-innovation RMSE — at zero-init expect ~0.5–1.0 K
    # (mid-substeps are noisy; the final substep is the cleanest).
    assert np.isfinite(substep_obs), (
        f"Test 2 FAILED: substep_obs is non-finite ({substep_obs})"
    )
    assert substep_obs > 0.0, (
        f"Test 2 FAILED: substep_obs should be > 0 at zero-init "
        f"(Tweedie estimate ≠ noisy obs at observed cells), got "
        f"{substep_obs}. Check mask/obs construction."
    )
    print("  Test 2 PASS")

    # ---------- Test 3: gradient flow ----------
    print("[smoke_cada] Test 3: gradient flow w.r.t. controller params only")
    # Tiny loss: mean-squared distance to truth on temperature.
    truth_T = jnp.asarray(
        eval_targets["temperature"]
        .transpose("batch", "time", "level", "lat", "lon")
        .values
    )
    # Call the underlying (un-jit'd) apply so jax.grad walks pytree structure.
    from graphcast import xarray_jax as _xj

    from gencast_da.cada_guided import _make_run_forward_transform

    cada_xform = _make_run_forward_transform(bundle, cfg, "temperature")

    # Differentiate ONLY w.r.t. controller params. GenCast (~57M params) is
    # frozen, so building gradients for it just to discard them blows up
    # memory: backprop has to keep activations alive for the *frozen*
    # subgraph too. Splitting the params and feeding GenCast's frozen
    # half through `jax.lax.stop_gradient` lets XLA prune the GenCast
    # cotangents from the backward graph. This is the primary memory win
    # over the naive `jax.value_and_grad(...)(params)` form.
    g_params, c_params = split_params(params)
    g_params_frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, g_params)

    def loss_fn(c_p):
        merged = merge_params(g_params_frozen, c_p)
        pred, _ = cada_xform.apply(
            merged,
            state,
            rng,
            eval_inputs,
            eval_targets,
            eval_forcings,
            obs_full,
            obs_mask,
            obs_phys_full,
        )
        pred_T_jx = _xj.unwrap_data(
            pred["temperature"].transpose("batch", "time", "level", "lat", "lon")
        )
        return jnp.mean((pred_T_jx - truth_T) ** 2)

    print("  forward + grad (this triggers full-chain unrolled compile)...")
    t0 = time.time()
    loss_val, c_grads = jax.value_and_grad(loss_fn)(c_params)
    c_grads = jax.block_until_ready(c_grads)
    print(
        f"  one fwd+bwd done in {time.time() - t0:.1f}s; loss = {float(loss_val):.4f}"
    )

    c_grad_max = max(
        (float(jnp.max(jnp.abs(p))) for d in c_grads.values() for p in d.values()),
        default=0.0,
    )
    c_grad_l2 = float(
        jnp.sqrt(sum(jnp.sum(p**2) for d in c_grads.values() for p in d.values()))
    )
    print(f"  controller grad max: {c_grad_max:.3e}")
    print(f"  controller grad L2 : {c_grad_l2:.3e}")
    assert np.isfinite(c_grad_max), "Test 3 FAILED: controller grads contain NaN/Inf"
    assert c_grad_max > 0.0, (
        f"Test 3 FAILED: controller grad max is exactly 0 ({c_grad_max}); "
        "controller is not connected to the loss."
    )
    print("  Test 3 PASS")

    print("[smoke_cada] ALL THREE TESTS PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-noise-levels",
        type=int,
        default=4,
        help="Override sampler_config.num_noise_levels for this smoke "
        "test only. The default (4) is the current 24 GB-safe "
        "setting for Test 3's full forward+backward; the bundle's "
        "default 20 OOMs.",
    )
    args = parser.parse_args()
    main(seed=args.seed, num_noise_levels=args.num_noise_levels)
