"""K-step autoregressive evaluation of a trained CADA controller on GenCast.

Loads a controller checkpoint produced by `train_cada_controller.py`,
then runs K assimilation steps starting from the example trajectory's
first 2 frames. At each step we generate one realisation of obs from
the truth, run the CADA-controlled sampler, and log:

  - cos(lat)-weighted full-field RMSE on temperature,
  - clean-truth RMSE on observed cells,
  - noisy-observation innovation RMSE,
  - clean-truth RMSE on unobserved cells,
  - runtime per 12 h step.

The default observation regime is temperature on a 4° grid at all 13
pressure levels with sigma_y=0.1 K.

Run:
    CUDA_VISIBLE_DEVICES=0 python -m gencast_da.run_cada_gencast \\
        --ckpt /path/to/controller.pkl \\
        --steps 4
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gencast_da.artifacts import write_evaluation_npz, write_metrics
from gencast_da.cada_features import build_step
from gencast_da.cada_guided import (
    CADA_VARIANT,
    CADAControllerConfig,
    init_cada,
    make_cada_run_forward,
    merge_params,
    split_params,
)
from gencast_da.checkpointing import (
    load_trusted_checkpoint,
    validate_compatibility,
)
from gencast_da.gencast_runner import (
    bundle_compatibility,
    load_bundle,
    load_example_batch,
)
from gencast_da.obs_operator import (
    ObsConfig,
    build_obs_indices,
    build_obs_mask,
)
from gencast_da.obs_operator import (
    predicted as obs_predicted,
)
from gencast_da.observation_arrays import make_obs_arrays
from gencast_da.rollout_utils import (
    build_step_args as _build_step_args,
)
from gencast_da.rollout_utils import (
    evolving_input_variables as _evolving_input_vars,
)
from gencast_da.rollout_utils import (
    initial_memory as _initial_memory,
)
from gencast_da.rollout_utils import (
    partition_rmse,
    weighted_rmse,
)
from gencast_da.rollout_utils import (
    to_temperature_array as _to_temp_array,
)
from gencast_da.rollout_utils import (
    update_memory as _update_memory,
)
from gencast_da.runtime import sha256_file


@dataclass
class CADAStepRecord:
    step: int
    lead_h: int
    rmse_full: float
    rmse_observed_truth: float | None
    rmse_observation_innovation: float | None
    rmse_unobserved_truth: float | None
    runtime_s: float


def _finite_metadata(value):
    """Encode legitimate infinite config bounds for strict JSON artifacts."""

    if isinstance(value, dict):
        return {key: _finite_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_metadata(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("metadata contains NaN")
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _load_controller_checkpoint(
    ckpt_path: Path,
) -> tuple[dict, CADAControllerConfig, int | None, dict | None, dict]:
    """Load a checkpoint saved by train_cada_controller.py.

    Returns controller parameters, controller config, diffusion schedule, and
    the effective training configuration when available.
    `train_num_noise_levels` is the schedule N the controller was trained on.
    Callers compare against inference N and refuse to proceed on mismatch.
    """
    payload, loaded_path = load_trusted_checkpoint(ckpt_path)
    if loaded_path != ckpt_path:
        print(f"[run_cada] primary checkpoint invalid; using {loaded_path}")
    controller_params = payload["controller_params"]
    ctrl_cfg_dict = payload["cfg"]
    ctrl_cfg = CADAControllerConfig(
        hidden_channels=ctrl_cfg_dict.get("hidden_channels", 32),
        num_blocks=ctrl_cfg_dict.get("num_blocks", 3),
        sigma_emb_dim=ctrl_cfg_dict.get("sigma_emb_dim", 64),
        gamma=ctrl_cfg_dict.get("gamma", 1.0),
        # Keep the same productive-pass remat path used during smoke/train.
        # Even without backward, this changes XLA's forward memory plan
        # enough to avoid OS-killed eval compiles on a 24 GB GPU.
        apply_remat=ctrl_cfg_dict.get("apply_remat", True),
    )
    train_num_noise_levels = payload.get("num_noise_levels")
    return (
        controller_params,
        ctrl_cfg,
        train_num_noise_levels,
        payload.get("training_config"),
        payload.get("compatibility"),
    )


def _check_noise_levels_match(
    train_n: int | None,
    inference_n: int,
    ckpt_path: Path,
) -> None:
    """Refuse to roll out a controller on a different sigma grid than the
    one it was trained on (the controller takes sigma_t as input, so any
    mismatch is OOD)."""
    if train_n is None:
        raise ValueError(f"checkpoint {ckpt_path} has no diffusion schedule")
    if train_n != inference_n:
        raise ValueError(
            f"sigma-grid mismatch: controller was trained at "
            f"num_noise_levels={train_n} but --num-noise-levels="
            f"{inference_n}. The controller is conditioned on sigma_t and "
            "the rho-spaced grid changes with N, so this would put every "
            "substep call OOD. Re-run with --num-noise-levels="
            f"{train_n}, or retrain at {inference_n}."
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="zero_init",
        help="Trained controller checkpoint. Pass 'zero_init' to skip "
        "checkpoint and use a freshly-initialised (zero) controller "
        "as a sanity baseline.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-noise-levels",
        type=int,
        default=4,
        help="DPM-Solver++ 2S chain length to use at inference. MUST "
        "match the value used to train the loaded controller: the "
        "controller takes sigma_t as a conditioning input, so a "
        "different N here puts every substep call OOD for the "
        "controller (rho-spaced sigma grid changes with N).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc",
        help="ERA5 NetCDF file name in DATA_ROOT/dataset/.",
    )
    parser.add_argument("--obs-lat-stride-deg", type=int, default=4)
    parser.add_argument("--obs-lon-stride-deg", type=int, default=4)
    parser.add_argument(
        "--obs-mask-mode",
        choices=("stride", "random"),
        default="stride",
    )
    parser.add_argument("--random-mask-fraction", type=float, default=None)
    parser.add_argument("--random-mask-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional run directory containing samples/ and metrics/.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.num_noise_levels < 2:
        raise ValueError("--num-noise-levels must be at least 2")

    bundle = load_bundle()
    # Match training schedule (see --num-noise-levels help).
    bundle = dataclasses.replace(
        bundle,
        sampler_config=dataclasses.replace(
            bundle.sampler_config,
            num_noise_levels=args.num_noise_levels,
        ),
    )
    print(
        f"[run_cada] sampler num_noise_levels = "
        f"{bundle.sampler_config.num_noise_levels}"
    )
    example = load_example_batch(dataset_name=args.dataset)
    if args.steps > example.sizes["time"] - 2:
        raise ValueError(
            f"--steps {args.steps} exceeds available forecast transitions"
        )
    obs_cfg = ObsConfig(
        seed=args.seed,
        lat_stride_deg=args.obs_lat_stride_deg,
        lon_stride_deg=args.obs_lon_stride_deg,
        mask_mode=args.obs_mask_mode,
        random_mask_fraction=args.random_mask_fraction,
        random_mask_seed=args.random_mask_seed,
    )
    var = obs_cfg.var

    print("[run_cada] init CADA params (compiles GenCast init pass once)...")
    init_arr = build_step(
        bundle,
        example,
        target_time_idx=2,
        obs_cfg=obs_cfg,
        rng_np=np.random.default_rng(args.seed),
    )
    rng = jax.random.PRNGKey(args.seed)
    active_compatibility = bundle_compatibility(
        bundle,
        controlled_variable=obs_cfg.var,
        cada_variant=CADA_VARIANT,
    )
    if args.ckpt == "zero_init":
        ctrl_cfg = CADAControllerConfig(apply_remat=True)
    else:
        (
            controller_params_loaded,
            ctrl_cfg,
            train_n,
            training_config,
            saved_compatibility,
        ) = _load_controller_checkpoint(Path(args.ckpt))
        _check_noise_levels_match(train_n, args.num_noise_levels, Path(args.ckpt))
        validate_compatibility(saved_compatibility, active_compatibility)
        if training_config is not None:
            expected_observation = {
                "obs_lat_stride_deg": args.obs_lat_stride_deg,
                "obs_lon_stride_deg": args.obs_lon_stride_deg,
                "obs_mask_mode": args.obs_mask_mode,
                "random_mask_fraction": args.random_mask_fraction,
            }
            changed = [
                key
                for key, value in expected_observation.items()
                if training_config.get(key) != value
            ]
            if changed:
                raise ValueError(
                    "evaluation observation settings differ from training: "
                    + ", ".join(changed)
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
    print(f"[run_cada] init done in {time.time() - t0:.1f}s")

    if args.ckpt != "zero_init":
        g_params, _ = split_params(params)
        params = merge_params(g_params, controller_params_loaded)
        print(f"[run_cada] loaded controller params from {args.ckpt}")
    else:
        print("[run_cada] zero_init mode: using freshly-initialised controller")

    cada_apply = make_cada_run_forward(bundle, ctrl_cfg, var)
    print(f"[run_cada] ctrl_cfg = {ctrl_cfg}")

    task_config = bundle.task_config
    evolving = _evolving_input_vars(task_config)
    observation_mask = build_obs_mask(example, obs_cfg)
    if obs_cfg.mask_mode == "stride":
        lat_idx, lon_idx = build_obs_indices(example, obs_cfg)
    else:
        lat_idx = lon_idx = None
    lat_deg = example.coords["lat"].values

    memory = _initial_memory(example, task_config).copy(deep=True)
    rng_np_obs = np.random.default_rng(args.seed * 17 + 5)

    records: list[CADAStepRecord] = []
    truth_outputs = []
    prediction_outputs = []
    observation_outputs = []
    for step in range(args.steps):
        target_time_idx = step + 2
        lead_h = (step + 1) * 12
        print(
            f"\n--- step {step + 1}/{args.steps}  "
            f"(lead {lead_h}h, target_time_idx={target_time_idx}) ---"
        )

        inputs_s, tt_s, forcings_s = _build_step_args(
            example, memory, target_time_idx, task_config
        )

        truth_slice = example.isel(time=[target_time_idx])
        truth_T = (
            truth_slice[var]
            .isel(batch=0)
            .transpose("time", "lat", "lon", "level")
            .values
        )

        truth_at_obs = obs_predicted(truth_slice, obs_cfg)
        obs_phys = truth_at_obs + rng_np_obs.normal(
            0.0, obs_cfg.noise_std, size=truth_at_obs.shape
        ).astype(truth_at_obs.dtype)

        obs_full_np, obs_mask_np, _ = make_obs_arrays(
            inputs_s, obs_phys, bundle.diffs_stddev_by_level, obs_cfg
        )
        obs_full = jnp.asarray(obs_full_np)
        obs_mask = jnp.asarray(obs_mask_np)

        # Physical-K obs zero-filled to (B, 1, L, lat, lon). The CADA
        # wrapper reads this for its substep obs loss diagnostic; eval
        # never gradients through it but it's a required apply argument.
        n_batch = inputs_s.sizes["batch"]
        n_lat = inputs_s.sizes["lat"]
        n_lon = inputs_s.sizes["lon"]
        n_level = inputs_s.sizes["level"]
        obs_phys_full_np = np.zeros(
            (n_batch, 1, n_level, n_lat, n_lon),
            dtype=np.float32,
        )
        obs_phys_arr = obs_phys.transpose("batch", "time", "level", "lat", "lon").values
        if obs_phys.sizes["lat"] == n_lat and obs_phys.sizes["lon"] == n_lon:
            obs_phys_full_np = obs_phys_arr.astype(np.float32)
            obs_phys_full_np *= observation_mask[None, None, None, :, :]
        else:
            assert lat_idx is not None and lon_idx is not None
            obs_phys_full_np[:, :, :, lat_idx[:, None], lon_idx[None, :]] = obs_phys_arr
        obs_phys_full = jnp.asarray(obs_phys_full_np)

        rng_step = jax.random.fold_in(rng, step + 31_337)
        t0 = time.time()
        pred, _final_state = cada_apply(
            params,
            state,
            rng_step,
            inputs_s,
            tt_s,
            forcings_s,
            obs_full,
            obs_mask,
            obs_phys_full,
        )
        jax.block_until_ready(pred)
        runtime = time.time() - t0
        pred_T = _to_temp_array(pred, var)
        full_observation_T = obs_phys_full_np[0, 0].transpose(1, 2, 0)[None]

        rmse_full = weighted_rmse(truth_T[0], pred_T[0], lat_deg)
        rmse_observed_truth = partition_rmse(
            pred_T,
            truth_T,
            observation_mask,
            observed=True,
        )
        rmse_observation_innovation = partition_rmse(
            pred_T,
            full_observation_T,
            observation_mask,
            observed=True,
        )
        rmse_unobserved_truth = partition_rmse(
            pred_T,
            truth_T,
            observation_mask,
            observed=False,
        )
        rec = CADAStepRecord(
            step=step + 1,
            lead_h=lead_h,
            rmse_full=rmse_full,
            rmse_observed_truth=rmse_observed_truth,
            rmse_observation_innovation=rmse_observation_innovation,
            rmse_unobserved_truth=rmse_unobserved_truth,
            runtime_s=runtime,
        )
        records.append(rec)
        truth_outputs.append(truth_T[0])
        prediction_outputs.append(pred_T[0])
        observation_outputs.append(full_observation_T[0])
        observed_text = (
            f"{rmse_observed_truth:.4f}" if rmse_observed_truth is not None else "n/a"
        )
        innovation_text = (
            f"{rmse_observation_innovation:.4f}"
            if rmse_observation_innovation is not None
            else "n/a"
        )
        unobserved_text = (
            f"{rmse_unobserved_truth:.4f}"
            if rmse_unobserved_truth is not None
            else "n/a"
        )
        print(
            f"  RMSE full={rmse_full:.4f}K  "
            f"observed_truth={observed_text}K  "
            f"innovation={innovation_text}K  "
            f"unobserved_truth={unobserved_text}K  ({runtime:.1f}s)"
        )

        memory = _update_memory(memory, pred, evolving)

    print()
    print("=" * 80)
    print(f"CADA-on-GenCast K-step summary  (K = {args.steps}, ckpt = {args.ckpt})")
    print("=" * 80)
    header = (
        f"{'step':<6}{'lead_h':<8}{'rmse_full':<14}"
        f"{'obs_truth':<14}{'innovation':<14}{'unobs_truth':<14}{'time_s':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        observed_text = (
            f"{r.rmse_observed_truth:.4f}"
            if r.rmse_observed_truth is not None
            else "n/a"
        )
        innovation_text = (
            f"{r.rmse_observation_innovation:.4f}"
            if r.rmse_observation_innovation is not None
            else "n/a"
        )
        unobserved_text = (
            f"{r.rmse_unobserved_truth:.4f}"
            if r.rmse_unobserved_truth is not None
            else "n/a"
        )
        print(
            f"{r.step:<6}{r.lead_h:<8}{r.rmse_full:<14.4f}"
            f"{observed_text:<14}{innovation_text:<14}"
            f"{unobserved_text:<14}{r.runtime_s:<10.2f}"
        )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        run_dir = os.environ.get("CADA_RUN_DIR")
        metadata = {
            "method": "cada",
            "checkpoint": args.ckpt,
            "checkpoint_sha256": (
                None
                if args.ckpt == "zero_init"
                else sha256_file(args.ckpt)
            ),
            "dataset": args.dataset,
            "seed": args.seed,
            "steps": args.steps,
            "num_noise_levels": args.num_noise_levels,
            "observation": dataclasses.asdict(obs_cfg),
            "compatibility": _finite_metadata(active_compatibility),
            "run_dir": run_dir,
            "run_id": Path(run_dir).name if run_dir else None,
            "config_sha256": os.environ.get("CADA_CONFIG_SHA256"),
        }
        coordinates = {
            "latitude": np.asarray(example.coords["lat"].values),
            "longitude": np.asarray(example.coords["lon"].values),
            "pressure_level": np.asarray(example.coords["level"].values),
            "lead_hours": np.asarray(
                [record.lead_h for record in records],
                dtype=np.int64,
            ),
            "observation_mask": np.asarray(observation_mask),
        }
        write_evaluation_npz(
            output_dir / "samples" / "evaluation.npz",
            truth=np.stack(truth_outputs),
            prediction=np.stack(prediction_outputs),
            observations=np.stack(observation_outputs),
            coordinates=coordinates,
            metadata=metadata,
        )
        write_metrics(
            output_dir / "metrics" / "evaluation.json",
            {"steps": [dataclasses.asdict(record) for record in records]},
            metadata=metadata,
        )
        print(f"[run_cada] wrote structured outputs under {output_dir}")


if __name__ == "__main__":
    main()
