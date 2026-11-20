"""End-to-end smoke test for GenCast Mini on a single GPU.

What it verifies
----------------
1. The pinned env is sane: jax sees a GPU, and `graphcast.gencast` imports.
2. The downloaded checkpoint, the 4 stats files, and the example ERA5 NetCDF
   are present and parse correctly.
3. A single 12h sample can be drawn without OOMing.

Run:
    conda activate ./.envs/gencast-da
    python -m gencast_da.smoke_load
"""

from __future__ import annotations

import dataclasses
import time

import jax
import numpy as np
from graphcast import data_utils

from gencast_da.gencast_runner import (
    load_bundle,
    load_example_batch,
    make_run_forward_jitted,
)


def main() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("smoke_load requires a JAX GPU backend")
    print("[smoke] jax devices:", jax.devices("gpu"))

    print("[smoke] Loading bundle (checkpoint + stats)...")
    t0 = time.time()
    bundle = load_bundle()
    print(f"[smoke]   loaded in {time.time() - t0:.1f}s")
    print("[smoke]   description:", bundle.description.strip().splitlines()[0])

    print("[smoke] Loading example ERA5 batch...")
    example = load_example_batch()
    print("[smoke]   dims:", dict(example.sizes))

    print(
        "[smoke] Extracting (inputs, targets_template, forcings) for 12h lead time..."
    )
    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example,
            target_lead_times=slice("12h", "12h"),
            **dataclasses.asdict(bundle.task_config),
        )
    )
    print("[smoke]   inputs   dims:", dict(eval_inputs.sizes))
    print("[smoke]   targets  dims:", dict(eval_targets.sizes))
    print("[smoke]   forcings dims:", dict(eval_forcings.sizes))

    print("[smoke] Building jitted run_forward...")
    run_forward = make_run_forward_jitted(bundle)

    print("[smoke] First call (compile + run, may take several minutes)...")
    t0 = time.time()
    pred = run_forward(jax.random.PRNGKey(0), eval_inputs, eval_targets, eval_forcings)
    jax.block_until_ready(pred)
    pred = pred.compute() if hasattr(pred, "compute") else pred
    print(f"[smoke]   first call wall time: {time.time() - t0:.1f}s")
    print("[smoke]   pred dims:", dict(pred.sizes))
    print("[smoke]   pred vars:", list(pred.data_vars))

    t = pred["2m_temperature"].values
    print(
        f"[smoke]   2m_temperature stats: shape={t.shape} "
        f"min={np.nanmin(t):.2f} mean={np.nanmean(t):.2f} max={np.nanmax(t):.2f}"
    )

    print("[smoke] Second call (cached compile)...")
    t0 = time.time()
    second = run_forward(
        jax.random.PRNGKey(1),
        eval_inputs,
        eval_targets,
        eval_forcings,
    )
    jax.block_until_ready(second)
    print(f"[smoke]   second call wall time: {time.time() - t0:.1f}s")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
