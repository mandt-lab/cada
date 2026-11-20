"""JAX-only CADA diagnostic reduction without GraphCast imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


def diagnostic_means(
    state: Mapping[str, Mapping[str, Any]],
) -> dict[str, jnp.ndarray]:
    """Return required diagnostic sums normalized by productive substeps."""

    values = {}
    for variables in state.values():
        for name in (
            "score_sum",
            "control_sum",
            "u_l2_sum",
            "substep_obs_sum",
            "substep_count",
        ):
            if name in variables:
                values[name] = variables[name]
    missing = sorted(
        {
            "score_sum",
            "control_sum",
            "u_l2_sum",
            "substep_obs_sum",
            "substep_count",
        }
        - set(values)
    )
    if missing:
        raise ValueError(f"missing CADA diagnostics: {missing}")
    count = jnp.maximum(values["substep_count"], 1.0)
    return {
        "score": values["score_sum"] / count,
        "control": values["control_sum"] / count,
        "u_l2": values["u_l2_sum"] / count,
        "substep_obs": values["substep_obs_sum"] / count,
        "substep_count": count,
    }
