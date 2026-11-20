"""Deterministic JAX key derivations shared by GenCast training and tests."""

from __future__ import annotations

import jax


def validation_key(root_key, example_index: int, rollout_index: int = 0):
    if example_index < 0 or rollout_index < 0:
        raise ValueError("validation indices must be non-negative")
    return jax.random.fold_in(
        root_key,
        99_991 + example_index * 100 + rollout_index,
    )
