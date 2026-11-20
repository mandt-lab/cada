from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cada.data import load_channel_minmax
from cada.preprocessing import (
    KSConfig,
    KSSplit,
    compute_dense_h5_channel_stats,
    compute_ks_training_minmax,
    generate_ks_hdf5,
    write_dense_channel_stats,
    write_ks_minmax_json,
)


def test_dense_training_stats_are_channelwise_and_loader_compatible(
    tmp_path: Path,
) -> None:
    values = np.empty((2, 2, 2, 2, 2), dtype=np.float32)
    ramp = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    values[:, :, 0] = ramp
    values[:, :, 1] = 100 + 2 * ramp
    h5_path = tmp_path / "dense.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("x", data=values)

    stats = compute_dense_h5_channel_stats(h5_path, batch_size=1)
    stats_path = write_dense_channel_stats(stats, tmp_path / "stats.pt")
    normalizer = load_channel_minmax(stats_path)

    assert stats.shape == values.shape
    assert stats.minimum == (0.0, 100.0)
    assert stats.maximum == (15.0, 130.0)
    assert torch.equal(normalizer.minimum, torch.tensor([0.0, 100.0]))
    assert torch.equal(normalizer.maximum, torch.tensor([15.0, 130.0]))


def test_tiny_ks_generation_schema_and_training_stats(tmp_path: Path) -> None:
    config = KSConfig(
        t_end=0.02,
        domain_length=8.0,
        spatial_points=8,
        integration_steps=4,
        saved_steps=3,
        initial_modes=2,
        mode_min=1,
        mode_max_exclusive=2,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
        h5_dtype="float32",
    )
    path = tmp_path / "ks.h5"

    result = generate_ks_hdf5(path, KSSplit(name="train", samples=1, seed=7), config)

    assert result.dataset == "train/pde_3-8"
    assert result.shape == (1, 3, 8)
    with h5py.File(path, "r") as handle:
        group = handle["train"]
        assert set(group) == {"dt", "dx", "pde_3-8", "t", "x"}
        assert group["pde_3-8"].shape == (1, 3, 8)
        assert group["x"].shape == (1, 8)
        assert group["t"].shape == (1, 3)

    stats = compute_ks_training_minmax(path)
    stats_path = write_ks_minmax_json(stats, tmp_path / "ks-stats.json")
    document = json.loads(stats_path.read_text(encoding="utf-8"))
    assert document["dataset"] == "train/pde_3-8"
    assert document["shape"] == [1, 3, 8]
    assert document["abs_min"] < document["abs_max"]
