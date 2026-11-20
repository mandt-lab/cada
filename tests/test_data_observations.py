from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from cada.data import (
    DenseH5TrajectoryDataset,
    KSH5TrajectoryDataset,
    MinMaxNormalizer,
    PreviewWindowDataset,
    ResumableRandomSampler,
)
from cada.observations import (
    AverageDownsample1D,
    AverageDownsample2D,
    BernoulliMask,
    RegularGridMask,
    TemporalCadence,
)


def test_minmax_normalization_round_trip_and_optional_clipping() -> None:
    normalizer = MinMaxNormalizer([0.0, 10.0], [2.0, 14.0], clip=True)
    value = torch.tensor([[[[-1.0]], [[15.0]]]])

    normalized = normalizer.normalize(value, channel_dim=1)

    assert torch.equal(normalized, torch.tensor([[[[0.0]], [[1.0]]]]))
    assert torch.equal(
        normalizer.denormalize(normalized, channel_dim=1),
        torch.tensor([[[[0.0]], [[14.0]]]]),
    )
    unclipped = MinMaxNormalizer([0.0], [2.0])
    outside = torch.tensor([[[-1.0, 3.0]]])
    normalized_outside = unclipped.normalize(outside, channel_dim=1)
    assert torch.equal(normalized_outside, torch.tensor([[[-0.5, 1.5]]]))
    assert torch.equal(
        unclipped.denormalize(normalized_outside, channel_dim=1),
        outside,
    )


def test_lazy_dense_and_grouped_ks_hdf5_schemas(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.h5"
    dense_values = np.arange(2 * 4 * 2 * 2 * 2, dtype=np.float32).reshape(2, 4, 2, 2, 2)
    with h5py.File(dense_path, "w") as handle:
        handle.create_dataset("x", data=dense_values)
    dense_stats = MinMaxNormalizer(
        dense_values.min(axis=(0, 1, 3, 4)),
        dense_values.max(axis=(0, 1, 3, 4)),
    )
    dense = DenseH5TrajectoryDataset(dense_path, dense_stats)
    assert dense._h5 is None
    assert dense.shape == (2, 4, 2, 2, 2)
    assert dense._h5 is None
    dense_window = dense.read_window(0, 1, 3)
    assert dense_window.shape == (2, 2, 2, 2)
    assert dense._h5 is not None
    assert torch.all((0 <= dense_window) & (dense_window <= 1))
    dense.close()

    ks_path = tmp_path / "ks.h5"
    ks_values = np.linspace(-2, 2, 2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    with h5py.File(ks_path, "w") as handle:
        group = handle.create_group("train")
        group.create_dataset("pde_4-6", data=ks_values)
    ks_stats = tmp_path / "ks.json"
    ks_stats.write_text(json.dumps({"abs_min": -2.0, "abs_max": 2.0}), encoding="utf-8")
    ks = KSH5TrajectoryDataset(ks_path, ks_stats, split="train")
    assert ks._h5 is None
    assert ks.dataset_key == "train/pde_4-6"
    assert ks._h5 is None
    assert ks[0].shape == (4, 1, 6)
    assert torch.all((0 <= ks[0]) & (ks[0] <= 1))
    ks.close()


def test_ds_ms_bernoulli_operators_and_temporal_cadence() -> None:
    state_1d = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])
    downsampled_1d = AverageDownsample1D(2)(state_1d)
    assert torch.equal(downsampled_1d.value, torch.tensor([[[2.0, 2.0, 6.0, 6.0]]]))
    assert torch.all(downsampled_1d.mask == 1)

    state_2d = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    downsampled_2d = AverageDownsample2D(2)(state_2d)
    assert torch.equal(
        downsampled_2d.value[0, 0, :2, :2],
        torch.full((2, 2), 2.5),
    )

    masked = RegularGridMask(spatial_dims=2, factor=2, phase=(1, 0))(state_2d)
    expected_mask = torch.zeros_like(state_2d)
    expected_mask[..., 1::2, 0::2] = 1
    assert torch.equal(masked.mask, expected_mask)
    assert torch.equal(masked.value, state_2d * expected_mask)

    bernoulli = BernoulliMask(spatial_dims=2, keep_probability=0.5, seed=17)
    state = torch.ones(2, 3, 4, 4)
    first = bernoulli(state)
    second = bernoulli(state)
    assert not torch.equal(first.mask, second.mask)
    assert torch.equal(first.mask[0, 0], first.mask[1, 2])
    bernoulli.reset()
    assert torch.equal(bernoulli(state).mask, first.mask)
    checkpoint_state = bernoulli.state_dict()
    expected_next = bernoulli(state).mask
    restored = BernoulliMask(
        spatial_dims=2,
        keep_probability=0.5,
        seed=999,
    )
    restored.load_state_dict(checkpoint_state)
    assert torch.equal(restored(state).mask, expected_next)

    cadence = TemporalCadence(interval=3, start=2)
    assert cadence.indices(10) == (2, 5, 8)
    assert not cadence.is_observation_time(4)
    assert cadence.is_observation_time(5)


def test_preview_window_count_locations_and_contents() -> None:
    class Source:
        n_trajectories = 2
        trajectory_length = 10

        def read_window(self, trajectory: int, start: int, stop: int) -> torch.Tensor:
            return (
                torch.arange(start, stop, dtype=torch.float32)
                + 100 * trajectory
            ).reshape(-1, 1, 1)

        def close(self) -> None:
            pass

    dataset = PreviewWindowDataset(Source(), horizon=4, stride=3)

    assert len(dataset) == 4
    assert [dataset.locate(index) for index in range(4)] == [
        (0, 0),
        (0, 3),
        (1, 0),
        (1, 3),
    ]
    assert torch.equal(dataset[0].flatten(), torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(dataset[3].flatten(), torch.tensor([103, 104, 105, 106, 107]))

    sampler = ResumableRandomSampler(
        dataset,
        torch.Generator().manual_seed(0),
    )
    with pytest.raises(ValueError, match="zero without a permutation"):
        sampler.load_state_dict(
            {
                "generator_state": sampler.generator.get_state(),
                "permutation": None,
                "position": 1,
            }
        )
