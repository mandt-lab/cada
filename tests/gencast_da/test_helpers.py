from __future__ import annotations

import numpy as np
import pytest
import xarray
from gencast_da.obs_operator import ObsConfig, apply, build_obs_mask
from gencast_da.observation_arrays import make_obs_arrays
from gencast_da.rollout_utils import (
    partition_rmse,
    update_memory,
    weighted_rmse,
)


def _state() -> xarray.Dataset:
    values = np.arange(2 * 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 2, 4, 4)
    return xarray.Dataset(
        {
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                values,
            )
        },
        coords={
            "batch": [0],
            "time": [0, 1],
            "level": [500, 850],
            "lat": [-1.0, 0.0, 1.0, 2.0],
            "lon": [0.0, 1.0, 2.0, 3.0],
        },
    )


def test_observation_arrays_match_stride_mask() -> None:
    state = _state()
    config = ObsConfig(lat_stride_deg=2, lon_stride_deg=2, noise_std=0.1)
    physical = state["temperature"].isel(time=[1], lat=[0, 2], lon=[0, 2])
    differences = xarray.Dataset(
        {"temperature": (("level",), np.array([2.0, 4.0], dtype=np.float32))},
        coords={"level": [500, 850]},
    )

    observation, mask, normalized_noise = make_obs_arrays(
        state,
        physical,
        differences,
        config,
    )

    expected_mask = build_obs_mask(state, config)
    assert observation.shape == (1, 1, 2, 4, 4)
    assert np.array_equal(mask[0, 0, 0], expected_mask)
    assert np.allclose(normalized_noise, [0.05, 0.025])


def test_observation_arrays_align_coordinate_labels() -> None:
    state = _state()
    config = ObsConfig(lat_stride_deg=2, lon_stride_deg=2, noise_std=0.1)
    physical = state["temperature"].isel(time=[1], lat=[0, 2], lon=[0, 2])
    differences = xarray.Dataset(
        {"temperature": (("level",), np.array([2.0, 4.0], dtype=np.float32))},
        coords={"level": [500, 850]},
    )
    expected = make_obs_arrays(state, physical, differences, config)

    reordered = physical.isel(
        level=[1, 0],
        lat=[1, 0],
        lon=[1, 0],
    )
    reordered_differences = differences.isel(level=[1, 0])
    actual = make_obs_arrays(
        state,
        reordered,
        reordered_differences,
        config,
    )

    for expected_array, actual_array in zip(expected, actual):
        assert np.array_equal(expected_array, actual_array)

    with pytest.raises(ValueError, match="exactly one time"):
        make_obs_arrays(
            state,
            state["temperature"],
            differences,
            config,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"lat_stride_deg": 0},
        {"lon_stride_deg": -1},
        {"mask_mode": "unknown"},
        {"random_mask_fraction": 0.0},
        {"random_mask_fraction": 1.1},
        {"random_mask_fraction": float("nan")},
        {"noise_std": 0.0},
    ],
)
def test_invalid_observation_configs_are_rejected(arguments: dict) -> None:
    with pytest.raises(ValueError):
        ObsConfig(**arguments)


def test_random_mask_has_exact_count_and_seed() -> None:
    state = _state().isel(lat=slice(0, 2), lon=slice(0, 3))
    config = ObsConfig(
        mask_mode="random",
        random_mask_fraction=0.5,
        random_mask_seed=0,
    )

    first = build_obs_mask(state, config)
    second = build_obs_mask(state, config)

    assert first.sum() == 3
    assert np.array_equal(first, second)
    assert set(np.flatnonzero(first)) == {3, 4, 5}


def test_nonuniform_observation_grid_is_rejected() -> None:
    state = _state().assign_coords(lat=[-1.0, 0.0, 1.5, 2.0])

    with pytest.raises(ValueError, match="uniform"):
        build_obs_mask(state, ObsConfig())


def test_observation_rng_is_reproducible_and_seed_sensitive() -> None:
    state = _state()
    config = ObsConfig()

    first = apply(state, config, np.random.default_rng(42))
    second = apply(state, config, np.random.default_rng(42))
    different = apply(state, config, np.random.default_rng(43))

    assert np.array_equal(first.values, second.values)
    assert not np.array_equal(first.values, different.values)


def test_rollout_helpers_update_memory_and_weight_rmse() -> None:
    state = _state()
    prediction = state.isel(time=[1]).assign_coords(time=[2])
    memory = update_memory(state, prediction, ["temperature"])

    assert memory.sizes["time"] == 2
    assert np.array_equal(
        memory["temperature"].isel(time=1).values,
        prediction["temperature"].isel(time=0).values,
    )
    truth = np.zeros((2, 3, 1), dtype=np.float32)
    forecast = np.ones_like(truth)
    assert weighted_rmse(truth, forecast, np.array([-60.0, 60.0])) == 1.0


def test_memory_requires_every_evolving_variable() -> None:
    state = _state().assign(
        extra=(
            _state()["temperature"].dims,
            _state()["temperature"].values,
        )
    )
    prediction = state[["temperature"]].isel(time=[1]).assign_coords(time=[2])

    with pytest.raises(ValueError, match="prediction.*extra"):
        update_memory(state, prediction, ["temperature", "extra"])


def test_three_step_memory_never_falls_back_to_truth() -> None:
    memory = _state()
    prediction_two = memory.isel(time=[1]).assign_coords(time=[2])
    prediction_two["temperature"] = (
        xarray.zeros_like(prediction_two["temperature"]) + 100
    )
    after_two = update_memory(memory, prediction_two, ["temperature"])
    prediction_three = prediction_two.assign_coords(time=[3])
    prediction_three["temperature"] = (
        xarray.zeros_like(prediction_three["temperature"]) + 200
    )
    after_three = update_memory(after_two, prediction_three, ["temperature"])

    assert np.all(after_two["temperature"].isel(time=1).values == 100)
    assert np.all(after_three["temperature"].isel(time=0).values == 100)
    assert np.all(after_three["temperature"].isel(time=1).values == 200)


def test_observed_truth_and_innovation_metrics_are_distinct() -> None:
    truth = np.zeros((1, 2, 2, 1), dtype=np.float32)
    prediction = truth.copy()
    observations = truth + 1
    mask = np.array([[True, False], [False, True]])

    assert partition_rmse(prediction, truth, mask, observed=True) == 0.0
    assert partition_rmse(prediction, observations, mask, observed=True) == 1.0


def test_partition_rmse_matches_global_count_weighting() -> None:
    truth = np.zeros((1, 2, 2, 1), dtype=np.float32)
    prediction = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    mask = np.array([[True, False], [False, True]])

    observed = partition_rmse(prediction, truth, mask, observed=True)
    unobserved = partition_rmse(prediction, truth, mask, observed=False)
    assert observed is not None and unobserved is not None
    reconstructed_mse = (
        mask.sum() * observed**2 + (~mask).sum() * unobserved**2
    ) / mask.size
    assert reconstructed_mse == pytest.approx(np.mean(prediction**2))
    assert (
        partition_rmse(
            prediction,
            truth,
            np.ones((2, 2), dtype=bool),
            observed=False,
        )
        is None
    )
