from __future__ import annotations

import numpy as np
import pytest
import xarray
from gencast_da.data_contract import (
    validate_example_dataset,
    validate_statistics,
)

LEVELS = (50, 100)


def _example() -> xarray.Dataset:
    time = np.arange(3) * np.timedelta64(12, "h")
    datetime = np.datetime64("2020-01-01") + time
    return xarray.Dataset(
        {
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                np.zeros((1, 3, 2, 181, 360), dtype=np.float32),
            )
        },
        coords={
            "batch": [0],
            "time": time,
            "datetime": (("batch", "time"), datetime[None]),
            "level": list(LEVELS),
            "lat": np.arange(-90, 91, dtype=np.float32),
            "lon": np.arange(360, dtype=np.float32),
        },
    )


def test_valid_gencast_dataset_contract() -> None:
    validate_example_dataset(
        _example(),
        expected_levels=LEVELS,
        expected_variables={"temperature"},
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.isel(lon=slice(None, None, -1)), "longitude"),
        (
            lambda data: data.assign_coords(time=np.arange(3) * np.timedelta64(6, "h")),
            "12-hour",
        ),
        (
            lambda data: xarray.concat([data, data], dim="batch"),
            "batch size",
        ),
        (lambda data: data.drop_vars("temperature"), "missing GenCast variables"),
        (
            lambda data: data.assign(
                temperature=data["temperature"].mean(("level", "lat", "lon"))
            ),
            "dimensions",
        ),
        (
            lambda data: data.assign(
                temperature=data["temperature"].astype(np.int32)
            ),
            "floating-point",
        ),
        (
            lambda data: data.assign_coords(
                datetime=data.coords["datetime"] + np.timedelta64(3, "h")
            ),
            "00/12 UTC",
        ),
    ],
)
def test_malformed_gencast_dataset_is_rejected(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_example_dataset(
            mutation(_example()),
            expected_levels=LEVELS,
            expected_variables={"temperature"},
        )


def test_statistics_levels_must_match_task_levels() -> None:
    statistics = xarray.Dataset(
        {"temperature": (("level",), np.ones(2, dtype=np.float32))},
        coords={"level": list(reversed(LEVELS))},
    )

    with pytest.raises(ValueError, match="misaligned"):
        validate_statistics((statistics,), expected_levels=LEVELS)

    nonfinite = statistics.assign(
        temperature=(("level",), np.array([1.0, np.nan]))
    ).assign_coords(level=list(LEVELS))
    with pytest.raises(ValueError, match="non-finite"):
        validate_statistics((nonfinite,), expected_levels=LEVELS)

    nonpositive = statistics.assign(
        temperature=(("level",), np.array([1.0, 0.0]))
    ).assign_coords(level=list(LEVELS))
    with pytest.raises(ValueError, match="strictly positive"):
        validate_statistics(
            (nonpositive,),
            expected_levels=LEVELS,
            require_positive=True,
        )

    malformed = xarray.Dataset(
        {"surface": (("lat",), np.ones(2, dtype=np.float32))},
        coords={"level": list(LEVELS), "lat": [0, 1]},
    )
    with pytest.raises(ValueError, match="scalar or level-only"):
        validate_statistics((malformed,), expected_levels=LEVELS)


def test_sst_allows_nan_mask_but_rejects_infinity() -> None:
    example = _example().drop_vars("temperature")
    values = np.zeros((1, 3, 181, 360), dtype=np.float32)
    values[:, :, 0, 0] = np.inf
    example["sea_surface_temperature"] = (
        ("batch", "time", "lat", "lon"),
        values,
    )

    with pytest.raises(ValueError, match="not infinity"):
        validate_example_dataset(
            example,
            expected_levels=LEVELS,
            expected_variables={"sea_surface_temperature"},
        )
