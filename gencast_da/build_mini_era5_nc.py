"""Build a GenCast 1p0deg Mini-compatible ERA5 NetCDF for an arbitrary window.

Reads from WeatherBench2's public 6-hour, 13-level, 0.25-degree ERA5 Zarr,
selects a time window, sub-samples the grid to 1deg by `isel(::4, ::4)`
(matches DeepMind's official 1deg construction), reformats to match the
2019 demo file's exact schema (variable list, dim names + ordering,
batch dim, timedelta time coord, datetime coord, dtypes, level int32),
adds year/day_progress forcings via `graphcast.data_utils.add_derived_vars`,
copies static fields (`land_sea_mask`, `geopotential_at_surface`) and the
SST land-NaN mask from a reference Mini file, and writes a single NetCDF.

Default builds Feb-Mar 2022 (118 12h target frames + 2 input frames):

  python -m gencast_da.build_mini_era5_nc \
      --start 2022-02-01 --end 2022-03-31 \
      --out "$GENCAST_DATA_ROOT"/dataset/\
source-era5_date-2022-01-31_res-1.0_levels-13_steps-118.nc \
      --validate

Memory: pressure-var chunks in the WB2 0.25deg Zarr are 54 MB per
12h timestep, so a 60-day window pulls ~40 GB across the network
before writing the ~1.4 GB Mini file. Use --start/--end smaller to
test the pipeline first.

Network access: tries anonymous GCS first (gcsfs token='anon'), then
plain HTTPS via fsspec. If both fail, see --source for alternatives.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# Public WeatherBench2 ERA5 Zarr (6h, 13 levels, 0.25deg with poles).
WB2_BUCKET_PATH = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
WB2_URL_GCS = "gs://" + WB2_BUCKET_PATH
WB2_URL_HTTPS = "https://storage.googleapis.com/" + WB2_BUCKET_PATH

# Mini contract (matches the 2019 demo file exactly).
MINI_LEVELS: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)
MINI_SURFACE_VARS: tuple[str, ...] = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
)
MINI_PRECIP_VAR = "total_precipitation_12hr"
MINI_PRESSURE_VARS: tuple[str, ...] = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
MINI_STATIC_VARS: tuple[str, ...] = (
    "geopotential_at_surface",
    "land_sea_mask",
)
MINI_FORCING_VARS: tuple[str, ...] = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)


def _load_nc(path: Path) -> xr.Dataset:
    """Load a NetCDF file by path, trying engines in order.

    Passing the path string (not a file handle) lets xarray use the .nc
    extension hint to guess the engine. We retry h5netcdf / netcdf4
    explicitly so we don't depend on which backend wrote the file.
    """
    last_err: Exception | None = None
    for engine in (None, "h5netcdf", "netcdf4", "scipy"):
        try:
            return xr.load_dataset(str(path), engine=engine)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"Failed to load {path} with any NetCDF engine; last error: {last_err}"
    )


def _open_wb2_zarr() -> xr.Dataset:
    """Open the public WB2 ERA5 Zarr lazily.

    Tries multiple access paths because the cryptography wheel in this
    env is built against newer glibc and `gcsfs` default auth may fail.
    Anonymous GCS access is fine since the bucket is public.
    """
    last_err: Exception | None = None
    attempts = [
        (
            "gcs anonymous",
            WB2_URL_GCS,
            dict(consolidated=True, storage_options={"token": "anon"}),
        ),
        (
            "gcs default",
            WB2_URL_GCS,
            dict(consolidated=True),
        ),
        (
            "https via fsspec",
            WB2_URL_HTTPS,
            dict(consolidated=True),
        ),
    ]
    for label, path, kwargs in attempts:
        try:
            print(f"[build] trying open_zarr ({label})  {path}")
            ds = xr.open_zarr(path, **kwargs)
            print(f"[build] opened via {label}: vars={len(ds.data_vars)}")
            return ds
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(
                f"[build]   {label} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
    raise RuntimeError(
        "All open_zarr access paths failed; check network/gcsfs install. "
        f"Last error: {last_err}"
    )


def _select_window(
    ds: xr.Dataset,
    start_dt: np.datetime64,
    end_dt: np.datetime64,
) -> tuple[xr.Dataset, str, str, str]:
    """Lazy-select required vars, time window, levels, sub-sample to 1deg.

    Returns (sub_dataset, lat_dim_name, lon_dim_name, precip_mode).
    `precip_mode` is "12hr" if the source already has it, else "6hr".
    """
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    # 12h cadence: include 2 frames before start_dt as input window.
    init_dt = start_dt - np.timedelta64(24, "h")
    times_12h = np.arange(
        init_dt,
        end_dt + np.timedelta64(12, "h"),
        np.timedelta64(12, "h"),
    )

    if "total_precipitation_12hr" in ds.data_vars:
        precip_mode = "12hr"
        precip_keep = ["total_precipitation_12hr"]
    elif "total_precipitation_6hr" in ds.data_vars:
        precip_mode = "6hr"
        precip_keep = ["total_precipitation_6hr"]
    else:
        raise RuntimeError(
            "Source Zarr lacks total_precipitation_12hr and _6hr; "
            f"have data_vars={sorted(ds.data_vars)[:30]}..."
        )
    print(f"[build] precipitation source: {precip_mode}")

    keep = (
        list(MINI_SURFACE_VARS)
        + list(MINI_PRESSURE_VARS)
        + [v for v in MINI_STATIC_VARS if v in ds.data_vars]
        + precip_keep
    )
    missing = [v for v in keep if v not in ds.data_vars]
    if missing:
        raise RuntimeError(f"Source Zarr is missing required Mini vars: {missing}")

    sub = ds[keep].sel(level=list(MINI_LEVELS))

    # Sub-sample 0.25deg -> 1deg using every 4th lat/lon point. This is
    # exactly the recipe DeepMind used to make the 1deg demo data
    # (per https://github.com/google-deepmind/graphcast/issues/131).
    sub = sub.isel(
        {
            lat_name: slice(None, None, 4),
            lon_name: slice(None, None, 4),
        }
    )

    if precip_mode == "12hr":
        sub = sub.sel(time=times_12h)
    else:
        # Keep all required vars at 12h cadence; pull 6h precip on a
        # superset of times so we can build 12h accumulations afterward.
        non_precip = [v for v in keep if v != "total_precipitation_6hr"]
        sub_12h = sub[non_precip].sel(time=times_12h)
        # 6h precip needed for time t (12h sum ending at t) is at
        # times {t-6h, t}. Build full 6h time grid covering both.
        times_6h = np.arange(
            init_dt - np.timedelta64(6, "h"),
            end_dt + np.timedelta64(12, "h"),
            np.timedelta64(6, "h"),
        )
        precip_6h = sub["total_precipitation_6hr"].sel(time=times_6h)
        sub = xr.merge([sub_12h, precip_6h.to_dataset()])

    return sub, lat_name, lon_name, precip_mode


def _build_12h_precip_from_6h(precip_6h: xr.DataArray, times_12h):
    """12h precip ending at t = 6h precip ending at (t-6h) + at t."""
    parts = []
    for t in times_12h:
        t_prev = t - np.timedelta64(6, "h")
        p = precip_6h.sel(time=t_prev) + precip_6h.sel(time=t)
        parts.append(p.expand_dims(time=[t]))
    out = xr.concat(parts, dim="time")
    out.name = "total_precipitation_12hr"
    return out


def _to_mini_schema(
    ds: xr.Dataset,
    lat_name: str,
    lon_name: str,
    precip_mode: str,
    init_datetime: np.datetime64,
) -> xr.Dataset:
    """Reformat to match the 2019 demo file: dim names, ordering,
    batch dim, timedelta time coord, datetime coord, dtypes."""
    rename_map = {}
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    # Lat ascending (-90 -> 90). 2019 file uses ascending; ERA5 native is descending.
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.reindex(lat=ds["lat"].values[::-1])

    # Coord dtypes: 2019 file has lat/lon float32, level int32, time timedelta64[ns].
    ds = ds.assign_coords(
        lat=np.asarray(np.round(ds["lat"].values), dtype=np.float32),
        lon=np.asarray(np.round(ds["lon"].values), dtype=np.float32),
        level=np.asarray(ds["level"].values, dtype=np.int32),
    )

    # If we kept 6h precip, build 12h precip now (after grid reshape).
    if precip_mode == "6hr":
        precip_12h_times = np.arange(
            np.datetime64(init_datetime, "ns"),
            ds["time"].values[-1] + np.timedelta64(1, "ns"),
            np.timedelta64(12, "h"),
        )
        # Drop frames in `time` that aren't multiples of 12h (the 6h scaffold).
        precip_12h = _build_12h_precip_from_6h(
            ds["total_precipitation_6hr"],
            precip_12h_times,
        )
        ds = ds.drop_vars("total_precipitation_6hr")
        # Re-align all other vars to the 12h time set in case they had
        # the broader 6h time range from the merge.
        ds = ds.sel(time=precip_12h_times)
        ds[MINI_PRECIP_VAR] = precip_12h.transpose("time", "lat", "lon")

    # Save absolute datetime, then convert time to timedelta from init_datetime.
    abs_times = ds["time"].values  # datetime64[ns]
    init_ns = np.datetime64(init_datetime, "ns")
    td_times = (abs_times - init_ns).astype("timedelta64[ns]")
    ds = ds.assign_coords(time=td_times)

    # Add batch dim of size 1 to vars that should have it.
    needs_batch = list(MINI_SURFACE_VARS) + [MINI_PRECIP_VAR] + list(MINI_PRESSURE_VARS)
    for v in needs_batch:
        if v in ds.data_vars and "batch" not in ds[v].dims:
            ds[v] = ds[v].expand_dims(batch=1)

    # Add datetime coord (batch, time).
    dt_coord = xr.DataArray(abs_times[None, :], dims=("batch", "time"))
    ds = ds.assign_coords(datetime=dt_coord)

    # Reorder dims to match the 2019 file exactly.
    for v in MINI_SURFACE_VARS + (MINI_PRECIP_VAR,):
        if v in ds.data_vars:
            ds[v] = ds[v].transpose("batch", "time", "lat", "lon")
    for v in MINI_PRESSURE_VARS:
        if v in ds.data_vars:
            ds[v] = ds[v].transpose("batch", "time", "level", "lat", "lon")
    for v in MINI_STATIC_VARS:
        if v in ds.data_vars:
            ds[v] = ds[v].transpose("lat", "lon")

    # Cast all data vars to float32; restore int32 level coord.
    for v in list(ds.data_vars):
        if ds[v].dtype != np.float32:
            ds[v] = ds[v].astype(np.float32)
    ds = ds.assign_coords(level=np.asarray(ds["level"].values, dtype=np.int32))

    return ds


def _add_forcings(ds: xr.Dataset) -> xr.Dataset:
    """Compute year/day_progress forcings via the official graphcast helper."""
    from graphcast import data_utils

    ds = ds.copy()
    data_utils.add_derived_vars(ds)
    # Cast to float32 (helper writes float64 sometimes).
    for v in MINI_FORCING_VARS:
        if v in ds.data_vars and ds[v].dtype != np.float32:
            ds[v] = ds[v].astype(np.float32)
    # Drop the bare "year_progress" / "day_progress" (we only need _sin/_cos).
    for v in ("year_progress", "day_progress"):
        if v in ds.data_vars:
            ds = ds.drop_vars(v)
    return ds


def _copy_static_and_sst_mask_from(ds: xr.Dataset, ref_path: Path) -> xr.Dataset:
    """Take time-invariant fields from a reference Mini file.

    `geopotential_at_surface` and `land_sea_mask` are constant in time,
    and `sea_surface_temperature` requires the reference Mini's NaN
    land mask (otherwise NaNCleaner / GenCast will treat land cells as
    valid and the model goes out-of-distribution). Pulling them
    directly from the 2019 file guarantees byte-identical statics.
    """
    if not ref_path.exists():
        print(
            f"[build] WARN reference Mini file not found at {ref_path}; "
            f"skipping static-var + SST-mask copy. Output will likely "
            f"NOT pass data_contract validation."
        )
        return ds
    ref = _load_nc(ref_path)

    for v in MINI_STATIC_VARS:
        if v not in ref.data_vars:
            print(f"[build] WARN reference is missing {v}")
            continue
        ds[v] = ref[v]
        print(f"[build] copied static {v} from reference")

    if "sea_surface_temperature" in ref.data_vars:
        sst_ref0 = ref["sea_surface_temperature"].isel(batch=0, time=0).values
        land_mask = np.isnan(sst_ref0)
        n_land = int(land_mask.sum())
        print(
            f"[build] applying reference SST land-NaN mask: "
            f"{n_land} cells ({n_land / land_mask.size:.1%})"
        )
        mask_da = xr.DataArray(land_mask, dims=("lat", "lon"))
        ds["sea_surface_temperature"] = ds["sea_surface_temperature"].where(~mask_da)

    return ds


def _selftest_6h_to_12h_precip() -> bool:
    """Verify the 6h -> 12h precip construction with a synthetic input."""
    times_6h = np.arange(
        np.datetime64("2022-01-30T18:00:00", "ns"),
        np.datetime64("2022-02-01T12:00:00", "ns") + np.timedelta64(1, "ns"),
        np.timedelta64(6, "h"),
    )
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 0.05, size=(len(times_6h), 4, 8)).astype(np.float32)
    da6 = xr.DataArray(
        vals,
        dims=("time", "lat", "lon"),
        coords={"time": times_6h},
    )
    times_12h = np.arange(
        np.datetime64("2022-01-31T00:00:00", "ns"),
        np.datetime64("2022-02-01T12:00:00", "ns") + np.timedelta64(1, "ns"),
        np.timedelta64(12, "h"),
    )
    da12 = _build_12h_precip_from_6h(da6, times_12h).values
    expected = np.stack(
        [
            da6.sel(time=t - np.timedelta64(6, "h")).values + da6.sel(time=t).values
            for t in times_12h
        ]
    )
    if not np.allclose(da12, expected):
        print(
            f"  FAIL: 6h->12h precip selftest "
            f"max|diff|={float(np.abs(da12 - expected).max()):.2e}"
        )
        return False
    print(
        f"  OK: 6h->12h precip selftest (n_times={len(times_12h)}, "
        f"max_val={float(da12.max()):.3f})"
    )
    return True


def _validate(out_path: Path, ref_path: Path) -> bool:
    """Check that the new file matches the Mini contract exactly."""
    print()
    print("=" * 60)
    print("VALIDATION  (against 2019 Mini reference)")
    print("=" * 60)
    new = _load_nc(out_path)
    ref = _load_nc(ref_path)

    ok = True

    # Variable set must be a superset of all required vars.
    required = (
        set(MINI_SURFACE_VARS)
        | {MINI_PRECIP_VAR}
        | set(MINI_PRESSURE_VARS)
        | set(MINI_STATIC_VARS)
        | set(MINI_FORCING_VARS)
    )
    missing = required - set(new.data_vars)
    if missing:
        print(f"  FAIL: missing data_vars: {sorted(missing)}")
        ok = False
    else:
        print(f"  OK: all {len(required)} required vars present")

    # Per-variable dim and dtype check vs reference.
    for v in sorted(required):
        if v not in ref.data_vars:
            print(f"  WARN: {v} not in reference file (skipping check)")
            continue
        if new[v].dims != ref[v].dims:
            print(f"  FAIL: {v} dims {new[v].dims} != ref {ref[v].dims}")
            ok = False
        if new[v].dtype != ref[v].dtype:
            print(f"  FAIL: {v} dtype {new[v].dtype} != ref {ref[v].dtype}")
            ok = False

    # Coord checks.
    for c in ("lat", "lon", "level"):
        if list(new[c].values) != list(ref[c].values):
            print(f"  FAIL: coord {c} values differ from reference")
            ok = False
        elif new[c].dtype != ref[c].dtype:
            print(f"  FAIL: coord {c} dtype {new[c].dtype} != ref {ref[c].dtype}")
            ok = False

    # Time must be monotonically increasing 12h timedeltas starting at 0.
    t = new["time"].values.astype("timedelta64[s]").astype(np.int64)
    deltas = np.diff(t)
    if not (deltas == 12 * 3600).all():
        print(f"  FAIL: time deltas not all 12h ({np.unique(deltas)})")
        ok = False
    if t[0] != 0:
        print(f"  FAIL: time[0] != 0 (got {t[0]} seconds)")
        ok = False

    # SST mask sanity: must be constant in time AND match the reference Mini
    # mask exactly, because GenCast's NaNCleaner expects the same land cells.
    if "sea_surface_temperature" in new.data_vars:
        sst_t0 = new["sea_surface_temperature"].isel(batch=0, time=0).values
        sst_tN = new["sea_surface_temperature"].isel(batch=0, time=-1).values
        ref_sst0 = ref["sea_surface_temperature"].isel(batch=0, time=0).values
        ref_mask = np.isnan(ref_sst0)
        new_mask_t0 = np.isnan(sst_t0)
        new_mask_tN = np.isnan(sst_tN)
        if not np.array_equal(new_mask_t0, new_mask_tN):
            print("  FAIL: SST land-NaN mask is not constant in time")
            ok = False
        elif not np.array_equal(new_mask_t0, ref_mask):
            n_diff = int((new_mask_t0 != ref_mask).sum())
            print(
                f"  FAIL: SST land-NaN mask differs from reference "
                f"({n_diff} cells differ)"
            )
            ok = False
        else:
            n_nan = int(new_mask_t0.sum())
            print(
                f"  OK: SST land mask constant in time, matches reference "
                f"({n_nan} NaN cells)"
            )
        # Sanity range on non-NaN SST (Kelvin, ocean only).
        valid = sst_t0[~new_mask_t0]
        if valid.size and (valid.min() < 250.0 or valid.max() > 320.0):
            print(
                f"  WARN: SST(non-NaN) range [{valid.min():.1f},"
                f" {valid.max():.1f}] K is outside expected [250, 320]"
            )

    # Static fields must be exactly the reference values (time-invariant).
    for v in MINI_STATIC_VARS:
        if v not in new.data_vars or v not in ref.data_vars:
            continue
        if not np.array_equal(new[v].values, ref[v].values):
            n_diff = int((new[v].values != ref[v].values).sum())
            print(f"  FAIL: static {v} differs from reference ({n_diff} cells)")
            ok = False
        else:
            print(f"  OK: static {v} matches reference exactly")

    # Precipitation: 12hr accumulation in meters; ERA5 has tiny negative
    # numerical noise but is bounded above by ~0.5 m even in extremes.
    if MINI_PRECIP_VAR in new.data_vars:
        tp = new[MINI_PRECIP_VAR].values
        if not np.isfinite(tp).all():
            n_bad = int((~np.isfinite(tp)).sum())
            print(f"  FAIL: total_precipitation_12hr has {n_bad} non-finite cells")
            ok = False
        else:
            tp_min, tp_max = float(tp.min()), float(tp.max())
            if tp_min < -1e-3 or tp_max > 1.0:
                print(
                    f"  FAIL: total_precipitation_12hr range "
                    f"[{tp_min:.2e}, {tp_max:.2e}] looks wrong "
                    f"(expected ~[0, 0.5] m)"
                )
                ok = False
            else:
                print(
                    f"  OK: total_precipitation_12hr range "
                    f"[{tp_min:.2e}, {tp_max:.2e}] m, all finite"
                )

    # Try the official extract pipeline at a target_time_idx in the middle.
    print()
    print("Running graphcast.data_utils.extract_inputs_targets_forcings...")
    try:
        from graphcast import data_utils

        from gencast_da.gencast_runner import load_bundle

        bundle = load_bundle()
        n_t = new.sizes["time"]
        target_idx = min(5, n_t - 1)
        slice_indices = [target_idx - 2, target_idx - 1, target_idx]
        base = new.isel(time=slice_indices)
        import dataclasses

        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            base,
            target_lead_times=slice("12h", "12h"),
            **dataclasses.asdict(bundle.task_config),
        )
        print(
            f"  OK: extract returned inputs={len(inputs.data_vars)} vars, "
            f"targets={len(targets.data_vars)} vars, "
            f"forcings={len(forcings.data_vars)} vars  "
            f"at target_idx={target_idx}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: extract raised {type(e).__name__}: {e}")
        ok = False

    print()
    print("Self-tests:")
    if not _selftest_6h_to_12h_precip():
        ok = False

    print()
    print(f"VALIDATION RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _print_summary(out_path: Path) -> None:
    ds = _load_nc(out_path)
    print()
    print("=" * 60)
    print(f"OUTPUT: {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    print("=" * 60)
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  data_vars ({len(ds.data_vars)}): {sorted(ds.data_vars)}")
    print(
        f"  datetime: {ds.datetime.isel(batch=0).values[0]}"
        f" ... {ds.datetime.isel(batch=0).values[-1]}"
    )
    print(
        f"  valid target indices [2..{ds.sizes['time'] - 1}] = "
        f"{ds.sizes['time'] - 2} transitions"
    )


def main():
    data_root = Path(
        os.environ.get(
            "GENCAST_DATA_ROOT",
            os.path.join(os.environ.get("CADA_ROOT", ""), "data", "gencast"),
        )
    )
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--start",
        type=str,
        default="2022-02-01",
        help="First TARGET frame date (00:00 UTC).",
    )
    p.add_argument(
        "--end",
        type=str,
        default="2022-03-31",
        help="Last TARGET frame date (12:00 UTC).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(
            data_root
            / "dataset"
            / "source-era5_date-2022-01-31_res-1.0_levels-13_steps-118.nc"
        ),
        help="Output NetCDF path. Default keeps Mini naming convention.",
    )
    p.add_argument(
        "--ref-mini-nc",
        type=str,
        default=str(
            data_root
            / "dataset"
            / "source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"
        ),
        help="Existing Mini-compatible NetCDF used to copy time-invariant "
        "static fields (land_sea_mask, geopotential_at_surface) and the "
        "SST land-NaN mask, and to validate the new file's schema.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run schema + extract_inputs_targets_forcings checks after build.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip building; just validate an existing file at --out.",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    ref_path = Path(args.ref_mini_nc)

    if args.validate_only:
        _print_summary(out_path)
        ok = _validate(out_path, ref_path)
        sys.exit(0 if ok else 1)

    start_dt = np.datetime64(args.start + "T00:00:00", "ns")
    end_dt = np.datetime64(args.end + "T12:00:00", "ns")
    init_datetime = start_dt - np.timedelta64(24, "h")
    n_target = int((end_dt - start_dt) / np.timedelta64(12, "h")) + 1
    print(
        f"[build] window: init={init_datetime}, target_range="
        f"[{start_dt}, {end_dt}] ({n_target} target frames @ 12h)"
    )

    t0 = time.time()
    ds_raw = _open_wb2_zarr()
    print(f"[build] WB2 dims: {dict(ds_raw.sizes)}")

    sub, lat_name, lon_name, precip_mode = _select_window(
        ds_raw,
        start_dt,
        end_dt,
    )
    print(f"[build] selected lazy: {dict(sub.sizes)}")

    print("[build] materializing (this is the slow network step)...")
    t1 = time.time()
    sub = sub.compute()
    print(
        f"[build] materialized in {time.time() - t1:.1f}s; "
        f"raw nbytes={sum(v.nbytes for v in sub.data_vars.values()) / 1e9:.2f} GB"
    )

    print("[build] reformatting to Mini schema...")
    sub = _to_mini_schema(sub, lat_name, lon_name, precip_mode, init_datetime)

    print("[build] copying static fields + SST mask from reference Mini file...")
    sub = _copy_static_and_sst_mask_from(sub, ref_path)

    print("[build] adding year/day_progress forcings...")
    sub = _add_forcings(sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build] writing {out_path}...")
    encoding = {v: {"dtype": "float32"} for v in sub.data_vars}
    sub.to_netcdf(out_path, encoding=encoding)
    print(f"[build] DONE in {time.time() - t0:.1f}s total")

    _print_summary(out_path)
    if args.validate:
        ok = _validate(out_path, ref_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
