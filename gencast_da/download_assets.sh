#!/usr/bin/env bash
# Download GenCast Mini checkpoint, GenCast normalization stats, and one
# example ERA5 1-degree NetCDF used as both the truth trajectory and the
# observation source.
#
# Default download root: $CADA_ROOT/data/gencast/
# Override with: GENCAST_DATA_ROOT=/some/other/dir bash gencast_da/download_assets.sh
#
# Requires `gsutil` from the Google Cloud SDK, OR will fall back to
# anonymous HTTPS via curl. The dm_graphcast bucket is public so no auth
# is required.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DATA_ROOT="${GENCAST_DATA_ROOT:-${CADA_ROOT:?set GENCAST_DATA_ROOT or CADA_ROOT}/data/gencast}"
mkdir -p "${DATA_ROOT}/params" "${DATA_ROOT}/stats" "${DATA_ROOT}/dataset"

CHECKPOINT_NAME="GenCast 1p0deg Mini <2019.npz"
STATS_FILES=(diffs_stddev_by_level.nc mean_by_level.nc stddev_by_level.nc min_by_level.nc)
EXAMPLE_NC="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"

GCS_BASE="https://storage.googleapis.com/dm_graphcast/gencast"

download() {
  local rel="$1"
  local dst="$2"
  local expected
  local actual
  local temporary
  expected="$(awk -v path="${rel}" 'substr($0, 67) == path {print $1}' \
    "${HERE}/assets.sha256")"
  [[ -n "${expected}" ]] || {
    echo "  missing checksum: ${rel}" >&2
    return 1
  }
  if [[ -f "${dst}" ]]; then
    actual="$(sha256sum "${dst}" | cut -d' ' -f1)"
    [[ "${actual}" == "${expected}" ]] || {
      echo "  checksum mismatch: ${dst}" >&2
      return 1
    }
    echo "  verified: ${dst}"
    return
  fi
  echo "  fetch:  ${rel} -> ${dst}"
  temporary="$(mktemp "${dst}.partial.XXXXXX")"
  trap 'rm -f -- "${temporary}"' RETURN
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -q cp "gs://dm_graphcast/gencast/${rel}" "${temporary}"
  else
    # URL-encode spaces and '<' in the rel path
    local enc; enc=${rel// /%20}
    enc=${enc//</%3C}
    curl -fsSL "${GCS_BASE}/${enc}" -o "${temporary}"
  fi
  actual="$(sha256sum "${temporary}" | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "  downloaded checksum mismatch: ${rel}" >&2
    return 1
  }
  mv -- "${temporary}" "${dst}"
  trap - RETURN
}

echo "[download] Checkpoint"
download "params/${CHECKPOINT_NAME}" "${DATA_ROOT}/params/${CHECKPOINT_NAME}"

echo "[download] Stats"
for f in "${STATS_FILES[@]}"; do
  download "stats/${f}" "${DATA_ROOT}/stats/${f}"
done

echo "[download] Example ERA5 NetCDF (30-step 1deg trajectory)"
download "dataset/${EXAMPLE_NC}" "${DATA_ROOT}/dataset/${EXAMPLE_NC}"

echo "[download] Done."
echo "DATA_ROOT=${DATA_ROOT}"
ls -lh "${DATA_ROOT}/params" "${DATA_ROOT}/stats" "${DATA_ROOT}/dataset"
