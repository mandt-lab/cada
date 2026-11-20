#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-${REPOSITORY_ROOT}/.envs/gencast-da/bin/python}"
[[ -n "${CADA_ROOT:-}" ]] || {
  printf 'cada_pipeline.sh: CADA_ROOT is not set\n' >&2
  exit 2
}
export CADA_ROOT
export GENCAST_DATA_ROOT="${GENCAST_DATA_ROOT:-${CADA_ROOT}/data/gencast}"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "${PYTHON_BIN}" ]] || {
  printf 'cada_pipeline.sh: Python not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
}

cd -- "${REPOSITORY_ROOT}"
dry_run_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  dry_run_args+=(--dry-run)
fi

run_id() {
  local config=$1
  local digest
  digest="$(sha256sum "${config}" | cut -c1-12)"
  printf '%s-%s' "$(date -u +%Y%m%dT%H%M%S%6NZ)" "${digest}"
}

smoke_config="gencast_da/configs/smoke.yaml"
controller_config="gencast_da/configs/controller.yaml"
evaluation_config="gencast_da/configs/evaluation.yaml"
figure_config="gencast_da/configs/figure.yaml"

smoke_id="$(run_id "${smoke_config}")"
controller_id="$(run_id "${controller_config}")"
evaluation_id="$(run_id "${evaluation_config}")"
figure_id="$(run_id "${figure_config}")"

"${PYTHON_BIN}" -m gencast_da smoke \
  --config "${smoke_config}" \
  --run-id "${smoke_id}" \
  "${dry_run_args[@]}"

"${PYTHON_BIN}" -m gencast_da train-controller \
  --config "${controller_config}" \
  --run-id "${controller_id}" \
  "${dry_run_args[@]}"

checkpoint="${CADA_ROOT}/runs/gencast/controller/gencast.controller.mini/${controller_id}/checkpoints/final.pkl"
"${PYTHON_BIN}" -m gencast_da evaluate \
  --config "${evaluation_config}" \
  --run-id "${evaluation_id}" \
  --checkpoint "${checkpoint}" \
  "${dry_run_args[@]}"

artifact="${CADA_ROOT}/runs/gencast/evaluation/gencast.evaluation.mini/${evaluation_id}/samples/evaluation.npz"
"${PYTHON_BIN}" -m gencast_da figure \
  --config "${figure_config}" \
  --run-id "${figure_id}" \
  --artifact "${artifact}" \
  "${dry_run_args[@]}"
