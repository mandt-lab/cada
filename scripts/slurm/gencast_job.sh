#!/usr/bin/env bash
#SBATCH --partition=spot.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

set -Eeuo pipefail

REPOSITORY_ROOT="${CADA_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
REPOSITORY_ROOT="$(cd -- "${REPOSITORY_ROOT}" && pwd -P)"
[[ -d "${REPOSITORY_ROOT}/gencast_da" ]] || {
  printf 'gencast_job.sh: repository not found: %s\n' "${REPOSITORY_ROOT}" >&2
  exit 2
}
GENCAST_ENV="${GENCAST_ENV:-${REPOSITORY_ROOT}/.envs/gencast-da}"
PYTHON_BIN="${GENCAST_PYTHON_BIN:-${GENCAST_ENV}/bin/python}"

[[ -x "${PYTHON_BIN}" ]] || {
  printf 'gencast_job.sh: Python not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
}

if [[ "${CADA_SKIP_MODULES:-0}" != "1" && -f /opt/Modules/init/bash ]]; then
  # shellcheck source=/dev/null
  set +u
  source /opt/Modules/init/bash
  set -u
  module load slurm
  module unload cuda >/dev/null 2>&1 || true
fi
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"

[[ -n "${CADA_ROOT:-}" ]] || {
  printf 'gencast_job.sh: CADA_ROOT is not set\n' >&2
  exit 2
}
export CADA_ROOT
export GENCAST_DATA_ROOT="${GENCAST_DATA_ROOT:-${CADA_ROOT}/data/gencast}"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
if [[ "${GENCAST_DETERMINISTIC:-0}" == "1" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_deterministic_ops=true"
fi
cd -- "${REPOSITORY_ROOT}"

declare -a command_args=("$@")
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && "${command_args[0]:-}" == "train-controller" ]]; then
  has_resume=false
  for argument in "${command_args[@]}"; do
    if [[ "${argument}" == "--resume" ]]; then
      has_resume=true
      break
    fi
  done
  [[ "${has_resume}" == true ]] || command_args+=(--resume)
fi

child_pid=

wait_for_child() {
  local status
  set +e
  wait "${child_pid}"
  status=$?
  set -e
  child_pid=
  return "${status}"
}

handle_usr1() {
  local child_status=0
  trap - USR1
  if [[ -n "${child_pid}" ]]; then
    kill -USR1 "${child_pid}" 2>/dev/null || true
    wait_for_child || child_status=$?
  fi
  if [[ "${command_args[0]:-}" != "train-controller" ]]; then
    exit "${child_status}"
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exit "${child_status}"
  fi
  if [[ "${child_status}" -ne 75 ]]; then
    printf 'gencast_job.sh: child exited %s without checkpoint acknowledgement\n' \
      "${child_status}" >&2
    exit "${child_status}"
  fi
  "${SCONTROL_BIN}" requeue "${SLURM_JOB_ID}"
  exit 0
}

handle_term() {
  local child_status=143
  trap - TERM
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait_for_child || child_status=$?
  fi
  exit "${child_status}"
}

trap handle_usr1 USR1
trap handle_term TERM

"${PYTHON_BIN}" -m gencast_da "${command_args[@]}" &
child_pid=$!
status=0
wait_for_child || status=$?
exit "${status}"
