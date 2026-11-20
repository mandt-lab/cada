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
#SBATCH --signal=B:USR1@120

set -Eeuo pipefail

if [[ -n "${CADA_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${CADA_PYTHON_BIN}"
else
    CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
    CONDA_ENV="${CONDA_ENV:-prediff}"
    # shellcheck source=/dev/null
    set +u
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
[[ -x "${PYTHON_BIN}" ]] || {
    printf 'job.sh: Python not found in Conda environment: %s\n' "${PYTHON_BIN}" >&2
    exit 1
}

REPOSITORY_ROOT="${CADA_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
REPOSITORY_ROOT="$(cd -- "${REPOSITORY_ROOT}" && pwd -P)"
[[ -d "${REPOSITORY_ROOT}/src/cada" ]] || {
    printf 'job.sh: repository not found: %s\n' "${REPOSITORY_ROOT}" >&2
    exit 2
}
[[ -n "${CADA_ROOT:-}" ]] || {
    printf 'job.sh: CADA_ROOT is not set\n' >&2
    exit 2
}
export CADA_ROOT
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${REPOSITORY_ROOT}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"

declare -a command_args=("$@")
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && "${command_args[0]:-}" == train-* ]]; then
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
    if [[ "${command_args[0]:-}" != train-* ]]; then
        exit "${child_status}"
    fi

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        printf 'job.sh: cannot requeue without SLURM_JOB_ID\n' >&2
        exit "${child_status}"
    fi
    if [[ "${child_status}" -ne 75 ]]; then
        printf 'job.sh: child exited %s without checkpoint acknowledgement\n' \
            "${child_status}" >&2
        exit "${child_status}"
    fi

    if ! "${SCONTROL_BIN}" requeue "${SLURM_JOB_ID}"; then
        printf 'job.sh: failed to requeue Slurm job %s\n' "${SLURM_JOB_ID}" >&2
        exit 1
    fi
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

"${PYTHON_BIN}" -m cada "${command_args[@]}" &
child_pid=$!

child_status=0
wait_for_child || child_status=$?
exit "${child_status}"
