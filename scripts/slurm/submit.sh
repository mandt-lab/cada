#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
JOB_SCRIPT="${SCRIPT_DIR}/job.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/slurm/submit.sh [OPTIONS] -- COMMAND [--config PATH] [COMMAND OPTIONS]

`--config` is required for training/evaluation and omitted for `smoke`.

Submit options:
  --profile PROFILE     Resource profile: smoke, spot (default), or long
  --job-name NAME       Slurm job name (default: cada)
  --account ACCOUNT     Slurm account (required, or set SLURM_ACCOUNT)
  --partition NAME      Override the profile partition
  --gpus COUNT          Override the profile GPU count
  --cpus COUNT          Override the profile CPU count
  --memory SIZE         Override the profile memory
  --time LIMIT          Override the profile time limit
  --nodelist HOSTS      Optional Slurm nodelist override
  --config-root DIR     Base for relative --config paths (default: repository)
  --sbatch-arg ARG      Pass one additional argument to sbatch; repeat as needed
  --submit              Validate with CADA --dry-run, then invoke sbatch
  -h, --help            Show this help

Without --submit, the shell-escaped sbatch command is printed and no directory
is created.
EOF
}

die() {
    printf 'submit.sh: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local option=$1
    local count=$2

    ((count >= 2)) || die "${option} requires a value"
}

profile=spot
job_name=cada
account=${SLURM_ACCOUNT:-}
config_root="${REPOSITORY_ROOT}"
partition_override=
gpus_override=
cpus_override=
memory_override=
time_override=
nodelist=
submit=false
declare -a extra_sbatch_args=()

while (($#)); do
    case "$1" in
        --profile)
            require_value "$1" "$#"
            profile=$2
            shift 2
            ;;
        --profile=*)
            profile=${1#*=}
            shift
            ;;
        --job-name)
            require_value "$1" "$#"
            job_name=$2
            shift 2
            ;;
        --job-name=*)
            job_name=${1#*=}
            shift
            ;;
        --account)
            require_value "$1" "$#"
            account=$2
            shift 2
            ;;
        --account=*)
            account=${1#*=}
            shift
            ;;
        --partition)
            require_value "$1" "$#"
            partition_override=$2
            shift 2
            ;;
        --partition=*)
            partition_override=${1#*=}
            shift
            ;;
        --gpus)
            require_value "$1" "$#"
            gpus_override=$2
            shift 2
            ;;
        --gpus=*)
            gpus_override=${1#*=}
            shift
            ;;
        --cpus)
            require_value "$1" "$#"
            cpus_override=$2
            shift 2
            ;;
        --cpus=*)
            cpus_override=${1#*=}
            shift
            ;;
        --memory)
            require_value "$1" "$#"
            memory_override=$2
            shift 2
            ;;
        --memory=*)
            memory_override=${1#*=}
            shift
            ;;
        --time)
            require_value "$1" "$#"
            time_override=$2
            shift 2
            ;;
        --time=*)
            time_override=${1#*=}
            shift
            ;;
        --nodelist)
            require_value "$1" "$#"
            nodelist=$2
            shift 2
            ;;
        --nodelist=*)
            nodelist=${1#*=}
            shift
            ;;
        --config-root)
            require_value "$1" "$#"
            config_root=$2
            shift 2
            ;;
        --config-root=*)
            config_root=${1#*=}
            shift
            ;;
        --sbatch-arg)
            require_value "$1" "$#"
            extra_sbatch_args+=("$2")
            shift 2
            ;;
        --sbatch-arg=*)
            extra_sbatch_args+=("${1#*=}")
            shift
            ;;
        --submit)
            submit=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "unknown option before --: $1"
            ;;
    esac
done

(($#)) || die "a CADA command is required after --"
declare -a cada_args=("$@")

case "${profile}" in
    smoke)
        partition=spot.p
        gpus=1
        cpus=4
        memory=32G
        time_limit=00:10:00
        signal_lead=120
        ;;
    spot)
        partition=spot.p
        gpus=1
        cpus=16
        memory=128G
        time_limit=48:00:00
        signal_lead=600
        ;;
    long)
        partition=long.p
        gpus=1
        cpus=16
        memory=128G
        time_limit=168:00:00
        signal_lead=600
        ;;
    *)
        die "invalid profile: ${profile}"
        ;;
esac

[[ -z "${partition_override}" ]] || partition=${partition_override}
[[ -z "${gpus_override}" ]] || gpus=${gpus_override}
[[ -z "${cpus_override}" ]] || cpus=${cpus_override}
[[ -z "${memory_override}" ]] || memory=${memory_override}
[[ -z "${time_override}" ]] || time_limit=${time_override}

[[ "${job_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    die "invalid job name: ${job_name}"
[[ -n "${account}" ]] || die "--account is required"
[[ "${account}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    die "invalid account: ${account}"
[[ -n "${CADA_ROOT:-}" ]] || die "CADA_ROOT is not set"
export CADA_ROOT
SLURM_LOG_DIR="${CADA_ROOT}/slurm"
[[ "${partition}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
    die "invalid partition: ${partition}"
[[ "${gpus}" =~ ^[1-9][0-9]*$ ]] || die "gpus must be a positive integer"
[[ "${cpus}" =~ ^[1-9][0-9]*$ ]] || die "cpus must be a positive integer"
[[ "${memory}" =~ ^[1-9][0-9]*([KMGTP][iI]?[bB]?)?$ ]] ||
    die "invalid memory: ${memory}"
[[ "${time_limit}" =~ ^([0-9]+-)?[0-9]+(:[0-5][0-9]){0,2}$ ]] ||
    die "invalid time limit: ${time_limit}"
if [[ -n "${nodelist}" ]]; then
    [[ "${nodelist}" != *[[:space:]]* ]] ||
        die "nodelist cannot contain whitespace"
fi
[[ -n "${config_root}" ]] || die "config root cannot be empty"
[[ "${config_root}" != *$'\n'* ]] || die "config root cannot contain a newline"

case "${cada_args[0]}" in
    smoke|train-prior|train-controller|evaluate)
        ;;
    *)
        die "command must be smoke, train-prior, train-controller, or evaluate"
        ;;
esac

if [[ "${cada_args[0]}" != "smoke" ]]; then
    config_index=
    config_value=
    config_style=
    for ((index = 1; index < ${#cada_args[@]}; index++)); do
        case "${cada_args[index]}" in
            --config)
                [[ -z "${config_index}" ]] ||
                    die "--config may be specified only once"
                ((index + 1 < ${#cada_args[@]})) ||
                    die "--config requires a path"
                [[ -n "${cada_args[index + 1]}" ]] ||
                    die "--config requires a path"
                config_index=$((index + 1))
                config_value=${cada_args[index + 1]}
                config_style=separate
                ((index++))
                ;;
            --config=*)
                [[ -z "${config_index}" ]] ||
                    die "--config may be specified only once"
                config_value=${cada_args[index]#*=}
                [[ -n "${config_value}" ]] || die "--config requires a path"
                config_index=$index
                config_style=joined
                ;;
        esac
    done
    [[ -n "${config_index}" ]] || die "the CADA command must include --config"

    if [[ "${config_value}" != /* ]]; then
        if [[ "${config_root}" != /* ]]; then
            config_root="${PWD}/${config_root}"
        fi
        config_value="${config_root%/}/${config_value}"
    fi
    if [[ "${config_style}" == joined ]]; then
        cada_args[config_index]="--config=${config_value}"
    else
        cada_args[config_index]=${config_value}
    fi

fi

declare -a sbatch_command=(
    sbatch
    "--job-name=${job_name}"
    "--account=${account}"
    "--partition=${partition}"
    --nodes=1
    --ntasks=1
    "--gres=gpu:${gpus}"
    "--cpus-per-task=${cpus}"
    "--mem=${memory}"
    "--time=${time_limit}"
    --requeue
    --open-mode=append
    "--signal=B:USR1@${signal_lead}"
    "--output=${SLURM_LOG_DIR}/%x-%j.out"
    "--error=${SLURM_LOG_DIR}/%x-%j.err"
)
if [[ -n "${nodelist}" ]]; then
    sbatch_command+=("--nodelist=${nodelist}")
fi
sbatch_command+=("${extra_sbatch_args[@]}")
sbatch_command+=("${JOB_SCRIPT}" "${cada_args[@]}")

dry_run_output="$(
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
    [[ -x "${PYTHON_BIN}" ]] ||
        die "Python not found in Conda environment: ${PYTHON_BIN}"

    export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    cd -- "${REPOSITORY_ROOT}"
    "${PYTHON_BIN}" -m cada "${cada_args[@]}" --dry-run
)"

if [[ "${cada_args[0]}" == train-* ]]; then
    has_run_id=false
    for argument in "${cada_args[@]}"; do
        if [[ "${argument}" == "--run-id" || "${argument}" == --run-id=* ]]; then
            has_run_id=true
            break
        fi
    done
    if [[ "${has_run_id}" == false ]]; then
        run_path="$(printf '%s\n' "${dry_run_output}" |
            awk -F'"' '/"run":/ {print $4; exit}')"
        [[ -n "${run_path}" ]] || die "dry-run did not return a run path"
        generated_run_id=${run_path##*/}
        cada_args+=(--run-id "${generated_run_id}")
        sbatch_command+=(--run-id "${generated_run_id}")
    fi
fi

printf '%q' "${sbatch_command[0]}"
printf ' %q' "${sbatch_command[@]:1}"
printf '\n'

[[ "${submit}" == true ]] || exit 0

mkdir -p -- "${SLURM_LOG_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    set +u
    source /opt/Modules/init/bash
    set -u
    module load slurm
fi
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
"${sbatch_command[@]}"
