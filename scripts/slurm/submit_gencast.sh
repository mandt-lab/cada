#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
JOB_SCRIPT="${SCRIPT_DIR}/gencast_job.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm/submit_gencast.sh [OPTIONS] -- COMMAND --config PATH [OPTIONS]

Options:
  --profile PROFILE     smoke, spot (default), or long
  --job-name NAME       Slurm job name (default: gencast-cada)
  --account ACCOUNT     Slurm account (required, or set SLURM_ACCOUNT)
  --partition NAME      Override profile partition
  --gpus COUNT          Override GPU count
  --cpus COUNT          Override CPU count
  --memory SIZE         Override memory
  --time LIMIT          Override time limit
  --nodelist HOSTS      Optional Slurm nodelist
  --sbatch-arg ARG      Additional sbatch argument; repeatable
  --submit              Validate with --dry-run, then submit
EOF
}

die() {
  printf 'submit_gencast.sh: %s\n' "$*" >&2
  exit 2
}

require_value() {
  (($2 >= 2)) || die "$1 requires a value"
}

profile=spot
job_name=gencast-cada
account=${SLURM_ACCOUNT:-}
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
    --profile) require_value "$1" "$#"; profile=$2; shift 2 ;;
    --profile=*) profile=${1#*=}; shift ;;
    --job-name) require_value "$1" "$#"; job_name=$2; shift 2 ;;
    --job-name=*) job_name=${1#*=}; shift ;;
    --account) require_value "$1" "$#"; account=$2; shift 2 ;;
    --account=*) account=${1#*=}; shift ;;
    --partition) require_value "$1" "$#"; partition_override=$2; shift 2 ;;
    --partition=*) partition_override=${1#*=}; shift ;;
    --gpus) require_value "$1" "$#"; gpus_override=$2; shift 2 ;;
    --gpus=*) gpus_override=${1#*=}; shift ;;
    --cpus) require_value "$1" "$#"; cpus_override=$2; shift 2 ;;
    --cpus=*) cpus_override=${1#*=}; shift ;;
    --memory) require_value "$1" "$#"; memory_override=$2; shift 2 ;;
    --memory=*) memory_override=${1#*=}; shift ;;
    --time) require_value "$1" "$#"; time_override=$2; shift 2 ;;
    --time=*) time_override=${1#*=}; shift ;;
    --nodelist) require_value "$1" "$#"; nodelist=$2; shift 2 ;;
    --nodelist=*) nodelist=${1#*=}; shift ;;
    --sbatch-arg) require_value "$1" "$#"; extra_sbatch_args+=("$2"); shift 2 ;;
    --sbatch-arg=*) extra_sbatch_args+=("${1#*=}"); shift ;;
    --submit) submit=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) die "unknown option before --: $1" ;;
  esac
done

(($#)) || die "a GenCast command is required after --"
declare -a command_args=("$@")
case "${command_args[0]}" in
  smoke|train-controller|evaluate) ;;
  *) die "command must be smoke, train-controller, or evaluate" ;;
esac
if [[ "${command_args[0]}" == "smoke" ]]; then
  export GENCAST_DETERMINISTIC=1
fi

case "${profile}" in
  smoke) partition=spot.p; gpus=1; cpus=8; memory=128G; time_limit=00:30:00 ;;
  spot) partition=spot.p; gpus=1; cpus=16; memory=128G; time_limit=48:00:00 ;;
  long) partition=long.p; gpus=1; cpus=16; memory=128G; time_limit=168:00:00 ;;
  *) die "invalid profile: ${profile}" ;;
esac
[[ -z "${partition_override}" ]] || partition=${partition_override}
[[ -z "${gpus_override}" ]] || gpus=${gpus_override}
[[ -z "${cpus_override}" ]] || cpus=${cpus_override}
[[ -z "${memory_override}" ]] || memory=${memory_override}
[[ -z "${time_override}" ]] || time_limit=${time_override}

[[ "${job_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid job name"
[[ -n "${account}" ]] || die "--account is required for GenCast submissions"
[[ "${account}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid account"
[[ -n "${CADA_ROOT:-}" ]] || die "CADA_ROOT is not set"
export CADA_ROOT
SLURM_LOG_DIR="${CADA_ROOT}/slurm"
[[ "${partition}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid partition"
[[ "${gpus}" =~ ^[1-9][0-9]*$ ]] || die "gpus must be positive"
[[ "${cpus}" =~ ^[1-9][0-9]*$ ]] || die "cpus must be positive"
[[ "${memory}" =~ ^[1-9][0-9]*([KMGTP][iI]?[bB]?)?$ ]] || die "invalid memory"
[[ "${time_limit}" =~ ^([0-9]+-)?[0-9]+(:[0-5][0-9]){0,2}$ ]] ||
  die "invalid time"
[[ -z "${nodelist}" || "${nodelist}" != *[[:space:]]* ]] ||
  die "nodelist cannot contain whitespace"

config_index=
for ((index = 1; index < ${#command_args[@]}; index++)); do
  case "${command_args[index]}" in
    --config)
      ((index + 1 < ${#command_args[@]})) || die "--config requires a path"
      config_index=$((index + 1))
      ((index++))
      ;;
    --config=*) config_index=$index ;;
  esac
done
[[ -n "${config_index}" ]] || die "command must include --config"
if [[ "${command_args[config_index]}" == --config=* ]]; then
  config_value=${command_args[config_index]#*=}
  [[ "${config_value}" = /* ]] || config_value="${REPOSITORY_ROOT}/${config_value}"
  command_args[config_index]="--config=${config_value}"
else
  config_value=${command_args[config_index]}
  [[ "${config_value}" = /* ]] || config_value="${REPOSITORY_ROOT}/${config_value}"
  command_args[config_index]=${config_value}
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
  --signal=B:USR1@600
  "--output=${SLURM_LOG_DIR}/%x-%j.out"
  "--error=${SLURM_LOG_DIR}/%x-%j.err"
)
[[ -z "${nodelist}" ]] || sbatch_command+=("--nodelist=${nodelist}")
sbatch_command+=("${extra_sbatch_args[@]}")
sbatch_command+=("${JOB_SCRIPT}" "${command_args[@]}")

GENCAST_ENV="${GENCAST_ENV:-${REPOSITORY_ROOT}/.envs/gencast-da}"
PYTHON_BIN="${GENCAST_PYTHON_BIN:-${GENCAST_ENV}/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || die "Python not found: ${PYTHON_BIN}"
export GENCAST_DATA_ROOT="${GENCAST_DATA_ROOT:-${CADA_ROOT}/data/gencast}"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${REPOSITORY_ROOT}"
dry_run_output="$(
  "${PYTHON_BIN}" -m gencast_da "${command_args[@]}" --dry-run
)"
if [[ "${command_args[0]}" == "train-controller" ]]; then
  has_run_id=false
  for argument in "${command_args[@]}"; do
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
    command_args+=(--run-id "${generated_run_id}")
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
