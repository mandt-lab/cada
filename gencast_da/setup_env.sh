#!/usr/bin/env bash

set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${HERE}/.." && pwd -P)"
GENCAST_ENV="${GENCAST_ENV:-${REPOSITORY_ROOT}/.envs/gencast-da}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"

[[ "${GENCAST_ENV}" = /* ]] || {
  printf 'setup_env.sh: GENCAST_ENV must be absolute: %s\n' "${GENCAST_ENV}" >&2
  exit 2
}
[[ ! -e "${GENCAST_ENV}" ]] || {
  printf 'setup_env.sh: refusing to reuse existing prefix: %s\n' "${GENCAST_ENV}" >&2
  exit 2
}
[[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || {
  printf 'setup_env.sh: Conda initialization not found under %s\n' "${CONDA_BASE}" >&2
  exit 2
}

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_SOLVER=classic

printf '[setup_env] Creating %s\n' "${GENCAST_ENV}"
conda create \
  --yes \
  --prefix "${GENCAST_ENV}" \
  --override-channels \
  --channel defaults \
  python=3.10 \
  pip \
  setuptools \
  wheel

PYTHON_BIN="${GENCAST_ENV}/bin/python"
[[ -x "${PYTHON_BIN}" ]] || {
  printf 'setup_env.sh: Python not found after environment creation\n' >&2
  exit 1
}

"${PYTHON_BIN}" -m pip install \
  --upgrade \
  pip==26.1 \
  setuptools==82.0.1 \
  wheel==0.47.0
"${PYTHON_BIN}" -m pip install \
  --requirement "${HERE}/requirements-linux-64-py310.lock"

"${PYTHON_BIN}" -m pip check
JAX_PLATFORMS=cpu "${PYTHON_BIN}" - <<'PY'
import json
import sys

import haiku
import jax
import optax
from graphcast import checkpoint, data_utils, denoiser, gencast, normalization

print(json.dumps({
    "python": sys.version,
    "prefix": sys.prefix,
    "jax": jax.__version__,
    "devices": [str(device) for device in jax.devices()],
    "sampler": str(gencast.SamplerConfig()),
}))
PY

printf '[setup_env] Ready: %s\n' "${GENCAST_ENV}"
printf 'Activate with: conda activate %q\n' "${GENCAST_ENV}"
