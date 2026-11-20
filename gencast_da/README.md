# CADA-on-GenCast

This directory is the JAX/Haiku implementation of the GenCast experiment. It
keeps its model code and dependency environment separate from the PyTorch
package under `src/cada`, while using the same `CADA_ROOT` run structure.

## Environment

Create the pinned repository-local environment:

```bash
bash gencast_da/setup_env.sh
conda activate "$PWD/.envs/gencast-da"
python -m pip install -r gencast_da/requirements-dev.txt
```

The installed GraphCast source is pinned to commit
`08cf73625c9d12bd9aaa038868bcb2fe488f2a22`. The local `.envs/` prefix is
never committed. JAX 0.6.2, Haiku 0.0.17, and the locked CUDA 12 runtime were
verified on RTX PRO 6000 Blackwell hardware.

GenCast assets are read through:

```bash
export CADA_ROOT=/absolute/path/to/cada-root
export GENCAST_DATA_ROOT=$CADA_ROOT/data/gencast
```

The default data path may be a symlink to an existing GenCast asset tree.
Download missing public assets with:

```bash
bash gencast_da/download_assets.sh
```

## Commands

All commands support a side-effect-free `--dry-run`:

```bash
python -m gencast_da data \
  --config gencast_da/configs/data.yaml \
  --dry-run

python -m gencast_da smoke \
  --config gencast_da/configs/smoke.yaml \
  --dry-run

python -m gencast_da train-controller \
  --config gencast_da/configs/controller.yaml \
  --dry-run

python -m gencast_da evaluate \
  --config gencast_da/configs/evaluation.yaml \
  --checkpoint /path/to/final.pkl \
  --dry-run

python -m gencast_da figure \
  --config gencast_da/configs/figure.yaml \
  --artifact /path/to/evaluation.npz \
  --dry-run
```

`cada_pipeline.sh` chains smoke, controller training, CADA evaluation, and a
CADA-only temperature figure. Set `DRY_RUN=1` to inspect all four commands
without creating outputs.

## Run layout

New outputs are written only under:

```text
$CADA_ROOT/runs/gencast/<stage>/<config-id>/<run-id>/
├── checkpoints/
├── samples/
├── metrics/
├── figures/
├── logs/
├── wandb/
├── resolved.yaml
└── metadata.json
```

Checkpoints remain trusted JAX/Optax pickle payloads, but writes are atomic and
include RNG state and SHA256 sidecars. Evaluations write structured metrics
JSON and a pickle-free NPZ containing truth, CADA predictions, observations,
coordinates, and metadata.

## Slurm

Inspect a submission without creating a run:

```bash
scripts/slurm/submit_gencast.sh \
  --profile smoke \
  --job-name gencast-smoke \
  --account YOUR_ACCOUNT \
  -- smoke \
  --config gencast_da/configs/smoke.yaml
```

Add `--submit` before `--` to validate and call `sbatch`. The separate GenCast
launcher activates `.envs/gencast-da`, forwards preemption signals, and keeps
the PyTorch Slurm path unchanged.

## Checks

CPU-only:

```bash
JAX_PLATFORMS=cpu python -m gencast_da._check_cada_controller
JAX_PLATFORMS=cpu python -m gencast_da.data_contract
```

The bounded GPU smoke command compiles one zero-initialized controller path
and checks parity, diagnostics, and gradient flow. Full training is never part
of CI.
