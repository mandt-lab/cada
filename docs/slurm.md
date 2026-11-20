# Slurm

The Slurm helpers are site-specific. Their defaults assume:

- partitions `spot.p` and `long.p`;
- Conda under `CONDA_BASE`, or `conda info --base` when it is unset;
- environment `prediff`.

`CADA_ROOT` and a Slurm account (`--account` or `SLURM_ACCOUNT`) are required.
Override partition, resource values, `CONDA_BASE`, and `CONDA_ENV` as needed
for another cluster. Submitted log paths are derived from `${CADA_ROOT}/slurm`.

## Profiles

- `smoke`: 1 GPU, 4 CPUs, 32 GiB, 10 minutes on `spot.p`.
- `spot` (default): 1 GPU, 16 CPUs, 128 GiB, 48 hours on `spot.p`.
- `long`: 1 GPU, 16 CPUs, 128 GiB, 168 hours on `long.p`.

All profiles request one node and one task, enable requeue, and append logs.
CADA smoke jobs send `USR1` 120 seconds before termination; longer CADA and
all GenCast helper submissions use 600 seconds. The wrappers forward that
signal and requeue training only after exit code 75 acknowledges a completed
signal checkpoint.

## Inspect before submission

Without `--submit`, `submit.sh` validates its arguments and prints the exact
shell-escaped `sbatch` command. It does not create directories or invoke
Slurm:

```bash
scripts/slurm/submit.sh \
  --profile smoke \
  --job-name cada-ks-smoke \
  -- train-prior \
  --config configs/priors/ks.yaml \
  --max-steps 2 \
  --device cuda
```

A data-free CUDA smoke uses the real tiny U-Net, diffusion, controller, and
forward/backward paths:

```bash
scripts/slurm/submit.sh \
  --profile smoke \
  --job-name cada-smoke \
  --account YOUR_ACCOUNT \
  -- smoke \
  --device cuda
```

## Submit

`--submit` activates `prediff`, exports repository `src` on `PYTHONPATH`, runs
the CADA command with `--dry-run`, creates the site-specific log directory,
and invokes `sbatch`:

```bash
export CADA_ROOT=/absolute/path/to/cada-artifacts

scripts/slurm/submit.sh \
  --profile spot \
  --job-name cada-ks-ms8 \
  --account YOUR_ACCOUNT \
  --partition YOUR_PARTITION \
  --submit \
  -- train-controller \
  --config configs/controllers/ks__ms_8.yaml \
  --device cuda
```

An evaluation uses the same wrapper:

```bash
scripts/slurm/submit.sh \
  --profile long \
  --job-name cada-ks-ms8-h640 \
  --account YOUR_ACCOUNT \
  --partition YOUR_PARTITION \
  --submit \
  -- evaluate \
  --config configs/evaluations/ks__ms_8__h640.yaml \
  --device cuda
```

`smoke`, `train-prior`, `train-controller`, and `evaluate` are accepted after
`--`. Data preparation and figure rendering should be run directly.

The two-step smoke command verifies plumbing only. Full prior and controller
schedules require substantial GPU time.

## GenCast jobs

GenCast has separate wrappers so JAX never runs in `prediff`:

```bash
scripts/slurm/submit_gencast.sh \
  --profile smoke \
  --job-name gencast-smoke \
  --account YOUR_ACCOUNT \
  -- smoke \
  --config gencast_da/configs/smoke.yaml
```

Add `--submit` before `--` after inspecting the command. The smoke profile
requests one GPU, 8 CPUs, 128 GiB, and 30 minutes. Spot and long profiles use
the same memory and wall-time defaults as the PyTorch launcher.

`gencast_job.sh` uses `.envs/gencast-da`, unloads inherited CUDA modules so
JAX uses its pinned CUDA wheels, forwards USR1/TERM, and requeues only after
the trainer has atomically checkpointed. Override `GENCAST_ENV`,
`GENCAST_DATA_ROOT`, account, and partition when required.
