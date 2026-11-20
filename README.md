# CADA

Research code for **Learning to Assimilate: Amortized Control for
Autoregressive Diffusion** (NeurIPS 2026). Its method, CADA (Control-Augmented
Data Assimilation), trains a lightweight controller on top of a frozen
autoregressive diffusion prior, then performs sparse-observation data
assimilation through controlled feed-forward rollouts.

The code covers Kuramoto–Sivashinsky (KS), Kolmogorov flow, and compact
64×64 ERA5 across downsampled and masked observation regimes, plus the
self-contained CADA-on-GenCast experiment under [`gencast_da/`](gencast_da/).

## Paper and citation status

The accompanying paper,
[Learning to Assimilate: Amortized Control for Autoregressive Diffusion](https://arxiv.org/abs/2510.06637),
is to appear at NeurIPS 2026. arXiv versions up to v3 (10 May 2026) were titled
*Control-Augmented Autoregressive Diffusion for Data Assimilation*. Use
[`CITATION.cff`](CITATION.cff) for the title, author list, and arXiv
identifier.

## Quick setup

The PyTorch environment is Python 3.10 in the Conda environment `prediff`:

```bash
conda env create -f environment.yml
conda activate prediff
python -m pip install --no-deps -e .
cada --help
```

Optional features can be installed with `python -m pip install -e
".[tracking]"` for Weights & Biases or `python -m pip install -e ".[era5]"`
for ERA5 netCDF conversion. See [environment details](docs/environment.md).

GenCast uses a separate pinned JAX/CUDA environment:

```bash
bash gencast_da/setup_env.sh
.envs/gencast-da/bin/python -m gencast_da --help
```

The two environments share artifact conventions, not Python imports. See the
[GenCast guide](gencast_da/README.md).

Set one absolute artifact root before running commands:

```bash
export CADA_ROOT=/absolute/path/to/cada-artifacts
```

All `${CADA_ROOT}` placeholders in manifests resolve against this value, and
commands fail if it is unset. Repository YAML remains under `configs/` and
`manifests/`; datasets, checkpoints, run outputs, and staged figure inputs live
under `CADA_ROOT`.

## Commands

Run bounded synthetic plumbing without data or checkpoints:

```bash
cada smoke --device cpu
```

Dry-run validates and resolves configuration without creating run outputs:

```bash
cada train-controller \
  --config configs/controllers/kolmogorov__ms_4.yaml \
  --device cpu \
  --dry-run
```

Train the prior, then the matching controller:

```bash
cada train-prior \
  --config configs/priors/kolmogorov.yaml \
  --device cuda

cada train-controller \
  --config configs/controllers/kolmogorov__ms_4.yaml \
  --device cuda
```

Evaluate the 180-step Kolmogorov MS-4 experiment:

```bash
cada evaluate \
  --config configs/evaluations/kolmogorov__ms_4__h180.yaml \
  --device cuda
```

The same evaluation config can run TTO-DA without a controller checkpoint:

```bash
cada evaluate \
  --config configs/evaluations/kolmogorov__ms_4__h180.yaml \
  --method tto-da \
  --tto-steps 50 \
  --tto-lr 0.01 \
  --device cuda
```

TTO-DA optimizes free per-transition/per-DDIM-step controls through the frozen
prior in each preview chunk. The config supplies `gamma`, `beta`, and the
deterministic diffusion seed. CADA and TTO-DA have distinct resolved run
identities and output paths.

`cada figure` plots a saved evaluation artifact as ground truth, uncontrolled
prior, and CADA panels. After copying an artifact to the path declared in the
YAML, plot Kolmogorov vorticity snapshots:

```bash
cada figure \
  --config configs/figures/kolmogorov__ms_4__h180__vorticity.yaml
```

Inspect a Slurm command without submitting:

```bash
scripts/slurm/submit.sh \
  --profile spot \
  --job-name cada-kolm-ms4 \
  -- train-controller \
  --config configs/controllers/kolmogorov__ms_4.yaml \
  --device cuda
```

Add `--submit` before `--` to run the built-in dry-run and call `sbatch`:

```bash
scripts/slurm/submit.sh \
  --profile spot \
  --job-name cada-kolm-ms4 \
  --submit \
  -- train-controller \
  --config configs/controllers/kolmogorov__ms_4.yaml \
  --device cuda
```

See [data preparation](docs/data.md), [experiment commands](docs/experiments.md),
and [Slurm usage](docs/slurm.md) before launching full runs.

## Artifact layout

Training and evaluation create immutable run directories:

```text
$CADA_ROOT/
├── data/<domain>/
├── runs/<domain>/<stage>/<config-id>/<run-id>/
│   ├── checkpoints/
│   ├── samples/
│   ├── metrics/
│   ├── figures/
│   ├── logs/
│   ├── wandb/
│   ├── resolved.yaml
│   └── metadata.json
└── publication/
    ├── artifacts/
    └── figures/
```

Evaluation writes `samples/evaluation.npz`, `metrics/evaluation.json`, and
SHA256 sidecars. Figure configs read separately staged artifacts and write
provenance sidecars. Details are in [artifacts](docs/artifacts.md) and
[metrics and figures](docs/metrics-and-figures.md).

This repository supports the TTO-DA evaluation ablation.

## Documentation and attribution

- [Architecture](docs/architecture.md)
- [Experiments](docs/experiments.md)
- [Data](docs/data.md)
- [Artifacts](docs/artifacts.md)
- [Metrics and figures](docs/metrics-and-figures.md)
- [Slurm](docs/slurm.md)
- [Environment](docs/environment.md)
- [CADA-on-GenCast](gencast_da/README.md)
- [Contributing](CONTRIBUTING.md)

Portions of the diffusion implementation are adapted from
`lucidrains/denoising-diffusion-pytorch` under the MIT License. See
[`NOTICE`](NOTICE) and [`LICENSE`](LICENSE); the CADA project must not be
attributed to the upstream author.
