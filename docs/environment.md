# Environment

CADA supports Python `>=3.10,<3.11`. The supported Conda environment is named
`prediff`:

```bash
conda env create -f environment.yml
conda activate prediff
python -m pip install --no-deps -e .
```

`environment.yml` is a direct dependency specification, not a solver lock.
It names Python 3.10, PyTorch, NumPy, SciPy, h5py, einops, Matplotlib, PyYAML,
and pip. Exact transitive versions and build strings depend on the solver,
channels, platform, and installation date.

[`prediff-freeze.txt`](prediff-freeze.txt) records the package versions in the
existing cluster environment used for repository verification. It is
provenance, not an installation lock.

The YAML pins PyTorch 2.7.1 with CUDA 12.8, the minimum stable build used here
with RTX PRO 6000 Blackwell (`sm_120`). If a system Python precedes Conda on
`PATH`, invoke `"$CONDA_PREFIX/bin/python"` explicitly. The Slurm launcher
already does this.

## Optional groups

Weights & Biases tracking is lazy and disabled by default:

```bash
python -m pip install -e ".[tracking]"
cada train-prior \
  --config configs/priors/ks.yaml \
  --tracking offline \
  --device cuda
```

ERA5 netCDF conversion requires xarray and a netCDF backend:

```bash
python -m pip install -e ".[era5]"
```

Development tools are isolated from runtime dependencies:

```bash
python -m pip install -e ".[dev]"
ruff check src
pytest
```

The `pyproject.toml` dependency groups are:

- base runtime: `torch`, `numpy`, `scipy`, `h5py`, `einops`, `matplotlib`,
  and `PyYAML`;
- `tracking`: `wandb`;
- `era5`: `xarray` and `netCDF4`;
- `dev`: `pytest`, `ruff`, and `xarray`.

## Import and command checks

After editable installation:

```bash
python -c "import cada; print(cada.__file__)"
cada --help
cada train-prior \
  --config configs/priors/ks.yaml \
  --device cpu \
  --dry-run
```

Dry-run does not initialize CUDA or require data/checkpoint files, but it does
validate and fully resolve the YAML.

## GenCast environment

GenCast remains isolated because its pinned JAX/CUDA stack is incompatible
with treating `prediff` as one combined environment:

```bash
bash gencast_da/setup_env.sh
conda activate "$PWD/.envs/gencast-da"
python -m gencast_da --help
python -m pip install -r gencast_da/requirements-dev.txt
```

`gencast_da/environment.yml` defines the minimal Python prefix and
`gencast_da/requirements-linux-64-py310.lock` pins the Linux JAX, CUDA,
GraphCast, Haiku, and data stack. GraphCast is installed from an immutable Git
commit. The JAX 0.6.2/CUDA 12 stack is verified on RTX PRO 6000 Blackwell.
The bootstrap refuses to reuse an existing prefix so stale copied environments
cannot be mistaken for a clean installation.

The pip lock pins versions and the GraphCast revision but does not include
wheel hashes; it protects dependency resolution, not package-index compromise.
Conda selects the latest Python 3.10 patch/build available from `defaults`.

CPU verification uses `JAX_PLATFORMS=cpu`; the GPU smoke check must run on an
allocated CUDA node. `.envs/` remains local and ignored.
