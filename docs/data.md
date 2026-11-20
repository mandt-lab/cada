# Data

All dataset paths are rooted at the absolute `CADA_ROOT` value:

```text
$CADA_ROOT/data/
├── ks/
│   ├── train.h5
│   ├── holdout.h5
│   └── normalization.json
├── kolmogorov/
│   ├── train.h5
│   ├── holdout.h5
│   └── normalization.pt
└── era5_small/
    ├── train.h5
    ├── holdout.h5
    └── normalization.pt
```

Inspect resolved paths without writing:

```bash
cada data \
  --config manifests/data/ks.yaml \
  --action inspect \
  --dry-run
```

## Kuramoto–Sivashinsky

The CLI can generate the fixed KS split and training-set min/max statistics:

```bash
cada data \
  --config manifests/data/ks.yaml \
  --action generate
```

The generated files use grouped HDF5 splits named `train` and `test`. The
manifest describes one channel, 256 spatial points, simulator time step 0.2,
training sequence length 140, and maximum evaluation horizon 640.

## Kolmogorov flow

The packaged CLI does not generate Kolmogorov trajectories. The reference
layout uses two source sets:

- `data/train.h5`, shape `[819, 64, 2, 64, 64]`, for prior/controller training;
- `data_long/test.h5`, shape `[1, 192, 2, 64, 64]`, for 60- and 180-step
  evaluation.

They are exposed without duplication through
`$CADA_ROOT/data/kolmogorov/train.h5` and `holdout.h5` symlinks. The regular
test set has only 64 states and cannot support a 180-step forecast, which
requires 181 states including the initial condition.

For another installation, supply equivalent `train.h5` and `holdout.h5`
files with a dense floating-point dataset named `x`, then compute training
statistics:

```bash
cada data \
  --config manifests/data/kolmogorov.yaml \
  --action stats
```

Expected states have two velocity channels on a 64×64 grid. The manifest
records simulator time step 0.2, training sequence length 64, and maximum
evaluation horizon 180.

## ERA5-small

The `request` action prints the paper-shaped CDS request JSON; it does not
authenticate with CDS or download files:

```bash
cada data \
  --config manifests/data/era5_small.yaml \
  --action request \
  --year 2006 \
  --year 2016
```

Install the ERA5 optional dependencies, obtain netCDF inputs independently,
then convert them:

```bash
python -m pip install -e ".[era5]"

cada data \
  --config manifests/data/era5_small.yaml \
  --action convert \
  --input /absolute/path/era5-2006.nc /absolute/path/era5-2016.nc
```

Conversion selects 500 hPa temperature, zonal wind, and meridional wind,
sorts coordinates, interpolates to 64×64, uses 6-hour cadence, writes
2006–2015 to training and 2016 to holdout, and derives channelwise training
min/max statistics.

## Integrity and normalization

PDE and ERA5 model inputs use channelwise min/max normalization. Statistics
must be fitted only on the declared training split. Normalization is linear
and does not clip holdout values outside the training extrema, preserving
physical values after denormalization. Dataset `data_sha256` and
`source_version` are `null` in the shipped configs because they depend on the
local copy of the data.

## GenCast assets

GenCast Mini uses its native 1-degree, 13-pressure-level checkpoint, NetCDF
statistics, and ERA5 trajectories. They are read from `$GENCAST_DATA_ROOT`,
which defaults to `$CADA_ROOT/data/gencast`; that path may be a symlink to an
existing shared asset tree to avoid duplicating more than 21 GiB.
`gencast_da/configs/data.yaml` and
`python -m gencast_da data --config gencast_da/configs/data.yaml --action
inspect` validate the live checkpoint and dataset contract. These native
assets are separate from the compact 64×64 `era5_small` dataset.
`gencast_da/assets.sha256` verifies the canonical checkpoint, statistics, and
example trajectory; downloads use a temporary file followed by atomic rename.
