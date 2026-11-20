# Artifacts

`CADA_ROOT` separates large, generated state from repository source. A normal
run is stored at:

```text
$CADA_ROOT/runs/<domain>/<stage>/<config-id>/<run-id>/
├── checkpoints/
├── samples/
├── metrics/
├── figures/
├── logs/
├── wandb/
├── resolved.yaml
└── metadata.json
```

The run ID contains a UTC timestamp and the first 12 characters of the
resolved-config SHA256. `resolved.yaml` captures the fully expanded config.
`metadata.json` records the config digest, runtime, repository state, seeds,
host, and available Slurm metadata.

## Checkpoints

Training writes numbered, atomic `.pt` checkpoints plus `latest.pt`. The
payload schema includes the trainer state, resolved configuration, random
number generator state, metadata, and optional W&B run ID. At successful
completion, the CLI copies the last checkpoint to the `checkpoint_path`
declared by the config, commonly:

```text
$CADA_ROOT/runs/<domain>/prior/paper/checkpoints/final.pt
$CADA_ROOT/runs/<domain>/controller/<regime>/checkpoints/final.pt
```

Use `--resume` to select the newest matching run or combine it with
`--resume-checkpoint` for an explicit checkpoint. A resumed controller cannot
replace its prior with `--prior-checkpoint`.

## Evaluation outputs

Each evaluation run writes:

- `samples/evaluation.npz`: pickle-free arrays `truth`, `uncontrolled`, and
  `controlled`, plus JSON metadata.
- `samples/evaluation.npz.sha256`: checksum sidecar.
- `metrics/evaluation.json`: schema-versioned metrics and provenance.
- `metrics/evaluation.json.sha256`: checksum sidecar.

The three trajectory arrays have identical shapes and floating-point dtypes.
They are stored in physical space; evaluation metrics may additionally report
normalized-model-space values according to their config.
Metadata includes the resolved evaluation config, checkpoint paths and hashes,
initial-noise seed, metric parameters, and run ID.

## Figure staging

Figure configs do not discover evaluation runs automatically. Copy the chosen
immutable artifact to the exact path in the config:

```text
$CADA_ROOT/publication/artifacts/<domain>/<regime>/<horizon>/evaluation.npz
$CADA_ROOT/publication/artifacts/<domain>/<regime>/<horizon>/evaluation.npz.sha256
```

Rendering writes figures to `publication/figures` and a `.provenance.json`
sidecar beside every output. The sidecar records resolved paths and input
SHA256 values.

Do not overwrite existing outputs unless replacement is deliberate and
`--overwrite` is supplied. Preserve the original evaluation run even after
staging a figure input.

## GenCast artifacts

GenCast runs use:

```text
$CADA_ROOT/runs/gencast/<stage>/<config-id>/<run-id>/
```

Controller checkpoints are native JAX/Optax `.pkl` payloads and must only be
loaded from trusted runs in the pinned GenCast environment. Atomic replacement
and SHA256 sidecars protect writes; controller and NumPy RNG state support
resume. Evaluations write `samples/evaluation.npz` with `truth`, `prediction`,
`observations`, coordinates, and encoded JSON metadata, plus
`metrics/evaluation.json`.
