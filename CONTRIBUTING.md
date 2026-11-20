# Contributing

This repository is research software tied to a fixed experiment matrix. Keep
changes small, testable, and explicit about whether a value comes from the
paper or is a repository default.

## Development setup

```bash
conda env create -f environment.yml
conda activate prediff
python -m pip install --no-deps -e .
python -m pip install -e ".[dev]"
```

Before proposing a change:

```bash
ruff check src tests gencast_da
pytest
cada --help
JAX_PLATFORMS=cpu .envs/gencast-da/bin/python \
  -m gencast_da._check_cada_controller
```

When changing commands or artifact schemas, update the relevant file under
`docs/` and run the documented `--dry-run` commands. Values listed under
`paper_fields` must match the paper.

See [the environment guide](docs/environment.md) before submitting changes.
