# Paper-canonical configurations

This directory contains static, standalone YAML files for the three-dataset paper matrix:

- `priors/`: 3 files, one ARDM prior for Kolmogorov, KS, and ERA5-small.
- `controllers/`: 18 files, six regimes (`DS/MS × 2/4/8`) per domain.
- `evaluations/`: 30 files: Kolmogorov has six regimes at horizons 60 and 180, KS has six at 140 and 640, and ERA5-small has six at 60.
- `../manifests/data/`: 3 standalone dataset manifests.

Every experiment embeds its complete dataset and model settings; there are no YAML anchors, inheritance, runtime composition, or external-baseline configs. Paths use `${CADA_ROOT}` and resolve below that root. Each dataset has distinct `train_data_path` and `holdout_data_path` values at `data/<domain>/train.h5` and `data/<domain>/holdout.h5`. Kolmogorov and ERA5-small use dense HDF5 key `x` with `train`/`holdout` labels; KS uses grouped `train`/`test` data. Normalization artifacts are `data/ks/normalization.json`, `data/kolmogorov/normalization.pt`, and `data/era5_small/normalization.pt`.

`source_provenance` records which fields follow the paper (`paper_fields`) and which are repository settings (`repository_default_fields`). Appendix H gives the controller architecture (`film_controlnet`, hidden width 768, and 8 group-normalization groups). Controller optimization uses Adam at `5e-5`, zero weight decay, a constant schedule, batch size 8, one-step gradient accumulation, fp16, gradient norm 1.0, and intermediate loss weight 1.0. Training runs for 200,000 steps on Kolmogorov, 100,000 on KS, and 150,000 on ERA5-small.

Prior optimization uses Adam at the paper's `3.2e-4` learning rate with zero weight decay and a constant schedule. PDE datasets use explicit channelwise min/max normalization. Seeds, path layout, mask phase/alignment, metric domains, and aggregation order are listed under `repository_default_fields` with concrete typed values. Dataset checksums and source versions depend on the local copy of the data, so they are `null` and listed under `unresolved_fields`.

Evaluation files are standalone runtime configurations: each embeds the Appendix H controller model, uses `gamma=0.1`, references both prior and controller checkpoints, and explicitly sets `beta=0.01`, `intermediate_weight=1.0`, and `evaluation_trajectories=8`.

Load a file with `cada.config.load_config(path, env={"CADA_ROOT": "/absolute/root"})`. Use `cada.config.to_resolved_dict(config)` for a JSON-compatible dictionary with environment variables resolved.
