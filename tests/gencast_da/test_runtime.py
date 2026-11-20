from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gencast_da.artifacts import (
    load_evaluation_npz,
    write_evaluation_npz,
    write_metrics,
)
from gencast_da.checkpointing import (
    fallback_path,
    load_trusted_checkpoint,
    save_trusted_checkpoint,
    truncate_history,
    validate_compatibility,
)
from gencast_da.cli import _log_structured_metrics, main
from gencast_da.config import ConfigError, load_config
from gencast_da.plot_cada_temperature import main as plot_cada_temperature
from gencast_da.runtime import RunLayout, config_digest, sha256_file
from gencast_da.tracking import init_tracker


@pytest.mark.parametrize(
    ("command", "config_name"),
    [
        ("data", "data.yaml"),
        ("smoke", "smoke.yaml"),
        ("train-controller", "controller.yaml"),
        ("evaluate", "evaluation.yaml"),
        ("figure", "figure.yaml"),
    ],
)
def test_gencast_cli_dry_runs_are_side_effect_free(
    repository_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    config_name: str,
) -> None:
    root = tmp_path / "cada"
    exit_code = main(
        [
            command,
            "--config",
            str(repository_root / "gencast_da" / "configs" / config_name),
            "--root",
            str(root),
            "--dry-run",
        ]
    )
    document = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert document["action"] == command
    assert document["root"] == str(root)
    assert not root.exists()


def test_gencast_cli_requires_cada_root_or_root_argument(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CADA_ROOT", raising=False)

    with pytest.raises(ValueError, match="CADA_ROOT is not set"):
        main(
            [
                "data",
                "--config",
                str(repository_root / "gencast_da/configs/data.yaml"),
                "--dry-run",
            ]
        )


def test_duplicate_gencast_config_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "duplicate.yaml"
    config.write_text("kind: data\nkind: smoke\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_config(config, cada_root=tmp_path)


def test_declarative_seeds_must_match_executable_seed(tmp_path: Path) -> None:
    config = tmp_path / "seed.yaml"
    config.write_text(
        """
kind: evaluation
config_id: gencast.evaluation.seed
stage: evaluation
module: gencast_da.run_cada_gencast
data_root: ${GENCAST_DATA_ROOT}
parameters:
  seed: 2
seeds:
  observation: 5
  diffusion: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="seed derivation"):
        load_config(config, cada_root=tmp_path)


def test_bounded_smoke_and_evaluation_ranges_are_validated(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(
        """
kind: smoke
config_id: gencast.smoke.invalid
stage: smoke
module: gencast_da.smoke_cada
data_root: ${GENCAST_DATA_ROOT}
parameters:
  seed: 0
  num-noise-levels: 20
seeds:
  model: 0
  observation: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="exactly four"):
        load_config(smoke, cada_root=tmp_path)


def test_checkpoint_override_is_part_of_resolved_identity(
    repository_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "controller.pkl"
    exit_code = main(
        [
            "evaluate",
            "--config",
            str(repository_root / "gencast_da/configs/evaluation.yaml"),
            "--root",
            str(tmp_path / "cada"),
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
        ]
    )
    document = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert document["config"]["parameters"]["ckpt"] == str(checkpoint)
    assert str(checkpoint) in document["command"]


def test_gencast_dry_run_returns_accepted_canonical_run_id(
    repository_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "train-controller",
        "--config",
        str(repository_root / "gencast_da/configs/controller.yaml"),
        "--root",
        str(tmp_path),
        "--dry-run",
    ]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    run_id = Path(first["paths"]["run"]).name

    assert main([*arguments, "--run-id", run_id]) == 0
    second = json.loads(capsys.readouterr().out)
    assert Path(second["paths"]["run"]).name == run_id


def test_gencast_run_layout_and_artifacts_are_atomic(tmp_path: Path) -> None:
    resolved = {"kind": "evaluation", "seed": 3}
    run_id = f"20260921T010203000000Z-{config_digest(resolved)[:12]}"
    layout = RunLayout.plan(
        root=tmp_path,
        stage="evaluation",
        config_id="gencast.test",
        resolved_config=resolved,
        run_id=run_id,
    )
    layout.create(resolved_config=resolved, repository=Path.cwd())
    assert layout.open_existing(resolved_config=resolved) == layout
    with pytest.raises(ValueError, match="resume config"):
        layout.open_existing(resolved_config={"kind": "changed"})

    artifact, artifact_checksum = write_evaluation_npz(
        layout.samples / "evaluation.npz",
        truth=np.ones((1, 2, 2, 1), dtype=np.float32),
        prediction=np.zeros((1, 2, 2, 1), dtype=np.float32),
        observations=np.ones((1, 2, 2, 1), dtype=np.float32),
        coordinates={
            "lead_hours": np.array([12]),
            "latitude": np.array([-1.0, 1.0]),
            "longitude": np.array([0.0, 1.0]),
            "pressure_level": np.array([500]),
            "observation_mask": np.array(
                [[1, 0], [0, 1]],
                dtype=np.uint8,
            ),
        },
        metadata={"method": "cada"},
    )
    metrics, metrics_checksum = write_metrics(
        layout.metrics / "evaluation.json",
        {"rmse": 1.0},
        metadata={"method": "cada"},
    )

    assert artifact_checksum.read_text().split()[0] == sha256_file(artifact)
    assert metrics_checksum.read_text().split()[0] == sha256_file(metrics)
    loaded = load_evaluation_npz(artifact)
    assert loaded.truth.shape == (1, 2, 2, 1)
    assert not tuple(layout.run_dir.rglob("*.tmp"))

    with pytest.raises(ValueError, match="truth and prediction"):
        write_evaluation_npz(
            layout.samples / "malformed.npz",
            truth=np.ones((1, 2, 2, 1), dtype=np.float32),
            prediction=np.ones((2, 2), dtype=np.float32),
            observations=np.ones((1, 2, 2, 1), dtype=np.float32),
            coordinates=loaded.coordinates,
            metadata={},
        )
    with pytest.raises(ValueError, match="non-empty"):
        write_evaluation_npz(
            layout.samples / "empty.npz",
            truth=np.empty((0, 2, 2, 1), dtype=np.float32),
            prediction=np.empty((0, 2, 2, 1), dtype=np.float32),
            observations=np.empty((0, 2, 2, 1), dtype=np.float32),
            coordinates={
                **loaded.coordinates,
                "lead_hours": np.array([], dtype=np.int64),
            },
            metadata={},
        )
    with pytest.raises(ValueError, match="reserved"):
        write_evaluation_npz(
            layout.samples / "reserved.npz",
            truth=loaded.truth,
            prediction=loaded.prediction,
            observations=loaded.observations,
            coordinates={**loaded.coordinates, "truth": np.ones(1)},
            metadata={},
        )
    with pytest.raises(ValueError, match="unsupported evaluation coordinates"):
        write_evaluation_npz(
            layout.samples / "extra-coordinate.npz",
            truth=loaded.truth,
            prediction=loaded.prediction,
            observations=loaded.observations,
            coordinates={
                **loaded.coordinates,
                "extra": np.array([object()], dtype=object),
            },
            metadata={},
        )
    with pytest.raises(ValueError, match="non-finite"):
        write_evaluation_npz(
            layout.samples / "nan-metadata.npz",
            truth=loaded.truth,
            prediction=loaded.prediction,
            observations=loaded.observations,
            coordinates=loaded.coordinates,
            metadata={"loss": float("nan")},
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        write_evaluation_npz(
            layout.samples / "reversed.npz",
            truth=loaded.truth,
            prediction=loaded.prediction,
            observations=loaded.observations,
            coordinates={
                **loaded.coordinates,
                "latitude": loaded.coordinates["latitude"][::-1],
            },
            metadata={},
        )

    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_evaluation_npz(artifact)


def test_gencast_temperature_figure_uses_artifact_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _ = write_evaluation_npz(
        tmp_path / "evaluation.npz",
        truth=np.ones((2, 2, 3, 1), dtype=np.float32),
        prediction=np.zeros((2, 2, 3, 1), dtype=np.float32),
        observations=np.ones((2, 2, 3, 1), dtype=np.float32),
        coordinates={
            "lead_hours": np.array([12, 24]),
            "latitude": np.array([-10.0, 10.0]),
            "longitude": np.array([0.0, 1.0, 2.0]),
            "pressure_level": np.array([700]),
            "observation_mask": np.ones((2, 3), dtype=np.uint8),
        },
        metadata={"method": "cada"},
    )
    output_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_cada_temperature",
            "--artifact",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--level-idx",
            "0",
            "--dpi",
            "40",
        ],
    )

    plot_cada_temperature()

    figure = output_dir / "temperature_level_0.png"
    provenance = output_dir / "temperature_level_0.provenance.json"
    assert figure.is_file() and figure.stat().st_size > 0
    document = json.loads(provenance.read_text(encoding="utf-8"))
    assert document["level_index"] == 0
    assert document["artifact_sha256"] == sha256_file(artifact)
    assert document["output_sha256"] == sha256_file(figure)
    assert document["pressure_level"] == 700
    assert document["lead_hours"] == [12, 24]
    assert document["rendering"]["dpi"] == 40


def test_resumed_tracking_does_not_replay_history(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    events = metrics / "events.jsonl"
    events.write_text('{"step": 1, "loss": 2.0}\n', encoding="utf-8")
    (metrics / "training-history.json").write_text(
        json.dumps([{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}]),
        encoding="utf-8",
    )

    class RecordingTracker:
        metrics_path = events

        def __init__(self) -> None:
            self.records = []

        def log(self, values, *, step=None) -> None:
            self.records.append((step, values))

    tracker = RecordingTracker()
    _log_structured_metrics(tracker, SimpleNamespace(metrics=metrics))

    assert tracker.records == [(2, {"step": 2, "loss": 1.0})]


def test_tracking_uses_supported_wandb_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wandb_dir = tmp_path / "wandb"
    metrics_dir = tmp_path / "metrics"
    wandb_dir.mkdir()
    metrics_dir.mkdir()
    init_kwargs = {}
    run = object()

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    monkeypatch.setenv("WANDB_DIR", "original")

    tracker = init_tracker(
        mode="online",
        wandb_dir=wandb_dir,
        metrics_dir=metrics_dir,
        config={"steps": 1},
        run_name="test-run",
    )

    assert tracker.run is run
    assert init_kwargs["mode"] == "online"
    assert "settings" not in init_kwargs


def test_checkpoint_corruption_falls_back_to_last_verified_copy(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "controller.pkl"
    save_trusted_checkpoint(
        checkpoint,
        {
            "schema_version": 1,
            "step": 1,
            "ema_total": 9.42,
            "ema_terminal": 8.75,
        },
    )
    save_trusted_checkpoint(
        checkpoint,
        {"schema_version": 1, "step": 2},
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")

    loaded, source = load_trusted_checkpoint(checkpoint)

    assert source == fallback_path(checkpoint)
    assert loaded["step"] == 1
    assert loaded["ema_total"] == 9.42
    assert loaded["ema_terminal"] == 8.75

    save_trusted_checkpoint(
        checkpoint,
        {"schema_version": 1, "step": 3},
    )
    recovered, source = load_trusted_checkpoint(checkpoint)
    assert source == checkpoint
    assert recovered["step"] == 3


def test_checkpoint_compatibility_rejects_sampler_or_source_changes() -> None:
    active = {
        "checkpoint_sha256": "a" * 64,
        "cada_source_sha256": "b" * 64,
        "sampler_config": {"rho": 7.0},
    }
    validate_compatibility(dict(active), active)
    for changed in (
        {**active, "checkpoint_sha256": "c" * 64},
        {**active, "cada_source_sha256": "d" * 64},
        {**active, "sampler_config": {"rho": 8.0}},
    ):
        with pytest.raises(ValueError, match="incompatible"):
            validate_compatibility(changed, active)


def test_fallback_resume_truncates_future_history() -> None:
    records = [
        {"step": 1, "loss": 3.0},
        {"step": 2, "loss": 2.0},
        {"step": 3, "loss": 1.0},
    ]

    assert truncate_history(records, 2) == records[:2]
