from __future__ import annotations

import json
import os
from math import pi
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
import yaml

import cada.evaluation as evaluation
from cada.artifacts import (
    build_evaluation_artifact,
    write_evaluation_artifact,
    write_sha256_sidecar,
)
from cada.cli import _dry_figure_manifest
from cada.config import load_config
from cada.data import MinMaxNormalizer
from cada.evaluation import compute_core_metrics, physical_artifact_trajectories
from cada.figures import (
    FigureManifest,
    _color_limits,
    _one_dimensional_matrix,
    _two_dimensional_series,
    load_evaluation_artifact,
    render_figure,
)
from cada.figures import sha256_file as figure_sha256_file
from cada.metrics import (
    high_correlation_time,
    periodic_total_variation_1d,
    rmse,
    vorticity_2d_finite_difference,
    vorticity_2d_spectral,
)


@pytest.mark.parametrize("suffix", [".pt", ".npz", ".PT", ".NPZ"])
def test_strict_evaluation_artifact_round_trip(tmp_path: Path, suffix: str) -> None:
    truth = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    path = tmp_path / f"evaluation{suffix}"

    write_evaluation_artifact(
        path,
        truth=truth,
        uncontrolled=truth + 1,
        controlled=truth + 0.5,
        metadata={
            "artifact_space": "physical_space",
            "config_id": "tiny",
            "seed": 7,
        },
    )
    loaded = load_evaluation_artifact(path)

    assert torch.equal(loaded.truth, truth)
    assert loaded.metadata["schema_version"] == 2
    assert loaded.metadata["config_id"] == "tiny"
    assert not list(tmp_path.glob("*.tmp"))

    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_evaluation_artifact(path)

    with pytest.raises(ValueError, match="identical shapes"):
        build_evaluation_artifact(
            truth=truth,
            uncontrolled=truth[:, :2],
            controlled=truth,
            metadata={},
        )


def test_artifacts_reject_nonfinite_and_unknown_schema(tmp_path: Path) -> None:
    truth = torch.zeros(1, 2, 1, 4)
    invalid = truth.clone()
    invalid[..., 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        build_evaluation_artifact(
            truth=invalid,
            uncontrolled=truth,
            controlled=truth,
            metadata={},
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_evaluation_artifact(
            truth=torch.empty(0, 2, 1, 4),
            uncontrolled=torch.empty(0, 2, 1, 4),
            controlled=torch.empty(0, 2, 1, 4),
            metadata={"artifact_space": "physical_space"},
        )
    with pytest.raises(ValueError, match="channels"):
        build_evaluation_artifact(
            truth=truth,
            uncontrolled=truth,
            controlled=truth,
            metadata={
                "artifact_space": "physical_space",
                "resolved_config": {
                    "dataset": {"physical_channels": 2},
                    "horizon": {"forecast_steps": 1},
                },
            },
        )

    valid_path = tmp_path / "valid.npz"
    write_evaluation_artifact(
        valid_path,
        truth=truth,
        uncontrolled=truth,
        controlled=truth,
        metadata={"artifact_space": "physical_space"},
    )
    with np.load(valid_path, allow_pickle=False) as payload:
        rewritten = {key: np.array(payload[key]) for key in payload.files}
    rewritten["schema_version"] = np.asarray(999, dtype=np.int64)
    invalid_path = tmp_path / "invalid.npz"
    np.savez_compressed(invalid_path, **rewritten)
    write_sha256_sidecar(invalid_path)

    with pytest.raises(ValueError, match="unsupported evaluation artifact"):
        load_evaluation_artifact(invalid_path)

    rewritten["schema_version"] = np.asarray(2, dtype=np.int64)
    metadata = json.loads(rewritten["metadata"].tobytes().decode("utf-8"))
    metadata.pop("artifact_space")
    rewritten["metadata"] = np.frombuffer(
        json.dumps(metadata).encode("utf-8"),
        dtype=np.uint8,
    )
    missing_space = tmp_path / "missing-space.npz"
    np.savez_compressed(missing_space, **rewritten)
    write_sha256_sidecar(missing_space)
    with pytest.raises(ValueError, match="physical_space"):
        load_evaluation_artifact(missing_space)


def test_metric_formulas_include_hct_equality_tv_and_vorticity() -> None:
    prediction = torch.tensor([[[0.0, 2.0]]])
    truth = torch.zeros_like(prediction)
    assert torch.allclose(rmse(prediction, truth), torch.tensor([[2**0.5]]))

    correlation = torch.tensor([0.95, 0.90, 0.89])
    hct = high_correlation_time(correlation, threshold=0.9, dt=0.2)
    assert hct.item() == pytest.approx(0.4)

    values = torch.tensor([[[[0.0, 1.0, 3.0]]]])
    assert periodic_total_variation_1d(values).item() == 6.0

    width = 16
    height = 8
    dx = 2 * pi / width
    x = torch.arange(width, dtype=torch.float32) * dx
    velocity = torch.zeros(1, 1, 2, height, width)
    velocity[..., 1, :, :] = torch.sin(x)
    expected_spectral = torch.cos(x).expand(1, 1, height, width)
    expected_finite = expected_spectral * (np.sin(dx) / dx)

    spectral = vorticity_2d_spectral(velocity, dx=dx, dy=1.0)
    finite = vorticity_2d_finite_difference(velocity, dx=dx, dy=1.0)

    assert torch.allclose(spectral, expected_spectral, atol=2e-5)
    assert torch.allclose(finite, expected_finite, atol=2e-6)


def test_figure_timestep_selection_and_shared_limits_are_exact() -> None:
    trajectory = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)
    selected = _one_dimensional_matrix(
        trajectory,
        trajectory=0,
        channel=0,
        timesteps=(0, 2, 4),
    )
    limits = _color_limits(
        [np.array([[-3.0, 1.0]]), np.array([[2.0]])],
        {"symmetric": True},
    )

    assert np.array_equal(selected[:, 0], np.array([0.0, 2.0, 4.0]))
    assert limits == (-3.0, 3.0)
    with pytest.raises(ValueError, match=r"\[B,T,C,H,W\]"):
        _two_dimensional_series(
            torch.zeros(1, 2, 1, 8),
            trajectory=0,
            rendering={"field": "channel"},
        )


def test_evaluation_artifacts_are_denormalized_per_channel() -> None:
    normalizer = MinMaxNormalizer([0.0, 100.0], [10.0, 200.0])
    normalized = torch.full((1, 2, 2, 1), 0.5)

    truth, uncontrolled, controlled = physical_artifact_trajectories(
        normalizer,
        truth=normalized,
        uncontrolled=normalized,
        controlled=normalized,
    )

    expected = torch.tensor([5.0, 150.0]).reshape(1, 1, 2, 1).expand_as(truth)
    assert torch.equal(truth, expected)
    assert torch.equal(uncontrolled, expected)
    assert torch.equal(controlled, expected)


def test_evaluation_noise_is_private_deterministic_and_seeded() -> None:
    trajectory = torch.zeros(2, 3, 1, 4)

    first = evaluation._deterministic_noises(
        trajectory,
        horizon=2,
        seed=11,
        device=torch.device("cpu"),
    )
    second = evaluation._deterministic_noises(
        trajectory,
        horizon=2,
        seed=11,
        device=torch.device("cpu"),
    )
    different = evaluation._deterministic_noises(
        trajectory,
        horizon=2,
        seed=12,
        device=torch.device("cpu"),
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_multichannel_metric_domains_match_independent_oracle(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    config = load_config(
        repository_root / "configs/evaluations/era5_small__ms_4__h60.yaml",
        env={"CADA_ROOT": str(tmp_path)},
    )
    truth = torch.full((1, 61, 3, 64, 64), 0.5)
    offsets = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 1, 3, 1, 1)
    controlled = truth + offsets
    normalizer = MinMaxNormalizer(
        [0.0, 10.0, 100.0],
        [2.0, 14.0, 110.0],
    )

    metrics = compute_core_metrics(
        config,
        truth=truth,
        uncontrolled=truth,
        controlled=controlled,
        normalizer=normalizer,
    )

    expected_model_rmse = float(
        torch.sqrt(torch.tensor((0.2**2 + 0.4**2 + 0.6**2) / 3))
    )
    assert metrics["controlled"]["rmse_mean"] == pytest.approx(
        expected_model_rmse
    )
    assert metrics["uncontrolled"]["rmse_mean"] == 0.0
    assert metrics["controlled"]["physics_mae_mean"] == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_agg_figure_renders_with_provenance(
    tmp_path: Path,
) -> None:
    assert matplotlib.get_backend().lower() == "agg"
    trajectory = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    artifact = tmp_path / "evaluation.npz"
    write_evaluation_artifact(
        artifact,
        truth=trajectory,
        uncontrolled=trajectory + 1,
        controlled=trajectory + 0.25,
        metadata={
            "artifact_space": "physical_space",
            "config_id": "tiny",
            "initial_noise_seed": 3,
        },
    )
    figure_output = tmp_path / "figure.png"
    figure_manifest = FigureManifest.from_dict(
        {
            "figure_type": "spacetime_1d",
            "artifact": str(artifact),
            "output": str(figure_output),
            "config_ids": ["tiny"],
            "seeds": [3],
            "rendering": {"dpi": 40, "colorbar": False},
        },
        cada_root=tmp_path,
    )

    rendered, provenance = render_figure(figure_manifest)

    assert rendered.is_file() and rendered.stat().st_size > 0
    figure_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    assert figure_provenance["config_ids"] == ["tiny"]
    assert len(figure_provenance["inputs"][0]["sha256"]) == 64
    assert figure_provenance["output_sha256"] == figure_sha256_file(rendered)
    assert len(figure_provenance["manifest_sha256"]) == 64
    assert figure_provenance["libraries"]["matplotlib"]


def test_svg_render_is_deterministic(tmp_path: Path) -> None:
    trajectory = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    artifact = tmp_path / "evaluation.npz"
    write_evaluation_artifact(
        artifact,
        truth=trajectory,
        uncontrolled=trajectory + 1,
        controlled=trajectory + 0.25,
        metadata={"artifact_space": "physical_space"},
    )

    outputs = []
    manifest_hashes = []
    output_path = tmp_path / "repeat.svg"
    for _ in range(2):
        manifest = FigureManifest.from_dict(
            {
                "figure_type": "spacetime_1d",
                "artifact": str(artifact),
                "output": str(output_path),
                "rendering": {"dpi": 40, "colorbar": False},
            },
            cada_root=tmp_path,
        )
        output, provenance = render_figure(
            manifest,
            overwrite=bool(outputs),
        )
        outputs.append(output.read_bytes())
        manifest_hashes.append(
            json.loads(provenance.read_text(encoding="utf-8"))["manifest_sha256"]
        )

    assert outputs[0] == outputs[1]
    assert manifest_hashes[0] == manifest_hashes[1]

    titled = FigureManifest.from_dict(
        {
            "figure_type": "spacetime_1d",
            "artifact": str(artifact),
            "output": str(output_path),
            "rendering": {
                "dpi": 40,
                "colorbar": False,
                "title": "Different title",
            },
        },
        cada_root=tmp_path,
    )
    _, titled_provenance = render_figure(titled, overwrite=True)
    titled_hash = json.loads(
        titled_provenance.read_text(encoding="utf-8")
    )["manifest_sha256"]
    assert titled_hash != manifest_hashes[0]


def test_manifest_identity_must_match_artifact_metadata(tmp_path: Path) -> None:
    trajectory = torch.zeros(1, 2, 1, 4)
    artifact = tmp_path / "evaluation.npz"
    write_evaluation_artifact(
        artifact,
        truth=trajectory,
        uncontrolled=trajectory,
        controlled=trajectory,
        metadata={
            "artifact_space": "physical_space",
            "config_id": "evaluation.correct",
            "checkpoint_ids": ["prior.correct", "controller.correct"],
            "initial_noise_seed": 2,
        },
    )
    manifest = FigureManifest.from_dict(
        {
            "figure_type": "spacetime_1d",
            "artifact": str(artifact),
            "output": str(tmp_path / "figure.png"),
            "config_ids": ["evaluation.wrong"],
            "checkpoint_ids": ["prior.correct", "controller.correct"],
            "seeds": [2],
        },
        cada_root=tmp_path,
    )
    output = tmp_path / "figure.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(ValueError, match="config_ids"):
        render_figure(manifest, overwrite=True)
    assert output.read_bytes() == b"sentinel"


def test_failed_render_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    output.write_bytes(b"sentinel")
    manifest = FigureManifest.from_dict(
        {
            "figure_type": "spacetime_1d",
            "artifact": str(tmp_path / "missing.npz"),
            "output": str(output),
        },
        cada_root=tmp_path,
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        render_figure(manifest)
    with pytest.raises(ValueError, match="missing"):
        render_figure(manifest, overwrite=True)
    assert output.read_bytes() == b"sentinel"

    lock = output.with_suffix(".png.lock")
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already in progress"):
        render_figure(manifest, overwrite=True)


def test_external_baseline_artifacts_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside this repository scope"):
        FigureManifest.from_dict(
            {
                "figure_type": "spacetime_1d",
                "artifact": str(tmp_path / "primary.npz"),
                "output": str(tmp_path / "figure.png"),
                "baseline_artifacts": [
                    {
                        "label": "External",
                        "path": str(tmp_path / "external.npz"),
                    }
                ],
            },
            cada_root=tmp_path,
        )

    manifest_path = tmp_path / "baseline.yaml"
    manifest_path.write_text(
        f"""
figure_type: spacetime_1d
artifact: {tmp_path / 'primary.npz'}
output: {tmp_path / 'figure.png'}
baseline_artifacts:
  - label: External
    path: {tmp_path / 'external.npz'}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside this repository scope"):
        _dry_figure_manifest(manifest_path, cada_root=tmp_path)


@pytest.mark.parametrize("timesteps", [[1.9], [True], [2, 1]])
def test_manifest_timesteps_are_strict_integers(
    tmp_path: Path,
    timesteps: list,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FigureManifest.from_dict(
            {
                "figure_type": "snapshot_grid_2d",
                "artifact": str(tmp_path / "artifact.npz"),
                "output": str(tmp_path / "figure.png"),
                "timesteps": timesteps,
            },
            cada_root=tmp_path,
        )


def test_manifest_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """
figure_type: spacetime_1d
artifact: first.npz
artifact: second.npz
output: figure.png
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        FigureManifest.from_yaml(path, cada_root=tmp_path)


@pytest.mark.parametrize("trajectory", [True, 1.9])
def test_manifest_trajectory_is_not_coerced(
    tmp_path: Path,
    trajectory,
) -> None:
    with pytest.raises(TypeError, match="trajectory must be an integer"):
        FigureManifest.from_dict(
            {
                "figure_type": "spacetime_1d",
                "artifact": str(tmp_path / "artifact.npz"),
                "output": str(tmp_path / "figure.png"),
                "trajectory": trajectory,
            },
            cada_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("channel", 1.9), ("u_channel", True), ("dpi", 0)],
)
def test_manifest_rendering_integers_are_strict(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FigureManifest.from_dict(
            {
                "figure_type": "spacetime_1d",
                "artifact": str(tmp_path / "artifact.npz"),
                "output": str(tmp_path / "figure.png"),
                "rendering": {field: value},
            },
            cada_root=tmp_path,
        )


def test_shipped_manifests_declare_no_external_methods(
    repository_root: Path,
) -> None:
    for path in sorted((repository_root / "configs/figures").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not document.get("baseline_artifacts")
        assert not document.get("baselines")
