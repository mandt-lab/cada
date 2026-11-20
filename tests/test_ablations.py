from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import cada.ablations as ablations
from cada.ablations import tto_da_rollout
from cada.cli import main
from cada.diffusion import GaussianDiffusion
from cada.observations import (
    AverageDownsample1D,
    Observation,
    TemporalCadence,
)


def test_tto_observation_cost_is_applied_only_at_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_step(
        diffusion,
        previous_state,
        active_observation,
        observation_operator,
        controls,
        initial_noise,
        time_pairs,
        **kwargs,
    ):
        del (
            diffusion,
            observation_operator,
            controls,
            initial_noise,
            time_pairs,
            kwargs,
        )
        next_state = previous_state + 1
        observation_loss = previous_state.flatten(1).sum(1) * 0
        if active_observation is not None:
            observation_loss = (
                (next_state - active_observation.value).square().flatten(1).mean(1)
            )
        return next_state, observation_loss, observation_loss * 0

    monkeypatch.setattr(ablations, "_free_control_step", fake_step)
    initial = torch.zeros(1, 1, 1)
    observation = Observation(
        value=torch.full_like(initial, 4.0),
        mask=torch.ones_like(initial),
    )
    states, objective = ablations._controlled_chunk(
        object(),
        initial,
        {4: observation},
        object(),
        torch.zeros(4, 1, 1, 1, 1),
        torch.zeros(4, 1, 1, 1),
        ((0, -1),),
        gamma=1.0,
        beta=0.0,
        intermediate_weight=1.0,
        clip_x_start=False,
    )

    assert torch.equal(states[-1], observation.value)
    assert objective == 0


def test_tto_improves_finite_objective_without_prior_gradients(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    trajectory = torch.tensor(
        [
            [
                [[0.1, 0.2, 0.3, 0.4]],
                [[0.2, 0.3, 0.4, 0.5]],
                [[0.7, 0.6, 0.5, 0.4]],
            ]
        ],
        dtype=torch.float32,
    )
    before = tuple(
        parameter.detach().clone() for parameter in tiny_diffusion.parameters()
    )

    result = tto_da_rollout(
        tiny_diffusion,
        trajectory,
        AverageDownsample1D(2),
        TemporalCadence(interval=2, start=2),
        preview_steps=2,
        iterations=8,
        lr=0.1,
        gamma=1.0,
        beta=1e-3,
        seed=23,
        sampling_steps=2,
        clip_x_start=False,
    )

    assert result.states.shape == trajectory.shape
    assert torch.isfinite(result.initial_objective)
    assert torch.isfinite(result.final_objective)
    assert result.final_objective < result.initial_objective
    assert result.gradient_norms
    assert all(torch.isfinite(torch.tensor(result.gradient_norms)))
    assert max(result.gradient_norms) > 0
    assert all(parameter.grad is None for parameter in tiny_diffusion.parameters())
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(tiny_diffusion.parameters(), before)
    )


def test_evaluation_dry_run_separates_method_paths(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository_root)
    root = tmp_path / "runs"
    config = "configs/evaluations/ks__ds_2__h140.yaml"
    documents = {}
    for method in ("cada", "tto-da"):
        exit_code = main(
            [
                "evaluate",
                "--config",
                config,
                "--method",
                method,
                "--root",
                str(root),
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.err == ""
        documents[method] = json.loads(captured.out)

    runs = {document["paths"]["run"] for document in documents.values()}
    assert len(runs) == 2
    assert "-cada-" in documents["cada"]["paths"]["run"]
    assert "-tto-da-" in documents["tto-da"]["paths"]["run"]
    assert "controller_checkpoint" in documents["cada"]["paths"]
    assert "controller_checkpoint" not in documents["tto-da"]["paths"]
    assert documents["tto-da"]["method"]["iterations"] == 50
    assert documents["tto-da"]["method"]["lr"] == 0.01
    assert not root.exists()
