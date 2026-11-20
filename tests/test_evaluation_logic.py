from __future__ import annotations

from types import SimpleNamespace

import torch

import cada.evaluation as evaluation
from cada.config import load_config
from cada.data import MinMaxNormalizer


class SyntheticSource:
    n_trajectories = 8
    trajectory_length = 141

    def read_window(self, trajectory: int, start: int, stop: int) -> torch.Tensor:
        values = torch.arange(start, stop, dtype=torch.float32)
        return values[:, None, None].expand(-1, 1, 256) + trajectory


class EvalOnly:
    def eval(self) -> None:
        pass


def test_controlled_and_uncontrolled_rollouts_receive_identical_noise(
    repository_root,
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config(
        repository_root / "configs/evaluations/ks__ms_4__h140.yaml",
        env={"CADA_ROOT": str(tmp_path)},
    )
    captured = {}

    def uncontrolled(
        diffusion,
        initial,
        horizon,
        *,
        initial_noises,
        **kwargs,
    ):
        del diffusion, kwargs
        captured["uncontrolled"] = initial_noises
        states = torch.cat(
            [
                initial[:, None],
                torch.zeros(
                    initial.shape[0],
                    horizon,
                    *initial.shape[1:],
                ),
            ],
            dim=1,
        )
        return SimpleNamespace(states=states)

    def controlled(
        diffusion,
        controller,
        truth,
        observation_operator,
        cadence,
        *,
        initial_noises,
        **kwargs,
    ):
        del diffusion, controller, observation_operator, cadence, kwargs
        captured["controlled"] = initial_noises
        return SimpleNamespace(
            states=truth.clone(),
            observation_indices=(),
            chunk_boundaries=(),
        )

    def timed(function, *args, **kwargs):
        return function(*args, **kwargs), 0.0

    monkeypatch.setattr(evaluation, "uncontrolled_rollout", uncontrolled)
    monkeypatch.setattr(evaluation, "preview_rollout", controlled)
    monkeypatch.setattr(evaluation, "synchronized_cuda_time", timed)
    monkeypatch.setattr(evaluation, "compute_core_metrics", lambda *args, **kwargs: {})

    layout = SimpleNamespace(
        samples=tmp_path / "samples",
        metrics=tmp_path / "metrics",
        run_id="test-run",
    )
    evaluation.run_evaluation(
        config,
        source=SyntheticSource(),
        normalizer=MinMaxNormalizer([0.0], [1.0]),
        diffusion=EvalOnly(),
        controller=EvalOnly(),
        observation_operator=object(),
        cadence=object(),
        layout=layout,
        device="cpu",
    )

    assert captured["controlled"] is captured["uncontrolled"]
    expected = evaluation._deterministic_noises(
        torch.zeros(8, 141, 1, 256),
        horizon=140,
        seed=config.seeds.diffusion,
        device=torch.device("cpu"),
    )
    assert torch.equal(captured["controlled"], expected)
