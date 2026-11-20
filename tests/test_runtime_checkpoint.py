from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import TinyController, TinyPrior
from torch.utils.data import DataLoader

from cada.artifacts import sha256_file
from cada.checkpoint import (
    CheckpointCallbackAdapter,
    CheckpointSignalHandler,
    capture_rng_state,
    discover_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    restore_trainer_checkpoint,
    save_checkpoint,
)
from cada.cli import _resume_trainer, validate_controller_prior
from cada.data import ResumableRandomSampler
from cada.diffusion import GaussianDiffusion
from cada.observations import BernoulliMask, TemporalCadence
from cada.runtime import RunLayout
from cada.tracking import flatten_scalars
from cada.training import ControllerTrainer, PriorTrainer


class RecordingTrainer:
    def __init__(self) -> None:
        self.loaded: dict[str, object] | None = None
        self.strict: bool | None = None
        self.device = torch.device("cpu")

    def load_state_dict(self, state: dict[str, object], *, strict: bool = True) -> None:
        self.loaded = state
        self.strict = strict


def test_run_layout_is_complete_and_resumable(tmp_path: Path) -> None:
    root = tmp_path / "cada"
    config = {"kind": "prior", "config_id": "prior.tiny", "value": 3}

    layout = RunLayout.create(
        dataset="ks",
        stage="prior",
        experiment_id="prior.tiny",
        resolved_config=config,
        root=root,
        repository=tmp_path,
    )
    resumed = RunLayout.create(
        dataset="ks",
        stage="prior",
        experiment_id="prior.tiny",
        resolved_config=config,
        root=root,
        resume_existing=True,
        repository=tmp_path,
    )

    assert resumed.run_dir == layout.run_dir
    assert all(path.is_dir() for path in layout.directories)
    assert layout.resolved_config_path.is_file()
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == layout.run_id
    assert not list(layout.run_dir.rglob("*.tmp"))


def test_atomic_checkpoint_save_load_restore_latest_and_signal(
    tmp_path: Path,
) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    rng_state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(2))
    restore_rng_state(rng_state, restore_cuda=False)
    actual = (random.random(), float(np.random.rand()), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    checkpoint_dir = tmp_path / "checkpoints"
    state = {"weight": torch.tensor([1.0])}
    first = save_checkpoint(
        checkpoint_dir,
        step=1,
        trainer_state=state,
        resolved_config={"kind": "tiny"},
        rng_state=rng_state,
    )
    second = save_checkpoint(
        checkpoint_dir,
        step=2,
        trainer_state={"weight": torch.tensor([2.0])},
        resolved_config={"kind": "tiny"},
        rng_state=rng_state,
    )

    assert first.name == "step_000000001.pt"
    assert discover_latest_checkpoint(checkpoint_dir) == second
    loaded = load_checkpoint(checkpoint_dir)
    assert loaded["step"] == 2
    assert torch.equal(loaded["trainer_state"]["weight"], torch.tensor([2.0]))
    trainer = RecordingTrainer()
    restore_trainer_checkpoint(trainer, loaded, restore_rng=False)
    assert trainer.strict is True
    assert trainer.loaded is not None
    assert torch.equal(trainer.loaded["weight"], torch.tensor([2.0]))

    handler = CheckpointSignalHandler(signals=(signal.SIGUSR1,))
    adapter = CheckpointCallbackAdapter(
        checkpoint_dir,
        {"kind": "tiny"},
        signal_handler=handler,
    )
    handler.request(signal.SIGUSR1)
    assert adapter.save_if_requested(3, state, {"loss": 0.5})
    assert not handler.requested
    assert discover_latest_checkpoint(checkpoint_dir).name == "step_000000003.pt"
    assert load_checkpoint(checkpoint_dir)["metadata"]["requested_by_signal"] == int(
        signal.SIGUSR1
    )
    assert not list(checkpoint_dir.glob("*.tmp"))


def test_overflowed_gradient_step_still_logs_and_checkpoints(
    tmp_path: Path,
) -> None:
    diffusion = GaussianDiffusion(TinyPrior(), timesteps=4, sampling_steps=2)
    diffusion.model.projection.weight.register_hook(
        lambda grad: torch.full_like(grad, float("inf"))
    )
    logged: list[dict[str, object]] = []
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = PriorTrainer(
        diffusion,
        [torch.linspace(0.0, 1.0, 2 * 2 * 1 * 8).reshape(2, 2, 1, 8)],
        learning_rate=1e-2,
        device="cpu",
        precision="fp32",
        log_callback=lambda metrics: logged.append(flatten_scalars(metrics)),
        checkpoint_callback=CheckpointCallbackAdapter(
            checkpoint_dir, {"kind": "tiny"}
        ),
        checkpoint_every=1,
    )

    metrics = trainer.train_step()

    assert metrics.grad_norm is None
    assert logged[0]["grad_norm"] is None
    assert load_checkpoint(checkpoint_dir)["metadata"]["metrics"]["grad_norm"] is None


def test_prior_resume_reproduces_next_batch_and_update() -> None:
    dataset = torch.linspace(0.0, 1.0, 4 * 2 * 1 * 8).reshape(4, 2, 1, 8)

    def make_trainer() -> PriorTrainer:
        generator = torch.Generator().manual_seed(91)
        sampler = ResumableRandomSampler(dataset, generator)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
        )
        diffusion = GaussianDiffusion(
            TinyPrior(),
            timesteps=4,
            sampling_steps=2,
        )
        return PriorTrainer(
            diffusion,
            loader,
            learning_rate=1e-2,
            device="cpu",
            precision="fp32",
            ema_decay=0.5,
        )

    torch.manual_seed(17)
    uninterrupted = make_trainer()
    uninterrupted.train_step()
    uninterrupted_metrics = uninterrupted.train_step()

    torch.manual_seed(17)
    interrupted = make_trainer()
    interrupted.train_step()
    saved_state = interrupted.state_dict()
    saved_rng = torch.get_rng_state()

    resumed = make_trainer()
    resumed.load_state_dict(saved_state)
    torch.set_rng_state(saved_rng)
    resumed_metrics = resumed.train_step()

    assert resumed_metrics.loss == uninterrupted_metrics.loss
    for name, expected in uninterrupted.state_dict()["diffusion"].items():
        assert torch.equal(resumed.state_dict()["diffusion"][name], expected)
    for name, expected in uninterrupted.state_dict()["ema_model"].items():
        assert torch.equal(resumed.state_dict()["ema_model"][name], expected)


def test_resume_rejects_config_and_step_mismatch(tmp_path: Path) -> None:
    checkpoint = save_checkpoint(
        tmp_path,
        step=2,
        trainer_state={"step": 2},
        resolved_config={"kind": "prior", "value": 1},
    )
    args = SimpleNamespace(resume=True, resume_checkpoint=str(checkpoint))
    layout = SimpleNamespace(checkpoints=tmp_path)

    with pytest.raises(ValueError, match="configuration"):
        _resume_trainer(
            args,
            RecordingTrainer(),
            layout,
            {"kind": "prior", "value": 2},
        )

    mismatched = save_checkpoint(
        tmp_path / "mismatch",
        step=2,
        trainer_state={"step": 1},
        resolved_config={"kind": "prior", "value": 1},
    )
    args.resume_checkpoint = str(mismatched)
    with pytest.raises(ValueError, match="step"):
        _resume_trainer(
            args,
            RecordingTrainer(),
            layout,
            {"kind": "prior", "value": 1},
        )


def test_controller_prior_identity_is_enforced(tmp_path: Path) -> None:
    expected_prior = tmp_path / "expected.pt"
    other_prior = tmp_path / "other.pt"
    expected_prior.write_bytes(b"expected")
    other_prior.write_bytes(b"other")
    checkpoint = {
        "metadata": {
            "prior_checkpoint_sha256": sha256_file(expected_prior),
        }
    }

    validate_controller_prior(checkpoint, expected_prior)
    with pytest.raises(ValueError, match="different prior"):
        validate_controller_prior(checkpoint, other_prior)
    with pytest.raises(ValueError, match="does not identify"):
        validate_controller_prior({"metadata": {}}, expected_prior)


def test_controller_resume_restores_sampler_optimizer_rng_and_observation() -> None:
    dataset = torch.linspace(0.0, 1.0, 4 * 3 * 1 * 8).reshape(4, 3, 1, 8)

    def make_trainer() -> ControllerTrainer:
        sampler = ResumableRandomSampler(
            dataset,
            torch.Generator().manual_seed(19),
        )
        loader = DataLoader(dataset, batch_size=1, sampler=sampler)
        diffusion = GaussianDiffusion(
            TinyPrior(),
            timesteps=4,
            sampling_steps=2,
        )
        return ControllerTrainer(
            diffusion,
            TinyController(),
            loader,
            BernoulliMask(spatial_dims=1, keep_probability=0.5, seed=23),
            TemporalCadence(interval=2, start=2),
            preview_steps=2,
            gamma=0.2,
            beta=0.01,
            learning_rate=1e-2,
            sampling_steps=2,
            clip_x_start=False,
            device="cpu",
            precision="fp32",
        )

    torch.manual_seed(29)
    uninterrupted = make_trainer()
    uninterrupted.train_step()
    expected_metrics = uninterrupted.train_step()

    torch.manual_seed(29)
    interrupted = make_trainer()
    interrupted.train_step()
    saved_state = interrupted.state_dict()
    saved_rng = torch.get_rng_state()

    resumed = make_trainer()
    resumed.load_state_dict(saved_state)
    torch.set_rng_state(saved_rng)
    actual_metrics = resumed.train_step()

    assert actual_metrics.total_loss == expected_metrics.total_loss
    assert actual_metrics.observation_loss == expected_metrics.observation_loss
    for name, expected in uninterrupted.controller.state_dict().items():
        assert torch.equal(resumed.controller.state_dict()[name], expected)
