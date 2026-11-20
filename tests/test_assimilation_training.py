from __future__ import annotations

import torch
from conftest import TinyController, TinyPrior
from torch import nn

import cada.assimilation as assimilation
from cada.assimilation import (
    RolloutResult,
    StepMetrics,
    preview_rollout,
)
from cada.control import ControlContext
from cada.diffusion import GaussianDiffusion
from cada.observations import (
    AverageDownsample1D,
    BernoulliMask,
    RegularGridMask,
    TemporalCadence,
)
from cada.training import ControllerTrainer, PriorTrainer


def _trajectory() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
                [[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
                [[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )


class RecordingController(nn.Module):
    physical_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.0))
        self.records: list[tuple[float, float, float]] = []
        self.masks: list[torch.Tensor] = []

    def forward(
        self,
        noisy_state: torch.Tensor,
        previous_state: torch.Tensor,
        observation: torch.Tensor,
        mask: torch.Tensor,
        previous_control: torch.Tensor,
        context: ControlContext,
    ) -> torch.Tensor:
        del previous_state, observation, previous_control
        self.records.append(
            (
                float(context.dt[0]),
                float(context.tau[0]),
                float(mask.sum()),
            )
        )
        self.masks.append(mask.detach().clone())
        return noisy_state * self.gain


def _step_metric(
    *,
    observation_loss: float,
    dense_loss: float,
    observation_available: bool,
) -> StepMetrics:
    zero = torch.tensor(0.0)
    return StepMetrics(
        controlled=True,
        observation_available=observation_available,
        target_index=1 if observation_available else None,
        total_loss=torch.tensor(observation_loss + dense_loss),
        observation_loss=torch.tensor(observation_loss),
        final_observation_mse=zero,
        intermediate_observation_mse=zero,
        final_full_state_rmse=zero,
        intermediate_full_state_rmse=zero,
        weighted_full_state_rmse=torch.tensor(dense_loss),
        control_rms=zero,
        denoising_steps=2,
    )


def test_preview_visibility_is_bounded_by_fixed_chunk(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    controller = RecordingController()
    trajectory = torch.linspace(0.0, 0.8, 9).reshape(1, 9, 1, 1).expand(-1, -1, -1, 8)
    rollout = preview_rollout(
        tiny_diffusion,
        controller,
        trajectory,
        AverageDownsample1D(2),
        TemporalCadence(interval=4, start=4),
        preview_steps=5,
        gamma=0.2,
        beta=0.01,
        initial_noises=torch.zeros(8, 1, 1, 8),
        sampling_steps=2,
        clip_x_start=False,
    )

    physical_records = controller.records[::2]
    assert len(physical_records) == 8
    assert rollout.chunk_boundaries == (5, 8)
    assert [metric.target_index for metric in rollout.step_metrics] == [
        4,
        4,
        4,
        4,
        None,
        8,
        8,
        8,
    ]
    assert [record[2] > 0 for record in physical_records] == [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert torch.allclose(
        torch.tensor([record[1] for record in physical_records]),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.5, 1.0]),
    )


def test_preview_reuses_one_bernoulli_mask_per_arrival(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    controller = RecordingController()
    trajectory = torch.rand(1, 5, 1, 8)
    preview_rollout(
        tiny_diffusion,
        controller,
        trajectory,
        BernoulliMask(spatial_dims=1, keep_probability=0.5, seed=9),
        TemporalCadence(interval=4, start=4),
        preview_steps=4,
        gamma=0.2,
        beta=0.01,
        initial_noises=torch.zeros(4, 1, 1, 8),
        sampling_steps=2,
        clip_x_start=False,
    )

    physical_masks = controller.masks[::2]
    assert len(physical_masks) == 4
    assert all(torch.equal(mask, physical_masks[0]) for mask in physical_masks[1:])


def test_rollout_loss_normalizes_arrivals_and_dense_steps_separately() -> None:
    metrics = (
        _step_metric(
            observation_loss=0.0,
            dense_loss=1.0,
            observation_available=False,
        ),
        _step_metric(
            observation_loss=0.0,
            dense_loss=1.0,
            observation_available=False,
        ),
        _step_metric(
            observation_loss=0.0,
            dense_loss=1.0,
            observation_available=False,
        ),
        _step_metric(
            observation_loss=8.0,
            dense_loss=1.0,
            observation_available=True,
        ),
    )
    rollout = RolloutResult(
        states=torch.zeros(1, 5, 1, 1),
        step_metrics=metrics,
        observation_indices=(4,),
        chunk_boundaries=(4,),
    )

    assert rollout.loss == 9.0


def test_exact_tweedie_aggregation_and_full_field_rmse() -> None:
    final, intermediate, total = assimilation._aggregate_substep_losses(
        [torch.tensor(1.0), torch.tensor(4.0), torch.tensor(9.0)],
        intermediate_weight=1.0,
    )
    full_rmse = assimilation._full_state_rmse(
        torch.tensor([[[3.0, 4.0]]]),
        torch.zeros(1, 1, 2),
    )

    assert final == 9.0
    assert intermediate == 2.5
    assert total == 11.5
    assert full_rmse == torch.sqrt(torch.tensor(12.5))


def test_unobserved_truth_changes_loss_but_not_forecast(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    controller = TinyController()
    first_truth = _trajectory()[:, :3].clone()
    second_truth = first_truth.clone()
    second_truth[:, 1:, :, 1::2] = 1.0 - second_truth[:, 1:, :, 1::2]
    noises = torch.zeros(2, 1, 1, 8)
    arguments = dict(
        diffusion=tiny_diffusion,
        controller=controller,
        observation_operator=RegularGridMask(1, 2, 0),
        cadence=TemporalCadence(interval=2, start=2),
        preview_steps=2,
        gamma=0.2,
        beta=0.01,
        initial_noises=noises,
        sampling_steps=2,
        clip_x_start=False,
    )

    first = preview_rollout(trajectory=first_truth, **arguments)
    second = preview_rollout(trajectory=second_truth, **arguments)

    assert torch.equal(first.states, second.states)
    assert not torch.equal(first.loss, second.loss)


def test_controlled_rollout_uses_dense_tail_without_reanchoring(
    tiny_diffusion: GaussianDiffusion,
) -> None:
    tiny_diffusion.requires_grad_(False)
    controller = TinyController()
    operator = AverageDownsample1D(2)
    trajectory = _trajectory()
    noises = torch.zeros(3, 1, 1, 8)

    rollout = preview_rollout(
        tiny_diffusion,
        controller,
        trajectory,
        operator,
        TemporalCadence(interval=2, start=2),
        preview_steps=2,
        gamma=0.2,
        beta=0.01,
        initial_noises=noises,
        sampling_steps=2,
        clip_x_start=False,
    )
    rollout.loss.backward()

    assert rollout.states.shape == trajectory.shape
    assert rollout.observation_indices == (2,)
    assert rollout.chunk_boundaries == (2, 3)
    unobserved, observed, tail = rollout.step_metrics
    assert unobserved.controlled
    assert not unobserved.observation_available
    assert unobserved.observation_loss == 0
    assert unobserved.final_full_state_rmse > 0
    assert unobserved.intermediate_full_state_rmse > 0
    assert torch.allclose(
        unobserved.weighted_full_state_rmse,
        0.01
        * (unobserved.final_full_state_rmse + unobserved.intermediate_full_state_rmse),
    )
    assert torch.allclose(
        unobserved.total_loss,
        unobserved.weighted_full_state_rmse,
    )
    assert observed.controlled
    assert observed.observation_available
    assert observed.final_observation_mse > 0
    assert observed.intermediate_observation_mse > 0
    assert torch.allclose(
        observed.observation_loss,
        observed.final_observation_mse + observed.intermediate_observation_mse,
    )
    assert observed.final_full_state_rmse > 0
    assert observed.intermediate_full_state_rmse > 0
    assert observed.weighted_full_state_rmse > 0
    assert torch.allclose(
        observed.weighted_full_state_rmse,
        0.01 * (observed.final_full_state_rmse + observed.intermediate_full_state_rmse),
    )
    assert torch.allclose(
        observed.total_loss,
        observed.observation_loss + observed.weighted_full_state_rmse,
    )
    assert tail.controlled
    assert not tail.observation_available
    assert tail.target_index is None
    assert tail.observation_loss == 0
    assert tail.weighted_full_state_rmse > 0
    assert controller.gain.grad is not None
    assert torch.isfinite(controller.gain.grad)
    assert not torch.allclose(
        operator(rollout.states[:, 2]).value,
        operator(trajectory[:, 2]).value,
        atol=1e-6,
    )


def test_prior_and_controller_trainers_each_take_one_cpu_step() -> None:
    prior_diffusion = GaussianDiffusion(TinyPrior(), timesteps=4, sampling_steps=2)
    prior_batch = _trajectory()[:, :2]
    prior_before = prior_diffusion.model.projection.weight.detach().clone()
    prior_trainer = PriorTrainer(
        prior_diffusion,
        [prior_batch],
        learning_rate=1e-2,
        device="cpu",
        precision="fp32",
        ema_decay=0.5,
    )

    prior_metrics = prior_trainer.train_step()

    assert prior_metrics.step == 1
    assert prior_metrics.loss >= 0
    assert prior_metrics.grad_norm > 0
    assert not torch.equal(
        prior_before, prior_diffusion.model.projection.weight.detach()
    )

    controller_diffusion = GaussianDiffusion(TinyPrior(), timesteps=4, sampling_steps=2)
    controller = TinyController()
    controller_before = controller.gain.detach().clone()
    controller_trainer = ControllerTrainer(
        controller_diffusion,
        controller,
        [_trajectory()[:, :3]],
        AverageDownsample1D(2),
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

    controller_metrics = controller_trainer.train_step()

    assert controller_metrics.step == 1
    assert controller_metrics.total_loss >= 0
    assert controller_metrics.grad_norm > 0
    assert not torch.equal(controller_before, controller.gain.detach())
    assert all(
        parameter.grad is None for parameter in controller_diffusion.model.parameters()
    )
