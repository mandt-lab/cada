"""Config knobs for CADA training (loss weights, optimizer, training
schedule). Defaults are chosen so the gradient signal mirrors the PDE
CADA `train_controlnet` at line 2373 (terminal-only loss).

The full loss is

    total = w_t * (w_full * full_mse + w_obs * obs_mse)
            + w_s * score_substep_mean
            + w_c * control_substep_mean
            + w_reg * u_l2_substep_mean

where score_substep_mean and control_substep_mean come from the Haiku
state accumulators in `cada_guided.controlled_pcd`.

Default values:
- `w_t = 1.0, w_full = 1.0, w_obs = 1.0` — terminal piece is the only
  signal.
- `w_s = w_c = 0.0` — score and control losses are computed and logged
  but contribute zero gradient. Flip to non-zero for "aggressive" mode.
- `w_reg = 1e-4` — small L2 on `u` to prevent runaway controls early in
  training (analog of PDE CADA's smoothness penalty).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CADALossConfig:
    """Weights mirroring PDE CADA `losses` dict (`tot_t/tot_s/tot_c`)."""

    w_t: float = 1.0
    w_full: float = 1.0
    w_obs: float = 1.0
    w_s: float = 0.0
    w_c: float = 0.0
    w_reg: float = 1e-4


@dataclass
class CADAOptConfig:
    """Optimizer and training-schedule knobs."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_train_steps: int = 1000
    log_every: int = 10
    val_every: int = 100
    seed: int = 0
