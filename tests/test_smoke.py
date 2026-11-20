from __future__ import annotations

import torch

from cada.cli import main
from cada.smoke import run_smoke


def test_cada_cpu_smoke() -> None:
    result = run_smoke(torch.device("cpu"), seed=7)

    assert result["device"] == "cpu"
    assert result["prior_loss"] >= 0
    assert result["controller_loss"] >= 0
    assert result["peak_memory_mib"] == 0
    assert result["cuda_rng_replay"] is False


def test_cada_smoke_cli_dry_run(capsys) -> None:
    assert main(["smoke", "--device", "cpu", "--seed", "3", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"action": "smoke"' in output
    assert '"seed": 3' in output
