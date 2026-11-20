from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from cada.cli import build_dataloader, main
from cada.config import ControllerExperimentConfig, load_config


class _TrajectorySource:
    n_trajectories = 2
    trajectory_length = 80

    def read_window(self, trajectory: int, start: int, stop: int) -> torch.Tensor:
        del trajectory
        return torch.zeros(stop - start, 1, 8)

    def close(self) -> None:
        pass


def test_cli_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "data",
        "smoke",
        "train-prior",
        "train-controller",
        "evaluate",
        "figure",
    ):
        assert command in help_text


@pytest.mark.parametrize(
    ("config_path", "expected_horizon", "expected_stride"),
    [
        ("configs/controllers/ks__ms_4.yaml", 54, 4),
        ("configs/controllers/kolmogorov__ms_4.yaml", 16, 4),
        ("configs/controllers/kolmogorov__ds_4.yaml", 16, 1),
    ],
)
def test_controller_loader_uses_anchored_preview_windows(
    repository_root: Path,
    tmp_path: Path,
    config_path: str,
    expected_horizon: int,
    expected_stride: int,
) -> None:
    config = load_config(
        repository_root / config_path,
        env={"CADA_ROOT": str(tmp_path)},
    )
    assert isinstance(config, ControllerExperimentConfig)

    loader = build_dataloader(
        config,
        _TrajectorySource(),
        workers=0,
        device="cpu",
    )

    assert loader.dataset.horizon == expected_horizon
    assert loader.dataset.stride == expected_stride


@pytest.mark.parametrize(
    ("name", "arguments", "expected_action"),
    [
        (
            "data-inspect",
            ["data", "--config", "manifests/data/ks.yaml", "--action", "inspect"],
            "data:inspect",
        ),
        (
            "data-generate",
            ["data", "--config", "manifests/data/ks.yaml", "--action", "generate"],
            "data:generate",
        ),
        (
            "data-request",
            [
                "data",
                "--config",
                "manifests/data/era5_small.yaml",
                "--action",
                "request",
                "--year",
                "2006",
            ],
            "data:request",
        ),
        (
            "data-convert",
            [
                "data",
                "--config",
                "manifests/data/era5_small.yaml",
                "--action",
                "convert",
                "--input",
                "not-downloaded.nc",
            ],
            "data:convert",
        ),
        (
            "data-stats",
            [
                "data",
                "--config",
                "manifests/data/kolmogorov.yaml",
                "--action",
                "stats",
            ],
            "data:stats",
        ),
        (
            "prior",
            ["train-prior", "--config", "configs/priors/ks.yaml"],
            "train-prior",
        ),
        (
            "controller",
            [
                "train-controller",
                "--config",
                "configs/controllers/ks__ds_2.yaml",
            ],
            "train-controller",
        ),
        (
            "evaluation",
            [
                "evaluate",
                "--config",
                "configs/evaluations/ks__ds_2__h140.yaml",
            ],
            "evaluate",
        ),
        (
            "figure",
            [
                "figure",
                "--config",
                "configs/figures/ks__ms_4__h640__spacetime.yaml",
            ],
            "figure",
        ),
    ],
)
def test_representative_dry_runs_are_json_and_side_effect_free(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    arguments: list[str],
    expected_action: str,
) -> None:
    monkeypatch.chdir(repository_root)
    root = tmp_path / name
    adjusted = [
        str(tmp_path / item) if item == "not-downloaded.nc" else item
        for item in arguments
    ]

    exit_code = main([*adjusted, "--root", str(root), "--device", "cpu", "--dry-run"])
    captured = capsys.readouterr()
    document = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert document["action"] == expected_action
    assert document["root"] == str(root)
    assert "/extra/" not in captured.out
    assert not root.exists()


def test_exact_checkpointing_rejects_multiworker_loading(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(repository_root)
    assert (
        main(
            [
                "train-controller",
                "--config",
                "configs/controllers/ks__ms_4.yaml",
                "--root",
                str(tmp_path),
                "--workers",
                "1",
                "--dry-run",
            ]
        )
        == 1
    )


def test_slurm_scripts_parse_and_submit_print_mode_never_launches(
    repository_root: Path, tmp_path: Path
) -> None:
    scripts = [
        repository_root / "scripts/slurm/job.sh",
        repository_root / "scripts/slurm/submit.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "sbatch-was-called"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        f"#!/usr/bin/env bash\ntouch {marker!s}\nexit 99\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["CADA_ROOT"] = str(tmp_path / "cada-root")
    environment["CADA_PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(repository_root / "scripts/slurm/submit.sh"),
            "--profile",
            "smoke",
            "--account",
            "test_account",
            "--",
            "train-prior",
            "--config",
            "configs/priors/ks.yaml",
            "--device",
            "cpu",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not marker.exists()
    assert result.stderr == ""
    assert result.stdout.startswith("sbatch ")
    assert "--account=test_account" in result.stdout
    assert "--gres=gpu:1" in result.stdout
    assert str(tmp_path / "cada-root" / "slurm") in result.stdout
    assert "configs/priors/ks.yaml" in result.stdout

    smoke = subprocess.run(
        [
            "bash",
            str(repository_root / "scripts/slurm/submit.sh"),
            "--profile",
            "smoke",
            "--account",
            "test_account",
            "--",
            "smoke",
            "--device",
            "cuda",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert " smoke --device cuda" in smoke.stdout


def test_training_dry_run_returns_an_accepted_canonical_run_id(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository_root)
    arguments = [
        "train-prior",
        "--config",
        "configs/priors/ks.yaml",
        "--root",
        str(tmp_path),
        "--device",
        "cpu",
        "--dry-run",
    ]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    run_id = Path(first["paths"]["run"]).name

    assert main([*arguments, "--run-id", run_id]) == 0
    second = json.loads(capsys.readouterr().out)
    assert Path(second["paths"]["run"]).name == run_id


def test_cli_requires_cada_root_or_root_argument(
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository_root)
    monkeypatch.delenv("CADA_ROOT", raising=False)

    exit_code = main(
        ["train-prior", "--config", "configs/priors/ks.yaml", "--dry-run"]
    )

    assert exit_code == 1
    assert "CADA_ROOT is not set" in capsys.readouterr().err


def test_gencast_slurm_print_mode_is_side_effect_free(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    for name in ("gencast_job.sh", "submit_gencast.sh"):
        subprocess.run(
            ["bash", "-n", str(repository_root / "scripts/slurm" / name)],
            check=True,
        )

    root = tmp_path / "cada"
    environment = dict(os.environ)
    environment["CADA_ROOT"] = str(root)
    environment["GENCAST_PYTHON_BIN"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            str(repository_root / "scripts/slurm/submit_gencast.sh"),
            "--profile",
            "smoke",
            "--account",
            "test_account",
            "--",
            "smoke",
            "--config",
            "gencast_da/configs/smoke.yaml",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
    assert result.stdout.startswith("sbatch ")
    assert "--account=test_account" in result.stdout
    assert "gencast_job.sh" in result.stdout
    assert not root.exists()

    training = subprocess.run(
        [
            "bash",
            str(repository_root / "scripts/slurm/submit_gencast.sh"),
            "--account",
            "test_account",
            "--",
            "train-controller",
            "--config",
            "gencast_da/configs/controller.yaml",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert re.search(r"--run-id [0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}", training.stdout)


@pytest.mark.parametrize(
    ("job_name", "python_variable", "command"),
    [
        ("job.sh", "CADA_PYTHON_BIN", "train-prior"),
        ("gencast_job.sh", "GENCAST_PYTHON_BIN", "train-controller"),
    ],
)
def test_slurm_usr1_requires_checkpoint_ack_and_requeues(
    repository_root: Path,
    tmp_path: Path,
    job_name: str,
    python_variable: str,
    command: str,
) -> None:
    arguments_path = tmp_path / f"{job_name}.arguments"
    requeue_path = tmp_path / f"{job_name}.requeue"
    fake_python = tmp_path / f"{job_name}.python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" > {arguments_path!s}\n"
        "trap 'exit 75' USR1\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_scontrol = tmp_path / f"{job_name}.scontrol"
    fake_scontrol.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" > {requeue_path!s}\n",
        encoding="utf-8",
    )
    fake_scontrol.chmod(0o755)
    environment = {
        **os.environ,
        "CADA_SKIP_MODULES": "1",
        "CADA_REPOSITORY_ROOT": str(repository_root),
        "CADA_ROOT": str(tmp_path / "runs"),
        "SCONTROL_BIN": str(fake_scontrol),
        "SLURM_JOB_ID": "12345",
        python_variable: str(fake_python),
    }
    process = subprocess.Popen(
        [
            "bash",
            str(repository_root / "scripts/slurm" / job_name),
            command,
            "--config",
            "unused.yaml",
        ],
        cwd=repository_root,
        env=environment,
    )
    for _ in range(100):
        if arguments_path.exists():
            break
        time.sleep(0.02)
    else:
        process.terminate()
        raise AssertionError("fake Slurm child did not start")

    process.send_signal(signal.SIGUSR1)
    assert process.wait(timeout=10) == 0
    assert requeue_path.read_text(encoding="utf-8").strip() == "requeue 12345"

    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" > {arguments_path!s}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment["SLURM_RESTART_COUNT"] = "1"
    subprocess.run(
        [
            "bash",
            str(repository_root / "scripts/slurm" / job_name),
            command,
            "--config",
            "unused.yaml",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
    )
    restarted_arguments = arguments_path.read_text(encoding="utf-8").split()
    assert restarted_arguments.count("--resume") == 1
