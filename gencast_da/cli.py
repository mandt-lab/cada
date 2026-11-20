"""Command-line orchestration for the isolated CADA-on-GenCast stack."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gencast_da.config import ExperimentConfig, load_config
from gencast_da.runtime import (
    RunLayout,
    atomic_write_json,
    config_digest,
    write_sha256_sidecar,
)
from gencast_da.tracking import init_tracker


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    parser.add_argument("--data-root")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--tracking",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gencast_da")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="inspect or download GenCast data")
    _common(data)
    data.add_argument(
        "--action",
        choices=("inspect", "download", "build"),
        default="inspect",
    )

    for name, help_text in (
        ("smoke", "run one bounded GenCast smoke check"),
        ("train-controller", "train the GenCast CADA controller"),
        ("evaluate", "evaluate a GenCast CADA controller"),
        ("figure", "render a saved GenCast CADA artifact"),
    ):
        child = commands.add_parser(name, help=help_text)
        _common(child)
        if name == "train-controller":
            child.add_argument("--resume", action="store_true")
        elif name == "evaluate":
            child.add_argument("--checkpoint")
        elif name == "figure":
            child.add_argument("--artifact")
    return parser


def _root(args: argparse.Namespace) -> Path:
    value = args.root or os.environ.get("CADA_ROOT")
    if not value:
        raise ValueError("CADA_ROOT is not set; export it or pass --root")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("CADA_ROOT must be absolute")
    return candidate.resolve(strict=False)


def _parameters_to_args(parameters: Mapping[str, Any]) -> list[str]:
    arguments: list[str] = []
    for name, value in parameters.items():
        option = f"--{name}"
        if value is None or value is False:
            continue
        if value is True:
            arguments.append(option)
            continue
        if isinstance(value, list):
            for item in value:
                arguments.extend((option, str(item)))
            continue
        if isinstance(value, (str, int, float)):
            arguments.extend((option, str(value)))
            continue
        raise TypeError(f"unsupported parameter value for {name!r}: {value!r}")
    return arguments


def _validate_kind(command: str, config: ExperimentConfig) -> None:
    expected = {
        "data": "data",
        "smoke": "smoke",
        "train-controller": "controller",
        "evaluate": "evaluation",
        "figure": "figure",
    }[command]
    if config.kind != expected:
        raise ValueError(f"{command} requires kind {expected!r}, got {config.kind!r}")


def _command(
    args: argparse.Namespace,
    layout: RunLayout,
    *,
    module: str,
    parameters: Mapping[str, Any],
) -> list[str]:
    repository = Path(__file__).resolve().parents[1]
    if args.command == "data" and args.action == "download":
        return ["bash", str(repository / "gencast_da" / "download_assets.sh")]

    command = [sys.executable, "-m", module]
    effective_parameters = dict(parameters)

    if args.command == "train-controller":
        effective_parameters.update(
            {
                "ckpt-dir": str(layout.checkpoints),
                "ckpt-name": "final.pkl",
                "history-path": str(layout.metrics / "training-history.json"),
            }
        )
        if getattr(args, "resume", False):
            effective_parameters["resume"] = str(layout.checkpoints / "final.pkl")
    elif args.command == "evaluate":
        effective_parameters["output-dir"] = str(layout.run_dir)
    elif args.command == "figure":
        effective_parameters["output-dir"] = str(layout.figures)
    return [*command, *_parameters_to_args(effective_parameters)]


def _effective_command_config(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> tuple[str, dict[str, Any]]:
    module = config.module
    parameters = dict(config.parameters)
    if args.command == "data" and args.action == "build":
        module = "gencast_da.build_mini_era5_nc"
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        parameters["ckpt"] = checkpoint
    artifact = getattr(args, "artifact", None)
    if artifact is not None:
        parameters["artifact"] = artifact
    return module, parameters


def _run_and_log(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
    append: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(environment),
        )

        def forward(signum, _frame) -> None:
            process.send_signal(signum)

        previous_usr1 = signal.signal(signal.SIGUSR1, forward)
        previous_term = signal.signal(signal.SIGTERM, forward)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return process.wait()
        finally:
            signal.signal(signal.SIGUSR1, previous_usr1)
            signal.signal(signal.SIGTERM, previous_term)


def _log_structured_metrics(tracker, layout: RunLayout) -> None:
    history_path = layout.metrics / "training-history.json"
    evaluation_path = layout.metrics / "evaluation.json"
    records: list[Mapping[str, Any]] = []
    if history_path.is_file():
        loaded = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = [record for record in loaded if isinstance(record, dict)]
    elif evaluation_path.is_file():
        loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
        steps = loaded.get("metrics", {}).get("steps", [])
        if isinstance(steps, list):
            records = [record for record in steps if isinstance(record, dict)]
    logged_steps: set[int] = set()
    if tracker.metrics_path.is_file():
        for line in tracker.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(existing.get("step"), int):
                logged_steps.add(existing["step"])
    for index, record in enumerate(records, start=1):
        step = record.get("step")
        effective_step = step if isinstance(step, int) else index
        if effective_step in logged_steps:
            continue
        tracker.log(
            record,
            step=effective_step,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root(args)
    data_root = args.data_root or os.environ.get("GENCAST_DATA_ROOT")
    config = load_config(
        args.config,
        cada_root=root,
        data_root=data_root,
    )
    _validate_kind(args.command, config)
    module, parameters = _effective_command_config(args, config)
    resolved = {
        **config.to_dict(),
        "module": module,
        "parameters": parameters,
        "command": args.command,
        "data_action": getattr(args, "action", None),
        "tracking": args.tracking,
    }
    layout = RunLayout.plan(
        root=root,
        stage=config.stage,
        config_id=config.config_id,
        resolved_config=resolved,
        run_id=args.run_id,
    )
    command = _command(
        args,
        layout,
        module=module,
        parameters=parameters,
    )
    plan = {
        "action": args.command,
        "config": resolved,
        "paths": layout.paths(),
        "command": command,
        "root": str(root),
        "data_root": config.data_root,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    repository = Path(__file__).resolve().parents[1]
    resuming = args.command == "train-controller" and args.resume
    if resuming:
        layout.open_existing(resolved_config=resolved)
        checkpoint = layout.checkpoints / "final.pkl"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    else:
        layout.create(resolved_config=resolved, repository=repository)
    tracker = init_tracker(
        mode=args.tracking,
        wandb_dir=layout.wandb,
        metrics_dir=layout.metrics,
        config=resolved,
        run_name=f"{config.config_id}-{layout.run_id}",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CADA_ROOT": str(root),
            "GENCAST_DATA_ROOT": config.data_root,
            "CADA_RUN_DIR": str(layout.run_dir),
            "CADA_CONFIG_SHA256": config_digest(resolved),
            "PYTHONPATH": str(repository)
            + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
        }
    )
    exit_code = _run_and_log(
        command,
        log_path=layout.logs / "process.log",
        environment=environment,
        append=resuming,
    )
    process_record = {
        "command": command,
        "exit_code": exit_code,
    }
    process_path = atomic_write_json(
        layout.metrics / "process.json",
        process_record,
    )
    write_sha256_sidecar(process_path)
    _log_structured_metrics(tracker, layout)
    tracker.finish(exit_code=exit_code)
    print(
        json.dumps(
            {
                "action": args.command,
                "run": str(layout.run_dir),
                "exit_code": exit_code,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
