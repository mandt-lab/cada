"""Optional scalar tracking for GenCast runs."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gencast_da.runtime import atomic_write_text

TrackingMode = Literal["disabled", "offline", "online"]


def _scalars(values: Mapping[str, Any]) -> dict[str, float | int | str | bool]:
    clean: dict[str, float | int | str | bool] = {}
    for key, value in values.items():
        if isinstance(value, bool | int | str):
            clean[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            clean[key] = value
    return clean


@dataclass
class Tracker:
    mode: TrackingMode
    metrics_path: Path
    run: Any = None

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        record = _scalars(values)
        if step is not None:
            record["step"] = step
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.run is not None:
            self.run.log(record, step=step)

    def finish(self, *, exit_code: int) -> None:
        self.log({"process_exit_code": exit_code})
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def init_tracker(
    *,
    mode: TrackingMode,
    wandb_dir: Path,
    metrics_dir: Path,
    config: Mapping[str, Any],
    run_name: str,
) -> Tracker:
    metrics_path = metrics_dir / "events.jsonl"
    if mode == "disabled":
        return Tracker(mode=mode, metrics_path=metrics_path)
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "tracking requires `python -m pip install wandb` in gencast-da"
        ) from error

    id_path = wandb_dir / "run_id.txt"
    if id_path.exists():
        run_id = id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = uuid.uuid4().hex
        atomic_write_text(id_path, run_id + "\n")
    run = wandb.init(
        project="cada",
        name=run_name,
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(wandb_dir),
        config=dict(config),
    )
    os.environ["WANDB_DIR"] = str(wandb_dir)
    return Tracker(mode=mode, metrics_path=metrics_path, run=run)
