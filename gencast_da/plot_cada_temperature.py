"""Render CADA-on-GenCast temperature truth, prediction, and error."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from gencast_da.artifacts import load_evaluation_npz
from gencast_da.runtime import (
    atomic_write_json,
    sha256_file,
    write_sha256_sidecar,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--level-idx", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    artifact_path = Path(args.artifact).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_evaluation_npz(artifact_path)
    truth = artifact.truth
    prediction = artifact.prediction
    latitude = artifact.coordinates["latitude"]
    longitude = artifact.coordinates["longitude"]
    pressure_levels = artifact.coordinates["pressure_level"]
    lead_hours = artifact.coordinates["lead_hours"]
    metadata = artifact.metadata

    if not 0 <= args.level_idx < truth.shape[-1]:
        raise ValueError("level-idx is outside the artifact pressure levels")

    selected_truth = truth[..., args.level_idx]
    selected_prediction = prediction[..., args.level_idx]
    error = np.abs(selected_prediction - selected_truth)
    minimum = float(min(selected_truth.min(), selected_prediction.min()))
    maximum = float(max(selected_truth.max(), selected_prediction.max()))
    error_maximum = max(float(error.max()), 0.5)

    figure, axes = plt.subplots(
        truth.shape[0],
        3,
        figsize=(13.5, 2.8 * truth.shape[0]),
        squeeze=False,
    )
    for index in range(truth.shape[0]):
        fields = (
            (selected_truth[index], "Truth", "RdYlBu_r", minimum, maximum),
            (selected_prediction[index], "CADA", "RdYlBu_r", minimum, maximum),
            (error[index], "|Error|", "hot_r", 0.0, error_maximum),
        )
        for column, (field, title, cmap, vmin, vmax) in enumerate(fields):
            axis = axes[index, column]
            image = axis.pcolormesh(
                longitude,
                latitude,
                field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            if index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"+{lead_hours[index]} h")
            figure.colorbar(image, ax=axis, shrink=0.8)

    level = pressure_levels[args.level_idx]
    figure.suptitle(f"CADA-on-GenCast temperature at {level} hPa")
    figure.tight_layout()
    output = output_dir / f"temperature_level_{args.level_idx}.png"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=args.dpi, bbox_inches="tight")
        os.replace(temporary, output)
        directory_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    write_sha256_sidecar(output)
    output_sha256 = sha256_file(output)
    provenance = atomic_write_json(
        output.with_suffix(".provenance.json"),
        {
            "schema_version": 1,
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "output": str(output),
            "output_sha256": output_sha256,
            "level_index": args.level_idx,
            "pressure_level": float(level),
            "lead_hours": lead_hours.tolist(),
            "rendering": {
                "dpi": args.dpi,
                "temperature_cmap": "RdYlBu_r",
                "error_cmap": "hot_r",
                "temperature_limits": [minimum, maximum],
                "error_limits": [0.0, error_maximum],
            },
            "metadata": metadata,
        },
    )
    write_sha256_sidecar(provenance)
    print(f"[plot_cada_temperature] wrote {output}")


if __name__ == "__main__":
    main()
