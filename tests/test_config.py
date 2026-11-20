from __future__ import annotations

import json
from pathlib import Path

import pytest

from cada.config import ConfigError, load_config, load_config_text, to_resolved_dict


def _paper_config_paths(repository_root: Path) -> dict[str, list[Path]]:
    return {
        "dataset": sorted((repository_root / "manifests/data").glob("*.yaml")),
        "prior": sorted((repository_root / "configs/priors").glob("*.yaml")),
        "controller": sorted((repository_root / "configs/controllers").glob("*.yaml")),
        "evaluation": sorted((repository_root / "configs/evaluations").glob("*.yaml")),
    }


def test_all_54_configs_load_strictly_with_exact_kind_counts(
    repository_root: Path, tmp_path: Path
) -> None:
    paths = _paper_config_paths(repository_root)
    assert {kind: len(items) for kind, items in paths.items()} == {
        "dataset": 3,
        "prior": 3,
        "controller": 18,
        "evaluation": 30,
    }

    root = tmp_path / "cada-root"
    loaded = []
    for expected_kind, items in paths.items():
        for path in items:
            config = load_config(path, env={"CADA_ROOT": str(root)})
            assert config.kind == expected_kind
            loaded.append(config)

    assert len(loaded) == 54
    resolved = json.dumps([to_resolved_dict(config) for config in loaded])
    assert str(root) in resolved
    assert "/extra/" not in resolved


def test_unknown_key_is_rejected_and_cada_root_is_expanded(
    repository_root: Path, tmp_path: Path
) -> None:
    source = repository_root / "manifests/data/ks.yaml"
    text = source.read_text(encoding="utf-8")
    root = tmp_path / "root"

    config = load_config_text(text, source=str(source), env={"CADA_ROOT": str(root)})
    assert config.train_data_path == str(root / "data/ks/train.h5")
    assert config.normalization.statistics_path == str(
        root / "data/ks/normalization.json"
    )

    with pytest.raises(ConfigError, match=r"unknown key\(s\): unexpected"):
        load_config_text(
            text + "\nunexpected: true\n",
            source=str(source),
            env={"CADA_ROOT": str(root)},
        )


def test_missing_cada_root_is_rejected(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CADA_ROOT", raising=False)

    with pytest.raises(ConfigError, match="CADA_ROOT is not set"):
        load_config(repository_root / "manifests/data/ks.yaml")
