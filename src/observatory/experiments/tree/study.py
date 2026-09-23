"""Fixed WDBC depth study over one verified dataset and split."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from observatory.experiments.tree.record import run_tree_experiment
from observatory.runtime.tree_schema import TreeRunManifest


class TreeStudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    experiment_id: str
    version_manifest: str
    depths: list[int]


def load_tree_study_config(path: Path) -> tuple[TreeStudyConfig, Path]:
    try:
        config = TreeStudyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid tree study config: {exc}") from exc
    if config.experiment_id != "wdbc-tree-depth-study" or config.depths != [1, 2, 3, 4, 5]:
        raise ValueError("tree study requires the fixed experiment and depths 1..5")
    version = (path.parent / config.version_manifest).resolve()
    return config, version


def run_tree_study(config_path: Path, runs_root: Path, repo_root: Path) -> list[TreeRunManifest]:
    config, version = load_tree_study_config(config_path)
    from observatory.datasets.tabular.breast_cancer import load_breast_cancer

    load_breast_cancer(version)  # Fail before creating any run.
    results: list[TreeRunManifest] = []
    for depth in config.depths:
        results.append(
            run_tree_experiment(version, max_depth=depth, runs_root=runs_root, repo_root=repo_root)
        )
        if results[-1].status != "completed":
            break
    return results
