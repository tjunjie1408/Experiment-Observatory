"""Configuration and execution entry point for the five-seed M5 study."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from observatory.data.kmeans_synthetic import KMeansSyntheticConfig
from observatory.runtime.kmeans_record import run_kmeans_experiment
from observatory.runtime.schema import KMeansRunManifest


class KMeansStudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    blob_centers: list[tuple[float, float]] = Field(min_length=2)
    blob_sizes: list[int] = Field(min_length=2)
    cluster_std: float = Field(ge=0, allow_inf_nan=False)
    data_seed: int = Field(ge=0, strict=True)
    n_clusters: int = Field(ge=2, strict=True)
    init_seeds: list[int] = Field(min_length=5, max_length=5)
    max_iterations: int = Field(ge=1, strict=True)


def load_kmeans_study_config(path: Path) -> KMeansStudyConfig:
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = KMeansStudyConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid K-means study config {path}: {exc}") from exc
    if len(config.blob_sizes) != len(config.blob_centers):
        raise ValueError("blob_sizes must contain one entry per blob center")
    if config.n_clusters > sum(config.blob_sizes):
        raise ValueError("n_clusters cannot exceed the number of generated samples")
    if len(set(config.init_seeds)) != len(config.init_seeds):
        raise ValueError("init_seeds must be unique")
    return config


def run_kmeans_study(
    config_path: Path, runs_root: Path, repo_root: Path
) -> list[KMeansRunManifest]:
    config = load_kmeans_study_config(config_path)
    data_config = KMeansSyntheticConfig(
        blob_centers=tuple(config.blob_centers),
        blob_sizes=tuple(config.blob_sizes),
        cluster_std=config.cluster_std,
        seed=config.data_seed,
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    return [
        run_kmeans_experiment(
            experiment_id=f"{config.experiment_id}-seed-{seed}",
            data_config=data_config,
            n_clusters=config.n_clusters,
            init_seed=seed,
            max_iterations=config.max_iterations,
            runs_root=runs_root,
            repo_root=repo_root,
        )
        for seed in config.init_seeds
    ]


__all__ = ["KMeansStudyConfig", "load_kmeans_study_config", "run_kmeans_study"]
