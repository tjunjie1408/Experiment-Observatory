"""Deterministic two-dimensional blob data for the M5 K-means study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

GENERATOR_ID = "synthetic_kmeans_v1"


@dataclass(frozen=True)
class KMeansSyntheticConfig:
    blob_centers: tuple[tuple[float, float], ...]
    blob_sizes: tuple[int, ...]
    cluster_std: float
    seed: int


@dataclass(frozen=True)
class KMeansSyntheticDataset:
    generator_id: str
    sample_ids: list[str]
    points: npt.NDArray[np.float64]


def generate_kmeans(config: KMeansSyntheticConfig) -> KMeansSyntheticDataset:
    centers = np.asarray(config.blob_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] != 2:
        raise ValueError("blob_centers must contain at least two two-dimensional centers")
    if len(config.blob_sizes) != len(centers) or any(size < 1 for size in config.blob_sizes):
        raise ValueError("blob_sizes must contain one positive size per center")
    if not np.isfinite(centers).all():
        raise ValueError("blob centers must be finite")
    if not np.isfinite(config.cluster_std) or config.cluster_std < 0:
        raise ValueError("cluster_std must be finite and non-negative")
    if isinstance(config.seed, bool) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(config.seed)
    parts = [
        rng.normal(loc=center, scale=config.cluster_std, size=(size, 2))
        for center, size in zip(centers, config.blob_sizes, strict=True)
    ]
    points = np.concatenate(parts)
    points = points[rng.permutation(len(points))]
    return KMeansSyntheticDataset(
        generator_id=GENERATOR_ID,
        sample_ids=[f"p{i:04d}" for i in range(len(points))],
        points=points,
    )
