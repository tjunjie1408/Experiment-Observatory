"""Small, inspectable Lloyd K-means implementation for the M5 experiment unit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
KMeansPhase = Literal["assignment", "update"]
KMeansStopReason = Literal["assignments_stable", "max_iterations"]


@dataclass(frozen=True)
class KMeansState:
    step: int
    iteration: int
    phase: KMeansPhase
    centers: FloatArray
    assignments: IntArray
    inertia: float
    empty_clusters: tuple[int, ...]


@dataclass(frozen=True)
class KMeansResult:
    snapshots: tuple[KMeansState, ...]
    stop_reason: KMeansStopReason


def _validate_points_centers(points: FloatArray, centers: FloatArray) -> None:
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2) with at least two samples")
    if centers.ndim != 2 or centers.shape[1] != points.shape[1]:
        raise ValueError("centers must have shape (n_clusters, 2)")
    if centers.shape[0] < 2 or centers.shape[0] > points.shape[0]:
        raise ValueError("n_clusters must be between 2 and n_samples")
    if not np.isfinite(points).all() or not np.isfinite(centers).all():
        raise ValueError("points and centers must contain only finite values")


def initialize_from_samples(points: FloatArray, n_clusters: int, seed: int) -> FloatArray:
    if isinstance(n_clusters, bool) or not 2 <= n_clusters <= len(points):
        raise ValueError("n_clusters must be between 2 and n_samples")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    candidate = np.asarray(points, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[1] != 2 or not np.isfinite(candidate).all():
        raise ValueError("points must be a finite two-dimensional matrix with two columns")
    indices = np.random.default_rng(seed).choice(len(candidate), size=n_clusters, replace=False)
    return candidate[indices].copy()


def assign(points: FloatArray, centers: FloatArray) -> IntArray:
    _validate_points_centers(points, centers)
    squared_distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    result: IntArray = np.argmin(squared_distances, axis=1).astype(np.int64)
    return result


def inertia(points: FloatArray, centers: FloatArray, assignments: IntArray) -> float:
    _validate_points_centers(points, centers)
    labels = np.asarray(assignments, dtype=np.int64)
    if labels.shape != (len(points),) or np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError("assignments must contain one valid cluster index per sample")
    residuals = points - centers[labels]
    value = float(np.sum(residuals * residuals))
    if not np.isfinite(value):
        raise FloatingPointError("K-means inertia became non-finite")
    return value


def update_centers(
    points: FloatArray, assignments: IntArray, old_centers: FloatArray
) -> tuple[FloatArray, tuple[int, ...]]:
    _validate_points_centers(points, old_centers)
    labels = np.asarray(assignments, dtype=np.int64)
    if labels.shape != (len(points),) or np.any(labels < 0) or np.any(labels >= len(old_centers)):
        raise ValueError("assignments must contain one valid cluster index per sample")
    updated = old_centers.copy()
    empty: list[int] = []
    for cluster in range(len(old_centers)):
        members = points[labels == cluster]
        if len(members) == 0:
            empty.append(cluster)
        else:
            updated[cluster] = np.mean(members, axis=0)
    return updated, tuple(empty)


def iter_lloyd(
    points: FloatArray, initial_centers: FloatArray, *, max_iterations: int
) -> KMeansResult:
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(initial_centers, dtype=np.float64).copy()
    _validate_points_centers(points, centers)
    if isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")

    snapshots: list[KMeansState] = []
    previous_assignments: IntArray | None = None
    step = 0
    for iteration in range(max_iterations + 1):
        assignments = assign(points, centers)
        snapshots.append(
            KMeansState(
                step=step,
                iteration=iteration,
                phase="assignment",
                centers=centers.copy(),
                assignments=assignments.copy(),
                inertia=inertia(points, centers, assignments),
                empty_clusters=(),
            )
        )
        step += 1
        if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
            return KMeansResult(tuple(snapshots), "assignments_stable")
        if iteration == max_iterations:
            return KMeansResult(tuple(snapshots), "max_iterations")

        updated, empty = update_centers(points, assignments, centers)
        snapshots.append(
            KMeansState(
                step=step,
                iteration=iteration,
                phase="update",
                centers=updated.copy(),
                assignments=assignments.copy(),
                inertia=inertia(points, updated, assignments),
                empty_clusters=empty,
            )
        )
        step += 1
        previous_assignments = assignments.copy()
        centers = updated

    raise AssertionError("bounded Lloyd loop must return")


__all__ = [
    "KMeansResult",
    "KMeansState",
    "assign",
    "inertia",
    "initialize_from_samples",
    "iter_lloyd",
    "update_centers",
]
