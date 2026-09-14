from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans

from observatory.models.kmeans.lloyd import assign, inertia, iter_lloyd, update_centers


def test_assignment_uses_nearest_center_and_lowest_index_for_ties() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    centers = np.array([[0.0, 0.0], [2.0, 0.0]])

    labels = assign(points, centers)

    np.testing.assert_array_equal(labels, [0, 0, 1])
    assert inertia(points, centers, labels) == pytest.approx(1.0)


def test_update_uses_cluster_means_and_retains_empty_center() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    old_centers = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 100.0]])
    labels = np.array([0, 0, 1])

    centers, empty = update_centers(points, labels, old_centers)

    np.testing.assert_allclose(centers, [[1.0, 0.0], [10.0, 0.0], [100.0, 100.0]])
    assert empty == (2,)


def test_snapshots_keep_assignment_and_update_phases_consistent() -> None:
    points = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]])
    initial = np.array([[0.0, 0.0], [10.0, 10.0]])

    result = iter_lloyd(points, initial, max_iterations=10)

    assert result.stop_reason == "assignments_stable"
    assert result.snapshots[-1].phase == "assignment"
    for snapshot in result.snapshots:
        assert snapshot.inertia == pytest.approx(
            inertia(points, snapshot.centers, snapshot.assignments)
        )
        if snapshot.phase == "assignment":
            np.testing.assert_array_equal(snapshot.assignments, assign(points, snapshot.centers))
        else:
            expected, empty = update_centers(
                points,
                snapshot.assignments,
                result.snapshots[snapshot.step - 1].centers,
            )
            np.testing.assert_allclose(snapshot.centers, expected)
            assert snapshot.empty_clusters == empty


def test_fixed_initialization_matches_sklearn_up_to_cluster_permutation() -> None:
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [9.0, 9.0], [9.0, 10.0], [10.0, 9.0]])
    initial = np.array([[0.0, 0.0], [10.0, 9.0]])

    ours = iter_lloyd(points, initial, max_iterations=20)
    reference = KMeans(
        n_clusters=2,
        init=initial,
        n_init=1,
        max_iter=20,
        tol=0.0,
        algorithm="lloyd",
    ).fit(points)

    ours_centers = ours.snapshots[-1].centers[np.argsort(ours.snapshots[-1].centers[:, 0])]
    ref_centers = reference.cluster_centers_[np.argsort(reference.cluster_centers_[:, 0])]
    np.testing.assert_allclose(ours_centers, ref_centers)
    assert ours.snapshots[-1].inertia == pytest.approx(reference.inertia_)


@pytest.mark.parametrize(
    ("points", "centers", "max_iterations"),
    [
        (np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), 1),
        (np.array([[0.0, np.nan], [1.0, 1.0]]), np.array([[0.0, 0.0]]), 1),
        (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0]]), 0),
    ],
)
def test_rejects_invalid_kmeans_inputs(
    points: np.ndarray, centers: np.ndarray, max_iterations: int
) -> None:
    with pytest.raises(ValueError):
        iter_lloyd(points, centers, max_iterations=max_iterations)
