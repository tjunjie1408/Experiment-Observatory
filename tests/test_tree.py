"""Independent examples for the hand-written binary CART learner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observatory.models.tree.cart import TreeCancelledError, fit_tree, iter_tree, predict_tree


def test_root_split_and_path_prediction() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    tree = fit_tree(x, y, ("a", "b", "c", "d"), max_depth=2)
    assert [node.node_id for node in tree.nodes] == ["r", "rL", "rR"]
    root = tree.nodes[0]
    assert root.split_feature_index == 0
    assert root.split_threshold == 1.5
    assert root.gini_parent == 0.5
    assert root.weighted_gini_decrease == 0.5
    assert root.class_counts == (2, 2)
    assert tree.nodes[1].sample_ids == ("a", "b")
    assert tree.nodes[2].sample_ids == ("c", "d")
    assert predict_tree(tree, x).tolist() == y.tolist()
    assert tree.path(x[2]).node_ids == ("r", "rR")


def test_feature_and_threshold_ties_are_deterministic() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    assert fit_tree(x, y, tuple("abcd"), max_depth=1).nodes[0].split_feature_index == 0
    tie = fit_tree(np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 0]), tuple("abc"), max_depth=1)
    assert tie.nodes[0].split_threshold == 0.5


def test_leaf_reasons_and_class_tie() -> None:
    ids = ("a", "b")
    assert (
        fit_tree(np.array([[0.0], [1.0]]), np.array([1, 1]), ids, max_depth=2).nodes[0].leaf_reason
        == "pure"
    )
    assert (
        fit_tree(np.array([[0.0], [0.0]]), np.array([0, 1]), ids, max_depth=2).nodes[0].leaf_reason
        == "no_positive_gain"
    )
    assert (
        fit_tree(np.array([[0.0], [0.0]]), np.array([0, 1]), ids, max_depth=2)
        .nodes[0]
        .predicted_class
        == 0
    )
    depth_tree = fit_tree(
        np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 1, 0, 1]), tuple("abcd"), max_depth=1
    )
    assert any(n.leaf_reason == "max_depth" for n in depth_tree.nodes)


def test_preorder_and_membership_conservation() -> None:
    x = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 1, 0, 1, 1])
    tree = fit_tree(x, y, tuple("abcdef"), max_depth=3)
    by_id = {n.node_id: n for n in tree.nodes}
    assert tree.nodes[0].node_id == "r"
    for node in tree.nodes:
        assert node.n_samples == len(node.sample_ids) == sum(node.class_counts)
        if node.is_leaf:
            continue
        left, right = by_id[node.left_child_id], by_id[node.right_child_id]
        assert left.step < right.step
        assert set(left.sample_ids).isdisjoint(right.sample_ids)
        assert set(left.sample_ids) | set(right.sample_ids) == set(node.sample_ids)
    assert predict_tree(tree, x).shape == (6,)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.0], [float("nan")]]),
        np.array([[0.0], [float("inf")]]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_invalid_feature_matrix_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_tree(bad, np.array([0, 1]), ("a", "b"), max_depth=2)


def test_invalid_labels_ids_and_depth_are_rejected() -> None:
    x = np.array([[0.0], [1.0]])
    for y in (np.array([0, 2]), np.array([0.0, 1.0]), np.array([0, 1, 0])):
        with pytest.raises(ValueError):
            fit_tree(x, y, ("a", "b"), max_depth=1)
    for ids in (("a", "a"), ("a",), ("a", "")):
        with pytest.raises(ValueError):
            fit_tree(x, np.array([0, 1]), ids, max_depth=1)
    for depth in (0, 6, True):
        with pytest.raises(ValueError):
            fit_tree(x, np.array([0, 1]), ("a", "b"), max_depth=depth)


def test_cancel_before_root_and_between_complete_expansions() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(TreeCancelledError):
        tuple(iter_tree(x, y, tuple("abcd"), max_depth=2, cancel_requested=lambda: True))
    emitted = []
    generator = iter_tree(x, y, tuple("abcd"), max_depth=2, cancel_requested=lambda: bool(emitted))
    emitted.append(next(generator))
    assert emitted[0].node_id == "r"
    with pytest.raises(TreeCancelledError):
        next(generator)


def test_wdbc_reference_is_diagnostic_not_tree_identity() -> None:
    from sklearn.tree import DecisionTreeClassifier

    from observatory.datasets.tabular.breast_cancer import load_breast_cancer

    data = load_breast_cancer(
        Path(__file__).resolve().parents[1] / "datasets/breast-cancer/versions/1.0.0.yaml"
    )
    tree = fit_tree(data.train.features, data.train.targets, data.train.sample_ids, max_depth=2)
    sklearn = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=2, random_state=0)
    sklearn.fit(data.train.features, data.train.targets)
    ours = predict_tree(tree, data.validation.features)
    assert ours.shape == sklearn.predict(data.validation.features).shape
    assert np.mean(ours == data.validation.targets) >= 0.75
