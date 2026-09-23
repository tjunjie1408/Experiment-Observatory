"""Independent exhaustive split and route checks on the pinned real dataset."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np

from observatory.datasets.tabular.breast_cancer import load_breast_cancer
from observatory.models.tree.cart import fit_tree


def _gini(targets: np.ndarray) -> Fraction:
    n = len(targets)
    n1 = int(np.count_nonzero(targets))
    return Fraction(2 * n1 * (n - n1), n * n)


def test_all_wdbc_splits_and_routes_match_independent_oracle() -> None:
    version = Path(__file__).resolve().parents[1] / "datasets/breast-cancer/versions/1.0.0.yaml"
    dataset = load_breast_cancer(version)
    x = dataset.train.features
    y = dataset.train.targets
    ids = dataset.train.sample_ids
    all_rows = (dataset.train, dataset.validation)
    for max_depth in range(1, 6):
        tree = fit_tree(x, y, ids, max_depth=max_depth)
        nodes = {node.node_id: node for node in tree.nodes}
        for node in tree.nodes:
            indices = np.array([ids.index(key) for key in node.sample_ids])
            node_y = y[indices]
            assert node.class_counts == (int(np.sum(node_y == 0)), int(np.sum(node_y == 1)))
            if node.is_leaf:
                continue
            parent_gini = _gini(node_y)
            best: tuple[Fraction, int, float, set[str]] | None = None
            for feature in range(x.shape[1]):
                values = sorted({float(value) for value in x[indices, feature]})
                for low, high in pairwise(values):
                    threshold = low / 2 + high / 2
                    if not low <= threshold < high:
                        threshold = low
                    left_mask = x[indices, feature] <= threshold
                    left_y, right_y = node_y[left_mask], node_y[~left_mask]
                    gain = parent_gini - (
                        len(left_y) * _gini(left_y) + len(right_y) * _gini(right_y)
                    ) / len(node_y)
                    if best is None or gain > best[0]:
                        best = (gain, feature, threshold, set(np.array(ids)[indices[left_mask]]))
            assert best is not None and best[0] > 0
            assert node.split_feature_index == best[1]
            assert node.split_threshold == best[2]
            assert set(nodes[node.left_child_id].sample_ids) == best[3]
            assert node.weighted_gini_decrease == float(best[0])
        for rows in all_rows:
            for features in rows.features:
                path = tree.path(features)
                cursor = nodes["r"]
                for node_id in path.node_ids:
                    assert node_id == cursor.node_id
                    if cursor.is_leaf:
                        assert cursor.predicted_class == path.predicted_class
                        break
                    cursor = nodes[
                        cursor.left_child_id
                        if features[cursor.split_feature_index] <= cursor.split_threshold
                        else cursor.right_child_id
                    ]
            assert len(rows.features) == len(rows.targets)
