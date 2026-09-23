"""Small numeric CART classifier with inspectable node expansions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt

LeafReason = Literal["pure", "max_depth", "insufficient_samples", "no_positive_gain"]
Side = Literal["left", "right"]


class TreeCancelledError(Exception):
    """A requested stop was observed at a node boundary."""


@dataclass(frozen=True)
class TreeNode:
    step: int
    node_id: str
    parent_id: str | None
    side: Side | None
    depth: int
    sample_ids: tuple[str, ...]
    n_samples: int
    class_counts: tuple[int, int]
    gini_parent: float
    is_leaf: bool
    split_feature_index: int | None = None
    split_threshold: float | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    left_count: int | None = None
    right_count: int | None = None
    gini_left: float | None = None
    gini_right: float | None = None
    weighted_gini_decrease: float | None = None
    predicted_class: int | None = None
    leaf_reason: LeafReason | None = None


@dataclass(frozen=True)
class TreePath:
    node_ids: tuple[str, ...]
    predicted_class: int


@dataclass(frozen=True)
class TreeModel:
    nodes: tuple[TreeNode, ...]
    n_features: int

    def path(self, row: npt.ArrayLike) -> TreePath:
        values = np.asarray(row)
        if values.shape != (self.n_features,) or values.dtype.kind not in "fiu":
            raise ValueError("row must contain one numeric value per feature")
        if not np.isfinite(values).all():
            raise ValueError("row contains non-finite features")
        lookup = {node.node_id: node for node in self.nodes}
        current = lookup["r"]
        visited: list[str] = []
        while True:
            visited.append(current.node_id)
            if current.is_leaf:
                assert current.predicted_class is not None
                return TreePath(tuple(visited), current.predicted_class)
            assert current.split_feature_index is not None
            assert current.split_threshold is not None
            child_id = (
                current.left_child_id
                if values[current.split_feature_index] <= current.split_threshold
                else current.right_child_id
            )
            assert child_id is not None
            current = lookup[child_id]


def _gini(counts: tuple[int, int]) -> Fraction:
    count = sum(counts)
    if count == 0:
        raise ValueError("empty tree node")
    return Fraction(2 * counts[0] * counts[1], count * count)


def _validate(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    sample_ids: tuple[str, ...],
    max_depth: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    x_input = np.asarray(features)
    y_input = np.asarray(targets)
    if (
        x_input.ndim != 2
        or x_input.shape[0] < 2
        or x_input.shape[1] < 1
        or x_input.dtype.kind not in "fiu"
        or not np.isfinite(x_input).all()
    ):
        raise ValueError("features must be a finite numeric matrix with at least two rows")
    if (
        y_input.shape != (len(x_input),)
        or y_input.dtype.kind not in "iu"
        or not np.isin(y_input, [0, 1]).all()
    ):
        raise ValueError("targets must be a binary integer vector aligned with features")
    if (
        len(sample_ids) != len(x_input)
        or any(not isinstance(key, str) or not key for key in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("sample IDs must be unique nonempty strings aligned with features")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 5:
        raise ValueError("max_depth must be an integer from 1 to 5")
    return np.asarray(x_input, dtype=np.float64), np.asarray(y_input, dtype=np.int64)


def iter_tree(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    sample_ids: tuple[str, ...],
    *,
    max_depth: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> Iterator[TreeNode]:
    """Yield complete selected node expansions in left-first DFS pre-order."""
    x, y = _validate(features, targets, sample_ids, max_depth)
    stack: list[tuple[str, str | None, Side | None, int, tuple[int, ...]]] = [
        ("r", None, None, 0, tuple(range(len(x))))
    ]
    step = 0
    while stack:
        if cancel_requested is not None and cancel_requested():
            raise TreeCancelledError("tree construction cancelled before next node")
        node_id, parent_id, side, depth, members = stack.pop()
        n = len(members)
        positives = sum(int(y[i]) for i in members)
        counts = (n - positives, positives)
        parent_gini = _gini(counts)
        base = TreeNode(
            step=step,
            node_id=node_id,
            parent_id=parent_id,
            side=side,
            depth=depth,
            sample_ids=tuple(sorted(sample_ids[i] for i in members)),
            n_samples=n,
            class_counts=counts,
            gini_parent=float(parent_gini),
            is_leaf=True,
        )
        step += 1
        reason: LeafReason | None = None
        if positives in (0, n):
            reason = "pure"
        elif depth >= max_depth:
            reason = "max_depth"
        elif n < 2:
            reason = "insufficient_samples"
        if reason is not None:
            yield replace(base, predicted_class=int(positives > n / 2), leaf_reason=reason)
            continue

        best_gain = Fraction(0)
        best_feature: int | None = None
        best_threshold = 0.0
        best_left: tuple[int, ...] = ()
        best_right: tuple[int, ...] = ()
        best_counts = ((0, 0), (0, 0))
        for feature in range(x.shape[1]):
            ordered = sorted(members, key=lambda i: (x[i, feature], sample_ids[i]))
            left_positive = 0
            for position in range(1, n):
                left_positive += int(y[ordered[position - 1]])
                a = float(x[ordered[position - 1], feature])
                b = float(x[ordered[position], feature])
                if a == b:
                    continue
                left_counts = (position - left_positive, left_positive)
                right_counts = (
                    counts[0] - left_counts[0],
                    counts[1] - left_counts[1],
                )
                gain = (
                    parent_gini
                    - Fraction(position, n) * _gini(left_counts)
                    - Fraction(n - position, n) * _gini(right_counts)
                )
                if gain <= best_gain:
                    continue
                threshold = a / 2 + b / 2
                if not np.isfinite(threshold):
                    raise FloatingPointError("tree threshold became non-finite")
                if not a <= threshold < b:
                    threshold = a
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left = tuple(ordered[:position])
                best_right = tuple(ordered[position:])
                best_counts = (left_counts, right_counts)
        if best_feature is None:
            yield replace(
                base,
                predicted_class=int(positives > n / 2),
                leaf_reason="no_positive_gain",
            )
            continue
        left_id = f"{node_id}L"
        right_id = f"{node_id}R"
        yield replace(
            base,
            is_leaf=False,
            split_feature_index=best_feature,
            split_threshold=best_threshold,
            left_child_id=left_id,
            right_child_id=right_id,
            left_count=len(best_left),
            right_count=len(best_right),
            gini_left=float(_gini(best_counts[0])),
            gini_right=float(_gini(best_counts[1])),
            weighted_gini_decrease=float(best_gain),
        )
        stack.append((right_id, node_id, "right", depth + 1, best_right))
        stack.append((left_id, node_id, "left", depth + 1, best_left))
    if cancel_requested is not None and cancel_requested():
        raise TreeCancelledError("tree construction cancelled before completion")


def fit_tree(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    sample_ids: tuple[str, ...],
    *,
    max_depth: int,
    cancel_requested: Callable[[], bool] | None = None,
) -> TreeModel:
    nodes = tuple(
        iter_tree(
            features, targets, sample_ids, max_depth=max_depth, cancel_requested=cancel_requested
        )
    )
    return TreeModel(nodes, np.asarray(features).shape[1])


def predict_tree(model: TreeModel, features: npt.ArrayLike) -> npt.NDArray[np.int64]:
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != model.n_features:
        raise ValueError("prediction features do not match fitted tree")
    return np.asarray([model.path(row).predicted_class for row in x], dtype=np.int64)
