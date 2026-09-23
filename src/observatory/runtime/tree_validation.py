"""Self-contained schema-v4 bundle validation shared by export and tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

from observatory.runtime.tree_schema import TreeEvent, TreeRunManifest, TreeSnapshot
from observatory.wdbc_contract import FEATURE_NAMES


class TreeValidationError(ValueError):
    """Invalid schema-v4 run or bundle."""


def _gini(counts: tuple[int, int]) -> float:
    n = sum(counts)
    if n <= 0:
        raise TreeValidationError("empty node")
    return 2 * counts[0] * counts[1] / (n * n)


def validate_tree_bundle(
    manifest_raw: object, events_raw: list[object], snapshots_raw: list[object]
) -> tuple[TreeRunManifest, list[TreeEvent], list[TreeSnapshot]]:
    try:
        manifest = TreeRunManifest.model_validate(manifest_raw)
        events = [TreeEvent.model_validate(item) for item in events_raw]
        snapshots = [TreeSnapshot.model_validate(item) for item in snapshots_raw]
    except ValueError as exc:
        raise TreeValidationError(f"schema-v4 shape invalid: {exc}") from exc
    if manifest.status != "completed" or manifest.stop_reason != "tree_complete":
        raise TreeValidationError("run is not completed")
    if manifest.train_evaluation is None or manifest.validation_evaluation is None:
        raise TreeValidationError("completed tree requires both final evaluations")
    cfg = manifest.data_config
    if cfg.feature_names != list(FEATURE_NAMES) or cfg.target_mapping != {"B": 0, "M": 1}:
        raise TreeValidationError("WDBC feature/target contract invalid")
    roster = manifest.dataset.roster
    keys = [row.sample_id for row in roster]
    if keys != sorted(set(keys)) or len(roster) != 569:
        raise TreeValidationError("dataset roster must contain 569 unique sorted IDs")
    labels = {row.sample_id: row.target for row in roster}
    training = {row.sample_id for row in roster if row.split == "train"}
    validation = set(keys) - training
    if len(training) != cfg.train_count or len(validation) != cfg.validation_count:
        raise TreeValidationError("split counts do not match dataset roster")
    observed = manifest.dataset.observed_rows
    observed_ids = [row.sample_id for row in observed]
    if observed_ids != sorted(set(observed_ids)) or observed_ids != manifest.observed_sample_ids:
        raise TreeValidationError("observed sample IDs are inconsistent")
    for row in observed:
        roster_row = next((item for item in roster if item.sample_id == row.sample_id), None)
        if roster_row is None or row.target != roster_row.target or row.split != roster_row.split:
            raise TreeValidationError("observed row does not match roster")
        if not all(math.isfinite(value) for value in row.features):
            raise TreeValidationError("observed row has non-finite features")
    for split in ("train", "validation"):
        for target in (0, 1):
            expected = [
                row.sample_id for row in roster if row.split == split and row.target == target
            ][:5]
            actual = [
                row.sample_id for row in observed if row.split == split and row.target == target
            ]
            if actual != expected:
                raise TreeValidationError("observed stratum selection invalid")
    if (
        not snapshots
        or manifest.n_snapshots_written != len(snapshots)
        or manifest.last_valid_step != len(snapshots) - 1
    ):
        raise TreeValidationError("snapshot count/last step is inconsistent")
    if [node.step for node in snapshots] != list(range(len(snapshots))):
        raise TreeValidationError("snapshot steps must be contiguous")
    if [event.seq for event in events] != list(range(1, len(events) + 1)) or any(
        event.run_id != manifest.run_id for event in events
    ):
        raise TreeValidationError("events have noncontiguous sequence or wrong run ID")
    if (
        len(events) != len(snapshots) + 3
        or [event.kind for event in events[:2]] != ["run.created", "run.started"]
        or events[-1].kind != "run.completed"
    ):
        raise TreeValidationError("tree lifecycle events are incomplete")
    nodes: dict[str, TreeSnapshot] = {}
    pending: list[str] = ["r"]
    for node in snapshots:
        if not pending or pending.pop() != node.node_id or node.node_id in nodes:
            raise TreeValidationError("node IDs violate DFS pre-order or repeat")
        if node.node_id == "r":
            if node.parent_id is not None or node.side is not None or node.depth != 0:
                raise TreeValidationError("root parent/depth invalid")
        else:
            parent = nodes.get(node.parent_id or "")
            if (
                parent is None
                or node.side not in ("left", "right")
                or node.depth != parent.depth + 1
            ):
                raise TreeValidationError("node parent/depth invalid")
            if node.node_id != f"{parent.node_id}{'L' if node.side == 'left' else 'R'}":
                raise TreeValidationError("node side/ID invalid")
        if node.sample_ids != sorted(set(node.sample_ids)) or not set(node.sample_ids) <= training:
            raise TreeValidationError("node membership has unknown, repeated or unsorted IDs")
        expected_counts = (
            sum(labels[key] == 0 for key in node.sample_ids),
            sum(labels[key] == 1 for key in node.sample_ids),
        )
        if (
            node.n_samples != len(node.sample_ids)
            or node.class_counts != expected_counts
            or not math.isclose(node.gini_parent, _gini(expected_counts), abs_tol=1e-12, rel_tol=0)
        ):
            raise TreeValidationError("node count, class distribution or Gini mismatch")
        event = events[node.step + 2]
        if (
            event.step != node.step
            or event.node_id != node.node_id
            or event.kind != ("tree.node_leaf" if node.is_leaf else "tree.node_split")
        ):
            raise TreeValidationError("node event does not match snapshot")
        if node.is_leaf:
            if node.predicted_class != int(expected_counts[1] > node.n_samples / 2):
                raise TreeValidationError("leaf prediction does not match class counts")
            if node.predicted_label != ("Malignant" if node.predicted_class == 1 else "Benign"):
                raise TreeValidationError("leaf display label does not match class")
        else:
            if node.depth >= manifest.training_config.max_depth:
                raise TreeValidationError("split exceeds configured depth")
            if node.split_feature_index is None or not 0 <= node.split_feature_index < 30:
                raise TreeValidationError("split feature is out of range")
            if node.split_feature_name != cfg.feature_names[node.split_feature_index]:
                raise TreeValidationError("split feature name does not match index")
            if (
                node.left_child_id != f"{node.node_id}L"
                or node.right_child_id != f"{node.node_id}R"
            ):
                raise TreeValidationError("split child references invalid")
            pending.extend((node.right_child_id, node.left_child_id))
        nodes[node.node_id] = node
    if pending or set(snapshots[0].sample_ids) != training:
        raise TreeValidationError("missing children or root membership mismatch")
    for node in snapshots:
        if node.is_leaf:
            continue
        left = nodes[node.left_child_id or ""]
        right = nodes[node.right_child_id or ""]
        if set(left.sample_ids) & set(right.sample_ids) or set(left.sample_ids) | set(
            right.sample_ids
        ) != set(node.sample_ids):
            raise TreeValidationError("child membership does not conserve parent")
        expected_gain = (
            node.gini_parent
            - (len(left.sample_ids) * left.gini_parent + len(right.sample_ids) * right.gini_parent)
            / node.n_samples
        )
        if (
            node.left_count != left.n_samples
            or node.right_count != right.n_samples
            or not all(
                math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
                for value, expected in (
                    (node.gini_left, left.gini_parent),
                    (node.gini_right, right.gini_parent),
                    (node.weighted_gini_decrease, expected_gain),
                )
                if value is not None
            )
            or expected_gain <= 0
        ):
            raise TreeValidationError("split child counts/Gini decrease invalid")
    predictions = manifest.predictions
    if [item.sample_id for item in predictions] != keys:
        raise TreeValidationError("predictions must cover sorted roster exactly")
    prediction_by_id = {item.sample_id: item for item in predictions}
    for row in observed:
        node = nodes["r"]
        while not node.is_leaf:
            assert node.split_feature_index is not None
            assert node.split_threshold is not None
            child_id = (
                node.left_child_id
                if row.features[node.split_feature_index] <= node.split_threshold
                else node.right_child_id
            )
            assert child_id is not None
            node = nodes[child_id]
        if prediction_by_id[row.sample_id].leaf_id != node.node_id:
            raise TreeValidationError("observed sample route does not match prediction")
    counts: dict[str, tuple[int, int]] = {"train": (0, 0), "validation": (0, 0)}
    for item, roster_row in zip(predictions, roster, strict=True):
        leaf = nodes.get(item.leaf_id)
        if leaf is None or not leaf.is_leaf or leaf.predicted_class != item.predicted_class:
            raise TreeValidationError("prediction references invalid leaf/class")
        if roster_row.split == "train" and item.sample_id not in leaf.sample_ids:
            raise TreeValidationError("training prediction leaf membership invalid")
        correct, total = counts[roster_row.split]
        counts[roster_row.split] = (
            correct + int(roster_row.target == item.predicted_class),
            total + 1,
        )
    for split, evaluation in (
        ("train", manifest.train_evaluation),
        ("validation", manifest.validation_evaluation),
    ):
        correct, total = counts[split]
        if (
            evaluation.correct != correct
            or evaluation.total != total
            or not math.isclose(evaluation.accuracy, correct / total, abs_tol=1e-12, rel_tol=0)
        ):
            raise TreeValidationError(f"{split} evaluation does not match predictions")
    return manifest, events, snapshots


def validate_tree_run(run_dir: Path) -> tuple[TreeRunManifest, list[TreeEvent], list[TreeSnapshot]]:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshots = json.loads((run_dir / "snapshots.json").read_text(encoding="utf-8"))
        if not isinstance(snapshots, list):
            raise TreeValidationError("snapshots.json must be an array")
        return validate_tree_bundle(manifest, events, snapshots)
    except (OSError, json.JSONDecodeError) as exc:
        raise TreeValidationError(f"cannot read tree artifacts: {exc}") from exc
