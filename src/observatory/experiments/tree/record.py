"""Record complete WDBC CART node expansions as schema-v4 artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from observatory.datasets.tabular.breast_cancer import BreastCancerDataset, load_breast_cancer
from observatory.models.tree.cart import TreeCancelledError, TreeModel, TreeNode, iter_tree
from observatory.runtime.storage import (
    append_jsonl,
    atomic_write_json,
    get_code_provenance,
    new_run_id,
)
from observatory.runtime.tree_schema import (
    TreeDataConfig,
    TreeDatasetSummary,
    TreeEvaluation,
    TreeEvent,
    TreeObservedRow,
    TreePrediction,
    TreeRosterRow,
    TreeRunManifest,
    TreeSnapshot,
    TreeTrainingConfig,
)


def _json(model: TreeRunManifest | TreeSnapshot | TreeEvent) -> dict[str, object]:
    return cast(dict[str, object], json.loads(model.model_dump_json(by_alias=True)))


def _manifest(
    dataset: BreastCancerDataset, run_id: str, depth: int, repo_root: Path
) -> TreeRunManifest:
    roster = [
        TreeRosterRow(
            sample_id=key, target=int(label), split=cast(Literal["train", "validation"], split)
        )
        for split, partition in (("train", dataset.train), ("validation", dataset.validation))
        for key, label in zip(partition.sample_ids, partition.targets, strict=True)
    ]
    roster.sort(key=lambda row: row.sample_id)
    observed = [
        TreeObservedRow(
            sample_id=key,
            target=int(partition.targets[i]),
            split=cast(Literal["train", "validation"], split),
            features=partition.features[i].tolist(),
        )
        for split, partition in (("train", dataset.train), ("validation", dataset.validation))
        for i, key in enumerate(partition.sample_ids)
        if key in dataset.observed_sample_ids
    ]
    observed.sort(key=lambda row: row.sample_id)
    return TreeRunManifest(
        run_id=run_id,
        experiment_id="wdbc-tree-depth-study",
        created_at=datetime.now(UTC).isoformat(),
        status="running",
        data_config=TreeDataConfig(
            version_manifest_sha256=dataset.version_manifest_sha256,
            processed_artifact_sha256=dataset.processed_artifact_sha256,
            split_sha256=dataset.split_sha256,
            feature_names=list(dataset.feature_names),
            target_mapping={"B": 0, "M": 1},
        ),
        dataset=TreeDatasetSummary(roster=roster, observed_rows=observed),
        training_config=TreeTrainingConfig(max_depth=depth),
        code_provenance=get_code_provenance(repo_root),
        observed_sample_ids=list(dataset.observed_sample_ids),
    )


def _evaluate(
    model: TreeModel, dataset: BreastCancerDataset
) -> tuple[TreeEvaluation, TreeEvaluation, list[TreePrediction]]:
    results: list[TreeEvaluation] = []
    predictions: list[TreePrediction] = []
    for partition in (dataset.train, dataset.validation):
        correct = 0
        for key, row, target in zip(
            partition.sample_ids, partition.features, partition.targets, strict=True
        ):
            path = model.path(row)
            correct += int(path.predicted_class == target)
            predictions.append(
                TreePrediction(
                    sample_id=key, leaf_id=path.node_ids[-1], predicted_class=path.predicted_class
                )
            )
        results.append(
            TreeEvaluation(
                correct=correct,
                total=len(partition.sample_ids),
                accuracy=correct / len(partition.sample_ids),
            )
        )
    predictions.sort(key=lambda item: item.sample_id)
    return results[0], results[1], predictions


def run_tree_experiment(
    version_manifest: Path,
    *,
    max_depth: int,
    runs_root: Path,
    repo_root: Path,
    cancel_requested: Callable[[], bool] | None = None,
) -> TreeRunManifest:
    """Validate dataset before creating a run; persist only whole node frames."""
    dataset = load_breast_cancer(version_manifest)
    training = TreeTrainingConfig(max_depth=max_depth)
    run_id = new_run_id(f"wdbc-tree-depth-{max_depth}")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(dataset, run_id, training.max_depth, repo_root)
    atomic_write_json(run_dir / "manifest.json", _json(manifest))
    seq = 0

    def event(
        kind: Literal[
            "run.created",
            "run.started",
            "tree.node_split",
            "tree.node_leaf",
            "run.completed",
            "run.cancelled",
            "run.failed",
        ],
        *,
        step: int | None = None,
        node_id: str | None = None,
    ) -> None:
        nonlocal seq
        seq += 1
        append_jsonl(
            run_dir / "events.jsonl",
            _json(TreeEvent(run_id=run_id, seq=seq, kind=kind, step=step, node_id=node_id)),
        )

    event("run.created")
    event("run.started")
    snapshots: list[TreeSnapshot] = []
    nodes: list[TreeNode] = []
    try:
        for node in iter_tree(
            dataset.train.features,
            dataset.train.targets,
            dataset.train.sample_ids,
            max_depth=max_depth,
            cancel_requested=cancel_requested,
        ):
            snapshot = TreeSnapshot.model_validate(
                {
                    **node.__dict__,
                    "split_feature_name": (
                        dataset.feature_names[node.split_feature_index]
                        if node.split_feature_index is not None
                        else None
                    ),
                    "predicted_label": ("Malignant" if node.predicted_class == 1 else "Benign")
                    if node.is_leaf
                    else None,
                }
            )
            snapshots.append(snapshot)
            nodes.append(node)
            atomic_write_json(run_dir / "snapshots.json", [_json(item) for item in snapshots])
            event(
                "tree.node_leaf" if node.is_leaf else "tree.node_split",
                step=node.step,
                node_id=node.node_id,
            )
            manifest = manifest.model_copy(
                update={"last_valid_step": node.step, "n_snapshots_written": len(snapshots)}
            )
            atomic_write_json(run_dir / "manifest.json", _json(manifest))
        model = TreeModel(tuple(nodes), len(dataset.feature_names))
        train, validation, predictions = _evaluate(model, dataset)
        completed = manifest.model_copy(
            update={
                "status": "completed",
                "stop_reason": "tree_complete",
                "train_evaluation": train,
                "validation_evaluation": validation,
                "predictions": predictions,
            }
        )
        event("run.completed")
        atomic_write_json(run_dir / "manifest.json", _json(completed))
        return completed
    except TreeCancelledError:
        cancelled = manifest.model_copy(
            update={"status": "cancelled", "stop_reason": "user_cancelled"}
        )
        event("run.cancelled")
        atomic_write_json(run_dir / "manifest.json", _json(cancelled))
        return cancelled
    except Exception as exc:
        failed = manifest.model_copy(
            update={"status": "failed", "stop_reason": "runtime_error", "error_message": str(exc)}
        )
        event("run.failed")
        atomic_write_json(run_dir / "manifest.json", _json(failed))
        raise
