from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from observatory.experiments.tree.study import run_tree_study
from observatory.runtime.tree_schema import TreeRunManifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tree/depth_study.yaml"


def _snapshots(runs_root: Path, manifest: TreeRunManifest) -> list[dict[str, object]]:
    path = runs_root / manifest.run_id / "snapshots.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def _shallow_structure(
    snapshots: list[dict[str, object]], below_depth: int
) -> dict[object, tuple[object, ...]]:
    return {
        node["nodeId"]: (node["isLeaf"], node["splitFeatureIndex"], node["splitThreshold"])
        for node in snapshots
        if isinstance(node["depth"], int) and node["depth"] < below_depth
    }


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[TreeRunManifest]]:
    runs_root = tmp_path_factory.mktemp("tree-study") / "runs"
    return runs_root, run_tree_study(CONFIG, runs_root, ROOT)


def test_study_records_five_completed_depths_on_one_split(
    study: tuple[Path, list[TreeRunManifest]],
) -> None:
    runs_root, manifests = study
    assert [m.training_config.max_depth for m in manifests] == [1, 2, 3, 4, 5]
    assert all(m.status == "completed" and m.stop_reason == "tree_complete" for m in manifests)
    assert len({m.run_id for m in manifests}) == 5
    assert {m.data_config.model_dump_json() for m in manifests} == {
        manifests[0].data_config.model_dump_json()
    }
    for manifest in manifests:
        snapshots = _snapshots(runs_root, manifest)
        assert manifest.n_snapshots_written == len(snapshots)
        assert max(int(str(node["depth"])) for node in snapshots) <= (
            manifest.training_config.max_depth
        )


def test_study_accuracy_matches_independent_recount(
    study: tuple[Path, list[TreeRunManifest]],
) -> None:
    runs_root, manifests = study
    for manifest in manifests:
        targets = {row.sample_id: (row.target, row.split) for row in manifest.dataset.roster}
        leaves = {
            node["nodeId"]: node["predictedClass"]
            for node in _snapshots(runs_root, manifest)
            if node["isLeaf"]
        }
        correct = {"train": 0, "validation": 0}
        total = {"train": 0, "validation": 0}
        for prediction in manifest.predictions:
            assert leaves[prediction.leaf_id] == prediction.predicted_class
            target, split = targets[prediction.sample_id]
            total[split] += 1
            correct[split] += int(prediction.predicted_class == target)
        assert manifest.train_evaluation is not None
        assert manifest.validation_evaluation is not None
        assert (manifest.train_evaluation.correct, manifest.train_evaluation.total) == (
            correct["train"],
            total["train"],
        )
        assert (manifest.validation_evaluation.correct, manifest.validation_evaluation.total) == (
            correct["validation"],
            total["validation"],
        )


def test_deeper_greedy_trees_extend_shallower_ones(
    study: tuple[Path, list[TreeRunManifest]],
) -> None:
    runs_root, manifests = study
    for shallow, deep in pairwise(manifests):
        depth = shallow.training_config.max_depth
        assert _shallow_structure(_snapshots(runs_root, shallow), depth) == _shallow_structure(
            _snapshots(runs_root, deep), depth
        )
        assert shallow.train_evaluation is not None and deep.train_evaluation is not None
        assert deep.train_evaluation.correct >= shallow.train_evaluation.correct


def test_study_rejects_non_fixed_depths_before_creating_runs(tmp_path: Path) -> None:
    config = tmp_path / "study.yaml"
    config.write_text(
        "experiment_id: wdbc-tree-depth-study\n"
        f"version_manifest: {(ROOT / 'datasets/breast-cancer/versions/1.0.0.yaml').as_posix()}\n"
        "depths: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"depths 1\.\.5"):
        run_tree_study(config, tmp_path / "runs", ROOT)
    assert not (tmp_path / "runs").exists()


def test_study_rejects_missing_dataset_before_creating_runs(tmp_path: Path) -> None:
    config = tmp_path / "study.yaml"
    config.write_text(
        "experiment_id: wdbc-tree-depth-study\n"
        "version_manifest: missing/1.0.0.yaml\n"
        "depths: [1, 2, 3, 4, 5]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version manifest"):
        run_tree_study(config, tmp_path / "runs", ROOT)
    assert not (tmp_path / "runs").exists()
