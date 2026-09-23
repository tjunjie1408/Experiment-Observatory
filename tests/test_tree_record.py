from __future__ import annotations

import json
from pathlib import Path

import pytest

from observatory.experiments.tree.record import run_tree_experiment
from observatory.runtime.export import ExportError, export_run

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / "datasets/breast-cancer/versions/1.0.0.yaml"


def test_real_tree_run_exports_and_matches_recorded_metrics(tmp_path: Path) -> None:
    manifest = run_tree_experiment(
        VERSION, max_depth=2, runs_root=tmp_path / "runs", repo_root=ROOT
    )
    assert manifest.status == "completed"
    assert (
        manifest.data_config.split_sha256
        == "0ef6330425ba5975465eabfd46565959ac859b694ac3f55598541a83cb41b3f1"
    )
    assert manifest.train_evaluation is not None
    assert manifest.validation_evaluation is not None
    assert manifest.train_evaluation.total == 397
    assert manifest.validation_evaluation.total == 172
    assert len(manifest.predictions) == 569
    run_dir = tmp_path / "runs" / manifest.run_id
    export_dir = tmp_path / "export"
    export_run(run_dir, export_dir)
    assert (export_dir / "manifest.json").is_file()
    snapshots = json.loads((export_dir / "snapshots.json").read_text())
    assert snapshots[0]["nodeId"] == "r"
    assert [item["step"] for item in snapshots] == list(range(len(snapshots)))


def test_cancelled_tree_remains_unexportable(tmp_path: Path) -> None:
    requested = iter([False, True]).__next__
    manifest = run_tree_experiment(
        VERSION,
        max_depth=2,
        runs_root=tmp_path / "runs",
        repo_root=ROOT,
        cancel_requested=requested,
    )
    assert manifest.status == "cancelled"
    assert manifest.stop_reason == "user_cancelled"
    with pytest.raises(ExportError, match="completed"):
        export_run(tmp_path / "runs" / manifest.run_id, tmp_path / "forbidden")


def test_export_rejects_tampered_membership(tmp_path: Path) -> None:
    manifest = run_tree_experiment(
        VERSION, max_depth=1, runs_root=tmp_path / "runs", repo_root=ROOT
    )
    source = tmp_path / "runs" / manifest.run_id / "snapshots.json"
    snapshots = json.loads(source.read_text())
    snapshots[1]["sampleIds"][0] = "unknown"
    source.write_text(json.dumps(snapshots))
    with pytest.raises(ExportError, match=r"membership|unknown"):
        export_run(source.parent, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
