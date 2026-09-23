"""Real SQLite tracking integration over a real WDBC recorder run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from observatory.analytics.records import AnalysisInputError, load_analysis_run
from observatory.analytics.recovery import (
    BackupError,
    backup_tracking,
    restore_tracking_backup,
    verify_tracking_backup,
)
from observatory.analytics.tracking import (
    inspect_tracking,
    read_tracking_link,
    sync_run,
    tracking_uri,
)
from observatory.analytics.warehouse import browser_catalog, build_batch, rebuild_catalog
from observatory.api.app import create_app
from observatory.api.catalog import read_catalog
from observatory.cli import build_parser, run_one
from observatory.datasets.synthetic.blobs import KMeansSyntheticConfig
from observatory.experiments.kmeans.record import run_kmeans_experiment
from observatory.experiments.tree.record import run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / "datasets/breast-cancer/versions/1.0.0.yaml"


def test_local_api_reads_verified_mlflow_step_history_without_a_new_file(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_tree_experiment(VERSION, max_depth=1, runs_root=runs_root, repo_root=ROOT)
    run_dir = runs_root / manifest.run_id
    database = tmp_path / "database/mlflow.db"
    app = create_app(
        runs_root=runs_root,
        repo_root=tmp_path,
        tracking_database=database,
    )

    with TestClient(app) as client:
        endpoint = f"/api/tracking/{manifest.run_id}/metrics"
        assert client.get(endpoint).status_code == 404
        synced = sync_run(run_dir, database)
        response = client.get(endpoint)
        assert response.status_code == 200
        payload = response.json()
        assert payload["runId"] == manifest.run_id
        assert payload["source"] == "mlflow"
        assert payload["state"] == "verified"
        series = {item["key"]: item for item in payload["series"]}
        assert series["accuracy.validation"]["points"] == [
            {"step": manifest.last_valid_step, "value": manifest.validation_evaluation.accuracy}
        ]
        assert series["nodes_recorded.all"]["points"][0] == {"step": 0, "value": 1.0}
        assert not (run_dir / "metrics.json").exists()

        MlflowClient(tracking_uri=tracking_uri(database)).log_metric(
            synced.mlflow_run_id, "accuracy.validation", 0.0, step=manifest.last_valid_step
        )
        assert client.get(endpoint).status_code == 409

    missing_app = create_app(
        runs_root=runs_root,
        repo_root=tmp_path,
        tracking_database=tmp_path / "missing.db",
    )
    with TestClient(missing_app) as client:
        assert client.get(endpoint).status_code == 503
        assert client.get("/api/tracking/../metrics").status_code == 404


def test_sync_is_idempotent_and_preserves_recorder_evidence(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_tree_experiment(VERSION, max_depth=1, runs_root=runs_root, repo_root=ROOT)
    run_dir = runs_root / manifest.run_id
    before = (run_dir / "manifest.json").read_bytes()
    db = tmp_path / "database/mlflow.db"

    first = sync_run(run_dir, db)
    second = sync_run(run_dir, db)

    assert first.created and not second.created
    assert first.mlflow_run_id == second.mlflow_run_id
    assert (run_dir / "manifest.json").read_bytes() == before
    source = load_analysis_run(run_dir)
    assert read_tracking_link(run_dir, source.manifest_sha256) == first.mlflow_run_id
    assert inspect_tracking(source, db) == ("verified", first.mlflow_run_id)
    assert inspect_tracking(source, tmp_path / "missing.db")[0] == "unavailable"
    batch = build_batch([run_dir], tmp_path / "warehouse", tracking_database=db)
    assert browser_catalog(batch.batch_dir)["runs"][0]["tracking_state"] == "verified"
    tracked = MlflowClient(tracking_uri=tracking_uri(db)).get_run(first.mlflow_run_id)
    assert tracked.info.status == "FINISHED"
    assert tracked.data.params["dataset_id"] == "uci-wdbc"
    assert tracked.data.params["split_identity"] == manifest.data_config.split_sha256
    assert tracked.data.metrics["accuracy.validation"] == manifest.validation_evaluation.accuracy
    assert (
        len(
            MlflowClient(tracking_uri=tracking_uri(db)).get_metric_history(
                first.mlflow_run_id, "accuracy.validation"
            )
        )
        == 1
    )
    backup_dir = tmp_path / "backup/mlflow"
    backup_tracking(db, backup_dir)
    verify_tracking_backup(backup_dir)
    restored = MlflowClient(tracking_uri=tracking_uri(backup_dir / "mlflow.db"))
    assert restored.get_run(first.mlflow_run_id).data.metrics["accuracy.validation"] == (
        manifest.validation_evaluation.accuracy
    )
    backup_db = backup_dir / "mlflow.db"
    backup_db.write_bytes(backup_db.read_bytes() + b"tampered")
    with pytest.raises(BackupError, match="hash mismatch"):
        verify_tracking_backup(backup_dir)


def test_corrupt_completed_run_is_not_synced(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_tree_experiment(VERSION, max_depth=1, runs_root=runs_root, repo_root=ROOT)
    run_dir = runs_root / manifest.run_id
    source = run_dir / "snapshots.json"
    snapshots = json.loads(source.read_text(encoding="utf-8"))
    snapshots[0]["sampleIds"][0] = "unknown"
    source.write_text(json.dumps(snapshots), encoding="utf-8")
    db = tmp_path / "database/mlflow.db"
    with pytest.raises(AnalysisInputError):
        sync_run(run_dir, db)
    assert not db.exists()


def test_linear_and_kmeans_share_tracking_timeline_contract(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    linear = run_one(ROOT / "configs/linear/converge.yaml", runs_root)
    kmeans = run_kmeans_experiment(
        experiment_id="tracking-kmeans",
        data_config=KMeansSyntheticConfig(
            blob_centers=((-4.0, -2.0), (4.0, 2.0)),
            blob_sizes=(12, 12),
            cluster_std=0.55,
            seed=2026,
        ),
        n_clusters=2,
        init_seed=0,
        max_iterations=10,
        runs_root=runs_root,
        repo_root=ROOT,
    )
    db = tmp_path / "database/mlflow.db"
    for manifest, key in ((linear, "mse.train"), (kmeans, "inertia.all")):
        run_dir = runs_root / manifest.run_id
        snapshots = json.loads((run_dir / "snapshots.json").read_text(encoding="utf-8"))
        first = sync_run(run_dir, db)
        second = sync_run(run_dir, db)
        assert first.mlflow_run_id == second.mlflow_run_id
        client = MlflowClient(tracking_uri=tracking_uri(db))
        history = client.get_metric_history(first.mlflow_run_id, key)
        assert [(item.step, item.value) for item in history] == [
            (item["step"], item["trainMse"] if key == "mse.train" else item["inertia"])
            for item in snapshots
        ]
        assert {
            item.path for item in client.list_artifacts(first.mlflow_run_id, "observatory-replay")
        } == {
            f"observatory-replay/{name}"
            for name in ("manifest.json", "events.jsonl", "snapshots.json")
        }


def test_backup_restores_artifacts_with_new_location_after_source_is_removed(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", runs_root)
    run_dir = runs_root / manifest.run_id
    old_root = tmp_path / "old-location"
    old_id = sync_run(run_dir, old_root / "mlflow.db").mlflow_run_id
    backup_dir = tmp_path / "portable-backup"
    backup_tracking(old_root / "mlflow.db", backup_dir)
    (old_root / "mlflow-artifacts").rename(old_root / "artifacts-no-longer-at-old-uri")
    new_root = tmp_path / "new-location"
    mapping = restore_tracking_backup(backup_dir, new_root)
    new_id = mapping[old_id]
    assert new_id != old_id
    restored = MlflowClient(tracking_uri=tracking_uri(new_root / "mlflow.db"))
    assert restored.get_run(new_id).data.tags["observatory_run_id"] == manifest.run_id
    assert len(restored.get_metric_history(new_id, "mse.train")) == manifest.n_snapshots_written
    downloaded = Path(restored.download_artifacts(new_id, "observatory-replay/snapshots.json"))
    assert downloaded.read_bytes() == (run_dir / "snapshots.json").read_bytes()
    assert sync_run(run_dir, new_root / "mlflow.db").mlflow_run_id == new_id


def test_invalid_backup_never_publishes_relocated_tracking(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", runs_root)
    db = tmp_path / "source/mlflow.db"
    sync_run(runs_root / manifest.run_id, db)
    backup_dir = tmp_path / "backup"
    backup_tracking(db, backup_dir)
    artifact_files = list(verify_tracking_backup(backup_dir)["artifactSha256"])
    assert artifact_files
    (backup_dir / "mlflow-artifacts" / artifact_files[0]).write_text("tampered", encoding="utf-8")
    target = tmp_path / "restored"
    with pytest.raises(BackupError):
        restore_tracking_backup(backup_dir, target)
    assert not target.exists()


def test_cli_common_tracking_hook_keeps_runtime_independent(tmp_path: Path) -> None:
    parser = build_parser()
    runs_root = tmp_path / "runs"
    db = tmp_path / "database/mlflow.db"
    args = parser.parse_args(
        [
            "run",
            str(ROOT / "configs/linear/slow.yaml"),
            "--runs-root",
            str(runs_root),
            "--tracking-database",
            str(db),
        ]
    )
    assert args.func(args) == 0
    run_dir = next(runs_root.iterdir())
    assert (run_dir / "tracking.json").is_file()
    assert len(MlflowClient(tracking_uri=tracking_uri(db)).search_experiments()) > 0
    for command, config in (
        ("run-dataset", "configs/linear/auto_mpg_weight.yaml"),
        ("run-all", "configs/linear/slow.yaml"),
        ("run-kmeans-study", "configs/kmeans/initialization_study.yaml"),
        ("run-tree-study", "configs/tree/depth_study.yaml"),
    ):
        parsed = parser.parse_args([command, str(ROOT / config), "--tracking-database", str(db)])
        assert parsed.tracking_database == str(db)


def test_local_stack_rebuilds_in_another_directory(tmp_path: Path) -> None:
    old = tmp_path / "original"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", old / "runs")
    source_run = old / "runs" / manifest.run_id
    old_db = old / "database/mlflow.db"
    sync_run(source_run, old_db)
    batch = build_batch([source_run], old / "warehouse", tracking_database=old_db)
    backup_tracking(old_db, old / "tracking-backup")

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    copied_run = relocated / "runs" / manifest.run_id
    shutil.copytree(source_run, copied_run)
    copied_batch = relocated / "warehouse" / batch.batch_id
    shutil.copytree(batch.batch_dir, copied_batch)
    copied_backup = relocated / "backup"
    shutil.copytree(old / "tracking-backup", copied_backup)
    (old / "database/mlflow-artifacts").rename(old / "database/original-artifacts-offline")

    new_tracking = relocated / "database"
    restored_ids = restore_tracking_backup(copied_backup, new_tracking)
    assert len(restored_ids) == 1
    assert sync_run(copied_run, new_tracking / "mlflow.db").mlflow_run_id in restored_ids.values()
    new_catalog = relocated / "catalog.duckdb"
    rebuild_catalog(copied_batch, new_catalog)
    assert read_catalog(new_catalog)["runs"][0]["run_id"] == manifest.run_id
    refreshed = build_batch(
        [copied_run], relocated / "new-warehouse", tracking_database=new_tracking / "mlflow.db"
    )
    assert browser_catalog(refreshed.batch_dir)["runs"][0]["tracking_state"] == "verified"


def test_restore_rejects_foreign_mlflow_run_instead_of_silently_omitting_it(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", runs_root)
    db = tmp_path / "source/mlflow.db"
    sync_run(runs_root / manifest.run_id, db)
    client = MlflowClient(tracking_uri=tracking_uri(db))
    foreign_id = client.create_experiment(
        "foreign", artifact_location=(tmp_path / "foreign-artifacts").resolve().as_uri()
    )
    client.create_run(foreign_id)
    backup_dir = tmp_path / "backup"
    backup_tracking(db, backup_dir)
    target = tmp_path / "relocated"
    with pytest.raises(BackupError, match="not an Observatory run"):
        restore_tracking_backup(backup_dir, target)
    assert not (target / "restore.json").exists()


def test_restore_rejects_deleted_run_instead_of_silently_omitting_it(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", runs_root)
    db = tmp_path / "source/mlflow.db"
    tracked = sync_run(runs_root / manifest.run_id, db)
    MlflowClient(tracking_uri=tracking_uri(db)).delete_run(tracked.mlflow_run_id)
    backup_dir = tmp_path / "backup"
    backup_tracking(db, backup_dir)
    target = tmp_path / "relocated"
    with pytest.raises(BackupError, match="deleted run"):
        restore_tracking_backup(backup_dir, target)
    assert not target.exists()
