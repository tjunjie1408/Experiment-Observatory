"""Parquet batch and DuckDB rebuild checks using an actual tree runtime run."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from observatory.analytics.warehouse import (
    WarehouseError,
    browser_catalog,
    build_batch,
    export_browser_catalog,
    rebuild_catalog,
)
from observatory.api.app import create_app
from observatory.cli import run_one
from observatory.experiments.tree.record import run_tree_experiment

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / "datasets/breast-cancer/versions/1.0.0.yaml"


def test_batch_is_idempotent_and_catalog_rebuilds_elsewhere(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = run_tree_experiment(VERSION, max_depth=2, runs_root=runs_root, repo_root=ROOT)
    source = runs_root / run.run_id
    warehouse = tmp_path / "warehouse"
    first = build_batch([source], warehouse, dataset_root=ROOT / "datasets")
    second = build_batch([source], warehouse, dataset_root=ROOT / "datasets")
    assert first == second
    assert first.run_count == 1 and first.metric_count == 2

    db = tmp_path / "database/observatory.duckdb"
    rebuild_catalog(first.batch_dir, db)
    with duckdb.connect(str(db), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone() == (569,)
        assert (
            con.execute(
                "SELECT value FROM metrics WHERE name='accuracy' AND split='validation'"
            ).fetchone()[0]
            == run.validation_evaluation.accuracy
        )
        assert con.execute("SELECT batch_id FROM catalog_meta").fetchone() == (first.batch_id,)

    moved = tmp_path / "restored/batch"
    shutil.copytree(first.batch_dir, moved)
    other_db = tmp_path / "restored/catalog.duckdb"
    rebuild_catalog(moved, other_db)
    with duckdb.connect(str(other_db), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone() == (569,)
    catalog = browser_catalog(moved)
    assert catalog["batchId"] == first.batch_id
    assert len(catalog["runs"]) == 1
    assert catalog["datasets"][0]["title"] == "Breast Cancer Wisconsin (Diagnostic)"
    assert catalog["datasetVersions"][0]["source_status"] == "verified_manifest"
    assert len(catalog["datasetArtifacts"]) == 4
    assert all(row["available"] for row in catalog["datasetArtifacts"])
    static_path = tmp_path / "public/catalog.json"
    export_browser_catalog(moved, static_path)
    first_static = static_path.read_bytes()
    export_browser_catalog(moved, static_path)
    assert static_path.read_bytes() == first_static
    with TestClient(
        create_app(runs_root=runs_root, repo_root=ROOT, catalog_database=other_db)
    ) as client:
        response = client.get("/api/catalog")
        assert response.status_code == 200
        assert response.json()["batchId"] == first.batch_id
        assert len(response.json()["artifacts"]) == 3
        assert len(response.json()["datasetArtifacts"]) == 4


def test_missing_raw_data_is_disclosed_and_present_bad_hash_is_rejected(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = run_tree_experiment(VERSION, max_depth=1, runs_root=runs_root, repo_root=ROOT)
    source = runs_root / run.run_id
    dataset_folder = tmp_path / "datasets/breast-cancer"
    (dataset_folder / "versions").mkdir(parents=True)
    shutil.copy2(ROOT / "datasets/breast-cancer/dataset.yaml", dataset_folder / "dataset.yaml")
    shutil.copy2(VERSION, dataset_folder / "versions/1.0.0.yaml")
    batch = build_batch([source], tmp_path / "warehouse", dataset_root=tmp_path / "datasets")
    assert all(not row["available"] for row in browser_catalog(batch.batch_dir)["datasetArtifacts"])
    (dataset_folder / "raw").mkdir()
    (dataset_folder / "raw/wdbc.data").write_text("wrong", encoding="utf-8")
    with pytest.raises(WarehouseError, match="hash mismatch"):
        build_batch([source], tmp_path / "warehouse", dataset_root=tmp_path / "datasets")


def test_missing_catalog_is_explicitly_unavailable(tmp_path: Path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path / "runs",
            repo_root=tmp_path,
            catalog_database=tmp_path / "missing.duckdb",
        )
    ) as client:
        assert client.get("/api/catalog").status_code == 503


def test_local_replay_serves_only_valid_completed_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    manifest = run_one(ROOT / "configs/linear/slow.yaml", runs_root)
    run_dir = runs_root / manifest.run_id
    with TestClient(create_app(runs_root=runs_root, repo_root=ROOT)) as client:
        base = f"/api/replay/{manifest.run_id}"
        assert client.get(f"{base}/manifest.json").json()["runId"] == manifest.run_id
        assert client.get(f"{base}/snapshots.json").json()[-1]["step"] == manifest.last_valid_step
        assert "run.completed" in client.get(f"{base}/events.jsonl").text
        assert client.get(f"{base}/tracking.json").status_code == 404
        assert client.get("/api/replay/not-a-run/manifest.json").status_code == 404
        snapshots = run_dir / "snapshots.json"
        snapshots.write_text("[]", encoding="utf-8")
        assert client.get(f"{base}/snapshots.json").status_code == 409


def test_corrupted_batch_cannot_replace_catalog(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = run_tree_experiment(VERSION, max_depth=1, runs_root=runs_root, repo_root=ROOT)
    batch = build_batch([runs_root / run.run_id], tmp_path / "warehouse")
    db = tmp_path / "catalog.duckdb"
    rebuild_catalog(batch.batch_dir, db)
    original = db.read_bytes()
    metric_file = batch.batch_dir / "metrics.parquet"
    metric_file.write_bytes(metric_file.read_bytes() + b"tampered")
    with pytest.raises(WarehouseError, match="hash mismatch"):
        rebuild_catalog(batch.batch_dir, db)
    assert db.read_bytes() == original
    static_path = tmp_path / "public/catalog.json"
    static_path.parent.mkdir()
    static_path.write_text("old catalog", encoding="utf-8")
    with pytest.raises(WarehouseError, match="hash mismatch"):
        export_browser_catalog(batch.batch_dir, static_path)
    assert static_path.read_text(encoding="utf-8") == "old catalog"
