"""Tests for the static export contract: bundle validation and rejection rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observatory.data.synthetic import SyntheticLinearConfig
from observatory.runtime.export import ExportError, export_run, validate_run_for_export
from observatory.runtime.record import create_run, run_training
from observatory.runtime.schema import ModelConfig


def make_completed_run(tmp_path: Path, *, learning_rate: float = 0.25, n_updates: int = 10) -> Path:
    data_cfg = SyntheticLinearConfig(
        n_samples=20, true_bias=1.0, true_weight=2.0, noise_std=0.3, seed=42
    )
    model_cfg = ModelConfig(
        algorithm="linear_regression_gradient_descent",
        initial_bias=0.0,
        initial_weight=0.0,
        learning_rate=learning_rate,
        n_updates=n_updates,
    )
    recorder = create_run(
        experiment_id="export-test",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path / "runs",
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)
    return tmp_path / "runs" / manifest.run_id


def make_failed_run(tmp_path: Path) -> Path:
    data_cfg = SyntheticLinearConfig(
        n_samples=20, true_bias=1.0, true_weight=2.0, noise_std=0.3, seed=42
    )
    model_cfg = ModelConfig(
        algorithm="linear_regression_gradient_descent",
        initial_bias=0.0,
        initial_weight=0.0,
        learning_rate=1e250,
        n_updates=5,
    )
    recorder = create_run(
        experiment_id="export-fail-test",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path / "runs",
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)
    return tmp_path / "runs" / manifest.run_id


def test_export_completed_run_produces_self_contained_bundle(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    target_dir = tmp_path / "export" / "bundle"

    manifest = export_run(run_dir, target_dir)

    assert manifest.status == "completed"
    assert (target_dir / "manifest.json").is_file()
    assert (target_dir / "events.jsonl").is_file()
    assert (target_dir / "snapshots.json").is_file()

    # Source run is untouched.
    original_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert original_manifest["runId"] == manifest.run_id


def test_export_rejects_existing_target_without_overwrite(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    target_dir = tmp_path / "export" / "bundle"
    target_dir.mkdir(parents=True)
    (target_dir / "sentinel.txt").write_text("do not touch", encoding="utf-8")

    with pytest.raises(ExportError):
        export_run(run_dir, target_dir)

    # Existing target contents are untouched.
    assert (target_dir / "sentinel.txt").read_text(encoding="utf-8") == "do not touch"
    assert not (target_dir / "manifest.json").exists()


def test_export_rejects_non_completed_run(tmp_path: Path) -> None:
    failed_run_dir = make_failed_run(tmp_path)
    target_dir = tmp_path / "export" / "bundle"

    with pytest.raises(ExportError, match="not completed"):
        export_run(failed_run_dir, target_dir)

    assert not target_dir.exists()


def test_export_rejects_missing_run_directory(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        export_run(tmp_path / "does-not-exist", tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_corrupt_manifest_json(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    (run_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ExportError, match="not valid JSON"):
        export_run(run_dir, tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schemaVersion"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportError, match="schemaVersion"):
        export_run(run_dir, tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_schema_version_data_config_mismatch(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schemaVersion"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportError, match="schemaVersion 2 requires external"):
        export_run(run_dir, tmp_path / "export" / "bundle")


def test_export_rejects_event_version_different_from_manifest(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["schemaVersion"] = 2
    lines[0] = json.dumps(event)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ExportError, match="does not match manifest schemaVersion"):
        export_run(run_dir, tmp_path / "export" / "bundle")


def test_export_rejects_non_finite_value_in_snapshot(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    snapshots_path = run_dir / "snapshots.json"
    snapshots = json.loads(snapshots_path.read_text())
    snapshots[0]["trainMse"] = float("nan")
    # allow_nan is required here only because we are hand-corrupting the
    # fixture to simulate a bad file; the recorder itself never writes NaN.
    snapshots_path.write_text(json.dumps(snapshots, allow_nan=True), encoding="utf-8")

    with pytest.raises(ExportError):
        export_run(run_dir, tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_snapshot_count_mismatch(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    snapshots_path = run_dir / "snapshots.json"
    snapshots = json.loads(snapshots_path.read_text())
    snapshots_path.write_text(json.dumps(snapshots[:-1]), encoding="utf-8")

    with pytest.raises(ExportError, match="snapshots"):
        export_run(run_dir, tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_event_from_different_run(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["runId"] = "some-other-run"
    lines[0] = json.dumps(tampered)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ExportError, match="different run"):
        export_run(run_dir, tmp_path / "export" / "bundle")

    assert not (tmp_path / "export" / "bundle").exists()


def test_export_rejects_missing_events_file(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    (run_dir / "events.jsonl").unlink()

    with pytest.raises(ExportError, match=r"events\.jsonl"):
        export_run(run_dir, tmp_path / "export" / "bundle")


def test_validate_run_for_export_does_not_write_anything(tmp_path: Path) -> None:
    run_dir = make_completed_run(tmp_path)
    before = sorted(run_dir.iterdir())

    manifest, events, snapshots = validate_run_for_export(run_dir)

    after = sorted(run_dir.iterdir())
    assert before == after
    assert manifest.status == "completed"
    assert len(snapshots) > 0
    assert len(events) > 0
