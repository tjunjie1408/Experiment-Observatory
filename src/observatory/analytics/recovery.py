"""Consistent local MLflow backup; remote DVC recovery is a separate gate."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from mlflow.entities import Metric, Param, ViewType
from mlflow.tracking import MlflowClient

from observatory.analytics.tracking import tracking_uri


class BackupError(ValueError):
    """Tracking backup cannot be created or verified safely."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink():
        raise BackupError("artifact store root is a symlink")
    files = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise BackupError("artifact store contains a symlink; backup refuses traversal")
    return [path for path in files if path.is_file()]


def backup_tracking(database_path: Path, target_dir: Path) -> dict[str, Any]:
    """Publish a new consistent SQLite snapshot and verified artifact copy."""
    if not database_path.is_file():
        raise BackupError(f"MLflow SQLite database not found: {database_path}")
    if target_dir.exists():
        raise BackupError(f"backup target already exists: {target_dir}")
    artifact_root = database_path.parent / "mlflow-artifacts"
    source_files = _artifact_files(artifact_root)
    before = {path.relative_to(artifact_root).as_posix(): _file_hash(path) for path in source_files}
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tracking-backup-", dir=target_dir.parent))
    try:
        source_db = sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            snapshot_db = sqlite3.connect(staging / "mlflow.db")
            try:
                source_db.backup(snapshot_db)
            finally:
                snapshot_db.close()
        finally:
            source_db.close()
        copied_root = staging / "mlflow-artifacts"
        if artifact_root.exists():
            shutil.copytree(artifact_root, copied_root)
        else:
            copied_root.mkdir()
        after = {
            path.relative_to(artifact_root).as_posix(): _file_hash(path)
            for path in _artifact_files(artifact_root)
        }
        copied = {
            path.relative_to(copied_root).as_posix(): _file_hash(path)
            for path in _artifact_files(copied_root)
        }
        if before != after or before != copied:
            raise BackupError("artifact store changed during backup")
        manifest = {
            "schemaVersion": 1,
            "databaseSha256": _file_hash(staging / "mlflow.db"),
            "artifactSha256": copied,
            "artifactRootUri": artifact_root.resolve().as_uri(),
        }
        (staging / "backup.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        verify_tracking_backup(staging)
        staging.replace(target_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_tracking_backup(backup_dir: Path) -> dict[str, Any]:
    """Validate every file before a restore points MLflow at this directory."""
    try:
        raw = json.loads((backup_dir / "backup.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
            raise BackupError("backup manifest version invalid")
        expected = raw["artifactSha256"]
        if not isinstance(expected, dict):
            raise BackupError("backup artifact manifest invalid")
        if _file_hash(backup_dir / "mlflow.db") != raw["databaseSha256"]:
            raise BackupError("MLflow SQLite backup hash mismatch")
        artifacts = backup_dir / "mlflow-artifacts"
        actual = {
            path.relative_to(artifacts).as_posix(): _file_hash(path)
            for path in _artifact_files(artifacts)
        }
        if actual != expected:
            raise BackupError("MLflow artifact backup hash mismatch")
        connection = sqlite3.connect(
            f"file:{(backup_dir / 'mlflow.db').resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            check = connection.execute("PRAGMA integrity_check").fetchone()
            if check != ("ok",):
                raise BackupError("MLflow SQLite backup failed integrity check")
        finally:
            connection.close()
        return raw
    except (OSError, KeyError, TypeError, json.JSONDecodeError, sqlite3.Error) as exc:
        raise BackupError(f"invalid MLflow backup: {exc}") from exc


def restore_tracking_backup(backup_dir: Path, target_dir: Path) -> dict[str, str]:
    """Rehydrate Observatory MLflow runs through public APIs at a new artifact root.

    MLflow assigns new run IDs. The returned old-to-new map is persisted in restore.json;
    recorder run IDs and their manifest tags remain stable.
    """
    manifest = verify_tracking_backup(backup_dir)
    if target_dir.exists():
        raise BackupError(f"restore target already exists: {target_dir}")
    old_root_uri = manifest.get("artifactRootUri")
    if not isinstance(old_root_uri, str) or not old_root_uri.startswith("file:///"):
        raise BackupError("backup lacks a supported source artifact root URI")
    old_client = MlflowClient(tracking_uri=tracking_uri(backup_dir / "mlflow.db"))
    # Preflight the entire source before creating any target database or artifact.
    planned: list[tuple[Any, list[tuple[Any, Path]]]] = []
    migrated_artifacts: set[str] = set()
    experiments = old_client.search_experiments(view_type=ViewType.ALL, max_results=1000)
    if len(experiments) == 1000:
        raise BackupError("too many MLflow experiments for this restore contract")
    for experiment in experiments:
        if experiment.lifecycle_stage != "active":
            raise BackupError(f"deleted experiment {experiment.name} is outside restore scope")
        old_runs = old_client.search_runs(
            [experiment.experiment_id], run_view_type=ViewType.ALL, max_results=1000
        )
        if len(old_runs) == 1000:
            raise BackupError(f"too many runs in experiment {experiment.name}")
        if not old_runs:
            continue
        if experiment.name == "Default":
            raise BackupError("default MLflow experiment is outside Observatory restore scope")
        run_entries: list[tuple[Any, Path]] = []
        for old_run in old_runs:
            old_id = old_run.info.run_id
            if not old_run.data.tags.get("observatory_run_id"):
                raise BackupError(f"run {old_id} is not an Observatory run")
            if old_run.info.lifecycle_stage != "active":
                raise BackupError(f"deleted run {old_id} is outside restore scope")
            if old_run.info.status not in {"FINISHED", "FAILED", "KILLED"}:
                raise BackupError(f"run {old_id} is not terminal")
            prefix = f"{old_root_uri}/"
            old_uri = old_run.info.artifact_uri
            if not old_uri.startswith(prefix):
                raise BackupError(f"run {old_id} has an unsupported artifact location")
            artifact_suffix = old_uri[len(prefix) :]
            parts = Path(artifact_suffix).parts
            if len(parts) < 2 or parts[-2:] != (old_id, "artifacts") or ".." in parts:
                raise BackupError(f"run {old_id} has an unsupported artifact location")
            source_artifacts = backup_dir / "mlflow-artifacts" / artifact_suffix
            for path in _artifact_files(source_artifacts):
                migrated_artifacts.add(path.relative_to(backup_dir / "mlflow-artifacts").as_posix())
            run_entries.append((old_run, source_artifacts))
        planned.append((experiment, run_entries))
    if migrated_artifacts != set(manifest["artifactSha256"]):
        raise BackupError("backup contains artifacts outside the migrated run set")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(exist_ok=False)
    try:
        new_client = MlflowClient(tracking_uri=tracking_uri(target_dir / "mlflow.db"))
        artifact_root = target_dir / "mlflow-artifacts"
        artifact_root.mkdir()
        mapping: dict[str, str] = {}
        for experiment, run_entries in planned:
            new_experiment_id = new_client.create_experiment(
                experiment.name,
                artifact_location=artifact_root.resolve().as_uri(),
                tags=experiment.tags,
            )
            for old_run, source_artifacts in run_entries:
                old_id = old_run.info.run_id
                new_run = new_client.create_run(
                    new_experiment_id,
                    start_time=old_run.info.start_time,
                    tags=old_run.data.tags,
                    run_name=old_run.info.run_name,
                )
                new_id = new_run.info.run_id
                mapping[old_id] = new_id
                params = [Param(key, value) for key, value in old_run.data.params.items()]
                if params:
                    new_client.log_batch(new_id, params=params)
                for key in old_run.data.metrics:
                    history = old_client.get_metric_history(old_id, key)
                    for start in range(0, len(history), 500):
                        new_client.log_batch(
                            new_id,
                            metrics=[
                                Metric(item.key, item.value, item.timestamp, item.step)
                                for item in history[start : start + 500]
                            ],
                        )
                for path in _artifact_files(source_artifacts):
                    relative = path.relative_to(source_artifacts)
                    parent = relative.parent.as_posix()
                    new_client.log_artifact(
                        new_id, str(path), artifact_path=None if parent == "." else parent
                    )
                new_client.set_terminated(
                    new_id, status=old_run.info.status, end_time=old_run.info.end_time
                )
                if new_client.get_run(new_id).data.metrics != old_run.data.metrics:
                    raise BackupError(f"run {old_id} metrics changed during relocation")
                for path in _artifact_files(source_artifacts):
                    relative_uri = path.relative_to(source_artifacts).as_posix()
                    restored = Path(new_client.download_artifacts(new_id, relative_uri))
                    if _file_hash(restored) != _file_hash(path):
                        raise BackupError(f"run {old_id} artifact changed during relocation")
        (target_dir / "restore.json").write_text(
            json.dumps({"schemaVersion": 1, "runIds": mapping}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return mapping
    except Exception as exc:
        try:
            shutil.rmtree(target_dir)
        except OSError as cleanup_error:
            raise BackupError(
                f"restore failed and incomplete target remains at {target_dir}: {cleanup_error}"
            ) from exc
        raise
