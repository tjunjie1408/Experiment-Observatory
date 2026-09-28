"""Immutable Parquet batch export and rebuildable DuckDB read-only catalog."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from observatory.analytics.records import AnalysisRun, load_analysis_run
from observatory.analytics.tracking import inspect_tracking
from observatory.runtime.tree_schema import TreeRunManifest


class WarehouseError(ValueError):
    """A source or published batch cannot be trusted for analysis."""


TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "datasets": pa.schema(
        [
            ("dataset_id", pa.string()),
            ("source", pa.string()),
            ("title", pa.string()),
            ("source_page", pa.string()),
            ("license", pa.string()),
            ("citation", pa.string()),
        ]
    ),
    "dataset_versions": pa.schema(
        [
            ("dataset_key", pa.string()),
            ("dataset_id", pa.string()),
            ("dataset_version", pa.string()),
            ("dataset_identity", pa.string()),
            ("source_url", pa.string()),
            ("source_status", pa.string()),
        ]
    ),
    "experiments": pa.schema([("experiment_id", pa.string())]),
    "runs": pa.schema(
        [
            ("run_id", pa.string()),
            ("experiment_id", pa.string()),
            ("dataset_key", pa.string()),
            ("split_identity", pa.string()),
            ("schema_version", pa.int32()),
            ("created_at", pa.string()),
            ("status", pa.string()),
            ("stop_reason", pa.string()),
            ("manifest_sha256", pa.string()),
            ("mlflow_run_id", pa.string()),
            ("tracking_state", pa.string()),
        ]
    ),
    "models": pa.schema(
        [
            ("run_id", pa.string()),
            ("algorithm", pa.string()),
            ("config_json", pa.string()),
        ]
    ),
    "metrics": pa.schema(
        [
            ("run_id", pa.string()),
            ("name", pa.string()),
            ("split", pa.string()),
            ("step", pa.int32()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("aggregation", pa.string()),
        ]
    ),
    "artifacts": pa.schema(
        [
            ("run_id", pa.string()),
            ("relative_path", pa.string()),
            ("role", pa.string()),
            ("bytes", pa.int64()),
            ("sha256", pa.string()),
        ]
    ),
    "predictions": pa.schema(
        [
            ("run_id", pa.string()),
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("target", pa.int8()),
            ("predicted_class", pa.int8()),
            ("leaf_id", pa.string()),
        ]
    ),
    "dataset_artifacts": pa.schema(
        [
            ("dataset_key", pa.string()),
            ("relative_path", pa.string()),
            ("role", pa.string()),
            ("bytes", pa.int64()),
            ("sha256", pa.string()),
            ("available", pa.bool_()),
        ]
    ),
}


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    batch_dir: Path
    run_count: int
    metric_count: int


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_key(run: AnalysisRun) -> str:
    raw = f"{run.dataset_id}\0{run.dataset_version}\0{run.dataset_identity}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_fingerprint(
    run: AnalysisRun, tracking_state: str, mlflow_run_id: str | None
) -> dict[str, object]:
    files = {}
    for name in ("manifest.json", "events.jsonl", "snapshots.json", "tracking.json"):
        path = run.run_dir / name
        if path.is_file():
            files[name] = _sha(path)
    return {
        "runId": run.run_id,
        "files": files,
        "trackingState": tracking_state,
        "mlflowRunId": mlflow_run_id,
    }


def _dataset_metadata(
    runs: list[AnalysisRun], dataset_root: Path | None
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    folders: dict[str, Path] = {}
    if dataset_root is not None and dataset_root.is_dir():
        for folder in sorted(dataset_root.iterdir()):
            meta_path = folder / "dataset.yaml"
            if folder.is_dir() and meta_path.is_file():
                source = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                if isinstance(source, dict) and isinstance(source.get("dataset_id"), str):
                    folders[source["dataset_id"]] = folder
    for run in runs:
        key = _dataset_key(run)
        if key in result:
            continue
        if run.dataset_version.startswith("synthetic-"):
            result[key] = {"status": "synthetic", "artifacts": []}
            continue
        dataset_folder = folders.get(run.dataset_id)
        if dataset_folder is None:
            result[key] = {"status": "manifest_missing", "artifacts": []}
            continue
        dataset_raw = yaml.safe_load((dataset_folder / "dataset.yaml").read_text(encoding="utf-8"))
        version_file = dataset_folder / "versions" / f"{run.dataset_version}.yaml"
        if not version_file.is_file():
            result[key] = {"status": "manifest_missing", "artifacts": []}
            continue
        if _sha(version_file) != run.dataset_identity:
            raise WarehouseError(f"dataset version manifest hash mismatch: {version_file}")
        version_raw = yaml.safe_load(version_file.read_text(encoding="utf-8"))
        if not isinstance(dataset_raw, dict) or not isinstance(version_raw, dict):
            raise WarehouseError("dataset/version manifest must be a mapping")
        if (
            version_raw.get("dataset_id") != run.dataset_id
            or str(version_raw.get("version")) != run.dataset_version
        ):
            raise WarehouseError("dataset/version manifest identity mismatch")
        artifacts: list[dict[str, object]] = []
        raw_artifacts = version_raw.get("artifacts", [])
        if not isinstance(raw_artifacts, list):
            raise WarehouseError("dataset artifacts must be a list")
        for item in raw_artifacts:
            if not isinstance(item, dict):
                raise WarehouseError("dataset artifact entry must be a mapping")
            relative = item.get("path")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise WarehouseError("dataset artifact path is unsafe")
            path = dataset_folder / relative
            available = path.is_file()
            try:
                expected_bytes = int(item["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WarehouseError(f"dataset artifact size is invalid: {relative}") from exc
            if expected_bytes < 0:
                raise WarehouseError(f"dataset artifact size is invalid: {relative}")
            if available and (
                path.stat().st_size != expected_bytes or _sha(path) != item.get("sha256")
            ):
                raise WarehouseError(f"dataset artifact hash mismatch or size mismatch: {path}")
            artifacts.append(
                {
                    "relative_path": relative,
                    "role": str(item.get("role", "unknown")),
                    "bytes": expected_bytes,
                    "sha256": str(item.get("sha256", "")),
                    "available": available,
                }
            )
        license_raw = dataset_raw.get("license")
        result[key] = {
            "status": "verified_manifest",
            "title": str(dataset_raw.get("title", run.dataset_id)),
            "source_page": str(dataset_raw.get("authoritative_page", "")),
            "license": str(license_raw.get("name", "unknown"))
            if isinstance(license_raw, dict)
            else "unknown",
            "citation": str(dataset_raw.get("citation", "")),
            "source_url": str(version_raw.get("source_url", "")),
            "artifacts": artifacts,
        }
    return result


def _tables(
    runs: list[AnalysisRun],
    tracking: dict[str, tuple[str, str | None]],
    metadata: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    tables: dict[str, list[dict[str, object]]] = {name: [] for name in TABLE_SCHEMAS}
    datasets: dict[str, dict[str, object]] = {}
    versions: dict[str, dict[str, object]] = {}
    experiments: set[str] = set()
    for run in runs:
        dataset_key = _dataset_key(run)
        source_meta = metadata[dataset_key]
        datasets[run.dataset_id] = {
            "dataset_id": run.dataset_id,
            "source": "external_dataset"
            if run.dataset_version.startswith("synthetic-") is False
            else "synthetic",
            "title": source_meta.get("title", run.dataset_id),
            "source_page": source_meta.get("source_page", ""),
            "license": source_meta.get("license", ""),
            "citation": source_meta.get("citation", ""),
        }
        if dataset_key not in versions:
            for artifact in cast(list[dict[str, object]], source_meta["artifacts"]):
                tables["dataset_artifacts"].append({"dataset_key": dataset_key, **artifact})
        versions[dataset_key] = {
            "dataset_key": dataset_key,
            "dataset_id": run.dataset_id,
            "dataset_version": run.dataset_version,
            "dataset_identity": run.dataset_identity,
            "source_url": source_meta.get("source_url", ""),
            "source_status": source_meta["status"],
        }
        experiments.add(run.experiment_id)
        tracking_state, mlflow_id = tracking[run.run_id]
        tables["runs"].append(
            {
                "run_id": run.run_id,
                "experiment_id": run.experiment_id,
                "dataset_key": dataset_key,
                "split_identity": run.split_identity,
                "schema_version": run.manifest.schema_version,
                "created_at": run.manifest.created_at,
                "status": run.manifest.status,
                "stop_reason": run.manifest.stop_reason,
                "manifest_sha256": run.manifest_sha256,
                "mlflow_run_id": mlflow_id,
                "tracking_state": tracking_state,
            }
        )
        tables["models"].append(
            {
                "run_id": run.run_id,
                "algorithm": run.algorithm,
                "config_json": json.dumps(
                    run.manifest.training_config.model_dump(mode="json"), sort_keys=True
                ),
            }
        )
        for metric in run.metrics:
            tables["metrics"].append(
                {
                    "run_id": run.run_id,
                    "name": metric.name,
                    "split": metric.split,
                    "step": metric.step,
                    "value": metric.value,
                    "unit": metric.unit,
                    "aggregation": metric.aggregation,
                }
            )
        for name in ("manifest.json", "events.jsonl", "snapshots.json"):
            path = run.run_dir / name
            if path.is_file():
                tables["artifacts"].append(
                    {
                        "run_id": run.run_id,
                        "relative_path": f"{run.run_id}/{name}",
                        "role": name.removesuffix(".json").removesuffix(".jsonl"),
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                )
        if isinstance(run.manifest, TreeRunManifest):
            roster = {row.sample_id: row for row in run.manifest.dataset.roster}
            for prediction in run.manifest.predictions:
                row = roster[prediction.sample_id]
                tables["predictions"].append(
                    {
                        "run_id": run.run_id,
                        "sample_id": prediction.sample_id,
                        "split": row.split,
                        "target": row.target,
                        "predicted_class": prediction.predicted_class,
                        "leaf_id": prediction.leaf_id,
                    }
                )
    tables["datasets"] = [datasets[key] for key in sorted(datasets)]
    tables["dataset_versions"] = [versions[key] for key in sorted(versions)]
    tables["experiments"] = [{"experiment_id": key} for key in sorted(experiments)]
    return tables


def _read_manifest(batch_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
        if manifest.get("schemaVersion") != 1 or set(manifest["files"]) != set(TABLE_SCHEMAS):
            raise WarehouseError("batch manifest has an unsupported file contract")
        for filename, digest in manifest["files"].items():
            if filename not in TABLE_SCHEMAS or _sha(batch_dir / f"{filename}.parquet") != digest:
                raise WarehouseError(f"batch artifact hash mismatch: {filename}")
        return cast(dict[str, Any], manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, WarehouseError):
            raise
        raise WarehouseError(f"invalid batch {batch_dir}: {exc}") from exc


def build_batch(
    run_dirs: list[Path],
    warehouse_root: Path,
    tracking_database: Path | None = None,
    dataset_root: Path | None = None,
) -> BatchResult:
    """Validate all inputs before publishing one immutable, explicit Parquet batch."""
    if not run_dirs:
        raise WarehouseError("at least one run is required")
    runs = [load_analysis_run(path) for path in run_dirs]
    runs.sort(key=lambda run: run.run_id)
    if len({run.run_id for run in runs}) != len(runs):
        raise WarehouseError("duplicate run ID in batch inputs")
    tracking = {run.run_id: inspect_tracking(run, tracking_database) for run in runs}
    source = [_source_fingerprint(run, *tracking[run.run_id]) for run in runs]
    metadata = _dataset_metadata(runs, dataset_root)
    source_bytes = json.dumps(
        {"runs": source, "datasets": metadata}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    batch_id = hashlib.sha256(source_bytes).hexdigest()[:20]
    batch_dir = warehouse_root / "batches" / batch_id
    if batch_dir.exists():
        manifest = _read_manifest(batch_dir)
        if manifest.get("sources") != source or manifest.get("datasets") != metadata:
            raise WarehouseError("existing batch ID has different inputs")
        return BatchResult(batch_id, batch_dir, len(runs), sum(len(run.metrics) for run in runs))
    batch_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{batch_id}-", dir=batch_dir.parent))
    try:
        tables = _tables(runs, tracking, metadata)
        files: dict[str, str] = {}
        for name, schema in TABLE_SCHEMAS.items():
            path = staging / f"{name}.parquet"
            pq.write_table(
                pa.Table.from_pylist(tables[name], schema=schema), path, compression="zstd"
            )
            files[name] = _sha(path)
        manifest = {
            "schemaVersion": 1,
            "batchId": batch_id,
            "sources": source,
            "datasets": metadata,
            "files": files,
        }
        (staging / "batch.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        _read_manifest(staging)
        staging.replace(batch_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return BatchResult(batch_id, batch_dir, len(runs), len(tables["metrics"]))


def rebuild_catalog(batch_dir: Path, database_path: Path) -> None:
    """Replace a catalog only after a fully validated batch and new DB are ready."""
    manifest = _read_manifest(batch_dir)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = database_path.with_name(f".{database_path.name}.{manifest['batchId']}.tmp")
    if temp_path.exists():
        raise WarehouseError(f"stale temporary catalog exists: {temp_path}")
    try:
        connection = duckdb.connect(str(temp_path))
        try:
            for name in TABLE_SCHEMAS:
                parquet_path = (
                    (batch_dir / f"{name}.parquet").resolve().as_posix().replace("'", "''")
                )
                connection.execute(
                    f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{parquet_path}')"
                )
            connection.execute("CREATE TABLE catalog_meta (batch_id VARCHAR)")
            connection.execute("INSERT INTO catalog_meta VALUES (?)", [manifest["batchId"]])
            row = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
            if row is None:
                raise WarehouseError("catalog count query returned no row")
            count = row[0]
            if count != len(manifest["sources"]):
                raise WarehouseError("catalog row count does not match batch sources")
        finally:
            connection.close()
        temp_path.replace(database_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def browser_catalog(batch_dir: Path) -> dict[str, object]:
    """Portable read-only catalog payload for the Svelte UI."""
    manifest = _read_manifest(batch_dir)

    def rows(name: str) -> list[dict[str, object]]:
        return cast(
            list[dict[str, object]], pq.read_table(batch_dir / f"{name}.parquet").to_pylist()
        )

    return {
        "schemaVersion": 1,
        "batchId": manifest["batchId"],
        "datasets": rows("datasets"),
        "datasetVersions": rows("dataset_versions"),
        "datasetArtifacts": rows("dataset_artifacts"),
        "runs": rows("runs"),
        "metrics": rows("metrics"),
        "models": rows("models"),
        "artifacts": rows("artifacts"),
    }


def export_browser_catalog(batch_dir: Path, target: Path) -> None:
    """Replace a static catalog only after the complete batch has been validated."""
    payload = browser_catalog(batch_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            staging = Path(stream.name)
            stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        staging.replace(target)
    finally:
        if staging is not None and staging.exists():
            staging.unlink()
