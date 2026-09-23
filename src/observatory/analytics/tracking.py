"""Idempotent MLflow projection of terminal Observatory file runs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from observatory.analytics.records import AnalysisRun, load_analysis_run
from observatory.runtime.storage import atomic_write_json

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

_STATUS = {
    "completed": "FINISHED",
    "failed": "FAILED",
    "interrupted": "FAILED",
    "cancelled": "KILLED",
}


class TrackingSyncError(ValueError):
    """Tracking state could not be reconciled with the recorder source."""


class TrackingHistoryNotFoundError(ValueError):
    """The run has no verified MLflow link."""


class TrackingHistoryUnavailableError(RuntimeError):
    """The configured local MLflow store cannot be read."""


@dataclass(frozen=True)
class SyncResult:
    run_id: str
    mlflow_run_id: str
    created: bool
    metric_count: int


def tracking_uri(database_path: Path) -> str:
    """Explicit SQLite location; no global MLflow tracking state is modified."""
    return f"sqlite:///{database_path.resolve().as_posix()}"


def _params(run: AnalysisRun) -> dict[str, str]:
    params = {
        "observatory_run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "dataset_version": run.dataset_version,
        "dataset_identity": run.dataset_identity,
        "algorithm": run.algorithm,
        "schema_version": str(run.manifest.schema_version),
    }
    if run.split_identity is not None:
        params["split_identity"] = run.split_identity
    training = run.manifest.training_config.model_dump(mode="json")
    params.update({f"training.{key}": str(value) for key, value in training.items()})
    return params


def _metric_series(run: AnalysisRun) -> dict[str, dict[int, float]]:
    """Project recorded states onto MLflow's keyed step series without inventing steps."""
    series: dict[str, dict[int, float]] = {}
    for item in (*run.timeline_metrics, *run.metrics):
        key = f"{item.name}.{item.split}"
        by_step = series.setdefault(key, {})
        old = by_step.get(item.step)
        if old is not None and old != item.value:
            raise TrackingSyncError(f"recorder metric conflict: {key} step {item.step}")
        by_step[item.step] = item.value
    return series


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync_replay_artifacts(client: MlflowClient, mlflow_id: str, run_dir: Path) -> None:
    """Keep the exact recorder bundle downloadable alongside metric curves."""
    published = {
        item.path.split("/")[-1]: item
        for item in client.list_artifacts(mlflow_id, "observatory-replay")
    }
    with tempfile.TemporaryDirectory(prefix="observatory-tracking-") as staging:
        for name in ("manifest.json", "events.jsonl", "snapshots.json"):
            source = run_dir / name
            if not source.is_file():
                continue
            if name in published:
                downloaded = Path(
                    client.download_artifacts(mlflow_id, f"observatory-replay/{name}", staging)
                )
                if _file_sha(downloaded) != _file_sha(source):
                    raise TrackingSyncError(f"MLflow replay artifact conflict: {name}")
            else:
                client.log_artifact(mlflow_id, str(source), artifact_path="observatory-replay")


def sync_run(run_dir: Path, database_path: Path) -> SyncResult:
    """Sync a terminal run; retries find the MLflow run by immutable source tag."""
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient

    run = load_analysis_run(run_dir)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_uri(database_path))
    experiment = client.get_experiment_by_name(run.experiment_id)
    if experiment is None:
        artifact_root = (database_path.parent / "mlflow-artifacts").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            run.experiment_id, artifact_location=artifact_root.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.observatory_run_id = '{run.run_id}'",
        max_results=2,
    )
    if len(matches) > 1:
        raise TrackingSyncError("multiple MLflow runs claim one Observatory run ID")
    created = not matches
    if created:
        tracked = client.create_run(
            experiment_id,
            tags={
                "observatory_run_id": run.run_id,
                "manifest_sha256": run.manifest_sha256,
                "recorder_status": run.manifest.status,
                "stop_reason": run.manifest.stop_reason or "",
            },
            run_name=run.run_id,
        )
    else:
        tracked = matches[0]
        if tracked.data.tags.get("manifest_sha256") != run.manifest_sha256:
            raise TrackingSyncError("source manifest changed for an already tracked run")
    mlflow_run_id = tracked.info.run_id
    expected_params = _params(run)
    for key, value in expected_params.items():
        existing = tracked.data.params.get(key)
        if existing is not None and existing != value:
            raise TrackingSyncError(f"MLflow parameter conflict: {key}")
    missing_params = [
        Param(key, value)
        for key, value in expected_params.items()
        if key not in tracked.data.params
    ]
    missing_metrics: list[Metric] = []
    for key, expected in _metric_series(run).items():
        history = client.get_metric_history(mlflow_run_id, key)
        actual = {item.step: item.value for item in history}
        if len(actual) != len(history) or any(
            step not in expected or expected[step] != value for step, value in actual.items()
        ):
            raise TrackingSyncError(f"MLflow metric conflict or duplicate history: {key}")
        missing_metrics.extend(
            Metric(key, value, 0, step)
            for step, value in sorted(expected.items())
            if step not in actual
        )
    metric_definitions = {
        f"{item.name}.{item.split}": item for item in (*run.timeline_metrics, *run.metrics)
    }
    tags = [
        RunTag(f"metric.{item.name}.{item.split}.unit", item.unit)
        for item in metric_definitions.values()
    ] + [
        RunTag(f"metric.{item.name}.{item.split}.aggregation", item.aggregation)
        for item in metric_definitions.values()
    ]
    if missing_params or tags:
        client.log_batch(mlflow_run_id, params=missing_params, tags=tags)
    for start in range(0, len(missing_metrics), 500):
        client.log_batch(mlflow_run_id, metrics=missing_metrics[start : start + 500])
    _sync_replay_artifacts(client, mlflow_run_id, run_dir)
    target_status = _STATUS[run.manifest.status]
    if tracked.info.status != target_status:
        client.set_terminated(mlflow_run_id, status=target_status)
    atomic_write_json(
        run_dir / "tracking.json",
        {
            "schemaVersion": 1,
            "runId": run.run_id,
            "manifestSha256": run.manifest_sha256,
            "mlflowRunId": mlflow_run_id,
            "status": "synced",
        },
    )
    return SyncResult(
        run.run_id, mlflow_run_id, created, sum(map(len, _metric_series(run).values()))
    )


def read_tracking_link(run_dir: Path, manifest_sha256: str) -> str | None:
    """A stale or invalid sidecar never changes the recorder's source truth."""
    try:
        raw = json.loads((run_dir / "tracking.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("manifestSha256") != manifest_sha256
        or raw.get("status") != "synced"
    ):
        return None
    value = raw.get("mlflowRunId")
    return value if isinstance(value, str) and value else None


def inspect_tracking(run: AnalysisRun, database_path: Path | None) -> tuple[str, str | None]:
    """Use MLflow's public API to distinguish a recorded link from verified state."""
    link = read_tracking_link(run.run_dir, run.manifest_sha256)
    if link is None:
        return "not_synced", None
    if database_path is None:
        return "link_recorded", link
    if not database_path.is_file():
        return "unavailable", link
    from mlflow.tracking import MlflowClient

    try:
        tracked = MlflowClient(tracking_uri=tracking_uri(database_path)).get_run(link)
    except Exception:
        return "unavailable", link
    if (
        tracked.data.tags.get("observatory_run_id") != run.run_id
        or tracked.data.tags.get("manifest_sha256") != run.manifest_sha256
        or tracked.info.status != _STATUS[run.manifest.status]
    ):
        return "conflict", link
    for metric in run.metrics:
        if tracked.data.metrics.get(f"{metric.name}.{metric.split}") != metric.value:
            return "conflict", link
    return "verified", link


def read_tracking_history(run: AnalysisRun, database_path: Path) -> dict[str, object]:
    """Read MLflow step metrics only when they still match the validated recorder."""
    state, mlflow_id = inspect_tracking(run, database_path)
    if state == "not_synced":
        raise TrackingHistoryNotFoundError("run is not synced to MLflow")
    if state in {"unavailable", "link_recorded"} or mlflow_id is None:
        raise TrackingHistoryUnavailableError("local MLflow tracking store is unavailable")
    if state != "verified":
        raise TrackingSyncError("MLflow run conflicts with its recorder source")

    from mlflow.tracking import MlflowClient

    expected = _metric_series(run)
    definitions = {
        f"{item.name}.{item.split}": item for item in (*run.timeline_metrics, *run.metrics)
    }
    client = MlflowClient(tracking_uri=tracking_uri(database_path))
    series: list[dict[str, object]] = []
    for key, steps in sorted(expected.items()):
        try:
            history = client.get_metric_history(mlflow_id, key)
        except Exception as exc:
            raise TrackingHistoryUnavailableError("MLflow metric history is unavailable") from exc
        actual = {item.step: item.value for item in history}
        if len(actual) != len(history) or actual != steps:
            raise TrackingSyncError(f"MLflow metric history conflicts with recorder: {key}")
        definition = definitions[key]
        series.append(
            {
                "key": key,
                "unit": definition.unit,
                "aggregation": definition.aggregation,
                "points": [
                    {"step": step, "value": value} for step, value in sorted(actual.items())
                ],
            }
        )
    return {"runId": run.run_id, "source": "mlflow", "state": "verified", "series": series}
