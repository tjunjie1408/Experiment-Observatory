"""Normalized, validated source records for tracking and warehouse exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from observatory.runtime.export import ExportError, validate_run_for_export
from observatory.runtime.schema import (
    DataConfig,
    ExternalDataConfig,
    KMeansDataConfig,
    KMeansRunManifest,
    KMeansSnapshot,
    RunManifest,
    Snapshot,
)
from observatory.runtime.tree_schema import TreeDataConfig, TreeRunManifest, TreeSnapshot

Manifest = RunManifest | KMeansRunManifest | TreeRunManifest
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


class AnalysisInputError(ValueError):
    """A run is not safe to use as a tracking or analysis source."""


@dataclass(frozen=True)
class MetricRecord:
    name: str
    split: str
    step: int
    value: float
    unit: str
    aggregation: str


@dataclass(frozen=True)
class AnalysisRun:
    run_dir: Path
    manifest: Manifest
    manifest_sha256: str
    dataset_id: str
    dataset_version: str
    dataset_identity: str
    split_identity: str | None
    algorithm: str
    metrics: tuple[MetricRecord, ...]
    timeline_metrics: tuple[MetricRecord, ...]

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def experiment_id(self) -> str:
        return self.manifest.experiment_id


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_analysis_run(run_dir: Path) -> AnalysisRun:
    """Read a terminal run, fully validating completed bundles before analysis."""
    source = run_dir / "manifest.json"
    try:
        raw_bytes = source.read_bytes()
        raw: Any = json.loads(raw_bytes)
        if not isinstance(raw, dict):
            raise AnalysisInputError("manifest must be a JSON object")
        version = raw.get("schemaVersion")
        model: type[Manifest]
        if version in (1, 2):
            model = RunManifest
        elif version == 3:
            model = KMeansRunManifest
        elif version == 4:
            model = TreeRunManifest
        else:
            raise AnalysisInputError(f"unsupported schema version: {version}")
        manifest = model.model_validate(raw)
        if manifest.status not in TERMINAL:
            raise AnalysisInputError(f"run is not terminal: {manifest.status}")
        if manifest.status == "completed":
            manifest, _, snapshots = validate_run_for_export(run_dir)
        else:
            events_path = run_dir / "events.jsonl"
            events = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            if not events or any(event.get("runId") != manifest.run_id for event in events):
                raise AnalysisInputError("terminal run events are missing or inconsistent")
            snapshots = []
        if run_dir.name != manifest.run_id:
            raise AnalysisInputError("run directory name does not match runId")
    except (OSError, json.JSONDecodeError, ExportError, ValueError) as exc:
        if isinstance(exc, AnalysisInputError):
            raise
        raise AnalysisInputError(f"invalid run {run_dir}: {exc}") from exc

    config = manifest.data_config
    metrics: tuple[MetricRecord, ...]
    algorithm: str
    split_identity: str | None
    dataset_id: str
    dataset_version: str
    if isinstance(manifest, TreeRunManifest):
        assert isinstance(config, TreeDataConfig)
        dataset_id = config.dataset_id
        dataset_version = config.dataset_version
        dataset_identity = config.version_manifest_sha256
        split_identity = config.split_sha256
        algorithm = manifest.training_config.algorithm
        metrics = (
            (
                MetricRecord(
                    "accuracy",
                    "train",
                    manifest.last_valid_step or 0,
                    manifest.train_evaluation.accuracy,
                    "fraction",
                    "correct/total",
                ),
                MetricRecord(
                    "accuracy",
                    "validation",
                    manifest.last_valid_step or 0,
                    manifest.validation_evaluation.accuracy,
                    "fraction",
                    "correct/total",
                ),
            )
            if manifest.status == "completed"
            and manifest.train_evaluation is not None
            and manifest.validation_evaluation is not None
            else ()
        )
        timeline_metrics = tuple(
            MetricRecord("nodes_recorded", "all", item.step, float(item.step + 1), "count", "count")
            for item in snapshots
            if isinstance(item, TreeSnapshot)
        ) + tuple(
            MetricRecord(
                "gini_decrease",
                "all",
                item.step,
                item.weighted_gini_decrease,
                "fraction",
                "weighted_impurity_decrease",
            )
            for item in snapshots
            if isinstance(item, TreeSnapshot) and item.weighted_gini_decrease is not None
        )
    elif isinstance(manifest, KMeansRunManifest):
        assert isinstance(config, KMeansDataConfig)
        dataset_id = config.generator
        dataset_identity = _identity(config.model_dump(mode="json"))
        dataset_version = f"synthetic-{dataset_identity[:12]}"
        split_identity = None
        algorithm = manifest.training_config.algorithm
        metrics = (
            (
                MetricRecord(
                    "inertia",
                    "all",
                    snapshots[-1].step,
                    snapshots[-1].inertia,
                    "squared_distance",
                    "sum",
                ),
            )
            if snapshots
            else ()
        )
        timeline_metrics = tuple(
            MetricRecord("inertia", "all", item.step, item.inertia, "squared_distance", "sum")
            for item in snapshots
            if isinstance(item, KMeansSnapshot)
        )
    else:
        assert isinstance(config, (DataConfig, ExternalDataConfig))
        if isinstance(config, ExternalDataConfig):
            dataset_id = config.dataset_id
            dataset_version = config.dataset_version
            dataset_identity = config.version_manifest_sha256
        else:
            dataset_id = config.generator
            dataset_identity = _identity(config.model_dump(mode="json"))
            dataset_version = f"synthetic-{dataset_identity[:12]}"
        split_identity = None
        algorithm = manifest.training_config.algorithm
        metrics = (
            (
                MetricRecord(
                    "mse",
                    "train",
                    snapshots[-1].step,
                    snapshots[-1].train_mse,
                    "target_squared",
                    "mean",
                ),
            )
            if snapshots
            else ()
        )
        timeline_metrics = tuple(
            MetricRecord("mse", "train", item.step, item.train_mse, "target_squared", "mean")
            for item in snapshots
            if isinstance(item, Snapshot)
        )
    return AnalysisRun(
        run_dir=run_dir,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_identity=dataset_identity,
        split_identity=split_identity,
        algorithm=algorithm,
        metrics=metrics,
        timeline_metrics=timeline_metrics,
    )
