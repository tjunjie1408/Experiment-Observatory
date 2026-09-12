"""Static export: package a completed run into a self-contained bundle a
static frontend can load without a Python backend.

- Only a completed, internally-consistent run is exported; the source run
  is never modified.
- An incomplete/failed/corrupt run, or an unsupported schemaVersion, is
  rejected before any output is written; a failed export never leaves a
  partial package on disk: the bundle is built in a temp directory and
  only renamed into the target once fully validated and written.
- The bundle is validated the same way a frontend consumer would: version,
  types, finite values, and cross-references between manifest/events/
  snapshots must agree. This is the Python-side half of the
  cross-language contract; web/src/lib/schema.ts implements the TS half.
- Exporting to an existing target directory is rejected; the caller must
  choose a new target rather than overwrite.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from observatory.runtime.schema import (
    EXTERNAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    DataConfig,
    Event,
    ExternalDataConfig,
    RunManifest,
    Snapshot,
)


class ExportError(ValueError):
    """Raised when a run cannot be exported as-is; no output is written."""


def _read_json(path: Path, *, label: str) -> object:
    if not path.is_file():
        raise ExportError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"{label} is not valid JSON: {exc}") from exc


def _read_events_jsonl(path: Path) -> list[Event]:
    if not path.is_file():
        raise ExportError(f"events.jsonl not found: {path}")
    events: list[Event] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExportError(f"events.jsonl could not be read: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportError(f"events.jsonl line {line_number} is not valid JSON: {exc}") from exc
        try:
            events.append(Event.model_validate(raw))
        except Exception as exc:  # pydantic ValidationError, kept broad for a single clear message
            raise ExportError(f"events.jsonl line {line_number} failed validation: {exc}") from exc
    return events


def _assert_snapshot_values_finite(snapshot: Snapshot) -> None:
    """Check finiteness directly on the parsed model's numeric fields.

    We check the Python floats themselves rather than round-tripping
    through model_dump_json(): pydantic serializes non-finite floats as
    JSON `null` by default, which would otherwise silently defeat a
    finiteness check performed after re-parsing the JSON.
    """
    numeric_fields = {
        "b": snapshot.b,
        "w": snapshot.w,
        "gradientB": snapshot.gradient_b,
        "gradientW": snapshot.gradient_w,
        "trainMse": snapshot.train_mse,
    }
    for field_name, value in numeric_fields.items():
        if not math.isfinite(value):
            raise ExportError(
                f"non-finite value at snapshot step {snapshot.step}.{field_name}: {value!r}"
            )
    for sample_id, prediction in snapshot.observed_predictions.items():
        if not math.isfinite(prediction):
            raise ExportError(
                f"non-finite value at snapshot step {snapshot.step}."
                f"observedPredictions[{sample_id!r}]: {prediction!r}"
            )


def _assert_manifest_values_finite(manifest: RunManifest) -> None:
    data = manifest.data_config
    if isinstance(data, DataConfig):
        for field_name, value in {
            "trueBias": data.true_bias,
            "trueWeight": data.true_weight,
            "noiseStd": data.noise_std,
        }.items():
            if not math.isfinite(value):
                raise ExportError(
                    f"non-finite value at manifest.dataConfig.{field_name}: {value!r}"
                )

    for index, value in enumerate(manifest.dataset.x):
        if not math.isfinite(value):
            raise ExportError(f"non-finite value at manifest.dataset.x[{index}]: {value!r}")
    for index, value in enumerate(manifest.dataset.y):
        if not math.isfinite(value):
            raise ExportError(f"non-finite value at manifest.dataset.y[{index}]: {value!r}")

    training = manifest.training_config
    for field_name, value in {
        "initialBias": training.initial_bias,
        "initialWeight": training.initial_weight,
        "learningRate": training.learning_rate,
    }.items():
        if not math.isfinite(value):
            raise ExportError(
                f"non-finite value at manifest.trainingConfig.{field_name}: {value!r}"
            )


def validate_run_for_export(run_dir: Path) -> tuple[RunManifest, list[Event], list[Snapshot]]:
    """Load and cross-validate a run's files without writing anything.

    Raises ExportError with a specific reason for any of: missing files,
    corrupt JSON, unsupported schemaVersion, incomplete/failed status,
    non-finite recorded values, or inconsistent cross-references between
    manifest/events/snapshots.
    """
    manifest_raw = _read_json(run_dir / "manifest.json", label="manifest.json")
    try:
        manifest = RunManifest.model_validate(manifest_raw)
    except Exception as exc:
        raise ExportError(f"manifest.json failed validation: {exc}") from exc

    if manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(version) for version in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ExportError(
            f"unsupported schemaVersion {manifest.schema_version}; expected one of {supported}"
        )
    if manifest.schema_version == SCHEMA_VERSION and not isinstance(
        manifest.data_config, DataConfig
    ):
        raise ExportError("schemaVersion 1 requires synthetic dataConfig")
    if manifest.schema_version == EXTERNAL_SCHEMA_VERSION and not isinstance(
        manifest.data_config, ExternalDataConfig
    ):
        raise ExportError("schemaVersion 2 requires external dataConfig")

    if manifest.status != "completed":
        raise ExportError(
            f"run is not completed (status={manifest.status}); only completed runs can be exported"
        )

    events = _read_events_jsonl(run_dir / "events.jsonl")

    snapshots_raw = _read_json(run_dir / "snapshots.json", label="snapshots.json")
    if not isinstance(snapshots_raw, list):
        raise ExportError("snapshots.json must contain a JSON array")
    try:
        snapshots = [Snapshot.model_validate(item) for item in snapshots_raw]
    except Exception as exc:
        raise ExportError(f"snapshots.json failed validation: {exc}") from exc

    if not (len(manifest.dataset.sample_ids) == len(manifest.dataset.x) == len(manifest.dataset.y)):
        raise ExportError("manifest.dataset sampleIds, x, and y must have the same length")
    if len(set(manifest.dataset.sample_ids)) != len(manifest.dataset.sample_ids):
        raise ExportError("manifest.dataset sampleIds must be unique")

    # Cross-reference checks: steps unique and increasing, run_id
    # consistent across all events, observed sample ids resolvable against
    # the dataset, and no non-finite values anywhere in the payload.
    steps = [s.step for s in snapshots]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ExportError("snapshots.json steps are not strictly increasing and unique")

    if manifest.n_snapshots_written != len(snapshots):
        raise ExportError(
            f"manifest declares {manifest.n_snapshots_written} snapshots "
            f"but snapshots.json has {len(snapshots)}"
        )

    for event in events:
        if event.run_id != manifest.run_id:
            raise ExportError(
                f"events.jsonl contains an event for a different run "
                f"({event.run_id} != {manifest.run_id})"
            )
        if event.schema_version != manifest.schema_version:
            raise ExportError(
                f"events.jsonl event schemaVersion {event.schema_version} "
                f"does not match manifest schemaVersion {manifest.schema_version}"
            )

    seqs = [e.seq for e in events]
    if seqs != sorted(seqs) or len(seqs) != len(set(seqs)):
        raise ExportError("events.jsonl seq values are not strictly increasing and unique")

    known_sample_ids = set(manifest.dataset.sample_ids)
    for observed_id in manifest.observed_sample_ids:
        if observed_id not in known_sample_ids:
            raise ExportError(f"observed sample id {observed_id!r} is not in the dataset")

    for snapshot in snapshots:
        for sample_id in snapshot.observed_predictions:
            if sample_id not in known_sample_ids:
                raise ExportError(
                    f"snapshot step {snapshot.step} predicts unknown sample id {sample_id!r}"
                )

    _assert_manifest_values_finite(manifest)
    for snapshot in snapshots:
        _assert_snapshot_values_finite(snapshot)

    return manifest, events, snapshots


def export_run(run_dir: Path, target_dir: Path) -> RunManifest:
    """Export a completed, valid run to `target_dir` as a self-contained
    static bundle: manifest.json, events.jsonl, snapshots.json.

    Raises ExportError (and writes nothing) if the run is invalid, or if
    `target_dir` already exists (reject on conflict, no overwrite).
    The source run under `run_dir` is never modified.
    """
    if target_dir.exists():
        raise ExportError(
            f"export target already exists: {target_dir}; choose a new target directory"
        )

    manifest, events, snapshots = validate_run_for_export(run_dir)

    # Build the bundle in a sibling temp directory first, then rename it
    # into place as a single operation, so a failure partway through
    # never leaves a partial package visible at `target_dir`.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_dir.parent, prefix=".export-tmp-") as tmp_name:
        tmp_dir = Path(tmp_name)

        (tmp_dir / "manifest.json").write_text(
            manifest.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
        )
        (tmp_dir / "snapshots.json").write_text(
            json.dumps(
                [json.loads(s.model_dump_json(by_alias=True)) for s in snapshots],
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        with (tmp_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.model_dump_json(by_alias=True))
                fh.write("\n")

        if target_dir.exists():
            # Re-check immediately before the rename: another process may
            # have created the target while we were building the bundle.
            raise ExportError(
                f"export target already exists: {target_dir}; choose a new target directory"
            )
        shutil.move(str(tmp_dir), str(target_dir))

    return manifest


__all__ = ["ExportError", "export_run", "validate_run_for_export"]
