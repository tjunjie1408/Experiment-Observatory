"""Linear-regression run lifecycle and recording layer.

- Each run gets its own runId directory under a runs root; two runs from the
  same config never share or overwrite files.
- manifest.json/events.jsonl/snapshots.json are written atomically (temp file
  + rename) so a crash or write failure never leaves a file that looks like
  a valid completed artifact.
- Non-finite values are never serialized; they are raised as errors before
  any snapshot reaches disk.
- SIGINT (Ctrl+C) during a run is caught at the per-step boundary and
  recorded as failed + user_cancelled, not silently swallowed as a
  successful exit.
"""

from __future__ import annotations

import contextlib
import json
import signal
import time
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from observatory.datasets.synthetic.linear import SyntheticLinearConfig, generate
from observatory.models.linear_regression.gradient_descent import iter_fit
from observatory.runtime.schema import (
    EXTERNAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DataConfig,
    DatasetSummary,
    Event,
    ExternalDataConfig,
    ExternalDatasetSummary,
    ModelConfig,
    RunManifest,
    Snapshot,
)
from observatory.runtime.storage import (
    RunIOError,
    append_jsonl,
    atomic_write_json,
    get_code_provenance,
    new_run_id,
)


class CancelledError(Exception):
    """Raised internally when SIGINT is observed at a safe step boundary."""


_REPLACE_MAX_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_S = 0.05


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    manifest: RunManifest


def load_run_manifest(run_dir: Path) -> RunManifest:
    """Load one persisted manifest through the authoritative Python schema."""
    try:
        path = run_dir / "manifest.json"
        last_exc: OSError | None = None
        for attempt in range(_REPLACE_MAX_ATTEMPTS):
            try:
                return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except PermissionError as exc:
                last_exc = exc
                if attempt < _REPLACE_MAX_ATTEMPTS - 1:
                    time.sleep(_REPLACE_RETRY_DELAY_S)
        assert last_exc is not None
        raise last_exc
    except (OSError, ValueError) as exc:
        raise RunIOError(f"failed to read manifest from {run_dir}: {exc}") from exc


def load_run_events(run_dir: Path) -> list[Event]:
    """Load complete persisted events; a malformed line is an explicit read failure."""
    try:
        path = run_dir / "events.jsonl"
        last_exc: OSError | None = None
        for attempt in range(_REPLACE_MAX_ATTEMPTS):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                break
            except PermissionError as exc:
                last_exc = exc
                if attempt < _REPLACE_MAX_ATTEMPTS - 1:
                    time.sleep(_REPLACE_RETRY_DELAY_S)
        else:
            assert last_exc is not None
            raise last_exc
        return [Event.model_validate_json(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise RunIOError(f"failed to read events from {run_dir}: {exc}") from exc


def interrupt_orphaned_run(run_dir: Path) -> RunManifest:
    """Finalize an unowned service run after restart without reconstructing training state."""
    manifest = load_run_manifest(run_dir)
    if manifest.status not in {"running", "cancelling"}:
        return manifest
    events = load_run_events(run_dir)
    next_seq = max((event.seq for event in events), default=0) + 1
    interrupted = manifest.model_copy(
        update={
            "status": "interrupted",
            "stop_reason": "service_restart",
            "error_message": "service restarted without an owning worker",
        }
    )
    atomic_write_json(
        run_dir / "manifest.json", json.loads(interrupted.model_dump_json(by_alias=True))
    )
    append_jsonl(
        run_dir / "events.jsonl",
        json.loads(
            Event(
                schema_version=manifest.schema_version,
                run_id=manifest.run_id,
                seq=next_seq,
                kind="run.interrupted",
                message="service restarted without an owning worker",
            ).model_dump_json(by_alias=True)
        ),
    )
    return interrupted


class LinearDataset(Protocol):
    """Minimal dataset shape required by the linear training recorder."""

    @property
    def sample_ids(self) -> Sequence[str]: ...

    @property
    def x(self) -> npt.NDArray[np.float64]: ...

    @property
    def y(self) -> npt.NDArray[np.float64]: ...


class RunRecorder:
    """Owns one run directory and writes manifest/events/snapshots for it."""

    def __init__(self, run_dir: Path, manifest: RunManifest, dataset: LinearDataset) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.dataset = dataset
        self._seq = 0
        self._snapshots: list[Snapshot] = []

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def snapshots_path(self) -> Path:
        return self.run_dir / "snapshots.json"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write_manifest(self, manifest: RunManifest) -> None:
        self.manifest = manifest
        atomic_write_json(self.manifest_path, json.loads(manifest.model_dump_json(by_alias=True)))

    def _write_event(self, event: Event) -> None:
        append_jsonl(self.events_path, json.loads(event.model_dump_json(by_alias=True)))

    def _write_snapshots(self) -> None:
        payload = [json.loads(s.model_dump_json(by_alias=True)) for s in self._snapshots]
        atomic_write_json(self.snapshots_path, payload)

    def record_created(self) -> None:
        self._write_manifest(self.manifest)
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.created",
            )
        )

    def record_started(self) -> None:
        self._write_manifest(self.manifest.model_copy(update={"status": "running"}))
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.started",
            )
        )

    def record_step(self, snapshot: Snapshot) -> None:
        self._snapshots.append(snapshot)
        self._write_snapshots()
        self._write_manifest(
            self.manifest.model_copy(update={"n_snapshots_written": len(self._snapshots)})
        )
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="step.recorded",
                step=snapshot.step,
            )
        )

    def record_completed(self, *, stop_reason: str) -> None:
        last_step = self._snapshots[-1].step if self._snapshots else None
        self._write_manifest(
            self.manifest.model_copy(
                update={
                    "status": "completed",
                    "stop_reason": stop_reason,
                    "last_valid_step": last_step,
                }
            )
        )
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.completed",
            )
        )

    def record_cancelling(self) -> None:
        if self.manifest.status == "cancelling":
            return
        self._write_manifest(self.manifest.model_copy(update={"status": "cancelling"}))
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.cancelling",
            )
        )

    def record_cancelled(self) -> None:
        last_step = self._snapshots[-1].step if self._snapshots else None
        self._write_manifest(
            self.manifest.model_copy(
                update={
                    "status": "cancelled",
                    "stop_reason": "user_cancelled",
                    "last_valid_step": last_step,
                }
            )
        )
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.cancelled",
            )
        )

    def record_interrupted(self, *, stop_reason: str, message: str) -> None:
        last_step = self._snapshots[-1].step if self._snapshots else None
        self._write_manifest(
            self.manifest.model_copy(
                update={
                    "status": "interrupted",
                    "stop_reason": stop_reason,
                    "last_valid_step": last_step,
                    "error_message": message,
                }
            )
        )
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.interrupted",
                message=message,
            )
        )

    def record_failed(self, *, stop_reason: str, message: str) -> None:
        last_step = self._snapshots[-1].step if self._snapshots else None
        self._write_manifest(
            self.manifest.model_copy(
                update={
                    "status": "failed",
                    "stop_reason": stop_reason,
                    "last_valid_step": last_step,
                    "error_message": message,
                }
            )
        )
        self._write_event(
            Event(
                schema_version=self.manifest.schema_version,
                run_id=self.manifest.run_id,
                seq=self._next_seq(),
                kind="run.failed",
                message=message,
            )
        )


@contextmanager
def _sigint_guard() -> Iterator[list[bool]]:
    """Install a SIGINT handler that records the interrupt instead of raising
    immediately, so the caller can stop at the next safe step boundary.
    There is no guaranteed millisecond deadline for the stop."""
    flag: list[bool] = [False]

    def _handler(signum: int, frame: types.FrameType | None) -> None:
        flag[0] = True

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


def _create_run_recorder(
    *,
    schema_version: int,
    experiment_id: str,
    data_config: DataConfig | ExternalDataConfig,
    dataset: LinearDataset,
    dataset_source_id: str,
    model_cfg: ModelConfig,
    runs_root: Path,
    repo_root: Path,
    max_observed_samples: int,
) -> RunRecorder:
    if dataset.x.ndim != 1 or dataset.x.shape != dataset.y.shape or dataset.x.size < 2:
        raise ValueError(
            "x and y must be one-dimensional arrays of equal shape with at least 2 samples"
        )
    if len(dataset.sample_ids) != dataset.x.size:
        raise ValueError("sample_ids, x, and y must have the same length")
    if len(set(dataset.sample_ids)) != len(dataset.sample_ids):
        raise ValueError("sample_ids must be unique")
    if not np.isfinite(dataset.x).all() or not np.isfinite(dataset.y).all():
        raise ValueError("x and y must contain only finite values")

    run_id = new_run_id(experiment_id)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    observed_sample_ids = list(
        dataset.sample_ids[: min(max_observed_samples, len(dataset.sample_ids))]
    )
    manifest = RunManifest(
        schema_version=schema_version,
        run_id=run_id,
        experiment_id=experiment_id,
        created_at=datetime.now(UTC).isoformat(),
        status="created",
        data_config=data_config,
        dataset=(
            DatasetSummary(
                generator_id=dataset_source_id,
                sample_ids=list(dataset.sample_ids),
                x=[float(v) for v in dataset.x],
                y=[float(v) for v in dataset.y],
            )
            if schema_version == SCHEMA_VERSION
            else ExternalDatasetSummary(
                source_id=dataset_source_id,
                sample_ids=list(dataset.sample_ids),
                x=[float(v) for v in dataset.x],
                y=[float(v) for v in dataset.y],
            )
        ),
        training_config=model_cfg,
        code_provenance=get_code_provenance(repo_root),
        observed_sample_ids=observed_sample_ids,
    )
    recorder = RunRecorder(run_dir, manifest, dataset)
    recorder.record_created()
    return recorder


def create_run(
    *,
    experiment_id: str,
    data_config: SyntheticLinearConfig,
    model_cfg: ModelConfig,
    runs_root: Path,
    repo_root: Path,
    max_observed_samples: int = 5,
) -> RunRecorder:
    """Validate synthetic config and write an unchanged schema-v1 created manifest."""
    dataset = generate(data_config)  # raises ValueError for invalid config
    return _create_run_recorder(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        data_config=DataConfig(
            generator=data_config.__class__.__name__,
            n_samples=data_config.n_samples,
            true_bias=data_config.true_bias,
            true_weight=data_config.true_weight,
            noise_std=data_config.noise_std,
            seed=data_config.seed,
        ),
        dataset=dataset,
        dataset_source_id=dataset.generator_id,
        model_cfg=model_cfg,
        runs_root=runs_root,
        repo_root=repo_root,
        max_observed_samples=max_observed_samples,
    )


def create_external_run(
    *,
    experiment_id: str,
    data_config: ExternalDataConfig,
    dataset: LinearDataset,
    dataset_source_id: str,
    model_cfg: ModelConfig,
    runs_root: Path,
    repo_root: Path,
    max_observed_samples: int = 5,
) -> RunRecorder:
    """Write a schema-v2 created manifest for a verified external linear dataset."""
    return _create_run_recorder(
        schema_version=EXTERNAL_SCHEMA_VERSION,
        experiment_id=experiment_id,
        data_config=data_config,
        dataset=dataset,
        dataset_source_id=dataset_source_id,
        model_cfg=model_cfg,
        runs_root=runs_root,
        repo_root=repo_root,
        max_observed_samples=max_observed_samples,
    )


def run_training(recorder: RunRecorder) -> RunManifest:
    """Execute gradient descent for the run, recording every step.

    A finite diverging trajectory that exhausts its budget is still
    `completed` with stop_reason=max_steps (never silently "converged"); a
    non-finite value or IO failure is `failed`; SIGINT observed at a step
    boundary is `failed` + `user_cancelled`.
    """
    dataset = recorder.dataset
    manifest = recorder.manifest
    cfg = manifest.training_config

    recorder.record_started()

    observed_ids = set(manifest.observed_sample_ids)
    id_index = {sid: i for i, sid in enumerate(dataset.sample_ids)}

    try:
        with _sigint_guard() as interrupted:
            try:
                step_iter = iter_fit(
                    dataset.x,
                    dataset.y,
                    learning_rate=cfg.learning_rate,
                    n_updates=cfg.n_updates,
                    b0=cfg.initial_bias,
                    w0=cfg.initial_weight,
                )
                for state in step_iter:
                    if interrupted[0]:
                        raise CancelledError

                    observed_predictions = {
                        sid: float(state.predictions[id_index[sid]]) for sid in observed_ids
                    }
                    snapshot = Snapshot(
                        step=state.step,
                        b=state.b,
                        w=state.w,
                        gradient_b=state.gradient_b,
                        gradient_w=state.gradient_w,
                        train_mse=state.mse,
                        observed_predictions=observed_predictions,
                    )
                    recorder.record_step(snapshot)
            except FloatingPointError as exc:
                recorder.record_failed(stop_reason="numerical_error", message=str(exc))
                return recorder.manifest

            if interrupted[0]:
                raise CancelledError

        recorder.record_completed(stop_reason="max_steps")
    except CancelledError:
        recorder.record_failed(stop_reason="user_cancelled", message="interrupted by user (SIGINT)")
    except RunIOError as exc:
        # Best-effort: try to still mark failed; if that also fails, the
        # manifest is left in its last successfully-written state, which
        # readers must treat as "incomplete, status unverified".
        with contextlib.suppress(RunIOError):
            recorder.record_failed(stop_reason="io_error", message=str(exc))
    return recorder.manifest


__all__ = [
    "SCHEMA_VERSION",
    "CancelledError",
    "RunIOError",
    "RunRecorder",
    "RunResult",
    "create_external_run",
    "create_run",
    "run_training",
]
