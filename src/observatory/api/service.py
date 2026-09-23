"""Single-worker process coordinator for the local real-time run service."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from observatory.api.models import RunRequest
from observatory.datasets.synthetic.linear import SyntheticLinearConfig
from observatory.experiments.linear_regression.record import (
    RunRecorder,
    create_run,
    interrupt_orphaned_run,
    load_run_events,
    load_run_manifest,
)
from observatory.models.linear_regression.gradient_descent import iter_fit
from observatory.runtime.schema import RunManifest, Snapshot

TERMINAL_STATUSES = frozenset({"completed", "cancelled", "interrupted", "failed"})
WorkerTarget = Callable[[dict[str, Any], Any, Any], None]


class RunConflictError(RuntimeError):
    """Raised when a second run is requested while the worker is occupied."""


class RunNotFoundError(LookupError):
    """Raised when a run id has no persisted directory."""


class WorkerStartError(RuntimeError):
    """Raised when the child process cannot be started."""


class EventCursorError(ValueError):
    """Raised when persisted event history cannot satisfy a requested cursor."""


def _training_worker(payload: dict[str, Any], messages: Any, cancel_requested: Any) -> None:
    """Compute snapshots in the child; the service process remains the sole artifact writer."""
    try:
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        cfg = payload["model"]
        observed_indices = payload["observedIndices"]
        sample_ids = payload["sampleIds"]
        for state in iter_fit(
            x,
            y,
            learning_rate=cfg["learningRate"],
            n_updates=cfg["nUpdates"],
            b0=cfg["initialBias"],
            w0=cfg["initialWeight"],
        ):
            if cancel_requested.is_set():
                messages.put(("cancelled", None))
                return
            messages.put(
                (
                    "snapshot",
                    {
                        "step": state.step,
                        "b": state.b,
                        "w": state.w,
                        "gradientB": state.gradient_b,
                        "gradientW": state.gradient_w,
                        "trainMse": state.mse,
                        "observedPredictions": {
                            sample_ids[index]: float(state.predictions[index])
                            for index in observed_indices
                        },
                    },
                )
            )
        messages.put(("cancelled" if cancel_requested.is_set() else "completed", None))
    except BaseException as exc:
        messages.put(("failed", f"{type(exc).__name__}: {exc}"))


@dataclass
class _ActiveRun:
    recorder: RunRecorder
    process: Any
    messages: Any
    cancel_requested: Any
    cancel_deadline: float | None = None
    monitor: threading.Thread | None = None


class RunService:
    """Own exactly one child worker and serialize all persistent run writes."""

    def __init__(
        self,
        *,
        runs_root: Path,
        repo_root: Path,
        worker_target: WorkerTarget = _training_worker,
        cancellation_timeout: float = 5.0,
    ) -> None:
        self.runs_root = runs_root
        self.repo_root = repo_root
        self.worker_target = worker_target
        self.cancellation_timeout = cancellation_timeout
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._active: _ActiveRun | None = None

    def recover_orphans(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        for run_dir in self.runs_root.iterdir():
            if run_dir.is_dir() and (run_dir / "manifest.json").is_file():
                raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("schemaVersion") in (3, 4):
                    continue  # M4 does not own K-means or tree lifecycle recovery.
                manifest = load_run_manifest(run_dir)
                if manifest.status in {"running", "cancelling"}:
                    interrupt_orphaned_run(run_dir)

    def _request(self, request: RunRequest | dict[str, Any]) -> RunRequest:
        if isinstance(request, RunRequest):
            return request
        try:
            return RunRequest.model_validate(request)
        except ValidationError:
            raise

    def start(self, request: RunRequest | dict[str, Any]) -> RunManifest:
        parsed = self._request(request)
        with self._lock:
            if self._active is not None:
                raise RunConflictError("a run is already active; requests are not queued")

            self.runs_root.mkdir(parents=True, exist_ok=True)
            recorder = create_run(
                experiment_id=parsed.experiment_id,
                data_config=SyntheticLinearConfig(**parsed.data.model_dump()),
                model_cfg=parsed.model.to_runtime_config(),
                runs_root=self.runs_root,
                repo_root=self.repo_root,
            )
            observed = set(recorder.manifest.observed_sample_ids)
            payload = {
                "x": recorder.dataset.x.tolist(),
                "y": recorder.dataset.y.tolist(),
                "sampleIds": list(recorder.dataset.sample_ids),
                "observedIndices": [
                    index
                    for index, sample_id in enumerate(recorder.dataset.sample_ids)
                    if sample_id in observed
                ],
                "model": parsed.model.model_dump(by_alias=True),
            }
            messages = self._context.Queue()
            cancel_requested = self._context.Event()
            process = self._context.Process(
                target=self.worker_target,
                args=(payload, messages, cancel_requested),
                daemon=True,
            )
            recorder.record_started()
            active = _ActiveRun(recorder, process, messages, cancel_requested)
            self._active = active
            try:
                process.start()
            except BaseException as exc:
                recorder.record_failed(
                    stop_reason="runtime_error", message=f"worker failed to start: {exc}"
                )
                self._active = None
                raise WorkerStartError(str(exc)) from exc
            monitor = threading.Thread(
                target=self._monitor_worker,
                args=(active,),
                name=f"observatory-{recorder.manifest.run_id}",
                daemon=True,
            )
            active.monitor = monitor
            monitor.start()
            return recorder.manifest

    def _finalize_active(self, active: _ActiveRun) -> None:
        active.process.join(timeout=1.0)
        with self._lock:
            if self._active is active:
                self._active = None
        active.messages.close()

    def _handle_message(self, active: _ActiveRun, kind: str, payload: Any) -> bool:
        with self._lock:
            recorder = active.recorder
            if recorder.manifest.status in TERMINAL_STATUSES:
                return True
            if kind == "snapshot":
                recorder.record_step(Snapshot.model_validate(payload))
                return False
            if kind == "completed":
                if recorder.manifest.status == "cancelling":
                    recorder.record_cancelled()
                else:
                    recorder.record_completed(stop_reason="max_steps")
                return True
            if kind == "cancelled":
                if recorder.manifest.status != "cancelling":
                    recorder.record_cancelling()
                recorder.record_cancelled()
                return True
            recorder.record_failed(stop_reason="runtime_error", message=str(payload))
            return True

    def _monitor_worker(self, active: _ActiveRun) -> None:
        while True:
            try:
                kind, payload = active.messages.get(timeout=0.05)
            except queue.Empty:
                kind = None
                payload = None

            if kind is not None and self._handle_message(active, kind, payload):
                self._finalize_active(active)
                return

            with self._lock:
                deadline = active.cancel_deadline
            if deadline is not None and time.monotonic() >= deadline and active.process.is_alive():
                active.process.terminate()
                active.process.join(timeout=1.0)
                with self._lock:
                    if active.recorder.manifest.status not in TERMINAL_STATUSES:
                        active.recorder.record_interrupted(
                            stop_reason="forced_termination",
                            message="worker did not stop within the cancellation timeout",
                        )
                self._finalize_active(active)
                return

            if not active.process.is_alive():
                try:
                    kind, payload = active.messages.get(timeout=0.2)
                except queue.Empty:
                    with self._lock:
                        if active.recorder.manifest.status not in TERMINAL_STATUSES:
                            active.recorder.record_interrupted(
                                stop_reason="worker_lost",
                                message="worker exited without a terminal result",
                            )
                    self._finalize_active(active)
                    return
                if self._handle_message(active, kind, payload):
                    self._finalize_active(active)
                    return

    def get(self, run_id: str) -> RunManifest:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
            raise RunNotFoundError(run_id)
        run_dir = self.runs_root / run_id
        if not run_dir.is_dir():
            raise RunNotFoundError(run_id)
        return load_run_manifest(run_dir)

    def cancel(self, run_id: str) -> tuple[RunManifest, bool]:
        with self._lock:
            manifest = self.get(run_id)
            if manifest.status in TERMINAL_STATUSES:
                return manifest, False
            active = self._active
            if active is None or active.recorder.manifest.run_id != run_id:
                raise RunNotFoundError(f"run {run_id} has no active worker")
            if active.recorder.manifest.status != "cancelling":
                active.recorder.record_cancelling()
                active.cancel_requested.set()
                active.cancel_deadline = time.monotonic() + self.cancellation_timeout
            return active.recorder.manifest, True

    def validate_event_cursor(self, run_id: str, after_seq: int) -> None:
        with self._lock:
            self.get(run_id)
            events = load_run_events(self.runs_root / run_id)
        for expected, event in enumerate(events, start=1):
            if event.seq != expected:
                raise EventCursorError(
                    f"persisted event history has a gap before sequence {event.seq}"
                )
        last_seq = events[-1].seq if events else 0
        if after_seq > last_seq:
            raise EventCursorError(
                f"event cursor {after_seq} is beyond the last persisted sequence {last_seq}"
            )

    async def stream_events(self, run_id: str, after_seq: int) -> AsyncIterator[str]:
        last_seq = after_seq
        while True:
            with self._lock:
                manifest = self.get(run_id)
                events = load_run_events(self.runs_root / run_id)
            pending = [event for event in events if event.seq > last_seq]
            for event in pending:
                if event.seq != last_seq + 1:
                    raise EventCursorError(f"event sequence {last_seq + 1} cannot be replayed")
                last_seq = event.seq
                yield (
                    f"id: {event.seq}\n"
                    f"event: {event.kind}\n"
                    f"data: {event.model_dump_json(by_alias=True)}\n\n"
                )
            if manifest.status in TERMINAL_STATUSES and not pending:
                return
            await asyncio.sleep(0.05)

    def shutdown(self) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
            if active.process.is_alive():
                active.process.terminate()
                active.process.join(timeout=1.0)
            if active.recorder.manifest.status not in TERMINAL_STATUSES:
                active.recorder.record_interrupted(
                    stop_reason="service_shutdown",
                    message="service stopped while worker was active",
                )
            self._active = None
