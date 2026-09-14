from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from observatory.api.app import create_app
from observatory.api.service import RunService
from observatory.datasets.synthetic.linear import SyntheticLinearConfig
from observatory.experiments.linear_regression.record import create_run
from observatory.runtime.schema import ModelConfig

VALID_REQUEST = {
    "experimentId": "api-linear",
    "data": {
        "nSamples": 8,
        "trueBias": 1.0,
        "trueWeight": 2.0,
        "noiseStd": 0.0,
        "seed": 7,
    },
    "model": {
        "algorithm": "linear_regression_gradient_descent",
        "initialBias": 0.0,
        "initialWeight": 0.0,
        "learningRate": 0.05,
        "nUpdates": 3,
    },
}


def _wait_for(
    predicate: Callable[[], bool], *, timeout: float = 5.0, interval: float = 0.01
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition was not met before timeout")


def _manifest(runs_root: Path, run_id: str) -> dict[str, Any]:
    path = runs_root / run_id / "manifest.json"
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)
    raise AssertionError("unreachable")


def _cooperative_worker(payload: dict[str, Any], messages: Any, cancel_requested: Any) -> None:
    del payload
    while not cancel_requested.is_set():
        time.sleep(0.01)
    messages.put(("cancelled", None))


def _uncooperative_worker(payload: dict[str, Any], messages: Any, cancel_requested: Any) -> None:
    del payload, messages, cancel_requested
    time.sleep(30)


def test_http_contract_validation_not_found_and_single_active_run(tmp_path: Path) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        worker_target=_cooperative_worker,
        cancellation_timeout=5.0,
    )

    with TestClient(app) as client:
        invalid = client.post("/api/runs", json={**VALID_REQUEST, "data": {"nSamples": 1}})
        assert invalid.status_code == 422
        unsafe_id = client.post("/api/runs", json={**VALID_REQUEST, "experimentId": "../escape"})
        invalid_algorithm = client.post(
            "/api/runs",
            json={
                **VALID_REQUEST,
                "model": {**VALID_REQUEST["model"], "algorithm": "arbitrary.module"},
            },
        )
        assert unsafe_id.status_code == 422
        assert invalid_algorithm.status_code == 422
        assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())

        missing = client.get("/api/runs/does-not-exist")
        assert missing.status_code == 404
        assert client.get("/api/runs/..").status_code == 404
        assert client.post("/api/runs/does-not-exist/cancel").status_code == 404

        started = client.post("/api/runs", json=VALID_REQUEST)
        assert started.status_code == 202
        run_id = started.json()["runId"]

        conflict = client.post("/api/runs", json=VALID_REQUEST)
        assert conflict.status_code == 409
        assert [path.name for path in (tmp_path / "runs").iterdir()] == [run_id]

        first_cancel = client.post(f"/api/runs/{run_id}/cancel")
        second_cancel = client.post(f"/api/runs/{run_id}/cancel")
        assert first_cancel.status_code == 202
        assert second_cancel.status_code in {200, 202}

        _wait_for(
            lambda: (
                _manifest(tmp_path / "runs", run_id)["status"]
                in {"cancelled", "interrupted", "failed"}
            ),
            timeout=10.0,
        )
        assert _manifest(tmp_path / "runs", run_id)["status"] == "cancelled"
        events = [
            json.loads(line)
            for line in (tmp_path / "runs" / run_id / "events.jsonl").read_text().splitlines()
        ]
        assert [event["kind"] for event in events].count("run.cancelling") == 1
        assert [event["kind"] for event in events].count("run.cancelled") == 1
        terminal_cancel = client.post(f"/api/runs/{run_id}/cancel")
        assert terminal_cancel.status_code == 200
        assert terminal_cancel.json()["status"] == "cancelled"


def test_real_worker_completes_and_sse_replays_after_cursor(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    app = create_app(runs_root=runs_root, repo_root=tmp_path)

    with TestClient(app) as client:
        started = client.post("/api/runs", json=VALID_REQUEST)
        assert started.status_code == 202
        run_id = started.json()["runId"]
        _wait_for(lambda: _manifest(runs_root, run_id)["status"] == "completed")

        all_events = client.get(f"/api/runs/{run_id}/events")
        assert all_events.status_code == 200
        assert all_events.headers["content-type"].startswith("text/event-stream")
        assert "id: 1\n" in all_events.text
        assert "event: run.completed\n" in all_events.text

        replayed = client.get(
            f"/api/runs/{run_id}/events?after_seq=1", headers={"Last-Event-ID": "1"}
        )
        assert replayed.status_code == 200
        assert "id: 1\n" not in replayed.text
        assert "id: 2\n" in replayed.text

        ambiguous = client.get(
            f"/api/runs/{run_id}/events?after_seq=1", headers={"Last-Event-ID": "2"}
        )
        assert ambiguous.status_code == 422
        unavailable = client.get(f"/api/runs/{run_id}/events?after_seq=999")
        assert unavailable.status_code == 409


def test_forced_termination_is_interrupted_not_cancelled(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    service = RunService(
        runs_root=runs_root,
        repo_root=tmp_path,
        worker_target=_uncooperative_worker,
        cancellation_timeout=0.1,
    )
    service.recover_orphans()
    manifest = service.start(VALID_REQUEST)

    service.cancel(manifest.run_id)
    _wait_for(lambda: _manifest(runs_root, manifest.run_id)["status"] == "interrupted")
    final = _manifest(runs_root, manifest.run_id)
    assert final["stopReason"] == "forced_termination"
    assert final["status"] != "cancelled"
    events = [
        json.loads(line)
        for line in (runs_root / manifest.run_id / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "run.interrupted"
    assert "run.cancelled" not in {event["kind"] for event in events}
    service.shutdown()


@pytest.mark.parametrize("orphan_status", ["running", "cancelling"])
def test_restart_marks_orphans_interrupted_but_preserves_terminal_runs(
    tmp_path: Path, orphan_status: str
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    created = create_run(
        experiment_id="orphan",
        data_config=SyntheticLinearConfig(
            n_samples=8,
            true_bias=1.0,
            true_weight=2.0,
            noise_std=0.0,
            seed=7,
        ),
        model_cfg=ModelConfig(
            algorithm="linear_regression_gradient_descent",
            initial_bias=0.0,
            initial_weight=0.0,
            learning_rate=0.05,
            n_updates=3,
        ),
        runs_root=runs_root,
        repo_root=tmp_path,
    )
    run_id = created.manifest.run_id
    run_dir = runs_root / run_id
    payload = _manifest(runs_root, run_id)
    payload["status"] = orphan_status
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    service = RunService(runs_root=runs_root, repo_root=tmp_path)
    service.recover_orphans()

    recovered = _manifest(runs_root, run_id)
    assert recovered["status"] == "interrupted"
    assert recovered["stopReason"] == "service_restart"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "run.interrupted"
    assert events[-1]["seq"] == events[-2]["seq"] + 1


def test_restart_does_not_rewrite_completed_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    app = create_app(runs_root=runs_root, repo_root=tmp_path)
    with TestClient(app) as client:
        run_id = client.post("/api/runs", json=VALID_REQUEST).json()["runId"]
        _wait_for(lambda: _manifest(runs_root, run_id)["status"] == "completed")

    before_manifest = (runs_root / run_id / "manifest.json").read_bytes()
    before_events = (runs_root / run_id / "events.jsonl").read_bytes()
    restarted = RunService(runs_root=runs_root, repo_root=tmp_path)
    restarted.recover_orphans()

    assert (runs_root / run_id / "manifest.json").read_bytes() == before_manifest
    assert (runs_root / run_id / "events.jsonl").read_bytes() == before_events


def test_sse_disconnect_does_not_request_cancellation(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    app = create_app(
        runs_root=runs_root,
        repo_root=tmp_path,
        worker_target=_cooperative_worker,
        cancellation_timeout=5.0,
    )

    with TestClient(app) as client:
        run_id = client.post("/api/runs", json=VALID_REQUEST).json()["runId"]
        stream = app.state.run_service.stream_events(run_id, 0)
        first_event = asyncio.run(anext(stream))
        assert "id: 1\n" in first_event
        asyncio.run(stream.aclose())

        assert _manifest(runs_root, run_id)["status"] == "running"
        client.post(f"/api/runs/{run_id}/cancel")
        _wait_for(lambda: _manifest(runs_root, run_id)["status"] == "cancelled")
