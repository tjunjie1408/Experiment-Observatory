"""Tests for the recording layer: run lifecycle, atomic writes, and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from observatory.data.synthetic import SyntheticLinearConfig
from observatory.models.linear_regression import mse, predict
from observatory.runtime.record import RunIOError, create_run, run_training
from observatory.runtime.schema import ModelConfig


def make_configs(
    *, learning_rate: float = 0.25, n_updates: int = 10
) -> tuple[SyntheticLinearConfig, ModelConfig]:
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
    return data_cfg, model_cfg


def test_completed_run_writes_consistent_files(tmp_path: Path) -> None:
    data_cfg, model_cfg = make_configs()
    recorder = create_run(
        experiment_id="test-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)

    assert manifest.status == "completed"
    assert manifest.stop_reason == "max_steps"
    assert manifest.last_valid_step == model_cfg.n_updates

    run_dir = tmp_path / manifest.run_id
    manifest_on_disk = json.loads((run_dir / "manifest.json").read_text())
    snapshots = json.loads((run_dir / "snapshots.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    assert manifest_on_disk["runId"] == manifest.run_id
    assert manifest_on_disk["status"] == "completed"
    assert len(snapshots) == model_cfg.n_updates + 1
    assert [s["step"] for s in snapshots] == list(range(model_cfg.n_updates + 1))

    # seq is strictly increasing and never resets.
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert events[0]["kind"] == "run.created"
    assert events[-1]["kind"] == "run.completed"


def test_snapshot_recomputes_from_recorded_b_w(tmp_path: Path) -> None:
    """Every value in a snapshot belongs to the same theta_k and can be
    independently recomputed from the recorded data and (b, w)."""
    data_cfg, model_cfg = make_configs(n_updates=5)
    recorder = create_run(
        experiment_id="test-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)
    run_dir = tmp_path / manifest.run_id

    x = np.array(manifest.dataset.x)
    y = np.array(manifest.dataset.y)
    snapshots = json.loads((run_dir / "snapshots.json").read_text())

    for snap in snapshots:
        recomputed_mse = mse(snap["b"], snap["w"], x, y)
        assert recomputed_mse == pytest.approx(snap["trainMse"], abs=1e-9)

        for sample_id, recorded_pred in snap["observedPredictions"].items():
            idx = manifest.dataset.sample_ids.index(sample_id)
            recomputed_pred = float(predict(snap["b"], snap["w"], x)[idx])
            assert recomputed_pred == pytest.approx(recorded_pred, abs=1e-9)


def test_same_config_twice_produces_distinct_runs_no_overwrite(tmp_path: Path) -> None:
    """Repeated identical config produces distinct runIds; neither is overwritten."""
    data_cfg, model_cfg = make_configs()
    repo_root = Path(__file__).resolve().parents[1]

    recorder_a = create_run(
        experiment_id="dup-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=repo_root,
    )
    manifest_a = run_training(recorder_a)

    recorder_b = create_run(
        experiment_id="dup-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=repo_root,
    )
    manifest_b = run_training(recorder_b)

    assert manifest_a.run_id != manifest_b.run_id
    assert (tmp_path / manifest_a.run_id).exists()
    assert (tmp_path / manifest_b.run_id).exists()
    assert manifest_a.status == "completed"
    assert manifest_b.status == "completed"


def test_invalid_config_rejected_before_run_dir_created(tmp_path: Path) -> None:
    """An invalid config is rejected before any run directory exists."""
    bad_data_cfg = SyntheticLinearConfig(
        n_samples=20, true_bias=1.0, true_weight=2.0, noise_std=-1.0, seed=42
    )
    _, model_cfg = make_configs()

    with pytest.raises(ValueError):
        create_run(
            experiment_id="bad-exp",
            data_config=bad_data_cfg,
            model_cfg=model_cfg,
            runs_root=tmp_path,
            repo_root=Path(__file__).resolve().parents[1],
        )

    assert list(tmp_path.iterdir()) == []


def test_numerical_overflow_marks_failed_not_completed(tmp_path: Path) -> None:
    """Non-finite values stop the run and it is recorded as failed, never a
    false completed claim; no illegal snapshot is persisted."""
    data_cfg, model_cfg = make_configs(learning_rate=1e250, n_updates=5)
    recorder = create_run(
        experiment_id="overflow-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)

    assert manifest.status == "failed"
    assert manifest.stop_reason == "numerical_error"

    run_dir = tmp_path / manifest.run_id
    snapshots = json.loads((run_dir / "snapshots.json").read_text())
    for snap in snapshots:
        assert np.isfinite(snap["trainMse"])
        assert np.isfinite(snap["b"])
        assert np.isfinite(snap["w"])


def test_io_failure_during_run_marks_failed_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write failure must not produce a false completed claim; the run is
    marked failed with stop_reason=io_error when the failure itself can
    still be persisted."""
    data_cfg, model_cfg = make_configs(n_updates=3)
    recorder = create_run(
        experiment_id="io-fail-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )

    call_count = {"n": 0}
    original_write_snapshots = recorder._write_snapshots

    def flaky_write_snapshots() -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RunIOError("simulated disk failure")
        original_write_snapshots()

    monkeypatch.setattr(recorder, "_write_snapshots", flaky_write_snapshots)

    manifest = run_training(recorder)

    assert manifest.status == "failed"
    assert manifest.stop_reason == "io_error"

    # The manifest on disk must reflect a terminal, readable state, not a
    # silently-vanished status.
    run_dir = tmp_path / manifest.run_id
    manifest_on_disk = json.loads((run_dir / "manifest.json").read_text())
    assert manifest_on_disk["status"] == "failed"


def test_diverging_but_finite_run_is_completed_not_converged(tmp_path: Path) -> None:
    """A finite error-growth trajectory that exhausts its budget is
    completed + max_steps; it must not be reported as converged."""
    data_cfg, model_cfg = make_configs(learning_rate=1.5, n_updates=80)
    recorder = create_run(
        experiment_id="diverge-exp",
        data_config=data_cfg,
        model_cfg=model_cfg,
        runs_root=tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    manifest = run_training(recorder)

    assert manifest.status == "completed"
    assert manifest.stop_reason == "max_steps"

    run_dir = tmp_path / manifest.run_id
    snapshots = json.loads((run_dir / "snapshots.json").read_text())
    assert snapshots[-1]["trainMse"] > snapshots[0]["trainMse"]
    assert np.isfinite(snapshots[-1]["trainMse"])
