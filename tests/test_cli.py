"""Tests for the CLI entry point: config validation, exit codes, and the
three-learning-rate comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from observatory.cli import ConfigError, build_parser, run_one

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "linear"


def write_config(path: Path, **overrides: object) -> Path:
    base: dict[str, object] = {
        "experiment_id": "test-cli-exp",
        "data": {
            "generator": "synthetic_linear_v1",
            "n_samples": 20,
            "true_bias": 1.0,
            "true_weight": 2.0,
            "noise_std": 0.3,
            "seed": 42,
        },
        "model": {
            "algorithm": "linear_regression_gradient_descent",
            "initial_bias": 0.0,
            "initial_weight": 0.0,
            "learning_rate": 0.25,
            "n_updates": 10,
        },
    }
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        base[section][field] = value  # type: ignore[index]
    config_path = path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return config_path


def test_run_one_completes_and_writes_files(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    runs_root = tmp_path / "runs"

    manifest = run_one(config_path, runs_root)

    assert manifest.status == "completed"
    run_dir = runs_root / manifest.run_id
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "snapshots.json").is_file()


def test_run_one_rejects_invalid_config_without_creating_run_dir(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, data__noise_std=-1.0)
    runs_root = tmp_path / "runs"

    with pytest.raises(ConfigError):
        run_one(config_path, runs_root)

    assert not runs_root.exists() or list(runs_root.iterdir()) == []


def test_run_one_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        run_one(tmp_path / "does_not_exist.yaml", tmp_path / "runs")


def test_run_one_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("not: valid: yaml: [", encoding="utf-8")

    with pytest.raises(ConfigError):
        run_one(config_path, tmp_path / "runs")


def test_run_one_rejects_missing_required_key(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(yaml.safe_dump({"experiment_id": "x", "data": {}}), encoding="utf-8")

    with pytest.raises(ConfigError):
        run_one(config_path, tmp_path / "runs")


def test_cli_run_subcommand_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    runs_root = tmp_path / "runs"

    good_config = write_config(tmp_path)
    args = parser.parse_args(["run", str(good_config), "--runs-root", str(runs_root)])
    assert args.func(args) == 0

    bad_config = write_config(tmp_path, data__noise_std=-1.0)
    args = parser.parse_args(["run", str(bad_config), "--runs-root", str(runs_root)])
    assert args.func(args) == 2


def test_cli_run_subcommand_numerical_failure_exit_code(tmp_path: Path) -> None:
    parser = build_parser()
    runs_root = tmp_path / "runs"
    config = write_config(tmp_path, model__learning_rate=1e250)

    args = parser.parse_args(["run", str(config), "--runs-root", str(runs_root)])
    assert args.func(args) == 1


def test_three_rate_configs_share_data_init_budget_and_diverge_behavior(tmp_path: Path) -> None:
    """Same data, initialization, and update budget across the three
    shipped configs; distinct real trajectories, not name-based fakes."""
    runs_root = tmp_path / "runs"
    manifests = {}
    for name in ("converge", "slow", "diverge"):
        manifest = run_one(CONFIGS_DIR / f"{name}.yaml", runs_root)
        manifests[name] = manifest

    # All completed (finite trajectories within budget).
    for manifest in manifests.values():
        assert manifest.status == "completed"
        assert manifest.stop_reason == "max_steps"

    # Same data config and same initial parameters across all three.
    data_configs = {m.data_config.model_dump_json() for m in manifests.values()}
    assert len(data_configs) == 1
    initial_params = {
        (m.training_config.initial_bias, m.training_config.initial_weight)
        for m in manifests.values()
    }
    assert len(initial_params) == 1
    budgets = {m.training_config.n_updates for m in manifests.values()}
    assert len(budgets) == 1

    # Only learning_rate differs.
    rates = {name: m.training_config.learning_rate for name, m in manifests.items()}
    assert len(set(rates.values())) == 3

    # Distinct, real trajectories: converge improves a lot, slow barely
    # moves, diverge grows but stays finite.
    def final_mse(name: str) -> float:
        run_dir = runs_root / manifests[name].run_id
        snapshots = json.loads((run_dir / "snapshots.json").read_text())
        return float(snapshots[-1]["trainMse"])

    def initial_mse(name: str) -> float:
        run_dir = runs_root / manifests[name].run_id
        snapshots = json.loads((run_dir / "snapshots.json").read_text())
        return float(snapshots[0]["trainMse"])

    initial_values = {initial_mse(name) for name in manifests}
    assert len(initial_values) == 1  # same data + same init -> same starting MSE

    assert final_mse("converge") < 1.0
    assert final_mse("slow") > 1.0
    assert final_mse("slow") < initial_mse("slow") * 2  # barely moved, not exploded
    assert final_mse("diverge") > initial_mse("diverge")
    assert np.isfinite(final_mse("diverge"))
