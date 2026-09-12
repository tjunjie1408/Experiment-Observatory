"""CLI entry point for observatory.

`observatory run <config.yaml>` creates a run, executes it, and reports the
final status and exit code:

- invalid config       -> no run created, exit code 2
- completed (any stop) -> exit code 0
- failed               -> exit code 1

`observatory run-all <config1.yaml> [config2.yaml ...]` runs each config in
turn against the same runs root, for the three-learning-rate comparison:
each config keeps its own runId and files; a later config failing does not
touch an earlier config's completed run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from observatory.data.auto_mpg import load_auto_mpg, standardize_weight
from observatory.data.synthetic import SyntheticLinearConfig
from observatory.models.linear_regression import fit, least_squares_reference, mse
from observatory.runtime.export import ExportError, export_run
from observatory.runtime.record import RunIOError, create_run, run_training
from observatory.runtime.schema import ModelConfig, RunManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"


class ConfigError(ValueError):
    """Raised for a malformed or invalid experiment config file."""


@dataclass(frozen=True)
class DatasetTrainingResult:
    dataset_id: str
    dataset_version: str
    sample_count: int
    b: float
    w: float
    train_mse: float
    reference_b: float
    reference_w: float
    reference_mse: float


def _load_config(config_path: Path) -> tuple[str, SyntheticLinearConfig, ModelConfig]:
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    try:
        experiment_id = raw["experiment_id"]
        data_raw: dict[str, Any] = raw["data"]
        model_raw: dict[str, Any] = raw["model"]
    except KeyError as exc:
        raise ConfigError(f"{config_path} is missing required key: {exc}") from exc

    if not isinstance(experiment_id, str) or not experiment_id:
        raise ConfigError(f"{config_path}: experiment_id must be a non-empty string")

    try:
        data_config = SyntheticLinearConfig(
            n_samples=data_raw["n_samples"],
            true_bias=data_raw["true_bias"],
            true_weight=data_raw["true_weight"],
            noise_std=data_raw["noise_std"],
            seed=data_raw["seed"],
        )
    except KeyError as exc:
        raise ConfigError(f"{config_path}: data section missing key: {exc}") from exc

    try:
        model_cfg = ModelConfig(
            algorithm=model_raw["algorithm"],
            initial_bias=model_raw["initial_bias"],
            initial_weight=model_raw["initial_weight"],
            learning_rate=model_raw["learning_rate"],
            n_updates=model_raw["n_updates"],
        )
    except KeyError as exc:
        raise ConfigError(f"{config_path}: model section missing key: {exc}") from exc

    return experiment_id, data_config, model_cfg


def train_dataset(config_path: Path) -> DatasetTrainingResult:
    """Train the supported external dataset config without creating a replay run.

    External provenance cannot be represented honestly by the synthetic-only
    schema-v1 DataConfig, so this command deliberately stops at verified
    training evidence until a versioned replay schema is designed.
    """
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"invalid dataset config in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    try:
        data_raw: dict[str, Any] = raw["data"]
        model_raw: dict[str, Any] = raw["model"]
        expected_data = {
            "dataset": "uci-auto-mpg",
            "feature": "weight",
            "target": "mpg",
            "preprocessing": "population_standardization",
            "split": "none",
        }
        for field, expected in expected_data.items():
            if data_raw.get(field) != expected:
                raise ConfigError(
                    f"{config_path}: data.{field} must be {expected!r} for this command"
                )
        version_manifest_value = data_raw["version_manifest"]
        if not isinstance(version_manifest_value, str) or not version_manifest_value:
            raise ConfigError(f"{config_path}: data.version_manifest must be a path string")
        model_cfg = ModelConfig(
            algorithm=model_raw["algorithm"],
            initial_bias=model_raw["initial_bias"],
            initial_weight=model_raw["initial_weight"],
            learning_rate=model_raw["learning_rate"],
            n_updates=model_raw["n_updates"],
        )
    except KeyError as exc:
        raise ConfigError(f"{config_path} is missing required key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{config_path}: {exc}") from exc

    version_manifest = Path(version_manifest_value)
    if not version_manifest.is_absolute():
        version_manifest = REPO_ROOT / version_manifest
    try:
        dataset = load_auto_mpg(version_manifest)
        prepared = standardize_weight(dataset)
        states = fit(
            prepared.x,
            prepared.y,
            learning_rate=model_cfg.learning_rate,
            n_updates=model_cfg.n_updates,
            b0=model_cfg.initial_bias,
            w0=model_cfg.initial_weight,
        )
    except (OSError, ValueError, FloatingPointError) as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc

    final = states[-1]
    reference_b, reference_w = least_squares_reference(prepared.x, prepared.y)
    reference_mse = mse(reference_b, reference_w, prepared.x, prepared.y)
    return DatasetTrainingResult(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        sample_count=len(dataset.sample_ids),
        b=final.b,
        w=final.w,
        train_mse=final.mse,
        reference_b=reference_b,
        reference_w=reference_w,
        reference_mse=reference_mse,
    )


def run_one(config_path: Path, runs_root: Path) -> RunManifest:
    """Create and execute a single run from a config file.

    Raises ConfigError before any run directory is created for invalid
    input; returns the final manifest for a created run, whether it ends
    up completed or failed.
    """
    experiment_id, data_config, model_cfg = _load_config(config_path)

    runs_root.mkdir(parents=True, exist_ok=True)

    try:
        recorder = create_run(
            experiment_id=experiment_id,
            data_config=data_config,
            model_cfg=model_cfg,
            runs_root=runs_root,
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    except RunIOError as exc:
        raise ConfigError(f"{config_path}: could not create run directory: {exc}") from exc

    return run_training(recorder)


def _cmd_train_dataset(args: argparse.Namespace) -> int:
    try:
        result = train_dataset(Path(args.config))
    except ConfigError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2

    print(
        f"dataset {result.dataset_id}@{result.dataset_version}: "
        f"samples={result.sample_count} b={result.b:.12g} w={result.w:.12g} "
        f"train_mse={result.train_mse:.12g} reference_mse={result.reference_mse:.12g}"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    runs_root = Path(args.runs_root)

    try:
        manifest = run_one(config_path, runs_root)
    except ConfigError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2

    print(f"run {manifest.run_id}: status={manifest.status} stop_reason={manifest.stop_reason}")
    return 0 if manifest.status == "completed" else 1


def _cmd_run_all(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root)
    exit_code = 0
    for config_str in args.configs:
        config_path = Path(config_str)
        try:
            manifest = run_one(config_path, runs_root)
        except ConfigError as exc:
            print(f"rejected: {exc}", file=sys.stderr)
            exit_code = max(exit_code, 2)
            continue

        print(f"run {manifest.run_id}: status={manifest.status} stop_reason={manifest.stop_reason}")
        if manifest.status != "completed":
            exit_code = max(exit_code, 1)
    return exit_code


def _cmd_export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    target_dir = Path(args.target_dir)

    try:
        manifest = export_run(run_dir, target_dir)
    except ExportError as exc:
        print(f"export rejected: {exc}", file=sys.stderr)
        return 2

    print(f"exported {manifest.run_id} to {target_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="observatory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_dataset_parser = subparsers.add_parser(
        "train-dataset",
        help="Train a supported external dataset config without creating a replay bundle.",
    )
    train_dataset_parser.add_argument(
        "config", help="Path to an external dataset config YAML file."
    )
    train_dataset_parser.set_defaults(func=_cmd_train_dataset)

    run_parser = subparsers.add_parser("run", help="Run a single synthetic experiment config.")
    run_parser.add_argument("config", help="Path to a linear regression config YAML file.")
    run_parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory under which run directories are created (default: ./runs).",
    )
    run_parser.set_defaults(func=_cmd_run)

    run_all_parser = subparsers.add_parser(
        "run-all", help="Run multiple experiment configs in sequence."
    )
    run_all_parser.add_argument("configs", nargs="+", help="Paths to config YAML files.")
    run_all_parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory under which run directories are created (default: ./runs).",
    )
    run_all_parser.set_defaults(func=_cmd_run_all)

    export_parser = subparsers.add_parser(
        "export", help="Export a completed run to a self-contained static bundle."
    )
    export_parser.add_argument("run_dir", help="Path to the source run directory.")
    export_parser.add_argument("target_dir", help="Path to the new export target directory.")
    export_parser.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
