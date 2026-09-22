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
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from observatory.datasets.synthetic.linear import SyntheticLinearConfig
from observatory.datasets.tabular.auto_mpg import (
    AutoMpgDataset,
    PreparedAutoMpgDataset,
    load_auto_mpg,
    prepare_auto_mpg,
    standardize_weight,
)
from observatory.datasets.tabular.breast_cancer import prepare_breast_cancer
from observatory.experiments.kmeans.study import run_kmeans_study
from observatory.experiments.linear_regression.record import (
    create_external_run,
    create_run,
    run_training,
)
from observatory.models.linear_regression.gradient_descent import (
    fit,
    least_squares_reference,
    mse,
)
from observatory.runtime.export import ExportError, export_run
from observatory.runtime.schema import ExternalDataConfig, ModelConfig, RunManifest
from observatory.runtime.storage import RunIOError

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


@dataclass(frozen=True)
class _DatasetRunInput:
    experiment_id: str
    dataset: AutoMpgDataset
    prepared: PreparedAutoMpgDataset
    model_config: ModelConfig


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_dataset_run_input(config_path: Path) -> _DatasetRunInput:
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"invalid dataset config in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    try:
        experiment_id = raw["experiment_id"]
        data_raw = raw["data"]
        model_raw = raw["model"]
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ConfigError(f"{config_path}: experiment_id must be a non-empty string")
        if not isinstance(data_raw, dict) or not isinstance(model_raw, dict):
            raise ConfigError(f"{config_path}: data and model must be mappings")
        version_manifest_value = data_raw["version_manifest"]
        processed_artifact_value = data_raw["processed_artifact"]
        if not isinstance(version_manifest_value, str) or not version_manifest_value:
            raise ConfigError(f"{config_path}: data.version_manifest must be a path string")
        if not isinstance(processed_artifact_value, str) or not processed_artifact_value:
            raise ConfigError(f"{config_path}: data.processed_artifact must be a path string")
        model_cfg = ModelConfig(
            algorithm=model_raw["algorithm"],
            initial_bias=model_raw["initial_bias"],
            initial_weight=model_raw["initial_weight"],
            learning_rate=model_raw["learning_rate"],
            n_updates=model_raw["n_updates"],
        )
        if any(
            isinstance(model_raw[field], bool)
            for field in ("initial_bias", "initial_weight", "learning_rate", "n_updates")
        ):
            raise ValueError("model numeric fields must not be booleans")
        if not all(
            math.isfinite(value)
            for value in (
                model_cfg.initial_bias,
                model_cfg.initial_weight,
                model_cfg.learning_rate,
            )
        ):
            raise ValueError("model parameters must be finite")
        if model_cfg.learning_rate <= 0:
            raise ValueError("model.learning_rate must be positive")
        if model_cfg.n_updates < 0:
            raise ValueError("model.n_updates must be non-negative")
    except KeyError as exc:
        raise ConfigError(f"{config_path} is missing required key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{config_path}: {exc}") from exc

    try:
        dataset = load_auto_mpg(_resolve_repo_path(version_manifest_value))
        expected_config_identity = {
            "dataset": dataset.dataset_id,
            "version": dataset.version,
            "feature": dataset.source_feature,
            "target": dataset.target,
            "preprocessing": dataset.preprocessing,
            "split": dataset.split_strategy,
        }
        for field, verified_value in expected_config_identity.items():
            configured_value = data_raw.get(field)
            if configured_value != verified_value:
                raise ValueError(
                    f"data.{field} {configured_value!r} does not match "
                    f"verified value {verified_value!r}"
                )
        processed_path = _resolve_repo_path(processed_artifact_value)
        processed_hash = hashlib.sha256(processed_path.read_bytes()).hexdigest()
        if processed_hash != dataset.processed_artifact_sha256:
            raise ValueError("configured processed artifact does not match the verified artifact")
        prepared = standardize_weight(dataset)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc

    return _DatasetRunInput(
        experiment_id=experiment_id,
        dataset=dataset,
        prepared=prepared,
        model_config=model_cfg,
    )


def train_dataset(config_path: Path) -> DatasetTrainingResult:
    """Train the supported external dataset without creating replay artifacts."""
    run_input = _load_dataset_run_input(config_path)
    dataset = run_input.dataset
    prepared = run_input.prepared
    model_cfg = run_input.model_config
    try:
        states = fit(
            prepared.x,
            prepared.y,
            learning_rate=model_cfg.learning_rate,
            n_updates=model_cfg.n_updates,
            b0=model_cfg.initial_bias,
            w0=model_cfg.initial_weight,
        )
    except (ValueError, FloatingPointError) as exc:
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


def run_dataset(config_path: Path, runs_root: Path) -> RunManifest:
    """Create and execute a schema-v2 replay run for the verified Auto MPG dataset."""
    run_input = _load_dataset_run_input(config_path)
    dataset = run_input.dataset
    prepared = run_input.prepared
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        recorder = create_external_run(
            experiment_id=run_input.experiment_id,
            data_config=ExternalDataConfig(
                source="external_dataset",
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                version_manifest_sha256=dataset.version_manifest_sha256,
                processed_artifact_sha256=dataset.processed_artifact_sha256,
                source_feature=dataset.source_feature,
                feature=dataset.feature,
                feature_unit=dataset.feature_unit,
                target=dataset.target,
                target_unit=dataset.target_unit,
                preprocessing=dataset.preprocessing,
                split=dataset.split,
            ),
            dataset=prepared,
            dataset_source_id=dataset.source_id,
            model_cfg=run_input.model_config,
            runs_root=runs_root,
            repo_root=REPO_ROOT,
        )
    except (OSError, ValueError, RunIOError) as exc:
        raise ConfigError(f"{config_path}: could not create dataset run: {exc}") from exc
    return run_training(recorder)


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


def _cmd_prepare_wdbc(args: argparse.Namespace) -> int:
    try:
        hashes = prepare_breast_cancer(Path(args.version_manifest), Path(args.output_dir))
    except ValueError as exc:
        print(f"preparation rejected: {exc}", file=sys.stderr)
        return 2
    print(f"prepared WDBC: rows=569 train=397 validation=172 hashes={hashes}")
    return 0


def _cmd_prepare_dataset(args: argparse.Namespace) -> int:
    try:
        result = prepare_auto_mpg(Path(args.raw_data), Path(args.output))
    except ValueError as exc:
        print(f"preparation rejected: {exc}", file=sys.stderr)
        return 2

    print(f"prepared Auto MPG: rows={result.row_count} sha256={result.sha256} output={args.output}")
    return 0


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


def _cmd_run_dataset(args: argparse.Namespace) -> int:
    try:
        manifest = run_dataset(Path(args.config), Path(args.runs_root))
    except ConfigError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2

    print(f"run {manifest.run_id}: status={manifest.status} stop_reason={manifest.stop_reason}")
    return 0 if manifest.status == "completed" else 1


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


def _cmd_run_kmeans_study(args: argparse.Namespace) -> int:
    try:
        manifests = run_kmeans_study(Path(args.config), Path(args.runs_root), REPO_ROOT)
    except (OSError, ValueError, RunIOError) as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2
    for manifest in manifests:
        print(
            f"run {manifest.run_id}: status={manifest.status} "
            f"stop_reason={manifest.stop_reason} inertia_seed={manifest.training_config.init_seed}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="observatory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wdbc_parser = subparsers.add_parser(
        "prepare-wdbc", help="Prepare verified WDBC data and split."
    )
    wdbc_parser.add_argument("version_manifest", help="Path to the pinned WDBC version YAML.")
    wdbc_parser.add_argument("output_dir", help="Directory for wdbc.csv and split.csv.")
    wdbc_parser.set_defaults(func=_cmd_prepare_wdbc)

    prepare_dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Prepare the deterministic Auto MPG DVC pipeline output.",
    )
    prepare_dataset_parser.add_argument("raw_data", help="Path to the pinned auto-mpg.data file.")
    prepare_dataset_parser.add_argument("output", help="Path for the processed weight-to-MPG CSV.")
    prepare_dataset_parser.set_defaults(func=_cmd_prepare_dataset)

    train_dataset_parser = subparsers.add_parser(
        "train-dataset",
        help="Train a supported external dataset config without creating a replay bundle.",
    )
    train_dataset_parser.add_argument(
        "config", help="Path to an external dataset config YAML file."
    )
    train_dataset_parser.set_defaults(func=_cmd_train_dataset)

    run_dataset_parser = subparsers.add_parser(
        "run-dataset", help="Run an external dataset config and record a schema-v2 replay bundle."
    )
    run_dataset_parser.add_argument("config", help="Path to an external dataset config YAML file.")
    run_dataset_parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory under which run directories are created (default: ./runs).",
    )
    run_dataset_parser.set_defaults(func=_cmd_run_dataset)

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

    kmeans_parser = subparsers.add_parser(
        "run-kmeans-study", help="Run the configured five-seed K-means initialization study."
    )
    kmeans_parser.add_argument("config", help="Path to a K-means study YAML file.")
    kmeans_parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory under which run directories are created (default: ./runs).",
    )
    kmeans_parser.set_defaults(func=_cmd_run_kmeans_study)

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
