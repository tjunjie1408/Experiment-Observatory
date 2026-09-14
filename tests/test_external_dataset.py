from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from observatory.cli import build_parser, run_dataset, train_dataset
from observatory.datasets.tabular.auto_mpg import (
    load_auto_mpg,
    prepare_auto_mpg,
    standardize_weight,
)
from observatory.models.linear_regression.gradient_descent import fit, mse, predict
from observatory.runtime.export import ExportError, export_run

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "auto-mpg"
DATASET_MANIFEST = DATASET_ROOT / "dataset.yaml"
VERSION_MANIFEST = DATASET_ROOT / "versions" / "1.0.0.yaml"
PROCESSED_DATA = DATASET_ROOT / "processed" / "weight-mpg.csv"
CONFIG_PATH = REPO_ROOT / "configs" / "linear" / "auto_mpg_weight.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_dataset_fixture(tmp_path: Path, manifest: dict[str, object]) -> Path:
    copied_root = tmp_path / "auto-mpg"
    for directory in ("raw", "processed", "versions"):
        (copied_root / directory).mkdir(parents=True, exist_ok=True)
    for relative_path in (
        "raw/auto+mpg.zip",
        "raw/auto-mpg.data",
        "processed/weight-mpg.csv",
    ):
        shutil.copyfile(DATASET_ROOT / relative_path, copied_root / relative_path)
    copied_manifest = copied_root / "versions" / "1.0.0.yaml"
    copied_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return copied_manifest


def _write_external_run_config(tmp_path: Path, version_manifest: Path) -> Path:
    config = _load_yaml(CONFIG_PATH)
    data = config["data"]
    assert isinstance(data, dict)
    data["version_manifest"] = str(version_manifest)
    data["processed_artifact"] = str(version_manifest.parent.parent / "processed/weight-mpg.csv")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _set_manifest_value(manifest: dict[str, object], path: str, value: object) -> None:
    if path.startswith("processed."):
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list)
        current = next(artifact for artifact in artifacts if artifact["role"] == "processed")
        parts = path.split(".")[1:]
    else:
        current = manifest
        parts = path.split(".")
    for part in parts[:-1]:
        nested = current[part]
        assert isinstance(nested, dict)
        current = nested
    current[parts[-1]] = value


def test_yaml_manifests_pin_attribution_provenance_and_artifacts() -> None:
    assert not (DATASET_ROOT / "dataset.json").exists()
    assert not (DATASET_ROOT / "versions" / "1.0.0.json").exists()

    dataset_manifest = _load_yaml(DATASET_MANIFEST)
    version_manifest = _load_yaml(VERSION_MANIFEST)

    assert dataset_manifest == {
        "dataset_id": "uci-auto-mpg",
        "title": "Auto MPG",
        "creator": "R. Quinlan",
        "citation": (
            "Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5859H."
        ),
        "authoritative_page": "https://archive.ics.uci.edu/dataset/9/auto+mpg",
        "doi": "10.24432/C5859H",
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "versions": ["1.0.0"],
    }
    assert version_manifest["dataset_id"] == "uci-auto-mpg"
    assert version_manifest["version"] == "1.0.0"
    assert version_manifest["creator"] == "R. Quinlan"
    assert version_manifest["citation"] == dataset_manifest["citation"]
    assert version_manifest["authoritative_page"] == dataset_manifest["authoritative_page"]
    assert version_manifest["doi"] == "10.24432/C5859H"
    assert version_manifest["license"] == dataset_manifest["license"]
    assert version_manifest["retrieved_at"] == "2026-09-12"
    assert version_manifest["source_url"] == (
        "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
    )
    assert version_manifest["row_count"] == 398
    assert version_manifest["selection"] == {"feature": "weight", "target": "mpg"}
    assert version_manifest["split"] == {"strategy": "none", "identity": "all-398-rows"}

    artifacts = version_manifest["artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        path = DATASET_ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]

    processed = next(artifact for artifact in artifacts if artifact["role"] == "processed")
    assert processed["path"] == "processed/weight-mpg.csv"
    assert processed["parent_version"] == "1.0.0"
    assert processed["row_count"] == 398
    assert processed["schema"] == [
        {"name": "sample_id", "type": "string", "nullable": False},
        {"name": "weight", "type": "float64", "nullable": False, "unit": "pounds"},
        {
            "name": "mpg",
            "type": "float64",
            "nullable": False,
            "unit": "miles per gallon",
        },
        {"name": "weight_standardized", "type": "float64", "nullable": False},
    ]
    assert processed["preprocessing"] == {
        "source_artifact": "raw/auto-mpg.data",
        "row_order": "source row order",
        "sample_id": "row-{1-based source row number, zero-padded to 4 digits}",
        "weight_mean": pytest.approx(2970.424623115578),
        "weight_std": pytest.approx(845.7772335198174),
        "standard_deviation_ddof": 0,
        "formula": "(weight - weight_mean) / weight_std",
        "target": "unchanged mpg",
        "numeric_type": "float64",
    }


def test_public_preparation_reproduces_checked_in_processed_artifact(tmp_path: Path) -> None:
    regenerated = tmp_path / "weight-mpg.csv"

    result = prepare_auto_mpg(DATASET_ROOT / "raw" / "auto-mpg.data", regenerated)

    assert regenerated.read_bytes() == PROCESSED_DATA.read_bytes()
    assert result.row_count == 398
    assert result.weight_mean == pytest.approx(2970.424623115578)
    assert result.weight_std == pytest.approx(845.7772335198174)
    assert result.bytes == 16456
    assert result.sha256 == "b9b64882d6714f38fa5e368e855b1a25996b67ad9d053efde594fa28aa541607"
    assert result.sha256 == _sha256(regenerated)


def test_prepare_dataset_cli_regenerates_the_dvc_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    regenerated = tmp_path / "weight-mpg.csv"
    args = build_parser().parse_args(
        [
            "prepare-dataset",
            str(DATASET_ROOT / "raw" / "auto-mpg.data"),
            str(regenerated),
        ]
    )

    assert args.func(args) == 0
    assert regenerated.read_bytes() == PROCESSED_DATA.read_bytes()
    output = capsys.readouterr().out
    assert "rows=398" in output
    assert "sha256=b9b64882d6714f38fa5e368e855b1a25996b67ad9d053efde594fa28aa541607" in output


@pytest.mark.parametrize("row_count", [397, 399])
def test_public_preparation_requires_exactly_398_raw_rows(tmp_path: Path, row_count: int) -> None:
    source_lines = (DATASET_ROOT / "raw" / "auto-mpg.data").read_bytes().splitlines(keepends=True)
    raw_path = tmp_path / "auto-mpg.data"
    if row_count == 397:
        raw_path.write_bytes(b"".join(source_lines[:-1]))
    else:
        raw_path.write_bytes(b"".join([*source_lines, source_lines[-1]]))

    with pytest.raises(ValueError, match=f"expected 398 raw rows, got {row_count}"):
        prepare_auto_mpg(raw_path, tmp_path / "weight-mpg.csv")


def test_public_preparation_rejects_non_finite_weight_or_mpg(tmp_path: Path) -> None:
    source = (DATASET_ROOT / "raw" / "auto-mpg.data").read_text(encoding="ascii")
    raw_path = tmp_path / "auto-mpg.data"
    raw_path.write_text(source.replace("18.0", "nan ", 1), encoding="ascii")

    with pytest.raises(ValueError, match="raw row 1 has non-finite mpg or weight"):
        prepare_auto_mpg(raw_path, tmp_path / "weight-mpg.csv")


def test_processed_csv_is_stable_and_in_source_row_order() -> None:
    with PROCESSED_DATA.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert list(rows[0]) == ["sample_id", "weight", "mpg", "weight_standardized"]
    assert len(rows) == 398
    assert rows[0]["sample_id"] == "row-0001"
    assert float(rows[0]["weight"]) == 3504.0
    assert float(rows[0]["mpg"]) == 18.0
    assert rows[-1]["sample_id"] == "row-0398"
    assert float(rows[-1]["weight"]) == 2720.0
    assert float(rows[-1]["mpg"]) == 31.0


def test_loader_consumes_verified_processed_weight_and_mpg() -> None:
    dataset = load_auto_mpg(VERSION_MANIFEST)

    assert dataset.dataset_id == "uci-auto-mpg"
    assert dataset.version == "1.0.0"
    assert dataset.source_feature == "weight"
    assert dataset.feature == "weight_standardized"
    assert dataset.feature_unit == "population standard deviations"
    assert dataset.target == "mpg"
    assert dataset.target_unit == "miles per gallon"
    assert dataset.preprocessing == "population_standardization"
    assert dataset.split == "all-398-rows"
    assert dataset.sample_ids[0] == "row-0001"
    assert dataset.sample_ids[-1] == "row-0398"
    assert dataset.x.shape == dataset.y.shape == dataset.standardized_x.shape == (398,)
    assert dataset.x.dtype == dataset.y.dtype == dataset.standardized_x.dtype == np.float64
    assert np.isfinite(dataset.x).all()
    assert np.isfinite(dataset.y).all()
    assert np.isfinite(dataset.standardized_x).all()
    assert (dataset.x[0], dataset.y[0]) == pytest.approx((3504.0, 18.0))
    assert (dataset.x[-1], dataset.y[-1]) == pytest.approx((2720.0, 31.0))
    assert dataset.weight_mean == pytest.approx(2970.424623115578)
    assert dataset.weight_scale == pytest.approx(845.7772335198174)
    assert not dataset.x.flags.writeable
    assert not dataset.y.flags.writeable
    assert not dataset.standardized_x.flags.writeable


def test_weight_standardization_uses_pinned_processed_values() -> None:
    dataset = load_auto_mpg(VERSION_MANIFEST)
    prepared = standardize_weight(dataset)

    np.testing.assert_array_equal(prepared.x, dataset.standardized_x)
    np.testing.assert_array_equal(prepared.y, dataset.y)
    assert prepared.weight_mean == dataset.weight_mean
    assert prepared.weight_scale == dataset.weight_scale
    assert np.mean(prepared.x) == pytest.approx(0.0, abs=1e-12)
    assert np.std(prepared.x, ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_loader_rejects_processed_artifact_hash_mismatch(tmp_path: Path) -> None:
    manifest = _load_yaml(VERSION_MANIFEST)
    copied_manifest = _write_dataset_fixture(tmp_path, manifest)
    processed_data = (tmp_path / "auto-mpg" / "processed" / "weight-mpg.csv").read_bytes()
    corrupted_data = bytes([processed_data[0] ^ 1]) + processed_data[1:]
    (tmp_path / "auto-mpg" / "processed" / "weight-mpg.csv").write_bytes(corrupted_data)

    with pytest.raises(ValueError, match=r"SHA-256 mismatch for processed/weight-mpg\.csv"):
        load_auto_mpg(copied_manifest)


@pytest.mark.parametrize("field", ["dataset_id", "version"])
def test_loader_rejects_missing_manifest_identity(tmp_path: Path, field: str) -> None:
    manifest = _load_yaml(VERSION_MANIFEST)
    manifest.pop(field)
    copied_manifest = _write_dataset_fixture(tmp_path, manifest)

    with pytest.raises(ValueError, match=f"version manifest {field} must be a non-empty string"):
        load_auto_mpg(copied_manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dataset_id", 9), ("dataset_id", True), ("version", 1), ("version", False)],
)
def test_loader_rejects_non_string_manifest_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = _load_yaml(VERSION_MANIFEST)
    manifest[field] = value
    copied_manifest = _write_dataset_fixture(tmp_path, manifest)

    with pytest.raises(ValueError, match=f"version manifest {field} must be a non-empty string"):
        load_auto_mpg(copied_manifest)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("weight_mean", True),
        ("weight_std", False),
        ("weight_mean", float("nan")),
        ("weight_std", float("inf")),
    ],
)
def test_loader_rejects_bool_or_non_finite_preprocessing_parameters(
    tmp_path: Path, parameter: str, value: object
) -> None:
    manifest = _load_yaml(VERSION_MANIFEST)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    processed = next(artifact for artifact in artifacts if artifact["role"] == "processed")
    processed["preprocessing"][parameter] = value
    copied_manifest = _write_dataset_fixture(tmp_path, manifest)

    with pytest.raises(ValueError, match=f"processed artifact {parameter} must be a finite number"):
        load_auto_mpg(copied_manifest)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("dataset_id", "other-dataset", "dataset_id must be 'uci-auto-mpg'"),
        ("selection.feature", "horsepower", "selection must be weight -> mpg"),
        ("selection.target", "horsepower", "selection must be weight -> mpg"),
        ("split.strategy", "train-test", "split must be none/all-398-rows"),
        ("split.identity", "first-300-rows", "split must be none/all-398-rows"),
        ("processed.schema", [], "processed artifact schema does not match"),
        (
            "processed.preprocessing.source_artifact",
            "raw/other.data",
            "source_artifact must identify a verified raw_data artifact",
        ),
        (
            "processed.preprocessing.formula",
            "weight / weight_std",
            "preprocessing formula must be",
        ),
        ("processed.preprocessing.standard_deviation_ddof", 1, "DDOF must be 0"),
        ("processed.preprocessing.numeric_type", "float32", "numeric_type must be 'float64'"),
    ],
)
def test_run_dataset_rejects_contradictory_manifest_before_creating_runs_root(
    tmp_path: Path, path: str, value: object, message: str
) -> None:
    manifest = _load_yaml(VERSION_MANIFEST)
    _set_manifest_value(manifest, path, value)
    version_manifest = _write_dataset_fixture(tmp_path, manifest)
    config_path = _write_external_run_config(tmp_path, version_manifest)
    runs_root = tmp_path / "runs"

    with pytest.raises(ValueError, match=message):
        run_dataset(config_path, runs_root)

    assert not runs_root.exists()


@pytest.mark.parametrize(("field", "value"), [("dataset", "other"), ("version", "2.0.0")])
def test_run_dataset_rejects_config_identity_mismatch_before_creating_runs_root(
    tmp_path: Path, field: str, value: str
) -> None:
    config = _load_yaml(CONFIG_PATH)
    data = config["data"]
    assert isinstance(data, dict)
    data[field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runs_root = tmp_path / "runs"

    with pytest.raises(ValueError):
        run_dataset(config_path, runs_root)

    assert not runs_root.exists()


def test_external_dataset_config_is_executable_without_creating_a_replay_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = train_dataset(CONFIG_PATH)

    assert result.dataset_id == "uci-auto-mpg"
    assert result.dataset_version == "1.0.0"
    assert result.sample_count == 398
    assert result.b == pytest.approx(result.reference_b, abs=1e-10)
    assert result.w == pytest.approx(result.reference_w, abs=1e-10)
    assert result.train_mse == pytest.approx(result.reference_mse, abs=1e-10)

    args = build_parser().parse_args(["train-dataset", str(CONFIG_PATH)])
    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert "dataset uci-auto-mpg@1.0.0" in output
    assert "samples=398" in output
    assert "reference_mse=" in output


def test_external_dataset_run_records_and_exports_schema_v2(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    manifest = run_dataset(CONFIG_PATH, runs_root)

    assert manifest.status == "completed"
    assert manifest.schema_version == 2
    run_dir = runs_root / manifest.run_id
    manifest_json = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["dataConfig"] == {
        "source": "external_dataset",
        "datasetId": "uci-auto-mpg",
        "datasetVersion": "1.0.0",
        "versionManifestSha256": _sha256(VERSION_MANIFEST),
        "processedArtifactSha256": _sha256(PROCESSED_DATA),
        "sourceFeature": "weight",
        "feature": "weight_standardized",
        "featureUnit": "population standard deviations",
        "target": "mpg",
        "targetUnit": "miles per gallon",
        "preprocessing": "population_standardization",
        "split": "all-398-rows",
    }
    assert not {"trueBias", "trueWeight", "noiseStd", "seed"} & manifest_json["dataConfig"].keys()
    assert manifest_json["dataset"]["sourceId"].startswith("uci-auto-mpg@1.0.0")
    assert "processed-sha256=" in manifest_json["dataset"]["sourceId"]
    assert "generatorId" not in manifest_json["dataset"]

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {event["schemaVersion"] for event in events} == {2}

    x = np.asarray(manifest_json["dataset"]["x"], dtype=np.float64)
    y = np.asarray(manifest_json["dataset"]["y"], dtype=np.float64)
    snapshots = json.loads((run_dir / "snapshots.json").read_text(encoding="utf-8"))
    assert x.shape == y.shape == (398,)
    assert x == pytest.approx(load_auto_mpg(VERSION_MANIFEST).standardized_x)
    for snapshot in snapshots:
        assert mse(snapshot["b"], snapshot["w"], x, y) == pytest.approx(
            snapshot["trainMse"], abs=1e-9
        )
        predictions = predict(snapshot["b"], snapshot["w"], x)
        for sample_id, recorded in snapshot["observedPredictions"].items():
            index = manifest.dataset.sample_ids.index(sample_id)
            assert predictions[index] == pytest.approx(recorded, abs=1e-9)

    exported = export_run(run_dir, tmp_path / "exported")
    assert exported.schema_version == 2


def test_export_rejects_schema_v2_with_synthetic_dataset_summary(tmp_path: Path) -> None:
    manifest = run_dataset(CONFIG_PATH, tmp_path / "runs")
    run_dir = tmp_path / "runs" / manifest.run_id
    manifest_path = run_dir / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_json["dataset"] = {
        "generatorId": manifest_json["dataset"].pop("sourceId"),
        **manifest_json["dataset"],
    }
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(ExportError, match="schemaVersion 2 requires external dataset summary"):
        export_run(run_dir, tmp_path / "exported")


def test_export_rejects_external_data_config_with_schema_v1(tmp_path: Path) -> None:
    manifest = run_dataset(CONFIG_PATH, tmp_path / "runs")
    run_dir = tmp_path / "runs" / manifest.run_id
    manifest_path = run_dir / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_json["schemaVersion"] = 1
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(ExportError, match="schemaVersion 1 requires synthetic"):
        export_run(run_dir, tmp_path / "exported")


def test_run_dataset_rejects_invalid_model_before_creating_artifacts(tmp_path: Path) -> None:
    config = _load_yaml(CONFIG_PATH)
    model = config["model"]
    assert isinstance(model, dict)
    model["learning_rate"] = -0.1
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runs_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        run_dataset(config_path, runs_root)

    assert not runs_root.exists()


def test_run_dataset_cli_returns_normal_run_status_and_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_parser().parse_args(
        ["run-dataset", str(CONFIG_PATH), "--runs-root", str(tmp_path / "runs")]
    )

    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert output.startswith("run auto-mpg-weight-")
    assert "status=completed stop_reason=max_steps" in output


def test_gradient_descent_matches_independent_least_squares_reference() -> None:
    config = _load_yaml(CONFIG_PATH)
    data_config = config["data"]
    model = config["model"]
    assert isinstance(data_config, dict)
    assert isinstance(model, dict)
    dataset = load_auto_mpg(REPO_ROOT / data_config["version_manifest"])
    prepared = standardize_weight(dataset)

    states = fit(
        prepared.x,
        prepared.y,
        learning_rate=model["learning_rate"],
        n_updates=model["n_updates"],
        b0=model["initial_bias"],
        w0=model["initial_weight"],
    )

    x_mean = float(np.mean(prepared.x))
    y_mean = float(np.mean(prepared.y))
    centered_x = prepared.x - x_mean
    centered_y = prepared.y - y_mean
    reference_w = float(np.sum(centered_x * centered_y) / np.sum(centered_x**2))
    reference_b = y_mean - reference_w * x_mean
    final = states[-1]

    assert final.b == pytest.approx(reference_b, abs=1e-10)
    assert final.w == pytest.approx(reference_w, abs=1e-10)
    assert final.mse == pytest.approx(
        mse(reference_b, reference_w, prepared.x, prepared.y), abs=1e-10
    )
