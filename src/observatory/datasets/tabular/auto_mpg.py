"""Verified loader and deterministic preparation for UCI Auto MPG weight vs. mpg."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import yaml

FloatArray = npt.NDArray[np.float64]

_DATASET_ID: Final = "uci-auto-mpg"
_SOURCE_FEATURE: Final = "weight"
_MODEL_FEATURE: Final = "weight_standardized"
_FEATURE_UNIT: Final = "population standard deviations"
_TARGET: Final = "mpg"
_TARGET_UNIT: Final = "miles per gallon"
_PREPROCESSING_IDENTITY: Final = "population_standardization"
_SPLIT_IDENTITY: Final = "all-398-rows"
_EXPECTED_PROCESSED_SCHEMA: list[dict[str, object]] = [
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


@dataclass(frozen=True)
class AutoMpgDataset:
    dataset_id: str
    version: str
    version_manifest_sha256: str
    processed_artifact_sha256: str
    source_id: str
    source_feature: Literal["weight"]
    feature: Literal["weight_standardized"]
    feature_unit: Literal["population standard deviations"]
    target: Literal["mpg"]
    target_unit: Literal["miles per gallon"]
    preprocessing: Literal["population_standardization"]
    split_strategy: Literal["none"]
    split: Literal["all-398-rows"]
    sample_ids: tuple[str, ...]
    x: FloatArray
    y: FloatArray
    standardized_x: FloatArray
    weight_mean: float
    weight_scale: float


@dataclass(frozen=True)
class PreparedAutoMpgDataset:
    sample_ids: tuple[str, ...]
    x: FloatArray
    y: FloatArray
    weight_mean: float
    weight_scale: float


@dataclass(frozen=True)
class AutoMpgPreparationResult:
    row_count: int
    weight_mean: float
    weight_std: float
    bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(dataset_root: Path, relative_path: str) -> Path:
    path = (dataset_root / relative_path).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes dataset root: {relative_path}") from exc
    return path


def _load_version_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read version manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("version manifest must contain a YAML mapping")
    return manifest


def prepare_auto_mpg(raw_path: Path, output_path: Path) -> AutoMpgPreparationResult:
    """Create the canonical weight/mpg CSV from the pinned UCI raw data file."""
    try:
        lines = raw_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read raw Auto MPG data {raw_path}: {exc}") from exc
    if len(lines) != 398:
        raise ValueError(f"expected 398 raw rows, got {len(lines)}")

    weights: list[float] = []
    mpg_values: list[float] = []
    for row_number, line in enumerate(lines, start=1):
        fields = line.split(maxsplit=8)
        if len(fields) != 9:
            raise ValueError(f"raw row {row_number} has {len(fields)} fields; expected 9")
        try:
            mpg = float(fields[0])
            weight = float(fields[4])
        except ValueError as exc:
            raise ValueError(f"raw row {row_number} has invalid mpg or weight") from exc
        if not np.isfinite(mpg) or not np.isfinite(weight):
            raise ValueError(f"raw row {row_number} has non-finite mpg or weight")
        weights.append(weight)
        mpg_values.append(mpg)

    x = np.asarray(weights, dtype=np.float64)
    y = np.asarray(mpg_values, dtype=np.float64)
    weight_mean = float(np.mean(x))
    weight_std = float(np.std(x, ddof=0))
    output_lines = ["sample_id,weight,mpg,weight_standardized"]
    output_lines.extend(
        f"row-{row_number:04d},{weight:.17g},{mpg:.17g},"
        f"{((weight - weight_mean) / weight_std):.17g}"
        for row_number, (weight, mpg) in enumerate(zip(x, y, strict=True), start=1)
    )
    output = ("\n".join(output_lines) + "\n").encode("utf-8")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output)
    except OSError as exc:
        raise ValueError(f"could not write processed Auto MPG data {output_path}: {exc}") from exc

    return AutoMpgPreparationResult(
        row_count=len(lines),
        weight_mean=weight_mean,
        weight_std=weight_std,
        bytes=len(output),
        sha256=hashlib.sha256(output).hexdigest(),
    )


def load_auto_mpg(version_manifest: Path) -> AutoMpgDataset:
    """Load selected columns after verifying every pinned source artifact."""
    manifest_path = version_manifest.resolve()
    manifest = _load_version_manifest(manifest_path)
    dataset_root = manifest_path.parent.parent

    dataset_id = manifest.get("dataset_id")
    version = manifest.get("version")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("version manifest dataset_id must be a non-empty string")
    if dataset_id != _DATASET_ID:
        raise ValueError(f"version manifest dataset_id must be {_DATASET_ID!r}")
    if not isinstance(version, str) or not version:
        raise ValueError("version manifest version must be a non-empty string")

    selection = manifest.get("selection")
    if selection != {"feature": _SOURCE_FEATURE, "target": _TARGET}:
        raise ValueError("version manifest selection must be weight -> mpg")
    split = manifest.get("split")
    if split != {"strategy": "none", "identity": _SPLIT_IDENTITY}:
        raise ValueError("version manifest split must be none/all-398-rows")

    row_count = manifest.get("row_count")
    if isinstance(row_count, bool) or row_count != 398:
        raise ValueError("version manifest row_count must be 398")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("version manifest artifacts must be a list")

    processed_path: Path | None = None
    processed_metadata: dict[str, Any] | None = None
    verified_artifact_roles: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("each version manifest artifact must be an object")
        relative_path = artifact.get("path")
        expected_hash = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError("artifact path and sha256 must be strings")
        if not isinstance(expected_bytes, int):
            raise ValueError("artifact bytes must be an integer")
        role = artifact.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("artifact role must be a non-empty string")
        if relative_path in verified_artifact_roles:
            raise ValueError(f"duplicate artifact path: {relative_path}")

        path = _artifact_path(dataset_root, relative_path)
        try:
            actual_bytes = path.stat().st_size
            actual_hash = _sha256(path)
        except OSError as exc:
            raise ValueError(f"could not read artifact {relative_path}: {exc}") from exc
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"byte-size mismatch for {relative_path}: expected {expected_bytes}, "
                f"got {actual_bytes}"
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        verified_artifact_roles[relative_path] = role
        if role == "processed":
            if processed_path is not None:
                raise ValueError("version manifest identifies multiple processed artifacts")
            processed_path = path
            processed_metadata = artifact

    if processed_path is None or processed_metadata is None:
        raise ValueError("version manifest does not identify a processed artifact")

    if processed_metadata.get("parent_version") != version:
        raise ValueError("processed artifact parent_version does not match manifest version")
    if processed_metadata.get("row_count") != row_count:
        raise ValueError("processed artifact row_count does not match manifest row_count")

    if processed_metadata.get("schema") != _EXPECTED_PROCESSED_SCHEMA:
        raise ValueError("processed artifact schema does not match the Auto MPG contract")

    preprocessing = processed_metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("processed artifact preprocessing must be a mapping")
    source_artifact = preprocessing.get("source_artifact")
    if (
        not isinstance(source_artifact, str)
        or verified_artifact_roles.get(source_artifact) != "raw_data"
    ):
        raise ValueError(
            "processed artifact source_artifact must identify a verified raw_data artifact"
        )
    if preprocessing.get("row_order") != "source row order":
        raise ValueError("processed artifact row_order must be 'source row order'")
    if preprocessing.get("sample_id") != "row-{1-based source row number, zero-padded to 4 digits}":
        raise ValueError("processed artifact sample_id identity is unsupported")
    if preprocessing.get("formula") != "(weight - weight_mean) / weight_std":
        raise ValueError(
            "processed artifact preprocessing formula must be '(weight - weight_mean) / weight_std'"
        )
    ddof = preprocessing.get("standard_deviation_ddof")
    if isinstance(ddof, bool) or ddof != 0:
        raise ValueError("processed artifact preprocessing DDOF must be 0")
    if preprocessing.get("target") != "unchanged mpg":
        raise ValueError("processed artifact preprocessing target must be 'unchanged mpg'")
    if preprocessing.get("numeric_type") != "float64":
        raise ValueError("processed artifact preprocessing numeric_type must be 'float64'")

    weight_mean_value = preprocessing.get("weight_mean")
    weight_scale_value = preprocessing.get("weight_std")
    if (
        isinstance(weight_mean_value, bool)
        or not isinstance(weight_mean_value, int | float)
        or not np.isfinite(weight_mean_value)
    ):
        raise ValueError("processed artifact weight_mean must be a finite number")
    if (
        isinstance(weight_scale_value, bool)
        or not isinstance(weight_scale_value, int | float)
        or not np.isfinite(weight_scale_value)
    ):
        raise ValueError("processed artifact weight_std must be a finite number")
    weight_mean = float(weight_mean_value)
    weight_scale = float(weight_scale_value)
    if weight_scale <= 0:
        raise ValueError("processed artifact weight_std must be positive")

    expected_header = ["sample_id", "weight", "mpg", "weight_standardized"]
    try:
        with processed_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != expected_header:
                raise ValueError(f"processed artifact header must be {expected_header}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"could not parse processed Auto MPG data: {exc}") from exc
    if len(rows) != row_count:
        raise ValueError(f"expected {row_count} processed rows, got {len(rows)}")

    sample_ids: list[str] = []
    weights: list[float] = []
    mpg_values: list[float] = []
    standardized_weights: list[float] = []
    for row_number, row in enumerate(rows, start=1):
        expected_sample_id = f"row-{row_number:04d}"
        if row["sample_id"] != expected_sample_id:
            raise ValueError(f"processed row {row_number} sample_id must be {expected_sample_id}")
        try:
            weight = float(row["weight"])
            mpg = float(row["mpg"])
            standardized_weight = float(row["weight_standardized"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"processed row {row_number} has invalid numeric data") from exc
        if not np.isfinite([weight, mpg, standardized_weight]).all():
            raise ValueError(f"processed row {row_number} has non-finite numeric data")
        sample_ids.append(expected_sample_id)
        weights.append(weight)
        mpg_values.append(mpg)
        standardized_weights.append(standardized_weight)

    x = np.asarray(weights, dtype=np.float64)
    y = np.asarray(mpg_values, dtype=np.float64)
    standardized_x = np.asarray(standardized_weights, dtype=np.float64)
    expected_standardized_x = (x - weight_mean) / weight_scale
    if not np.array_equal(standardized_x, expected_standardized_x):
        raise ValueError("processed standardized weights do not match manifest parameters")
    if not np.isclose(np.mean(x), weight_mean, rtol=0.0, atol=1e-12):
        raise ValueError("processed weights do not match manifest weight_mean")
    if not np.isclose(np.std(x, ddof=0), weight_scale, rtol=0.0, atol=1e-12):
        raise ValueError("processed weights do not match manifest weight_std")

    x.flags.writeable = False
    y.flags.writeable = False
    standardized_x.flags.writeable = False
    processed_hash = str(processed_metadata["sha256"])
    return AutoMpgDataset(
        dataset_id=dataset_id,
        version=version,
        version_manifest_sha256=_sha256(manifest_path),
        processed_artifact_sha256=processed_hash,
        source_id=f"{dataset_id}@{version};processed-sha256={processed_hash}",
        source_feature=_SOURCE_FEATURE,
        feature=_MODEL_FEATURE,
        feature_unit=_FEATURE_UNIT,
        target=_TARGET,
        target_unit=_TARGET_UNIT,
        preprocessing=_PREPROCESSING_IDENTITY,
        split_strategy="none",
        split=_SPLIT_IDENTITY,
        sample_ids=tuple(sample_ids),
        x=x,
        y=y,
        standardized_x=standardized_x,
        weight_mean=weight_mean,
        weight_scale=weight_scale,
    )


def standardize_weight(dataset: AutoMpgDataset) -> PreparedAutoMpgDataset:
    """Apply the manifest's population standardization without changing row order."""
    x = np.array(dataset.standardized_x, dtype=np.float64, copy=True)
    y = np.array(dataset.y, dtype=np.float64, copy=True)
    x.flags.writeable = False
    y.flags.writeable = False
    return PreparedAutoMpgDataset(
        sample_ids=dataset.sample_ids,
        x=x,
        y=y,
        weight_mean=dataset.weight_mean,
        weight_scale=dataset.weight_scale,
    )
