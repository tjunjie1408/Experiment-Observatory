"""Pinned UCI WDBC data and deterministic, model-independent preparation."""

from __future__ import annotations

import csv
import hashlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = tuple(
    f"{name}_{group}"
    for group in ("mean", "se", "worst")
    for name in (
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    )
)
SPLIT_STRATEGY = "stratified-sha256-2026-v1"
_ARTIFACTS = {
    "raw/wdbc.data": "raw_data",
    "raw/wdbc.names": "source_description",
    "processed/wdbc.csv": "processed",
    "processed/split.csv": "split",
}


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    role: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Version(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: Literal["uci-wdbc"]
    version: Literal["1.0.0"]
    source_url: Literal[
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    ]
    description_url: Literal[
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"
    ]
    retrieved_at: str
    row_count: Literal[569]
    feature_names: list[str]
    target_mapping: dict[str, int]
    preprocessing: Literal["none"]
    split_strategy: Literal["stratified-sha256-2026-v1"]
    train_count: Literal[397]
    validation_count: Literal[172]
    serialization: Literal["utf8-lf-id-sorted-float17g-v1"]
    artifacts: list[_Artifact]


@dataclass(frozen=True)
class DatasetPartition:
    sample_ids: tuple[str, ...]
    features: npt.NDArray[np.float64]
    targets: npt.NDArray[np.int64]


@dataclass(frozen=True)
class BreastCancerDataset:
    dataset_id: str
    version: str
    feature_names: tuple[str, ...]
    version_manifest_sha256: str
    processed_artifact_sha256: str
    split_sha256: str
    train: DatasetPartition
    validation: DatasetPartition
    observed_sample_ids: tuple[str, ...]


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_breast_cancer(raw: bytes) -> dict[str, bytes]:
    """Pure preparation: validate official rows, return canonical CSV bytes.

    This accepts reordered raw rows for reproducibility tests. The public
    preparation/loader entry points additionally require pinned source hashes.
    """
    try:
        records = list(csv.reader(io.StringIO(raw.decode("ascii"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"invalid raw WDBC CSV: {exc}") from exc
    if len(records) != 569:
        raise ValueError("expected 569 raw WDBC rows")
    rows: dict[str, tuple[int, list[float]]] = {}
    for index, fields in enumerate(records, start=1):
        if len(fields) != 32:
            raise ValueError(f"raw row {index}: expected 32 fields")
        source_id, label = fields[:2]
        if not source_id or not source_id.isascii() or not source_id.isdigit():
            raise ValueError(f"raw row {index}: invalid source ID")
        sample_id = f"wdbc-{source_id}"
        if sample_id in rows:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        if label not in {"B", "M"}:
            raise ValueError(f"raw row {index}: unknown diagnosis")
        try:
            values = [float(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f"raw row {index}: invalid feature") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"raw row {index}: non-finite feature")
        rows[sample_id] = (int(label == "M"), values)
    class_ids = [[key for key, (target, _) in rows.items() if target == c] for c in (0, 1)]
    if [len(ids) for ids in class_ids] != [357, 212]:
        raise ValueError("expected class counts B=357, M=212")
    train_ids: set[str] = set()
    for ids in class_ids:
        ordered = sorted(ids, key=lambda key: (_digest(f"uci-wdbc:2026:{key}".encode()), key))
        train_ids.update(ordered[: len(ordered) * 7 // 10])
    data_lines = [",".join(("sample_id", "target", *FEATURE_NAMES))]
    split_lines = ["sample_id,split"]
    for sample_id in sorted(rows):
        target, features = rows[sample_id]
        data_lines.append(
            ",".join((sample_id, str(target), *(format(value, ".17g") for value in features)))
        )
        split_lines.append(f"{sample_id},{'train' if sample_id in train_ids else 'validation'}")
    return {
        "wdbc.csv": ("\n".join(data_lines) + "\n").encode("utf-8"),
        "split.csv": ("\n".join(split_lines) + "\n").encode("utf-8"),
    }


def _read_manifest(path: Path) -> tuple[_Version, str]:
    try:
        payload = path.read_bytes()
        version = _Version.model_validate(yaml.safe_load(payload))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid WDBC version manifest: {exc}") from exc
    if version.feature_names != list(FEATURE_NAMES):
        raise ValueError("manifest feature order does not match WDBC")
    if version.target_mapping != {"B": 0, "M": 1}:
        raise ValueError("manifest target mapping must be B=0, M=1")
    if (
        len(version.artifacts) != len(_ARTIFACTS)
        or {artifact.path: artifact.role for artifact in version.artifacts} != _ARTIFACTS
    ):
        raise ValueError("manifest must identify exactly the four canonical artifacts")
    return version, _digest(payload)


def _verify_payload(artifact: _Artifact, payload: bytes) -> None:
    if len(payload) != artifact.bytes:
        raise ValueError(f"artifact byte-size mismatch: {artifact.path}")
    if _digest(payload) != artifact.sha256:
        raise ValueError(f"artifact SHA-256 mismatch: {artifact.path}")


def _read_artifact(root: Path, artifact: _Artifact) -> bytes:
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes dataset root")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read artifact {artifact.path}: {exc}") from exc
    _verify_payload(artifact, payload)
    return payload


def _canonical_outputs(root: Path, version: _Version) -> dict[str, bytes]:
    raw = {a.path: _read_artifact(root, a) for a in version.artifacts if a.path.startswith("raw/")}
    return render_breast_cancer(raw["raw/wdbc.data"])


def prepare_breast_cancer(version_manifest: Path, output_dir: Path) -> dict[str, str]:
    """Verify source pins and expected output hashes before writing any output.

    Each file is replaced atomically. A crash between the two replacements is
    detectable by load_breast_cancer; this is not a multi-file transaction.
    """
    version, _ = _read_manifest(version_manifest)
    root = version_manifest.resolve().parent.parent
    output = output_dir.resolve()
    if output == root / "raw" or output.is_relative_to(root / "raw"):
        raise ValueError("output must not overwrite raw source data")
    rendered = _canonical_outputs(root, version)
    for artifact in version.artifacts:
        if artifact.path.startswith("processed/"):
            _verify_payload(artifact, rendered[Path(artifact.path).name])
    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".wdbc-", dir=output) as temporary:
            for name, payload in rendered.items():
                staged = Path(temporary) / name
                staged.write_bytes(payload)
            for name in rendered:
                (Path(temporary) / name).replace(output / name)
    except OSError as exc:
        raise ValueError(f"could not write prepared WDBC artifacts: {exc}") from exc
    return {name: _digest(payload) for name, payload in rendered.items()}


def load_breast_cancer(version_manifest: Path) -> BreastCancerDataset:
    """Verify pins and independently reconstruct prepared artifacts before use."""
    version, manifest_hash = _read_manifest(version_manifest)
    root = version_manifest.resolve().parent.parent
    payloads = {a.path: _read_artifact(root, a) for a in version.artifacts}
    canonical = render_breast_cancer(payloads["raw/wdbc.data"])
    for name, expected in canonical.items():
        if payloads[f"processed/{name}"] != expected:
            raise ValueError(f"processed {name} does not match canonical preparation")
    rows = list(csv.reader(io.StringIO(canonical["wdbc.csv"].decode())))[1:]
    split_rows = list(csv.reader(io.StringIO(canonical["split.csv"].decode())))[1:]
    partitions: dict[str, DatasetPartition] = {}
    observed: list[str] = []
    for split in ("train", "validation"):
        selected = [row for row, member in zip(rows, split_rows, strict=True) if member[1] == split]
        ids = tuple(row[0] for row in selected)
        x = np.array([[float(v) for v in row[2:]] for row in selected], dtype=np.float64)
        y = np.array([int(row[1]) for row in selected], dtype=np.int64)
        x.flags.writeable = False
        y.flags.writeable = False
        partitions[split] = DatasetPartition(ids, x, y)
        for target in (0, 1):
            observed.extend([key for key, label in zip(ids, y, strict=True) if label == target][:5])
    return BreastCancerDataset(
        dataset_id=version.dataset_id,
        version=version.version,
        feature_names=FEATURE_NAMES,
        version_manifest_sha256=manifest_hash,
        processed_artifact_sha256=_digest(canonical["wdbc.csv"]),
        split_sha256=_digest(canonical["split.csv"]),
        train=partitions["train"],
        validation=partitions["validation"],
        observed_sample_ids=tuple(sorted(observed)),
    )
