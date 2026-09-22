from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from observatory.cli import build_parser
from observatory.datasets.tabular.breast_cancer import (
    FEATURE_NAMES,
    load_breast_cancer,
    prepare_breast_cancer,
    render_breast_cancer,
)

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "breast-cancer"
MANIFEST = ROOT / "versions" / "1.0.0.yaml"


def test_real_dataset_partitions_and_feature_identity() -> None:
    dataset = load_breast_cancer(MANIFEST)
    assert dataset.train.features.shape == (397, 30)
    assert dataset.validation.features.shape == (172, 30)
    assert np.bincount(dataset.train.targets).tolist() == [249, 148]
    assert np.bincount(dataset.validation.targets).tolist() == [108, 64]
    assert not set(dataset.train.sample_ids) & set(dataset.validation.sample_ids)
    assert len(set(dataset.train.sample_ids) | set(dataset.validation.sample_ids)) == 569
    assert len(dataset.observed_sample_ids) == 20
    assert FEATURE_NAMES[:2] == ("radius_mean", "texture_mean")
    assert FEATURE_NAMES[-1] == "fractal_dimension_worst"
    # Official first record; locate by ID rather than relying on source ordering.
    partition = next(
        p for p in (dataset.train, dataset.validation) if "wdbc-842302" in p.sample_ids
    )
    index = partition.sample_ids.index("wdbc-842302")
    assert partition.targets[index] == 1
    assert partition.features[index, :3].tolist() == [17.99, 10.38, 122.8]
    assert not partition.features.flags.writeable
    assert not partition.targets.flags.writeable
    assert (
        dataset.split_sha256
        == hashlib.sha256((ROOT / "processed/split.csv").read_bytes()).hexdigest()
    )


def test_prepare_reproduces_pinned_bytes_and_ignores_raw_row_order(tmp_path: Path) -> None:
    prepare_breast_cancer(MANIFEST, tmp_path / "prepared")
    raw = (ROOT / "raw/wdbc.data").read_bytes()
    reversed_raw = b"\n".join(reversed(raw.splitlines())) + b"\n"
    rendered = render_breast_cancer(reversed_raw)
    for name in ("wdbc.csv", "split.csv"):
        expected = (ROOT / "processed" / name).read_bytes()
        assert (tmp_path / "prepared" / name).read_bytes() == expected
        assert rendered[name] == expected
        assert b"\r" not in expected
        assert expected.endswith(b"\n")


@pytest.mark.parametrize("mutation", ["duplicate", "label", "nan", "width", "id", "count"])
def test_malformed_raw_is_rejected(mutation: str) -> None:
    rows = (ROOT / "raw/wdbc.data").read_text(encoding="ascii").splitlines()
    fields = rows[0].split(",")
    if mutation == "duplicate":
        fields[0] = rows[1].split(",")[0]
    elif mutation == "label":
        fields[1] = "X"
    elif mutation == "nan":
        fields[2] = "nan"
    elif mutation == "width":
        fields.pop()
    elif mutation == "id":
        fields[0] = "not-an-id"
    else:
        rows.pop()
    rows[0] = ",".join(fields)
    with pytest.raises(ValueError):
        render_breast_cancer(("\n".join(rows) + "\n").encode("ascii"))


@pytest.fixture
def copied_dataset(tmp_path: Path) -> Path:
    target = tmp_path / "breast-cancer"
    shutil.copytree(ROOT, target)
    return target / "versions/1.0.0.yaml"


@pytest.mark.parametrize(
    "path", ["raw/wdbc.data", "raw/wdbc.names", "processed/wdbc.csv", "processed/split.csv"]
)
def test_loader_rejects_corrupted_or_missing_artifacts(copied_dataset: Path, path: str) -> None:
    artifact = copied_dataset.parent.parent / path
    original = artifact.read_bytes()
    artifact.write_bytes(b"x" + original[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        load_breast_cancer(copied_dataset)
    artifact.unlink()
    with pytest.raises(ValueError, match="artifact"):
        load_breast_cancer(copied_dataset)


def test_rehashed_split_tampering_still_rejected(copied_dataset: Path) -> None:
    manifest = yaml.safe_load(copied_dataset.read_text())
    path = copied_dataset.parent.parent / "processed/split.csv"
    payload = path.read_bytes().replace(b",train\n", b",validation\n", 1)
    path.write_bytes(payload)
    entry = next(a for a in manifest["artifacts"] if a["path"] == "processed/split.csv")
    entry.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    copied_dataset.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_breast_cancer(copied_dataset)


@pytest.mark.parametrize("mutation", ["features", "mapping", "split", "path", "duplicate"])
def test_manifest_contract_tampering_rejected(copied_dataset: Path, mutation: str) -> None:
    manifest = yaml.safe_load(copied_dataset.read_text())
    if mutation == "features":
        manifest["feature_names"].reverse()
    elif mutation == "mapping":
        manifest["target_mapping"] = {"B": 1, "M": 0}
    elif mutation == "split":
        manifest["split_strategy"] = "random"
    elif mutation == "path":
        manifest["artifacts"][0]["path"] = "../../outside.data"
    else:
        manifest["artifacts"].append(manifest["artifacts"][0])
    copied_dataset.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_breast_cancer(copied_dataset)


def test_cli_prepares_and_rejects_bad_source_without_output(
    copied_dataset: Path, tmp_path: Path
) -> None:
    parser = build_parser()
    output = tmp_path / "output"
    args = parser.parse_args(["prepare-wdbc", str(copied_dataset), str(output)])
    assert args.func(args) == 0
    assert (output / "split.csv").is_file()
    (copied_dataset.parent.parent / "raw/wdbc.data").write_bytes(b"invalid")
    other = tmp_path / "rejected"
    args = parser.parse_args(["prepare-wdbc", str(copied_dataset), str(other)])
    assert args.func(args) == 2
    assert not other.exists()
