from __future__ import annotations

from pathlib import Path

import pytest

from observatory.cli import build_parser
from observatory.kmeans_study import load_kmeans_study_config, run_kmeans_study


def test_study_generates_five_completed_runs_from_same_dataset(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "kmeans" / "initialization_study.yaml"

    manifests = run_kmeans_study(config, tmp_path / "runs", tmp_path)

    assert [manifest.training_config.init_seed for manifest in manifests] == [0, 1, 2, 3, 4]
    assert all(manifest.status == "completed" for manifest in manifests)
    assert len({tuple(map(tuple, manifest.dataset.points)) for manifest in manifests}) == 1
    assert len(list((tmp_path / "runs").iterdir())) == 5


def test_study_config_requires_exactly_five_unique_initializations(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """experiment_id: invalid-study
blob_centers: [[-1.0, 0.0], [1.0, 0.0]]
blob_sizes: [5, 5]
cluster_std: 0.2
data_seed: 1
n_clusters: 2
init_seeds: [0, 1, 2, 3, 3]
max_iterations: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="init_seeds must be unique"):
        load_kmeans_study_config(config)

    config.write_text(
        config.read_text(encoding="utf-8").replace("[0, 1, 2, 3, 3]", "[0, 1]"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least 5 items"):
        load_kmeans_study_config(config)


def test_cli_kmeans_study_command_runs_five_experiments(tmp_path: Path) -> None:
    parser = build_parser()
    config = Path(__file__).parents[1] / "configs" / "kmeans" / "initialization_study.yaml"
    runs_root = tmp_path / "runs"

    args = parser.parse_args(["run-kmeans-study", str(config), "--runs-root", str(runs_root)])

    assert args.func(args) == 0
    assert len(list(runs_root.iterdir())) == 5
