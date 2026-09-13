from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from observatory.data.kmeans_synthetic import KMeansSyntheticConfig
from observatory.runtime.export import export_run
from observatory.runtime.kmeans_record import run_kmeans_experiment
from observatory.runtime.schema import KMeansRunManifest, KMeansSnapshot


def test_kmeans_run_writes_consistent_schema_v3_bundle(tmp_path: Path) -> None:
    manifest = run_kmeans_experiment(
        experiment_id="kmeans-seed-0",
        data_config=KMeansSyntheticConfig(
            blob_centers=((-4.0, -2.0), (-4.0, 2.0), (4.0, -2.0), (4.0, 2.0)),
            blob_sizes=(20, 20, 20, 10),
            cluster_std=0.55,
            seed=2026,
        ),
        n_clusters=3,
        init_seed=0,
        max_iterations=20,
        runs_root=tmp_path,
        repo_root=tmp_path,
    )
    run_dir = tmp_path / manifest.run_id

    persisted = KMeansRunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    snapshots = [
        KMeansSnapshot.model_validate(item)
        for item in json.loads((run_dir / "snapshots.json").read_text(encoding="utf-8"))
    ]
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    assert persisted.schema_version == 3
    assert persisted.status == "completed"
    assert persisted.n_snapshots_written == len(snapshots)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["kind"] == "run.completed"
    points = np.asarray(persisted.dataset.points)
    for snapshot in snapshots:
        centers = np.asarray(snapshot.centers)
        labels = np.asarray(snapshot.assignments)
        expected = float(np.sum((points - centers[labels]) ** 2))
        assert snapshot.inertia == expected

    exported = export_run(run_dir, tmp_path / "exported")
    assert exported.schema_version == 3
    assert (tmp_path / "exported" / "manifest.json").is_file()
