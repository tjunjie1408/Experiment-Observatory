"""Schema-v3 artifact recording for the M5 K-means experiment unit."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from observatory.datasets.synthetic.blobs import KMeansSyntheticConfig, generate_kmeans
from observatory.models.kmeans.lloyd import initialize_from_samples, iter_lloyd
from observatory.runtime.schema import (
    KMeansConfig,
    KMeansDataConfig,
    KMeansDatasetSummary,
    KMeansEvent,
    KMeansRunManifest,
    KMeansSnapshot,
)
from observatory.runtime.storage import (
    append_jsonl,
    atomic_write_json,
    get_code_provenance,
    new_run_id,
)


def run_kmeans_experiment(
    *,
    experiment_id: str,
    data_config: KMeansSyntheticConfig,
    n_clusters: int,
    init_seed: int,
    max_iterations: int,
    runs_root: Path,
    repo_root: Path,
) -> KMeansRunManifest:
    dataset = generate_kmeans(data_config)
    initial = initialize_from_samples(dataset.points, n_clusters, init_seed)
    result = iter_lloyd(dataset.points, initial, max_iterations=max_iterations)

    run_id = new_run_id(experiment_id)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    training = KMeansConfig(
        algorithm="kmeans_lloyd",
        n_clusters=n_clusters,
        init_seed=init_seed,
        max_iterations=max_iterations,
    )
    manifest = KMeansRunManifest(
        run_id=run_id,
        experiment_id=experiment_id,
        created_at=datetime.now(UTC).isoformat(),
        status="running",
        data_config=KMeansDataConfig(
            generator="synthetic_kmeans_v1",
            blob_centers=[list(center) for center in data_config.blob_centers],
            blob_sizes=list(data_config.blob_sizes),
            cluster_std=data_config.cluster_std,
            seed=data_config.seed,
        ),
        dataset=KMeansDatasetSummary(
            generator_id="synthetic_kmeans_v1",
            sample_ids=dataset.sample_ids,
            points=dataset.points.tolist(),
        ),
        training_config=training,
        code_provenance=get_code_provenance(repo_root),
        observed_sample_ids=dataset.sample_ids[:5],
    )
    atomic_write_json(
        run_dir / "manifest.json", json.loads(manifest.model_dump_json(by_alias=True))
    )
    events = [
        KMeansEvent(run_id=run_id, seq=1, kind="run.created"),
        KMeansEvent(run_id=run_id, seq=2, kind="run.started"),
    ]
    snapshots: list[KMeansSnapshot] = []
    seq = 2
    for state in result.snapshots:
        snapshot = KMeansSnapshot(
            step=state.step,
            iteration=state.iteration,
            phase=state.phase,
            centers=state.centers.tolist(),
            assignments=state.assignments.tolist(),
            inertia=state.inertia,
            empty_clusters=list(state.empty_clusters),
        )
        snapshots.append(snapshot)
        seq += 1
        events.append(
            KMeansEvent(
                run_id=run_id,
                seq=seq,
                kind=("iteration.assigned" if state.phase == "assignment" else "iteration.updated"),
                step=state.step,
                iteration=state.iteration,
            )
        )
        if state.empty_clusters:
            seq += 1
            events.append(
                KMeansEvent(
                    run_id=run_id,
                    seq=seq,
                    kind="cluster.empty",
                    step=state.step,
                    iteration=state.iteration,
                    message=f"retained centers for empty clusters {list(state.empty_clusters)}",
                )
            )
    seq += 1
    events.append(KMeansEvent(run_id=run_id, seq=seq, kind="run.completed"))
    completed = manifest.model_copy(
        update={
            "status": "completed",
            "stop_reason": result.stop_reason,
            "last_valid_step": snapshots[-1].step,
            "n_snapshots_written": len(snapshots),
        }
    )
    atomic_write_json(
        run_dir / "snapshots.json",
        [json.loads(snapshot.model_dump_json(by_alias=True)) for snapshot in snapshots],
    )
    for event in events:
        append_jsonl(run_dir / "events.jsonl", json.loads(event.model_dump_json(by_alias=True)))
    atomic_write_json(
        run_dir / "manifest.json", json.loads(completed.model_dump_json(by_alias=True))
    )
    return completed


__all__ = ["run_kmeans_experiment"]
