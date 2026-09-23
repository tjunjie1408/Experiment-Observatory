"""Fixed read-only DuckDB projection for the local catalog endpoint."""

from __future__ import annotations

from pathlib import Path

import duckdb


class CatalogUnavailableError(ValueError):
    """No published, readable catalog exists at the configured location."""


def read_catalog(database_path: Path) -> dict[str, object]:
    if not database_path.is_file():
        raise CatalogUnavailableError("catalog has not been built")
    try:
        with duckdb.connect(str(database_path), read_only=True) as connection:
            meta = connection.execute("SELECT batch_id FROM catalog_meta").fetchone()
            if meta is None:
                raise CatalogUnavailableError("catalog has no published batch")

            def rows(view: str) -> list[dict[str, object]]:
                result = connection.execute(f"SELECT * FROM {view}")
                names = [column[0] for column in result.description]
                return [dict(zip(names, values, strict=True)) for values in result.fetchall()]

            return {
                "schemaVersion": 1,
                "batchId": meta[0],
                "datasets": rows("datasets"),
                "datasetVersions": rows("dataset_versions"),
                "datasetArtifacts": rows("dataset_artifacts"),
                "runs": rows("runs"),
                "metrics": rows("metrics"),
                "models": rows("models"),
                "artifacts": rows("artifacts"),
            }
    except duckdb.Error as exc:
        raise CatalogUnavailableError(f"catalog cannot be read: {exc}") from exc
