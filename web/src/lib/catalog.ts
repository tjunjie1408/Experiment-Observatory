/** Read-only D5 catalog contract, separate from schema-v1..v4 replay bundles. */

export interface DatasetRow { dataset_id: string; source: string; title: string; source_page: string; license: string; citation: string }
export interface DatasetVersionRow {
  dataset_key: string;
  dataset_id: string;
  dataset_version: string;
  dataset_identity: string;
  source_url: string;
  source_status: string;
}
export interface DatasetArtifact { dataset_key: string; relative_path: string; role: string; bytes: number; sha256: string; available: boolean }
export interface CatalogRun {
  run_id: string;
  experiment_id: string;
  dataset_key: string;
  split_identity: string | null;
  schema_version: number;
  status: string;
  mlflow_run_id: string | null;
  tracking_state: string;
}
export interface CatalogMetric {
  run_id: string;
  name: string;
  split: string;
  step: number;
  value: number;
  unit: string;
  aggregation: string;
}
export interface CatalogArtifact { run_id: string; relative_path: string; sha256: string }
export interface BrowserCatalog {
  schemaVersion: 1;
  batchId: string;
  datasets: DatasetRow[];
  datasetVersions: DatasetVersionRow[];
  datasetArtifacts: DatasetArtifact[];
  runs: CatalogRun[];
  metrics: CatalogMetric[];
  artifacts: CatalogArtifact[];
}

export interface ComparedMetric {
  name: string;
  split: string;
  unit: string;
  baselineValue: number;
  candidateValue: number;
}

export interface RunMetricComparison {
  comparable: ComparedMetric[];
  reasons: string[];
}

/** Compare recorded final metrics, not model quality or unseen-data performance. */
export function compareCatalogRuns(
  catalog: BrowserCatalog,
  baseline: CatalogRun,
  candidate: CatalogRun,
): RunMetricComparison {
  const incomparable = (reason: string): RunMetricComparison => ({ comparable: [], reasons: [reason] });
  if (baseline.status !== "completed" || candidate.status !== "completed") {
    return incomparable("Only completed runs can be compared.");
  }
  if (baseline.dataset_key !== candidate.dataset_key) {
    return incomparable("Different dataset versions.");
  }
  if (baseline.split_identity !== candidate.split_identity) {
    return incomparable("Different split identity; scores use different observations.");
  }

  const baselineMetrics = catalog.metrics.filter((metric) => metric.run_id === baseline.run_id);
  const candidateMetrics = catalog.metrics.filter((metric) => metric.run_id === candidate.run_id);
  if (baselineMetrics.length === 0) return incomparable(`${baseline.run_id} has no final metrics.`);
  if (candidateMetrics.length === 0) return incomparable(`${candidate.run_id} has no final metrics.`);

  const comparable: ComparedMetric[] = [];
  const reasons: string[] = [];
  for (const metric of baselineMetrics) {
    const label = `${metric.name} (${metric.split})`;
    const peers = candidateMetrics.filter(
      (item) => item.name === metric.name && item.split === metric.split,
    );
    const match = peers.find(
      (item) => item.unit === metric.unit && item.aggregation === metric.aggregation,
    );
    if (match) {
      comparable.push({
        name: metric.name,
        split: metric.split,
        unit: metric.unit,
        baselineValue: metric.value,
        candidateValue: match.value,
      });
    } else if (peers.length === 0) {
      reasons.push(`${label}: no matching metric and split.`);
    } else {
      const sameUnit = peers.find((item) => item.unit === metric.unit);
      reasons.push(sameUnit
        ? `${label}: aggregation differs (${metric.aggregation} vs ${sameUnit.aggregation}).`
        : `${label}: unit differs (${metric.unit} vs ${peers[0]!.unit}).`);
    }
  }
  return { comparable, reasons };
}

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("catalog object expected");
  return value as Record<string, unknown>;
}
function string(value: unknown): string {
  if (typeof value !== "string" || !value) throw new Error("catalog string expected");
  return value;
}
function array(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("catalog array expected");
  return value;
}
function optionalString(value: unknown): string {
  if (typeof value !== "string") throw new Error("catalog string expected");
  return value;
}

export function parseCatalog(raw: unknown): BrowserCatalog {
  const input = record(raw);
  if (input.schemaVersion !== 1) throw new Error("unsupported catalog version");
  const datasets = array(input.datasets).map((item): DatasetRow => {
    const row = record(item);
    return { dataset_id: string(row.dataset_id), source: string(row.source), title: string(row.title), source_page: optionalString(row.source_page), license: optionalString(row.license), citation: optionalString(row.citation) };
  });
  const datasetVersions = array(input.datasetVersions).map((item): DatasetVersionRow => {
    const row = record(item);
    return { dataset_key: string(row.dataset_key), dataset_id: string(row.dataset_id), dataset_version: string(row.dataset_version), dataset_identity: string(row.dataset_identity), source_url: optionalString(row.source_url), source_status: string(row.source_status) };
  });
  const datasetArtifacts = array(input.datasetArtifacts).map((item): DatasetArtifact => {
    const row = record(item);
    if (typeof row.bytes !== "number" || !Number.isSafeInteger(row.bytes) || row.bytes < 0 || typeof row.available !== "boolean") throw new Error("invalid dataset artifact");
    return { dataset_key: string(row.dataset_key), relative_path: string(row.relative_path), role: string(row.role), bytes: row.bytes, sha256: string(row.sha256), available: row.available };
  });
  const runs = array(input.runs).map((item): CatalogRun => {
    const row = record(item);
    if (typeof row.schema_version !== "number" || !Number.isInteger(row.schema_version) || row.schema_version < 1 || row.schema_version > 4) throw new Error("invalid run schema version");
    return { run_id: string(row.run_id), experiment_id: string(row.experiment_id), dataset_key: string(row.dataset_key), split_identity: row.split_identity === null ? null : string(row.split_identity), schema_version: row.schema_version, status: string(row.status), mlflow_run_id: row.mlflow_run_id === null ? null : string(row.mlflow_run_id), tracking_state: string(row.tracking_state) };
  });
  const metrics = array(input.metrics).map((item): CatalogMetric => {
    const row = record(item);
    if (typeof row.step !== "number" || !Number.isInteger(row.step) || typeof row.value !== "number" || !Number.isFinite(row.value)) throw new Error("invalid catalog metric");
    return { run_id: string(row.run_id), name: string(row.name), split: string(row.split), step: row.step, value: row.value, unit: string(row.unit), aggregation: string(row.aggregation) };
  });
  const artifacts = array(input.artifacts).map((item): CatalogArtifact => {
    const row = record(item);
    return { run_id: string(row.run_id), relative_path: string(row.relative_path), sha256: string(row.sha256) };
  });
  const ids = new Set(datasets.map((row) => row.dataset_id));
  const versions = new Set(datasetVersions.map((row) => row.dataset_key));
  const runIds = new Set(runs.map((row) => row.run_id));
  if (ids.size !== datasets.length || versions.size !== datasetVersions.length || runIds.size !== runs.length) throw new Error("duplicate catalog identity");
  if (datasetVersions.some((row) => !ids.has(row.dataset_id)) || datasetArtifacts.some((row) => !versions.has(row.dataset_key)) || runs.some((row) => !versions.has(row.dataset_key)) || metrics.some((row) => !runIds.has(row.run_id)) || artifacts.some((row) => !runIds.has(row.run_id))) throw new Error("broken catalog relationship");
  return { schemaVersion: 1, batchId: string(input.batchId), datasets, datasetVersions, datasetArtifacts, runs, metrics, artifacts };
}

export async function loadCatalog(): Promise<BrowserCatalog> {
  for (const url of ["catalog/catalog.json", "/api/catalog"]) {
    try {
      const response = await fetch(url);
      if (response.ok) return parseCatalog(await response.json());
    } catch {
      // Static export can be absent while the local API is available, or vice versa.
    }
  }
  throw new Error("No published catalog is available. Replay bundles remain independent.");
}
