import { describe, expect, it } from "vitest";
import { compareCatalogRuns, parseCatalog } from "../lib/catalog";

const valid = {
  schemaVersion: 1,
  batchId: "batch-1",
  datasets: [{ dataset_id: "uci-wdbc", source: "external_dataset", title: "WDBC", source_page: "https://example.org", license: "CC BY 4.0", citation: "UCI" }],
  datasetVersions: [{ dataset_key: "v1", dataset_id: "uci-wdbc", dataset_version: "1.0.0", dataset_identity: "hash", source_url: "https://example.org/data", source_status: "verified_manifest" }],
  datasetArtifacts: [{ dataset_key: "v1", relative_path: "raw/data", role: "raw_data", bytes: 12, sha256: "digest", available: true }],
  runs: [{ run_id: "tree-1", experiment_id: "depth-study", dataset_key: "v1", split_identity: "split-hash", schema_version: 4, status: "completed", mlflow_run_id: null, tracking_state: "not_synced" }],
  metrics: [{ run_id: "tree-1", name: "accuracy", split: "validation", step: 2, value: 0.9, unit: "fraction", aggregation: "correct/total" }],
  artifacts: [{ run_id: "tree-1", relative_path: "tree-1/manifest.json", sha256: "digest" }],
};

describe("browser catalog contract", () => {
  it("keeps the data-version-run-metric chain", () => {
    const catalog = parseCatalog(valid);
    expect(catalog.datasetVersions[0]?.dataset_id).toBe("uci-wdbc");
    expect(catalog.runs[0]?.dataset_key).toBe("v1");
    expect(catalog.metrics[0]?.value).toBe(0.9);
    expect(catalog.datasetArtifacts[0]?.available).toBe(true);
  });

  it("rejects a run pointing to a missing dataset version", () => {
    expect(() => parseCatalog({ ...valid, datasetVersions: [] })).toThrow(/relationship/);
  });

  it("rejects a non-finite metric", () => {
    expect(() => parseCatalog({ ...valid, metrics: [{ ...valid.metrics[0], value: Infinity }] })).toThrow(/metric/);
  });

  it("rejects an artifact pointing to a missing dataset version", () => {
    expect(() => parseCatalog({ ...valid, datasetArtifacts: [{ ...valid.datasetArtifacts[0], dataset_key: "missing" }] })).toThrow(/relationship/);
  });
});

describe("catalog metric comparisons", () => {
  const peer = { ...valid.runs[0], run_id: "tree-2" };
  const peerMetric = { ...valid.metrics[0], run_id: "tree-2", value: 0.8 };

  it("shows numbers only when dataset, split and metric contract match", () => {
    const catalog = parseCatalog({ ...valid, runs: [...valid.runs, peer], metrics: [...valid.metrics, peerMetric] });
    expect(compareCatalogRuns(catalog, catalog.runs[0]!, catalog.runs[1]!)).toEqual({
      comparable: [{ name: "accuracy", split: "validation", unit: "fraction", baselineValue: 0.9, candidateValue: 0.8 }],
      reasons: [],
    });
  });

  it("blocks a different split or dataset version even with the same metric name", () => {
    const catalog = parseCatalog({ ...valid, runs: [...valid.runs, { ...peer, split_identity: "other-split" }], metrics: [...valid.metrics, peerMetric] });
    const result = compareCatalogRuns(catalog, catalog.runs[0]!, catalog.runs[1]!);
    expect(result.comparable).toEqual([]);
    expect(result.reasons).toContain("Different split identity; scores use different observations.");
    expect(compareCatalogRuns(catalog, { ...catalog.runs[0]!, dataset_key: "other-version" }, catalog.runs[1]!).reasons).toContain("Different dataset versions.");
  });

  it("names a unit mismatch instead of comparing the numerical values", () => {
    const catalog = parseCatalog({ ...valid, runs: [...valid.runs, peer], metrics: [...valid.metrics, { ...peerMetric, unit: "percent" }] });
    const result = compareCatalogRuns(catalog, catalog.runs[0]!, catalog.runs[1]!);
    expect(result.comparable).toEqual([]);
    expect(result.reasons).toContain("accuracy (validation): unit differs (fraction vs percent).");
  });

  it("names a different aggregation even when the unit matches", () => {
    const catalog = parseCatalog({ ...valid, runs: [...valid.runs, peer], metrics: [...valid.metrics, { ...peerMetric, aggregation: "mean" }] });
    const result = compareCatalogRuns(catalog, catalog.runs[0]!, catalog.runs[1]!);
    expect(result.comparable).toEqual([]);
    expect(result.reasons).toContain("accuracy (validation): aggregation differs (correct/total vs mean).");
  });

  it("explains missing metrics and incomplete runs", () => {
    const catalog = parseCatalog({ ...valid, runs: [...valid.runs, peer], metrics: valid.metrics });
    expect(compareCatalogRuns(catalog, catalog.runs[0]!, catalog.runs[1]!).reasons).toContain("tree-2 has no final metrics.");
    expect(compareCatalogRuns(catalog, catalog.runs[0]!, { ...catalog.runs[1]!, status: "failed" }).reasons).toContain("Only completed runs can be compared.");
  });
});
