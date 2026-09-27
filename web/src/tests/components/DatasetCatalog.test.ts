// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import DatasetCatalog from "../../components/DatasetCatalog.svelte";

const catalog = {
  schemaVersion: 1,
  batchId: "published-batch",
  datasets: [{ dataset_id: "uci-wdbc", source: "external_dataset", title: "WDBC", source_page: "https://example.org", license: "CC BY 4.0", citation: "UCI" }],
  datasetVersions: [{ dataset_key: "v1", dataset_id: "uci-wdbc", dataset_version: "1.0.0", dataset_identity: "hash", source_url: "https://example.org/data", source_status: "verified_manifest" }],
  datasetArtifacts: [{ dataset_key: "v1", relative_path: "raw/data.csv", role: "raw_data", bytes: 12, sha256: "digest", available: false }],
  runs: [{ run_id: "tree-1", experiment_id: "depth-study", dataset_key: "v1", split_identity: "split-hash", schema_version: 4, status: "completed", mlflow_run_id: null, tracking_state: "not_synced" }],
  metrics: [],
  artifacts: [],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DatasetCatalog disclosure", () => {
  it("starts collapsed with a summary of the selected dataset and local state", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) => {
      if (String(input) === "/api/catalog") throw new TypeError("Failed to fetch");
      return new Response(JSON.stringify(catalog), { status: 200 });
    }));

    const { container } = render(DatasetCatalog, { onSelectReplay: vi.fn() });

    expect(await screen.findByText("WDBC · 1.0.0")).toBeTruthy();
    expect(screen.getByText("1 run")).toBeTruthy();
    expect(await screen.findByText("Local API offline")).toBeTruthy();
    expect(container.querySelector("details")?.open).toBe(false);
  });

  it("collapses after a replay is opened from the catalog", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify(catalog), { status: 200 }),
    ));
    const onSelectReplay = vi.fn();
    const { container } = render(DatasetCatalog, { onSelectReplay });
    const details = container.querySelector("details")!;
    await fireEvent.click(details.querySelector("summary")!);
    // Browsers fire `toggle` after a summary click; jsdom flips `open` without it.
    await fireEvent(details, new Event("toggle"));
    expect(details.open).toBe(true);

    await fireEvent.click(await screen.findByRole("button", { name: "Open local recorded replay" }));

    expect(onSelectReplay).toHaveBeenCalledWith("/api/replay/tree-1");
    await waitFor(() => expect(details.open).toBe(false));
  });
});

describe("DatasetCatalog availability", () => {
  it("identifies stale local indexes and unavailable source artifacts", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) =>
      new Response(JSON.stringify(String(input) === "/api/catalog" ? { ...catalog, batchId: "newer-batch" } : catalog), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });

    expect(await screen.findByText(/Local catalog batch differs/)).toBeTruthy();
    expect(screen.getByText(/raw\/data\.csv/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Open local recorded replay" })).toBeNull();
  });

  it("keeps the static index visible when the local service is offline", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) => {
      if (String(input) === "/api/catalog") throw new TypeError("Failed to fetch");
      return new Response(JSON.stringify(catalog), { status: 200 });
    }));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });

    expect(await screen.findByText(/Local catalog API is unavailable/)).toBeTruthy();
    expect(screen.getByText("tree-1")).toBeTruthy();
  });

  it("does not call a malformed API response a stale batch", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) =>
      new Response(JSON.stringify(String(input) === "/api/catalog" ? {} : catalog), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });

    expect(await screen.findByText(/Local catalog API is unavailable/)).toBeTruthy();
    expect(screen.queryByText(/Local catalog batch differs/)).toBeNull();
  });

  it("offers local replay only for a matching batch", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify(catalog), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });

    await waitFor(() => expect(screen.getByRole("button", { name: "Open local recorded replay" })).toBeTruthy());
    expect(screen.queryByText(/Local catalog batch differs/)).toBeNull();
  });

  it("shows matching values and explains a split mismatch without ranking models", async () => {
    const comparisonCatalog = {
      ...catalog,
      runs: [
        catalog.runs[0],
        { ...catalog.runs[0], run_id: "tree-2" },
        { ...catalog.runs[0], run_id: "tree-other-split", split_identity: "other-split" },
      ],
      metrics: [
        { run_id: "tree-1", name: "accuracy", split: "validation", step: 2, value: 0.90001, unit: "fraction", aggregation: "correct/total" },
        { run_id: "tree-2", name: "accuracy", split: "validation", step: 2, value: 0.90002, unit: "fraction", aggregation: "correct/total" },
        { run_id: "tree-other-split", name: "accuracy", split: "validation", step: 2, value: 0.95, unit: "fraction", aggregation: "correct/total" },
      ],
    };
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async () =>
      new Response(JSON.stringify(comparisonCatalog), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });

    expect(await screen.findByText(/accuracy \(validation, fraction\): 0\.90001 ↔ 0\.90002/)).toBeTruthy();
    expect(screen.getByText(/Not comparable: Different split identity/)).toBeTruthy();
    expect(screen.getByText(/not a model-quality verdict/)).toBeTruthy();

    await fireEvent.change(screen.getByRole("combobox", { name: "Comparison baseline" }), { target: { value: "tree-2" } });
    expect(await screen.findByText(/accuracy \(validation, fraction\): 0\.90002 ↔ 0\.90001/)).toBeTruthy();
  });

  it("opens verified MLflow step metrics inside the catalog without a dashboard", async () => {
    const tracked = { ...catalog, runs: [{ ...catalog.runs[0], tracking_state: "verified" }] };
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) =>
      new Response(JSON.stringify(String(input).includes("/api/tracking/")
        ? { runId: "tree-1", source: "mlflow", state: "verified", series: [{ key: "nodes_recorded.all", unit: "count", aggregation: "count", points: [{ step: 0, value: 1 }, { step: 1, value: 2 }] }] }
        : tracked), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });
    await fireEvent.click(await screen.findByRole("button", { name: "View MLflow metrics" }));

    expect(await screen.findByText("nodes_recorded.all")).toBeTruthy();
    expect(screen.getByRole("img", { name: /nodes_recorded\.all.*metric history/ })).toBeTruthy();
    expect(screen.getAllByText("Step 1: 2 count").length).toBeGreaterThan(0);
  });

  it("shows an API failure without losing the catalog or static replay", async () => {
    const tracked = { ...catalog, runs: [{ ...catalog.runs[0], tracking_state: "verified" }] };
    vi.stubGlobal("fetch", vi.fn<typeof fetch>(async (input) =>
      String(input).includes("/api/tracking/")
        ? new Response("Unavailable", { status: 503 })
        : new Response(JSON.stringify(tracked), { status: 200 }),
    ));

    render(DatasetCatalog, { onSelectReplay: vi.fn() });
    await fireEvent.click(await screen.findByRole("button", { name: "View MLflow metrics" }));

    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.getByText("tree-1")).toBeTruthy();
  });
});
