<script lang="ts">
  import { onMount } from "svelte";
  import { AVAILABLE_RUNS } from "../lib/availableRuns";
  import { compareCatalogRuns, loadCatalog, type BrowserCatalog } from "../lib/catalog";
  import TrackedMetrics from "./TrackedMetrics.svelte";

  export let onSelectReplay: (path: string) => void;

  let catalog: BrowserCatalog | null = null;
  let message = "Loading catalog…";
  let datasetId = "";
  let versionKey = "";
  let baselineRunId = "";
  let trackedRunId = "";
  let localCatalogState: "checking" | "matched" | "unavailable" | "different" = "checking";
  $: versions = catalog?.datasetVersions.filter((item) => item.dataset_id === datasetId) ?? [];
  $: selectedVersion = versions.find((item) => item.dataset_key === versionKey);
  $: selectedDataset = catalog?.datasets.find((item) => item.dataset_id === datasetId);
  $: datasetArtifacts = catalog?.datasetArtifacts.filter((item) => item.dataset_key === versionKey) ?? [];
  $: unavailableArtifacts = datasetArtifacts.filter((item) => !item.available);
  $: runs = catalog?.runs.filter((run) => run.dataset_key === versionKey) ?? [];
  $: baselineRun = runs.find((run) => run.run_id === baselineRunId);

  onMount(() => {
    let live = true;
    void loadCatalog().then((loaded) => {
      if (!live) return;
      catalog = loaded;
      datasetId = loaded.datasets[0]?.dataset_id ?? "";
      versionKey = loaded.datasetVersions.find((item) => item.dataset_id === datasetId)?.dataset_key ?? "";
      baselineRunId = loaded.runs.find((run) => run.dataset_key === versionKey)?.run_id ?? "";
      message = `${loaded.runs.length} indexed runs from batch ${loaded.batchId}`;
      void fetch("/api/catalog").then(async (response) => {
        if (!response.ok) throw new Error(`Catalog API returned HTTP ${response.status}`);
        return await response.json() as { batchId?: unknown };
      }).then((local) => {
        if (live) localCatalogState = typeof local.batchId !== "string"
          ? "unavailable"
          : local.batchId === loaded.batchId ? "matched" : "different";
      }).catch(() => {
        if (live) localCatalogState = "unavailable";
      });
    }).catch((error: unknown) => {
      if (live) message = error instanceof Error ? error.message : String(error);
    });
    return () => { live = false; };
  });

  function chooseDataset(value: string): void {
    datasetId = value;
    trackedRunId = "";
    versionKey = catalog?.datasetVersions.find((item) => item.dataset_id === value)?.dataset_key ?? "";
    baselineRunId = catalog?.runs.find((run) => run.dataset_key === versionKey)?.run_id ?? "";
  }

  function chooseVersion(value: string): void {
    versionKey = value;
    trackedRunId = "";
    baselineRunId = catalog?.runs.find((run) => run.dataset_key === value)?.run_id ?? "";
  }

  function metricText(runId: string): string {
    return catalog?.metrics.filter((metric) => metric.run_id === runId)
      .map((metric) => `${metric.name} (${metric.split}) ${metric.value.toFixed(4)} ${metric.unit}`)
      .join(" · ") || "No final metrics recorded";
  }
</script>

<section class="dataset-catalog" aria-label="Dataset and experiment catalog">
  <header><h2>Data & experiments</h2><span role="status">{message}</span></header>
  {#if catalog}
    {#if localCatalogState === "different"}
      <p role="status">Local catalog batch differs from the displayed index. Local replay is disabled until the catalog is refreshed.</p>
    {:else if localCatalogState === "unavailable"}
      <p role="status">Local catalog API is unavailable. The loaded index can still be browsed; its freshness cannot be confirmed.</p>
    {/if}
    <div class="catalog-controls">
      <label>Dataset
        <select aria-label="Catalog dataset" value={datasetId} on:change={(event) => chooseDataset(event.currentTarget.value)}>
          {#each catalog.datasets as dataset}
            <option value={dataset.dataset_id}>{dataset.title} · {dataset.source}</option>
          {/each}
        </select>
      </label>
      <label>Version
        <select aria-label="Catalog version" value={versionKey} on:change={(event) => chooseVersion(event.currentTarget.value)}>
          {#each versions as version}
            <option value={version.dataset_key}>{version.dataset_version}</option>
          {/each}
        </select>
      </label>
    </div>
    {#if selectedVersion}
      {#if selectedDataset}
        <p class="source-line">{selectedDataset.title} · {selectedDataset.license || "License unavailable"}</p>
        {#if selectedDataset.source_page.startsWith("https://")}
          <a href={selectedDataset.source_page} target="_blank" rel="noopener noreferrer">Authoritative dataset page</a>
        {/if}
      {/if}
      <p class="source-line">Version identity: <code>{selectedVersion.dataset_identity}</code></p>
      <p class="source-line">Source manifest: {selectedVersion.source_status} · Raw files: {datasetArtifacts.filter((item) => item.available).length}/{datasetArtifacts.length} available</p>
      {#if unavailableArtifacts.length > 0}
        <p class="source-line">Unavailable when indexed: {unavailableArtifacts.map((item) => item.relative_path).join(", ")}. This is a snapshot, not a live file check.</p>
      {/if}
      {#if selectedVersion.source_url.startsWith("https://")}
        <a href={selectedVersion.source_url} target="_blank" rel="noopener noreferrer">Version source file</a>
      {/if}
      {#if runs.length === 0}<p>No validated runs are indexed for this version.</p>{/if}
      <div class="catalog-runs">
        {#each runs as run (run.run_id)}
          {@const replay = AVAILABLE_RUNS.find((item) => item.id === run.run_id)}
          <article>
            <h3>{run.run_id}</h3>
            <p>{run.experiment_id} · {run.status} · {run.tracking_state}</p>
            <p>{metricText(run.run_id)}</p>
            <small>Split: {run.split_identity ?? "not applicable"}</small>
            {#if replay && run.status === "completed"}
              <button type="button" on:click={() => onSelectReplay(replay.path)}>Open recorded replay</button>
            {:else if localCatalogState === "matched" && run.status === "completed"}
              <button type="button" on:click={() => onSelectReplay(`/api/replay/${encodeURIComponent(run.run_id)}`)}>Open local recorded replay</button>
            {:else}
              <small>Replay bundle is not published in this static site. Start the local service to open its recorded timeline.</small>
            {/if}
            {#if localCatalogState === "matched" && run.tracking_state === "verified"}
              <button type="button" on:click={() => trackedRunId = trackedRunId === run.run_id ? "" : run.run_id}>
                {trackedRunId === run.run_id ? "Hide MLflow metrics" : "View MLflow metrics"}
              </button>
              {#if trackedRunId === run.run_id}<TrackedMetrics runId={run.run_id} />{/if}
            {/if}
          </article>
        {/each}
      </div>
      {#if runs.length > 1 && baselineRun}
        <section aria-label="Run metric comparison" class="metric-comparison">
          <h3>Recorded metric comparison</h3>
          <p>Matching recorded fields are shown side by side; this is not a model-quality verdict.</p>
          <label>Reference run
            <select aria-label="Comparison baseline" bind:value={baselineRunId}>
              {#each runs as run}
                <option value={run.run_id}>{run.run_id}</option>
              {/each}
            </select>
          </label>
          {#each runs.filter((run) => run.run_id !== baselineRunId) as peer (peer.run_id)}
            {@const comparison = compareCatalogRuns(catalog, baselineRun, peer)}
            <article>
              <h4>{baselineRun.run_id} ↔ {peer.run_id}</h4>
              {#each comparison.comparable as metric}
                <p>{metric.name} ({metric.split}, {metric.unit}): {metric.baselineValue} ↔ {metric.candidateValue}</p>
              {/each}
              {#each comparison.reasons as reason}
                <p>Not comparable: {reason}</p>
              {/each}
            </article>
          {/each}
        </section>
      {/if}
    {/if}
  {/if}
</section>

<style>
  .dataset-catalog { margin: 1rem 0; padding: 1rem; border: 1px solid #334155; border-radius: .75rem; background: #111e2e; color: #e2e8f0; }
  header { display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; align-items: baseline; }
  h2 { margin: 0; font-size: 1.1rem; }
  header span, small { color: #94a3b8; }
  .catalog-controls { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
  label { display: grid; gap: .3rem; }
  select { min-width: 12rem; max-width: 100%; padding: .45rem; color: #e2e8f0; background: #172334; border: 1px solid #475569; }
  .source-line { overflow-wrap: anywhere; font-size: .85rem; }
  .catalog-runs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .75rem; }
  article { border: 1px solid #334155; border-radius: .5rem; padding: .8rem; }
  .metric-comparison { margin-top: 1rem; border-top: 1px solid #334155; padding-top: .8rem; }
  .metric-comparison h4 { margin: 0; overflow-wrap: anywhere; }
  h3 { margin: 0; font-size: .88rem; overflow-wrap: anywhere; }
  p { margin: .45rem 0; }
  small { display: block; overflow-wrap: anywhere; }
  button { margin-top: .6rem; padding: .35rem .55rem; cursor: pointer; }
</style>
