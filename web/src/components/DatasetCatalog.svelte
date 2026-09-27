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
  let open = false;
  $: versions = catalog?.datasetVersions.filter((item) => item.dataset_id === datasetId) ?? [];
  $: selectedVersion = versions.find((item) => item.dataset_key === versionKey);
  $: selectedDataset = catalog?.datasets.find((item) => item.dataset_id === datasetId);
  $: datasetArtifacts = catalog?.datasetArtifacts.filter((item) => item.dataset_key === versionKey) ?? [];
  $: unavailableArtifacts = datasetArtifacts.filter((item) => !item.available);
  $: runs = catalog?.runs.filter((run) => run.dataset_key === versionKey) ?? [];
  $: baselineRun = runs.find((run) => run.run_id === baselineRunId);
  $: localNote = localCatalogState === "different"
    ? "Local index stale"
    : localCatalogState === "unavailable" ? "Local API offline" : "";

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

  function openReplay(path: string): void {
    onSelectReplay(path);
    open = false;
  }

  function metricText(runId: string): string {
    return catalog?.metrics.filter((metric) => metric.run_id === runId)
      .map((metric) => `${metric.name} (${metric.split}) ${metric.value.toFixed(4)} ${metric.unit}`)
      .join(" · ") || "No final metrics recorded";
  }
</script>

<section class="dataset-catalog" aria-label="Dataset and experiment catalog">
  <details bind:open>
    <summary>
      <span class="disclosure-chevron" aria-hidden="true"></span>
      <h2>Data & experiments</h2>
      {#if selectedDataset && selectedVersion}
        <span class="catalog-chip">{selectedDataset.title} · {selectedVersion.dataset_version}</span>
        <span class="catalog-chip">{runs.length} {runs.length === 1 ? "run" : "runs"}</span>
      {/if}
      {#if localNote}<span class="catalog-chip warning-chip">{localNote}</span>{/if}
      <span class="catalog-message" role="status">{message}</span>
    </summary>

    {#if catalog}
      <div class="catalog-body">
        {#if localCatalogState === "different"}
          <p class="catalog-notice" role="status">Local catalog batch differs from the displayed index. Local replay is disabled until the catalog is refreshed.</p>
        {:else if localCatalogState === "unavailable"}
          <p class="catalog-notice" role="status">Local catalog API is unavailable. The loaded index can still be browsed; its freshness cannot be confirmed.</p>
        {/if}
        <div class="catalog-controls">
          <label><span>Dataset</span>
            <select aria-label="Catalog dataset" value={datasetId} on:change={(event) => chooseDataset(event.currentTarget.value)}>
              {#each catalog.datasets as dataset}
                <option value={dataset.dataset_id}>{dataset.title} · {dataset.source}</option>
              {/each}
            </select>
          </label>
          <label><span>Version</span>
            <select aria-label="Catalog version" value={versionKey} on:change={(event) => chooseVersion(event.currentTarget.value)}>
              {#each versions as version}
                <option value={version.dataset_key}>{version.dataset_version}</option>
              {/each}
            </select>
          </label>
          {#if selectedVersion}
            <div class="catalog-links">
              {#if selectedDataset?.source_page.startsWith("https://")}
                <a href={selectedDataset.source_page} target="_blank" rel="noopener noreferrer">Authoritative dataset page</a>
              {/if}
              {#if selectedVersion.source_url.startsWith("https://")}
                <a href={selectedVersion.source_url} target="_blank" rel="noopener noreferrer">Version source file</a>
              {/if}
            </div>
          {/if}
        </div>
        {#if selectedVersion}
          <dl class="provenance-grid">
            {#if selectedDataset}
              <div><dt>License</dt><dd>{selectedDataset.license || "License unavailable"}</dd></div>
            {/if}
            <div class="wide"><dt>Version identity</dt><dd><code title={selectedVersion.dataset_identity}>{selectedVersion.dataset_identity}</code></dd></div>
            <div><dt>Source manifest</dt><dd>{selectedVersion.source_status}</dd></div>
            <div><dt>Raw files</dt><dd>{datasetArtifacts.length - unavailableArtifacts.length}/{datasetArtifacts.length} available</dd></div>
          </dl>
          {#if unavailableArtifacts.length > 0}
            <p class="catalog-note">Unavailable when indexed: {unavailableArtifacts.map((item) => item.relative_path).join(", ")}. This is a snapshot, not a live file check.</p>
          {/if}
          {#if runs.length === 0}<p class="catalog-note">No validated runs are indexed for this version.</p>{/if}
          <div class="catalog-runs">
            {#each runs as run (run.run_id)}
              {@const replay = AVAILABLE_RUNS.find((item) => item.id === run.run_id)}
              <article>
                <h3 title={run.run_id}>{run.run_id}</h3>
                <p class="run-meta">{run.experiment_id} · {run.status} · {run.tracking_state}</p>
                <p class="run-metrics">{metricText(run.run_id)}</p>
                <small>Split: {run.split_identity ?? "not applicable"}</small>
                <div class="run-actions">
                  {#if replay && run.status === "completed"}
                    <button type="button" class="primary-action" on:click={() => openReplay(replay.path)}>Open recorded replay</button>
                  {:else if localCatalogState === "matched" && run.status === "completed"}
                    <button type="button" class="primary-action" on:click={() => openReplay(`/api/replay/${encodeURIComponent(run.run_id)}`)}>Open local recorded replay</button>
                  {:else}
                    <small>Replay bundle is not published in this static site. Start the local service to open its recorded timeline.</small>
                  {/if}
                  {#if localCatalogState === "matched" && run.tracking_state === "verified"}
                    <button type="button" on:click={() => trackedRunId = trackedRunId === run.run_id ? "" : run.run_id}>
                      {trackedRunId === run.run_id ? "Hide MLflow metrics" : "View MLflow metrics"}
                    </button>
                  {/if}
                </div>
                {#if localCatalogState === "matched" && trackedRunId === run.run_id}<TrackedMetrics runId={run.run_id} />{/if}
              </article>
            {/each}
          </div>
          {#if runs.length > 1 && baselineRun}
            <details class="metric-comparison">
              <summary>
                <span class="disclosure-chevron" aria-hidden="true"></span>
                <h3>Recorded metric comparison</h3>
                <span class="catalog-chip">{runs.length - 1} {runs.length === 2 ? "peer" : "peers"}</span>
              </summary>
              <section aria-label="Run metric comparison">
                <p class="catalog-note">Matching recorded fields are shown side by side; this is not a model-quality verdict.</p>
                <label><span>Reference run</span>
                  <select aria-label="Comparison baseline" bind:value={baselineRunId}>
                    {#each runs as run}
                      <option value={run.run_id}>{run.run_id}</option>
                    {/each}
                  </select>
                </label>
                <div class="comparison-list">
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
                </div>
              </section>
            </details>
          {/if}
        {/if}
      </div>
    {/if}
  </details>
</section>

<style>
  .dataset-catalog { margin: 12px 16px 0; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }
  summary { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 14px; cursor: pointer; list-style: none; }
  summary::-webkit-details-marker { display: none; }
  summary:hover .disclosure-chevron { border-color: var(--text); }
  summary:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: var(--radius); }
  .disclosure-chevron { width: 7px; height: 7px; margin-right: 2px; border-right: 1.5px solid var(--text-muted); border-bottom: 1.5px solid var(--text-muted); transform: rotate(-45deg); transition: transform 120ms ease; }
  details[open] > summary > .disclosure-chevron { transform: rotate(45deg); }
  h2 { margin: 0; font-size: 13px; }
  h3, h4 { margin: 0; font-size: 12px; overflow-wrap: anywhere; }
  .catalog-chip { border: 1px solid var(--border); border-radius: 5px; background: var(--surface-muted); padding: 2px 6px; color: var(--text-muted); font: 500 10px/1.3 var(--font-data); white-space: nowrap; }
  .warning-chip { border-color: #e3bd83; background: #fff7e8; color: #6e4700; }
  .catalog-message { min-width: 0; margin-left: auto; overflow: hidden; color: var(--text-muted); font: 500 10px/1.3 var(--font-data); text-overflow: ellipsis; white-space: nowrap; }
  .catalog-body { display: grid; gap: 12px; padding: 12px 14px 14px; border-top: 1px solid var(--border); }
  .catalog-notice { margin: 0; padding: 7px 10px; border: 1px solid #e3bd83; border-radius: var(--radius-control); background: #fff7e8; color: #6e4700; font-size: 12px; }
  .catalog-controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; }
  label { display: grid; gap: 4px; }
  label > span, dt { color: var(--text-muted); font: 600 10px/1 var(--font-data); letter-spacing: .07em; text-transform: uppercase; }
  select { min-width: 12rem; max-width: 100%; }
  .catalog-links { display: flex; flex-wrap: wrap; gap: 12px; margin-left: auto; padding-bottom: 8px; font-size: 12px; }
  a { color: var(--run-a); }
  .provenance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 0; }
  .provenance-grid div { display: grid; gap: 5px; min-width: 0; padding: 8px 10px; border-radius: 7px; background: var(--surface-muted); }
  .provenance-grid .wide { grid-column: span 2; }
  dd { margin: 0; overflow: hidden; font: 500 11px/1.3 var(--font-data); text-overflow: ellipsis; white-space: nowrap; }
  code { font: inherit; }
  .catalog-note { margin: 0; color: var(--text-muted); font-size: 12px; overflow-wrap: anywhere; }
  .catalog-runs { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
  article { display: grid; align-content: start; gap: 5px; min-width: 0; padding: 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
  .catalog-runs h3 { font-family: var(--font-data); font-size: 11px; }
  article p { margin: 0; font-size: 12px; }
  .run-meta { color: var(--text-muted); }
  .run-metrics { font-family: var(--font-data); font-size: 11px; }
  small { display: block; color: var(--text-muted); font-size: 11px; overflow-wrap: anywhere; }
  .run-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 4px; }
  .primary-action { border-color: var(--accent); background: var(--accent); color: #fff; }
  .primary-action:hover:not(:disabled) { border-color: var(--accent-hover); background: var(--accent-hover); }
  .metric-comparison { border: 1px solid var(--border); border-radius: 8px; }
  .metric-comparison summary { padding: 8px 10px; }
  .metric-comparison section { display: grid; gap: 10px; padding: 10px; border-top: 1px solid var(--border); }
  .comparison-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; }
  .comparison-list p { font-family: var(--font-data); font-size: 11px; overflow-wrap: anywhere; }
  @media (max-width: 760px) {
    .dataset-catalog { margin-inline: 10px; }
    .catalog-message { flex-basis: 100%; margin-left: 0; }
    .catalog-links { margin-left: 0; }
    .provenance-grid .wide { grid-column: auto; }
    .catalog-controls label { flex: 1 1 100%; }
    select { min-width: 0; width: 100%; }
  }
</style>
