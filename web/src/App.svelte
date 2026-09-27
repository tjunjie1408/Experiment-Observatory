<script lang="ts">
  import AppHeader from "./components/AppHeader.svelte";
  import ComparisonNotice from "./components/ComparisonNotice.svelte";
  import DatasetCatalog from "./components/DatasetCatalog.svelte";
  import RunPicker from "./components/RunPicker.svelte";
  import RunWorkspace from "./components/RunWorkspace.svelte";
  import { AVAILABLE_RUNS, studyLabelForPath } from "./lib/availableRuns";
  import { getComparisonNotice } from "./lib/comparison";
  import type { RunBundle } from "./lib/schema";

  let runAPath = AVAILABLE_RUNS[0]?.path ?? "";
  let runBPath = "";
  let bundleA: RunBundle | null = null;
  let bundleB: RunBundle | null = null;

  $: comparisonActive = runBPath !== "";
  $: comparisonNotice = comparisonActive
    ? getComparisonNotice(bundleA, bundleB)
    : null;
  $: if (!comparisonActive) bundleB = null;
  $: studyLabel = studyLabelForPath(runAPath) ?? "Recorded replay";
</script>

<AppHeader {comparisonActive} {studyLabel} />
<main>
  <RunPicker bind:runAPath bind:runBPath />
  <DatasetCatalog onSelectReplay={(path: string) => (runAPath = path)} />
  {#if comparisonNotice}
    <ComparisonNotice notice={comparisonNotice} />
  {/if}
  <div class:comparison-grid={comparisonActive} class="workspace-collection">
    <RunWorkspace
      path={runAPath}
      label="Run A"
      onBundleChange={(bundle: RunBundle | null) => (bundleA = bundle)}
    />
    {#if comparisonActive}
      <RunWorkspace
        path={runBPath}
        label="Run B"
        onBundleChange={(bundle: RunBundle | null) => (bundleB = bundle)}
      />
    {/if}
  </div>
</main>
<footer class="app-footer">
  <span>Deterministic static replay</span>
  <span>Keyboard: left/right step, space play</span>
</footer>
