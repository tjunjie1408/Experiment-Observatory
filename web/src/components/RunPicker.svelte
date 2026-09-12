<script lang="ts">
  import { AVAILABLE_RUNS } from "../lib/availableRuns";

  export let runAPath: string;
  export let runBPath: string;
</script>

<section class="run-picker" aria-label="Run selection">
  <label>
    <span>Run A</span>
    <select bind:value={runAPath} aria-label="Run A">
      {#each AVAILABLE_RUNS as run}
        <option value={run.path}>{run.label}</option>
      {/each}
    </select>
  </label>
  <label>
    <span>Compare with</span>
    <select bind:value={runBPath} aria-label="Compare with Run B">
      <option value="">None</option>
      {#each AVAILABLE_RUNS as run}
        <option value={run.path}>{run.label}</option>
      {/each}
    </select>
  </label>
  <div class="replay-contract">
    <span class="status-indicator success" aria-hidden="true"></span>
    Recorded steps only
  </div>
</section>
