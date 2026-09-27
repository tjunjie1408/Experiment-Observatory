<script lang="ts">
  import { groupAvailableRuns } from "../lib/availableRuns";

  export let runAPath: string;
  export let runBPath: string;

  const groups = groupAvailableRuns();
</script>

<section class="run-picker" aria-label="Run selection">
  <label>
    <span>Run A</span>
    <select bind:value={runAPath} aria-label="Run A">
      {#each groups as group (group.group)}
        <optgroup label={group.label}>
          {#each group.runs as run (run.id)}
            <option value={run.path}>{run.label}</option>
          {/each}
        </optgroup>
      {/each}
    </select>
  </label>
  <label>
    <span>Compare with</span>
    <select bind:value={runBPath} aria-label="Compare with Run B">
      <option value="">None</option>
      {#each groups as group (group.group)}
        <optgroup label={group.label}>
          {#each group.runs as run (run.id)}
            <option value={run.path}>{run.label}</option>
          {/each}
        </optgroup>
      {/each}
    </select>
  </label>
  <div class="replay-contract">
    <span class="status-indicator success" aria-hidden="true"></span>
    Recorded steps only
  </div>
</section>
