<script lang="ts">
  import { onMount } from "svelte";
  import { linearScale, niceDomain, segmentConsecutivePoints, serializePoints } from "../lib/chartGeometry";
  import { loadTrackingHistory, type TrackingHistory } from "../lib/trackingMetrics";

  export let runId: string;

  let history: TrackingHistory | null = null;
  let error = "";

  onMount(() => {
    const controller = new AbortController();
    void loadTrackingHistory(runId, controller.signal).then((loaded) => {
      if (!controller.signal.aborted) history = loaded;
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) error = reason instanceof Error ? reason.message : String(reason);
    });
    return () => controller.abort();
  });
</script>

<section aria-label={`MLflow metrics for ${runId}`} class="tracked-metrics">
  {#if error}
    <p role="alert">{error}. Recorded replay remains available independently.</p>
  {:else if history}
    <p>Verified MLflow history for {runId} · completed run, not a live training stream.</p>
    {#if history.series.length === 0}<p>No tracked metric series.</p>{/if}
    {#each history.series as series (series.key)}
      {@const xScale = linearScale(niceDomain(series.points.map((point) => point.step), 0.02), [28, 340])}
      {@const yScale = linearScale(niceDomain(series.points.map((point) => point.value)), [88, 12])}
      {@const segments = segmentConsecutivePoints(series.points.map((point) => ({ x: point.step, y: point.value, step: point.step })))}
      {@const last = series.points.at(-1)}
      <div class="metric-series">
        <h4>{series.key}</h4>
        <small>{series.unit} · {series.aggregation}</small>
        {#if series.points.length > 0}
          <svg viewBox="0 0 360 108" role="img" aria-label={`${series.key} metric history in ${series.unit}`}>
            {#each segments as segment}
              {#if segment.length > 1}
                <polyline points={serializePoints(segment, xScale, yScale)} data-recorded-steps={segment.map((point) => point.step).join(" ")} fill="none" stroke="currentColor" stroke-width="2" />
              {/if}
            {/each}
            {#each series.points as point (point.step)}
              <circle cx={xScale(point.step)} cy={yScale(point.value)} r="2.5" fill="currentColor">
                <title>{`Step ${point.step}: ${point.value} ${series.unit}`}</title>
              </circle>
            {/each}
            <text x="180" y="105" text-anchor="middle">step</text>
          </svg>
          {#if last}<p>Step {last.step}: {last.value} {series.unit}</p>{/if}
        {:else}
          <p>No recorded steps.</p>
        {/if}
      </div>
    {/each}
  {:else}
    <p role="status">Loading MLflow metric history…</p>
  {/if}
</section>

<style>
  .tracked-metrics { margin-top: .7rem; padding: .7rem; border: 1px solid var(--border); border-radius: 7px; background: var(--surface-muted); font-size: 12px; }
  .metric-series { margin-top: .7rem; }
  .metric-series h4 { margin: 0; overflow-wrap: anywhere; }
  svg { display: block; width: 100%; max-width: 360px; color: var(--run-a); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }
  svg text { fill: var(--text-muted); font-size: 9px; }
</style>
