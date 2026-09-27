<script lang="ts">
  import type { KMeansRunBundle, KMeansSnapshot } from "../lib/schema";

  export let bundle: KMeansRunBundle;
  export let snapshot: KMeansSnapshot;

  const colors = ["#4656a6", "#d9772b", "#8a4fb3", "#2f8a5f", "#c2415d"];
  const colorFor = (index: number) => colors[index % colors.length] ?? "#626873";
  const width = 680;
  const height = 430;
  const padding = 34;
  $: all = [...bundle.manifest.dataset.points, ...snapshot.centers];
  $: xs = all.map((point) => point[0]);
  $: ys = all.map((point) => point[1]);
  $: minX = Math.min(...xs);
  $: maxX = Math.max(...xs);
  $: minY = Math.min(...ys);
  $: maxY = Math.max(...ys);
  $: scaleX = (value: number) =>
    padding + ((value - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2);
  $: scaleY = (value: number) =>
    height - padding - ((value - minY) / Math.max(maxY - minY, 1)) * (height - padding * 2);
</script>

<div class="kmeans-layout">
  <article class="chart-card kmeans-chart">
    <header>
      <h3>Cluster assignment & centers</h3>
      <span>Recorded {snapshot.phase} phase</span>
    </header>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="K-means points and centers">
      <rect x="0" y="0" {width} {height} class="plot-bg" />
      {#each bundle.manifest.dataset.points as point, index}
        <circle
          cx={scaleX(point[0])}
          cy={scaleY(point[1])}
          r="5"
          fill={colorFor(snapshot.assignments[index] ?? 0)}
          opacity="0.82"
        />
      {/each}
      {#each snapshot.centers as center, index}
        <g transform={`translate(${scaleX(center[0])} ${scaleY(center[1])})`}>
          <circle r="11" fill={colorFor(index)} class="center-marker" />
          <text y="4" text-anchor="middle">{index}</text>
        </g>
      {/each}
    </svg>
    <footer>
      <span>N={bundle.manifest.dataset.points.length}, K={snapshot.centers.length}</span>
      <strong>Inertia {snapshot.inertia.toFixed(4)}</strong>
    </footer>
  </article>

  <aside class="kmeans-inspector">
    <div class="panel-heading">
      <h3>Lloyd state</h3>
      <span class="status-tag">Iteration {snapshot.iteration}</span>
    </div>
    <dl>
      <div><dt>Phase</dt><dd>{snapshot.phase}</dd></div>
      <div><dt>Recorded step</dt><dd>{snapshot.step}</dd></div>
      <div><dt>Init seed</dt><dd>{bundle.manifest.trainingConfig.initSeed}</dd></div>
      <div><dt>Stop reason</dt><dd>{bundle.manifest.stopReason}</dd></div>
      <div><dt>Empty clusters</dt><dd>{snapshot.emptyClusters.join(", ") || "None"}</dd></div>
    </dl>
    <p>
      {snapshot.phase === "assignment"
        ? "Assignments are nearest to the centers shown in this frame."
        : "Centers are means of the recorded assignments; reassignment happens in the next frame."}
    </p>
  </aside>
</div>

<style>
  .kmeans-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(250px, 1fr); gap: 16px; padding: 16px; }
  .kmeans-chart { min-height: 0; }
  svg { display: block; width: 100%; height: auto; margin-top: 10px; }
  .plot-bg { fill: var(--surface); stroke: var(--border); }
  .center-marker { stroke: var(--text); stroke-width: 2; }
  text { fill: #fff; font: 700 10px var(--font-data); }
  .kmeans-inspector { align-self: start; min-width: 0; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }
  dl { margin: 10px 0 0; color: var(--text-muted); font: 500 11px/1.3 var(--font-data); }
  dl div { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--grid); }
  dd { margin: 0; color: var(--text); text-align: right; overflow-wrap: anywhere; }
  p { margin: 12px 0 0; color: var(--text-muted); font-size: 12px; line-height: 1.5; }
  @media (max-width: 1080px) { .kmeans-layout { grid-template-columns: 1fr; } }
  @media (max-width: 760px) { .kmeans-layout { gap: 12px; padding: 10px; } }
</style>
