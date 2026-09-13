<script lang="ts">
  import type { KMeansRunBundle, KMeansSnapshot } from "../lib/schema";

  export let bundle: KMeansRunBundle;
  export let snapshot: KMeansSnapshot;

  const colors = ["#38bdf8", "#f97316", "#a78bfa", "#22c55e", "#f43f5e"];
  const colorFor = (index: number) => colors[index % colors.length] ?? "#e2e8f0";
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
          <circle r="11" fill={colorFor(index)} stroke="white" stroke-width="3" />
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
    <span class="eyebrow">Lloyd state</span>
    <h3>Iteration {snapshot.iteration}</h3>
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
  .kmeans-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(230px, 1fr); gap: 1rem; padding: 1rem; }
  .kmeans-chart { min-height: 0; }
  svg { display: block; width: 100%; height: auto; }
  .plot-bg { fill: rgba(15, 23, 42, 0.72); stroke: rgba(148, 163, 184, 0.2); }
  text { fill: #07111f; font-size: 10px; font-weight: 800; }
  .kmeans-inspector { border: 1px solid var(--border, #334155); border-radius: 12px; padding: 1rem; background: rgba(15, 23, 42, 0.7); }
  .eyebrow { color: #38bdf8; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }
  dl div { display: flex; justify-content: space-between; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.16); }
  dt { color: #94a3b8; }
  dd { margin: 0; text-align: right; }
  p { color: #cbd5e1; line-height: 1.5; }
  @media (max-width: 780px) { .kmeans-layout { grid-template-columns: 1fr; } }
</style>
