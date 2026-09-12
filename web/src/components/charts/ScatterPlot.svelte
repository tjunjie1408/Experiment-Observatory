<script lang="ts">
  import {
    CHART_HEIGHT,
    CHART_MARGIN,
    CHART_WIDTH,
    finitePoints,
    linearScale,
    niceDomain,
  } from "../../lib/chartGeometry";
  import { predict } from "../../lib/math";
  import type { RunBundle, Snapshot } from "../../lib/schema";

  export let bundle: RunBundle;
  export let snapshot: Snapshot;

  $: dataset = bundle.manifest.dataset;
  $: predictions = predict(snapshot.b, snapshot.w, dataset.x);
  $: points = finitePoints(
    dataset.x.map((x, index) => ({
      x,
      y: dataset.y[index] as number,
      sampleId: dataset.sampleIds[index],
    })),
  );
  $: xDomain = niceDomain(dataset.x);
  $: yDomain = niceDomain([...dataset.y, ...predictions]);
  $: xScale = linearScale(xDomain, [
    CHART_MARGIN.left,
    CHART_WIDTH - CHART_MARGIN.right,
  ]);
  $: yScale = linearScale(yDomain, [
    CHART_HEIGHT - CHART_MARGIN.bottom,
    CHART_MARGIN.top,
  ]);
  $: line = {
    x1: xScale(xDomain[0]),
    y1: yScale(snapshot.b + snapshot.w * xDomain[0]),
    x2: xScale(xDomain[1]),
    y2: yScale(snapshot.b + snapshot.w * xDomain[1]),
  };
  $: hasFiniteLine = Object.values(line).every(Number.isFinite);
</script>

<div class="chart-viewport">
  <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} width="100%" height="100%" role="img">
    {#each points as point}
      <circle
        cx={xScale(point.x)}
        cy={yScale(point.y)}
        r="3"
        class="sample-point"
        data-sample-id={point.sampleId}
      />
    {/each}
    {#if hasFiniteLine}
      <line
        x1={line.x1}
        y1={line.y1}
        x2={line.x2}
        y2={line.y2}
        class="regression-line"
      />
    {/if}
    <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 6} class="axis-label">x</text>
    <text
      x="12"
      y={CHART_HEIGHT / 2}
      class="axis-label"
      transform={`rotate(-90 12 ${CHART_HEIGHT / 2})`}>y</text
    >
  </svg>
</div>
