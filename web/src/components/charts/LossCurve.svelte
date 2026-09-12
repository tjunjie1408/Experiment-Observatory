<script lang="ts">
  import {
    CHART_HEIGHT,
    CHART_MARGIN,
    CHART_WIDTH,
    finitePoints,
    linearScale,
    niceDomain,
    segmentConsecutivePoints,
    serializePoints,
  } from "../../lib/chartGeometry";
  import type { RunBundle } from "../../lib/schema";

  export let bundle: RunBundle;
  export let currentStep: number;

  $: recorded = finitePoints(
    bundle.snapshots.map((snapshot) => ({
      x: snapshot.step,
      y: snapshot.trainMse,
      step: snapshot.step,
    })),
  );
  $: xScale = linearScale(
    niceDomain(
      bundle.snapshots.map((snapshot) => snapshot.step),
      0.02,
    ),
    [CHART_MARGIN.left, CHART_WIDTH - CHART_MARGIN.right],
  );
  $: yScale = linearScale(
    niceDomain(bundle.snapshots.map((snapshot) => snapshot.trainMse)),
    [CHART_HEIGHT - CHART_MARGIN.bottom, CHART_MARGIN.top],
  );
  $: segments = segmentConsecutivePoints(recorded);
  $: current = bundle.snapshots[currentStep];
  $: hasFiniteCurrent =
    current !== undefined &&
    Number.isFinite(current.step) &&
    Number.isFinite(current.trainMse);
</script>

<div class="chart-viewport">
  <svg
    viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
    width="100%"
    height="100%"
    role="img"
  >
    {#each segments as segment}
      <polyline
        points={serializePoints(segment, xScale, yScale)}
        class="loss-curve-line"
        data-recorded-steps={segment.map((point) => point.step).join(" ")}
      />
    {/each}
    {#each recorded as point}
      <circle
        cx={xScale(point.x)}
        cy={yScale(point.y)}
        r="1.5"
        class="recorded-step-point"
        data-step={point.step}
      />
    {/each}
    {#if current && hasFiniteCurrent}
      <circle
        cx={xScale(current.step)}
        cy={yScale(current.trainMse)}
        r="4"
        class="current-step-marker"
        data-step={current.step}
      />
    {/if}
    <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 6} class="axis-label">step</text
    >
    <text
      x="12"
      y={CHART_HEIGHT / 2}
      class="axis-label"
      transform={`rotate(-90 12 ${CHART_HEIGHT / 2})`}>train MSE</text
    >
  </svg>
</div>
