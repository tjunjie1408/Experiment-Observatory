<script lang="ts">
  import {
    CHART_HEIGHT,
    CHART_MARGIN,
    CHART_WIDTH,
    finitePoints,
    linearScale,
    niceDomain,
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
  $: curvePoints = serializePoints(recorded, xScale, yScale);
  $: current = bundle.snapshots[currentStep];
  $: hasFiniteCurrent =
    current !== undefined &&
    Number.isFinite(current.step) &&
    Number.isFinite(current.trainMse);
</script>

<div class="chart-viewport">
  <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} width="100%" height="100%" role="img">
    <polyline
      points={curvePoints}
      class="loss-curve-line"
      data-recorded-steps={recorded.map((point) => point.step).join(" ")}
    />
    {#if current && hasFiniteCurrent}
      <circle
        cx={xScale(current.step)}
        cy={yScale(current.trainMse)}
        r="4"
        class="current-step-marker"
        data-step={current.step}
      />
    {/if}
    <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 6} class="axis-label">step</text>
    <text
      x="12"
      y={CHART_HEIGHT / 2}
      class="axis-label"
      transform={`rotate(-90 12 ${CHART_HEIGHT / 2})`}>train MSE</text
    >
  </svg>
</div>
