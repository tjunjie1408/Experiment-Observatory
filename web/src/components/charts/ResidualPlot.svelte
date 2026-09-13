<script lang="ts">
  import {
    CHART_HEIGHT,
    CHART_MARGIN,
    CHART_WIDTH,
    finitePoints,
    linearScale,
    niceDomain,
  } from "../../lib/chartGeometry";
  import { residuals } from "../../lib/math";
  import {
    featureLabel,
    isExternalDataConfig,
    type LinearRunBundle,
    type Snapshot,
  } from "../../lib/schema";

  export let bundle: LinearRunBundle;
  export let snapshot: Snapshot;

  $: dataset = bundle.manifest.dataset;
  $: xAxisLabel = featureLabel(bundle.manifest.dataConfig);
  $: residualLabel = isExternalDataConfig(bundle.manifest.dataConfig)
    ? `residual (${bundle.manifest.dataConfig.targetUnit})`
    : "residual";
  $: values = residuals(snapshot.b, snapshot.w, dataset.x, dataset.y);
  $: points = finitePoints(
    dataset.x.map((x, index) => ({
      x,
      y: values[index] as number,
      sampleId: dataset.sampleIds[index],
    })),
  );
  $: xScale = linearScale(niceDomain(dataset.x), [
    CHART_MARGIN.left,
    CHART_WIDTH - CHART_MARGIN.right,
  ]);
  $: residualScale = linearScale(niceDomain(values), [
    CHART_HEIGHT - CHART_MARGIN.bottom,
    CHART_MARGIN.top,
  ]);
  $: zeroY = residualScale(0);
</script>

<div class="chart-viewport">
  <svg
    viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
    width="100%"
    height="100%"
    role="img"
  >
    {#if Number.isFinite(zeroY)}
      <line
        x1={CHART_MARGIN.left}
        y1={zeroY}
        x2={CHART_WIDTH - CHART_MARGIN.right}
        y2={zeroY}
        class="zero-line"
      />
    {/if}
    {#each points as point}
      <circle
        cx={xScale(point.x)}
        cy={residualScale(point.y)}
        r="3"
        class="residual-point"
        data-sample-id={point.sampleId}
      />
    {/each}
    <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 6} class="axis-label"
      >{xAxisLabel}</text
    >
    <text
      x="12"
      y={CHART_HEIGHT / 2}
      class="axis-label"
      transform={`rotate(-90 12 ${CHART_HEIGHT / 2})`}>{residualLabel}</text
    >
  </svg>
</div>
