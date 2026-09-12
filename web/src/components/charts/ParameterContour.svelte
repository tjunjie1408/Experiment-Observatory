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
  import { computeContourGrid } from "../../lib/math";
  import type { RunBundle } from "../../lib/schema";

  export let bundle: RunBundle;
  export let currentStep: number;

  const resolution = 24;

  $: dataset = bundle.manifest.dataset;
  $: trajectory = finitePoints(
    bundle.snapshots.map((snapshot) => ({
      x: snapshot.b,
      y: snapshot.w,
      step: snapshot.step,
    })),
  );
  $: bDomain = niceDomain(
    bundle.snapshots.map((snapshot) => snapshot.b),
    0.3,
  );
  $: wDomain = niceDomain(
    bundle.snapshots.map((snapshot) => snapshot.w),
    0.3,
  );
  $: bScale = linearScale(bDomain, [
    CHART_MARGIN.left,
    CHART_WIDTH - CHART_MARGIN.right,
  ]);
  $: wScale = linearScale(wDomain, [
    CHART_HEIGHT - CHART_MARGIN.bottom,
    CHART_MARGIN.top,
  ]);
  $: contour = computeContourGrid(
    dataset.x,
    dataset.y,
    bDomain,
    wDomain,
    resolution,
  );
  $: finiteMse = contour.grid.flat().filter((value) => Number.isFinite(value));
  $: maxMse = finiteMse.length > 0 ? Math.max(...finiteMse) : 0;
  $: cells = contour.bValues.flatMap((b, bIndex) =>
    contour.wValues.flatMap((w, wIndex) => {
      const value = contour.grid[bIndex]?.[wIndex];
      if (value === undefined || !Number.isFinite(value)) return [];
      const intensity = maxMse > 0 ? Math.min(1, value / maxMse) : 0;
      return [
        {
          x: bScale(b) - cellWidth / 2,
          y: wScale(w) - cellHeight / 2,
          fill: `hsl(35 ${12 + intensity * 45}% ${96 - intensity * 30}%)`,
        },
      ];
    }),
  );
  $: trajectorySegments = segmentConsecutivePoints(trajectory);
  $: current = bundle.snapshots[currentStep];
  $: hasFiniteCurrent =
    current !== undefined &&
    Number.isFinite(current.b) &&
    Number.isFinite(current.w);

  const cellWidth =
    (CHART_WIDTH - CHART_MARGIN.left - CHART_MARGIN.right) / resolution;
  const cellHeight =
    (CHART_HEIGHT - CHART_MARGIN.bottom - CHART_MARGIN.top) / resolution;
</script>

<div class="chart-viewport">
  <svg
    viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
    width="100%"
    height="100%"
    role="img"
  >
    {#each cells as cell}
      <rect
        x={cell.x}
        y={cell.y}
        width={cellWidth}
        height={cellHeight}
        fill={cell.fill}
        stroke="none"
      />
    {/each}
    {#each trajectorySegments as segment}
      <polyline
        points={serializePoints(segment, bScale, wScale)}
        class="trajectory-line"
        data-recorded-steps={segment.map((point) => point.step).join(" ")}
      />
    {/each}
    {#each trajectory as point}
      <circle
        cx={bScale(point.x)}
        cy={wScale(point.y)}
        r="1.5"
        class="recorded-step-point"
        data-step={point.step}
      />
    {/each}
    {#if current && hasFiniteCurrent}
      <circle
        cx={bScale(current.b)}
        cy={wScale(current.w)}
        r="4"
        class="current-step-marker"
        data-step={current.step}
      />
    {/if}
    <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 6} class="axis-label">b</text>
    <text
      x="12"
      y={CHART_HEIGHT / 2}
      class="axis-label"
      transform={`rotate(-90 12 ${CHART_HEIGHT / 2})`}>w</text
    >
  </svg>
</div>
