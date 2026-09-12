<script lang="ts">
  import { afterUpdate, onMount } from "svelte";
  import type { RunBundle, Snapshot } from "../../lib/schema";
  import {
    renderContourView,
    renderLossCurveView,
    renderResidualsView,
    renderScatterView,
  } from "../../lib/views";

  export let kind: "scatter" | "residuals" | "loss" | "contour";
  export let bundle: RunBundle;
  export let snapshot: Snapshot;
  export let currentStep: number;

  let container: HTMLDivElement;

  function render(): void {
    if (!container) return;
    if (kind === "scatter") renderScatterView(container, bundle, snapshot);
    else if (kind === "residuals") renderResidualsView(container, bundle, snapshot);
    else if (kind === "loss") renderLossCurveView(container, bundle, currentStep);
    else renderContourView(container, bundle, currentStep);
  }

  onMount(render);
  afterUpdate(render);
</script>

<div class="chart-viewport" bind:this={container}></div>
