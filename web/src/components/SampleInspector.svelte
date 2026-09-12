<script lang="ts">
  import type { PlaybackController } from "../lib/playback";
  import {
    featureLabel,
    targetLabel,
    type RunBundle,
    type Snapshot,
  } from "../lib/schema";

  export let controller: PlaybackController;
  export let bundle: RunBundle;
  export let snapshot: Snapshot;
  export let selectedSampleId: string | null;

  $: inputLabel = featureLabel(bundle.manifest.dataConfig);
  $: outputLabel = targetLabel(bundle.manifest.dataConfig);
  $: sampleIndex =
    selectedSampleId === null
      ? -1
      : bundle.manifest.dataset.sampleIds.indexOf(selectedSampleId);
  $: actual =
    sampleIndex >= 0 ? bundle.manifest.dataset.y[sampleIndex] : undefined;
  $: input =
    sampleIndex >= 0 ? bundle.manifest.dataset.x[sampleIndex] : undefined;
  $: prediction =
    selectedSampleId === null
      ? undefined
      : snapshot.observedPredictions[selectedSampleId];
  $: residual =
    prediction !== undefined && actual !== undefined
      ? prediction - actual
      : undefined;

  function format(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value)
      ? "Unavailable"
      : value.toFixed(4);
  }
</script>

<section class="sample-inspector" aria-labelledby="sample-title">
  <div class="panel-heading">
    <h3 id="sample-title">Sample inspector</h3>
    <label>
      <span class="sr-only">Selected sample</span>
      <select
        value={selectedSampleId ?? ""}
        on:change={(event) =>
          controller.selectSample(event.currentTarget.value || null)}
      >
        {#each bundle.manifest.observedSampleIds as sampleId}
          <option value={sampleId}>{sampleId}</option>
        {/each}
      </select>
    </label>
  </div>
  {#if selectedSampleId !== null && prediction !== undefined}
    <dl class="sample-values">
      <div>
        <dt>Input {inputLabel}</dt>
        <dd>{format(input)}</dd>
      </div>
      <div>
        <dt>Actual {outputLabel}</dt>
        <dd>{format(actual)}</dd>
      </div>
      <div>
        <dt>Prediction ŷ</dt>
        <dd>{format(prediction)}</dd>
      </div>
      <div>
        <dt>Residual</dt>
        <dd class:negative={(residual ?? 0) < 0}>{format(residual)}</dd>
      </div>
    </dl>
  {:else}
    <p class="empty-message">
      This sample is not tracked in the current snapshot.
    </p>
  {/if}
</section>
