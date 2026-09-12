<script lang="ts">
  import type { PlaybackController, PlaybackState } from "../lib/playback";

  export let controller: PlaybackController;
  export let state: PlaybackState;

  $: lastStep = Math.max(0, (state.bundle?.snapshots.length ?? 1) - 1);
  $: disabled = state.bundle === null;
  $: playing = state.phase === "playing";

  function togglePlayback(): void {
    if (playing) controller.pause();
    else controller.play();
  }
</script>

<div class="playback-controls" role="group" aria-label="Playback controls">
  <div class="transport">
    <button
      type="button"
      on:click={() => controller.seekToStep(0)}
      disabled={disabled || state.currentStep === 0}
      aria-label="First step">|&lt;</button
    >
    <button
      type="button"
      on:click={() => controller.stepBy(-1)}
      disabled={disabled || state.currentStep === 0}
      aria-label="Step back">&lt;&lt;</button
    >
    <button
      class="play-button"
      type="button"
      on:click={togglePlayback}
      {disabled}
      aria-pressed={playing}
    >
      {playing ? "Pause" : "Play"}
    </button>
    <button
      type="button"
      on:click={() => controller.stepBy(1)}
      disabled={disabled || state.currentStep === lastStep}
      aria-label="Step forward">&gt;&gt;</button
    >
    <button
      type="button"
      on:click={() => controller.seekToStep(lastStep)}
      disabled={disabled || state.currentStep === lastStep}
      aria-label="Last step">&gt;|</button
    >
  </div>
  <label class="scrubber-label">
    <span class="sr-only">Step scrubber</span>
    <input
      type="range"
      min="0"
      max={lastStep}
      value={state.currentStep}
      {disabled}
      on:input={(event) =>
        controller.seekToStep(Number(event.currentTarget.value))}
    />
    <span class="step-readout" aria-live="polite">
      Step <strong>{String(state.currentStep).padStart(3, "0")}</strong> / {String(
        lastStep,
      ).padStart(3, "0")}
    </span>
  </label>
  <span class="phase-tag" data-phase={state.phase}>{state.phase}</span>
</div>
