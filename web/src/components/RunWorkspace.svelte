<script lang="ts">
  import { onMount } from "svelte";
  import { BundleLoadError, loadRunBundle } from "../lib/loader";
  import { PlaybackController, type PlaybackState } from "../lib/playback";
  import type { RunBundle } from "../lib/schema";
  import PlaybackControls from "./PlaybackControls.svelte";
  import SampleInspector from "./SampleInspector.svelte";
  import TelemetryPanel from "./TelemetryPanel.svelte";
  import LossCurve from "./charts/LossCurve.svelte";
  import ParameterContour from "./charts/ParameterContour.svelte";
  import ResidualPlot from "./charts/ResidualPlot.svelte";
  import ScatterPlot from "./charts/ScatterPlot.svelte";

  export let path: string;
  export let label: "Run A" | "Run B";
  export let onBundleChange: (bundle: RunBundle | null) => void = () =>
    undefined;

  const controller = new PlaybackController();
  let state: PlaybackState = controller.getState();
  let statusMessage = "No run selected";
  let mounted = false;
  let requestedPath = "";
  let loadToken = 0;
  let activeRequest: AbortController | null = null;

  $: snapshot = state.bundle?.snapshots[state.currentStep];
  $: if (mounted && path && path !== requestedPath) void load(path);

  async function load(nextPath: string): Promise<void> {
    requestedPath = nextPath;
    const token = ++loadToken;
    activeRequest?.abort();
    activeRequest = new AbortController();
    controller.startLoading();
    onBundleChange(null);
    statusMessage = `Loading ${label}`;

    try {
      const bundle = await loadRunBundle(nextPath, activeRequest.signal);
      if (token !== loadToken) return;
      controller.setReady(bundle);
      statusMessage = `${bundle.manifest.runId} loaded with ${bundle.snapshots.length} recorded snapshots`;
      onBundleChange(bundle);
    } catch (error) {
      if (token !== loadToken || activeRequest.signal.aborted) return;
      const message =
        error instanceof BundleLoadError ? error.message : String(error);
      controller.setError(message);
      statusMessage = `Failed to load ${label}: ${message}`;
      onBundleChange(null);
    }
  }

  function handleKeyboard(event: KeyboardEvent): void {
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLSelectElement ||
      event.target instanceof HTMLButtonElement
    )
      return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      controller.stepBy(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      controller.stepBy(1);
    } else if (event.key === " ") {
      event.preventDefault();
      if (state.phase === "playing") controller.pause();
      else controller.play();
    }
  }

  onMount(() => {
    mounted = true;
    const unsubscribe = controller.subscribe((nextState) => {
      state = nextState;
    });
    if (path) void load(path);

    return () => {
      mounted = false;
      activeRequest?.abort();
      unsubscribe();
      controller.dispose();
    };
  });
</script>

<section
  class:run-b={label === "Run B"}
  class="run-workspace"
  aria-label={`${label} workspace`}
>
  <div
    class="workspace-toolbar"
    role="toolbar"
    aria-label={`Replay toolbar for ${label}`}
    tabindex="0"
    on:keydown={handleKeyboard}
  >
    <div class="run-identity">
      <span class="series-tag">{label}</span>
      <span role="status" aria-live="polite">{statusMessage}</span>
    </div>
    <PlaybackControls {controller} {state} />
  </div>

  {#if state.phase === "loading"}
    <div class="loading-state" aria-hidden="true">
      <div class="loading-bar"></div>
      <div class="loading-grid">
        <span></span><span></span><span></span><span></span>
      </div>
    </div>
  {:else if state.phase === "error"}
    <div class="error-state" role="alert">
      <strong>Run artifact unavailable</strong>
      <p>{state.errorMessage}</p>
      <button type="button" on:click={() => load(path)}>Retry load</button>
    </div>
  {:else if state.bundle && snapshot}
    <div class="workspace-layout">
      <div class="charts-grid">
        <article class="chart-card chart-primary">
          <header>
            <h3>Observed samples & linear fit</h3>
            <span>ŷ = w·x + b</span>
          </header>
          <ScatterPlot bundle={state.bundle} {snapshot} />
          <footer>
            <span>N={state.bundle.manifest.dataset.x.length}</span><strong
              >w={snapshot.w.toFixed(4)} b={snapshot.b.toFixed(4)}</strong
            >
          </footer>
        </article>
        <article class="chart-card">
          <header>
            <h3>Residual distribution</h3>
            <span>e = ŷ - y</span>
          </header>
          <ResidualPlot bundle={state.bundle} {snapshot} />
          <footer>
            <span>Derived from recorded parameters</span><strong
              >Step {snapshot.step}</strong
            >
          </footer>
        </article>
        <article class="chart-card">
          <header>
            <h3>MSE loss vs. step</h3>
            <span>Recorded trace</span>
          </header>
          <LossCurve bundle={state.bundle} currentStep={state.currentStep} />
          <footer>
            <span>Initial {state.bundle.snapshots[0]?.trainMse.toFixed(4)}</span
            ><strong>Current {snapshot.trainMse.toFixed(4)}</strong>
          </footer>
        </article>
        <article class="chart-card">
          <header>
            <h3>Parameter trajectory</h3>
            <span>Derived loss surface</span>
          </header>
          <ParameterContour
            bundle={state.bundle}
            currentStep={state.currentStep}
          />
          <footer>
            <span>Bias b / weight w</span><strong>Recorded path</strong>
          </footer>
        </article>
      </div>
      <div class="inspector-stack">
        <TelemetryPanel bundle={state.bundle} {snapshot} />
        <SampleInspector
          {controller}
          bundle={state.bundle}
          {snapshot}
          selectedSampleId={state.selectedSampleId}
        />
      </div>
    </div>
  {:else}
    <div class="empty-state">Select a recorded run to begin replay.</div>
  {/if}
</section>
