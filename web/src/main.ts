/**
 * App shell: run selection, playback controls, sample inspector, and A/B
 * compare. No Python backend; everything here reads static exported
 * bundles.
 */

import "./style.css";
import { loadRunBundle, BundleLoadError } from "./lib/loader";
import { PlaybackController, type PlaybackState } from "./lib/playback";
import { renderScatterView, renderResidualsView, renderLossCurveView, renderContourView } from "./lib/views";
import type { RunBundle } from "./lib/schema";

const AVAILABLE_RUNS: { id: string; label: string; path: string }[] = [
  { id: "converge", label: "Converge (lr=0.25)", path: "/runs/converge" },
  { id: "slow", label: "Slow (lr=0.001)", path: "/runs/slow" },
  { id: "diverge", label: "Diverge (lr=1.5)", path: "/runs/diverge" },
];

const app = document.querySelector<HTMLDivElement>("#app");
if (app === null) throw new Error("missing #app root element");

app.innerHTML = `
  <header>
    <h1>AI Experiment Observatory &mdash; Linear Regression Replay</h1>
    <p class="subtitle">Static replay of a real recorded training run. No training happens in the browser.</p>
  </header>
  <section class="run-picker" aria-label="Run selection">
    <label for="run-select-a">Run A</label>
    <select id="run-select-a"></select>
    <label for="run-select-b">Compare with</label>
    <select id="run-select-b">
      <option value="">(none)</option>
    </select>
  </section>
  <section id="status-a" class="status" role="status" aria-live="polite"></section>
  <div id="panel-a" class="run-panel"></div>
  <section id="status-b" class="status" role="status" aria-live="polite" hidden></section>
  <div id="panel-b" class="run-panel" hidden></div>
  <section id="compare-note" class="compare-note" hidden></section>
`;

function buildRunPanel(container: HTMLElement, label: string): void {
  container.innerHTML = `
    <h2>${label}</h2>
    <div class="controls" role="group" aria-label="Playback controls for ${label}">
      <button type="button" class="btn-step-back" aria-label="Step back">&#9664;&#9664;</button>
      <button type="button" class="btn-play-pause" aria-pressed="false">Play</button>
      <button type="button" class="btn-step-forward" aria-label="Step forward">&#9654;&#9654;</button>
      <input type="range" class="scrubber" min="0" max="0" value="0" aria-label="Step scrubber for ${label}" />
      <span class="step-readout" aria-live="polite">step 0 / 0</span>
    </div>
    <div class="views-grid">
      <div class="view-cell"><h3>Scatter &amp; regression line</h3><div class="view scatter"></div></div>
      <div class="view-cell"><h3>Residuals</h3><div class="view residuals"></div></div>
      <div class="view-cell"><h3>Loss curve (train MSE vs step)</h3><div class="view loss-curve"></div></div>
      <div class="view-cell"><h3>Parameter trajectory over loss contour</h3><div class="view contour"></div></div>
    </div>
    <div class="sample-inspector">
      <label>Selected sample</label>
      <select class="sample-select"></select>
      <div class="sample-readout"></div>
    </div>
  `;
}

class RunPanel {
  private controller = new PlaybackController();
  private container: HTMLElement;
  private statusEl: HTMLElement;
  private currentLoadToken = 0;

  constructor(container: HTMLElement, statusEl: HTMLElement, private label: string) {
    this.container = container;
    this.statusEl = statusEl;
    buildRunPanel(this.container, label);
    this.wireControls();
    this.controller.subscribe((state) => this.render(state));
  }

  async load(path: string): Promise<void> {
    const token = ++this.currentLoadToken;
    this.controller.startLoading();
    this.statusEl.textContent = `Loading ${this.label}…`;
    const controller = new AbortController();
    try {
      const bundle = await loadRunBundle(path, controller.signal);
      if (token !== this.currentLoadToken) return; // a newer selection superseded this load
      this.controller.setReady(bundle);
      this.statusEl.textContent = `${this.label}: ${bundle.manifest.runId} (status: ${bundle.manifest.status})`;
    } catch (exc) {
      if (token !== this.currentLoadToken) return;
      const message = exc instanceof BundleLoadError ? exc.message : String(exc);
      this.controller.setError(message);
      this.statusEl.textContent = `${this.label}: failed to load — ${message}`;
    }
  }

  getBundle(): RunBundle | null {
    return this.controller.getState().bundle;
  }

  getState() {
    return this.controller.getState();
  }

  private wireControls(): void {
    const playPauseBtn = this.container.querySelector<HTMLButtonElement>(".btn-play-pause");
    const stepBackBtn = this.container.querySelector<HTMLButtonElement>(".btn-step-back");
    const stepForwardBtn = this.container.querySelector<HTMLButtonElement>(".btn-step-forward");
    const scrubber = this.container.querySelector<HTMLInputElement>(".scrubber");
    const sampleSelect = this.container.querySelector<HTMLSelectElement>(".sample-select");

    playPauseBtn?.addEventListener("click", () => {
      const state = this.controller.getState();
      if (state.phase === "playing") this.controller.pause();
      else this.controller.play();
    });
    stepBackBtn?.addEventListener("click", () => this.controller.stepBy(-1));
    stepForwardBtn?.addEventListener("click", () => this.controller.stepBy(1));
    scrubber?.addEventListener("input", () => {
      this.controller.seekToStep(Number(scrubber.value));
    });
    sampleSelect?.addEventListener("change", () => {
      this.controller.selectSample(sampleSelect.value || null);
    });

    this.container.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        this.controller.stepBy(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        this.controller.stepBy(1);
      } else if (event.key === " ") {
        event.preventDefault();
        const state = this.controller.getState();
        if (state.phase === "playing") this.controller.pause();
        else this.controller.play();
      }
    });
  }

  private render(state: PlaybackState): void {
    const playPauseBtn = this.container.querySelector<HTMLButtonElement>(".btn-play-pause");
    const scrubber = this.container.querySelector<HTMLInputElement>(".scrubber");
    const readout = this.container.querySelector<HTMLElement>(".step-readout");
    const sampleSelect = this.container.querySelector<HTMLSelectElement>(".sample-select");
    const sampleReadout = this.container.querySelector<HTMLElement>(".sample-readout");

    if (playPauseBtn !== null) {
      const isPlaying = state.phase === "playing";
      playPauseBtn.textContent = isPlaying ? "Pause" : "Play";
      playPauseBtn.setAttribute("aria-pressed", String(isPlaying));
      playPauseBtn.disabled = state.bundle === null;
    }

    if (state.bundle === null) {
      if (scrubber !== null) {
        scrubber.disabled = true;
        scrubber.max = "0";
        scrubber.value = "0";
      }
      if (readout !== null) readout.textContent = "step — / —";
      if (sampleSelect !== null) sampleSelect.innerHTML = "";
      if (sampleReadout !== null) sampleReadout.textContent = "";
      return;
    }

    const lastStep = state.bundle.snapshots.length - 1;
    if (scrubber !== null) {
      scrubber.disabled = false;
      scrubber.max = String(lastStep);
      scrubber.value = String(state.currentStep);
    }
    if (readout !== null) {
      readout.textContent = `step ${state.currentStep} / ${lastStep} — ${state.phase}`;
    }

    if (sampleSelect !== null && sampleSelect.options.length === 0) {
      for (const sampleId of state.bundle.manifest.observedSampleIds) {
        const option = document.createElement("option");
        option.value = sampleId;
        option.textContent = sampleId;
        sampleSelect.appendChild(option);
      }
    }
    if (sampleSelect !== null && state.selectedSampleId !== null) {
      sampleSelect.value = state.selectedSampleId;
    }

    const snapshot = state.bundle.snapshots[state.currentStep];
    if (snapshot === undefined) return;

    if (sampleReadout !== null) {
      const sampleId = state.selectedSampleId;
      if (sampleId !== null && sampleId in snapshot.observedPredictions) {
        const idx = state.bundle.manifest.dataset.sampleIds.indexOf(sampleId);
        const actualY = state.bundle.manifest.dataset.y[idx];
        const pred = snapshot.observedPredictions[sampleId];
        sampleReadout.textContent = `${sampleId}: predicted=${pred?.toFixed(4)} actual=${actualY?.toFixed(4)}`;
      } else if (sampleId !== null) {
        sampleReadout.textContent = `${sampleId}: not tracked in this run's observed samples`;
      } else {
        sampleReadout.textContent = "";
      }
    }

    const scatterEl = this.container.querySelector<HTMLElement>(".scatter");
    const residualsEl = this.container.querySelector<HTMLElement>(".residuals");
    const lossCurveEl = this.container.querySelector<HTMLElement>(".loss-curve");
    const contourEl = this.container.querySelector<HTMLElement>(".contour");
    if (scatterEl !== null) renderScatterView(scatterEl, state.bundle, snapshot);
    if (residualsEl !== null) renderResidualsView(residualsEl, state.bundle, snapshot);
    if (lossCurveEl !== null) renderLossCurveView(lossCurveEl, state.bundle, state.currentStep);
    if (contourEl !== null) renderContourView(contourEl, state.bundle, state.currentStep);
  }
}

const panelAEl = document.querySelector<HTMLElement>("#panel-a")!;
const panelBEl = document.querySelector<HTMLElement>("#panel-b")!;
const statusAEl = document.querySelector<HTMLElement>("#status-a")!;
const statusBEl = document.querySelector<HTMLElement>("#status-b")!;
const compareNoteEl = document.querySelector<HTMLElement>("#compare-note")!;
const runSelectA = document.querySelector<HTMLSelectElement>("#run-select-a")!;
const runSelectB = document.querySelector<HTMLSelectElement>("#run-select-b")!;

for (const run of AVAILABLE_RUNS) {
  const optionA = document.createElement("option");
  optionA.value = run.path;
  optionA.textContent = run.label;
  runSelectA.appendChild(optionA);

  const optionB = document.createElement("option");
  optionB.value = run.path;
  optionB.textContent = run.label;
  runSelectB.appendChild(optionB);
}

const panelA = new RunPanel(panelAEl, statusAEl, "Run A");
const panelB = new RunPanel(panelBEl, statusBEl, "Run B");

function updateCompareNote(): void {
  const bundleA = panelA.getBundle();
  const bundleB = panelB.getBundle();
  if (bundleB === null) {
    compareNoteEl.hidden = true;
    return;
  }
  if (bundleA === null) {
    compareNoteEl.hidden = false;
    compareNoteEl.textContent = "Run A has not finished loading; comparison unavailable.";
    return;
  }

  const dataA = bundleA.manifest.dataConfig;
  const dataB = bundleB.manifest.dataConfig;
  const sameData =
    dataA.seed === dataB.seed &&
    dataA.nSamples === dataB.nSamples &&
    dataA.trueBias === dataB.trueBias &&
    dataA.trueWeight === dataB.trueWeight &&
    dataA.noiseStd === dataB.noiseStd;
  const sameInit =
    bundleA.manifest.trainingConfig.initialBias === bundleB.manifest.trainingConfig.initialBias &&
    bundleA.manifest.trainingConfig.initialWeight === bundleB.manifest.trainingConfig.initialWeight;

  compareNoteEl.hidden = false;
  if (!sameData || !sameInit) {
    compareNoteEl.textContent =
      "These runs use different data or initialization; comparison is shown side by side only, values are not aligned or interpolated.";
    return;
  }

  const lenA = bundleA.snapshots.length;
  const lenB = bundleB.snapshots.length;
  const rateA = bundleA.manifest.trainingConfig.learningRate;
  const rateB = bundleB.manifest.trainingConfig.learningRate;
  let note = `Same data and initialization. Learning rate: A=${rateA}, B=${rateB}.`;
  if (lenA !== lenB) {
    note += ` Different step budgets (A has ${lenA} steps, B has ${lenB} steps): steps beyond the shorter run show no recorded data for that run, not an extrapolated value.`;
  }
  compareNoteEl.textContent = note;
}

runSelectA.addEventListener("change", () => {
  void panelA.load(runSelectA.value).then(updateCompareNote);
});
runSelectB.addEventListener("change", () => {
  if (runSelectB.value === "") {
    panelBEl.hidden = true;
    statusBEl.hidden = true;
    compareNoteEl.hidden = true;
    return;
  }
  panelBEl.hidden = false;
  statusBEl.hidden = false;
  void panelB.load(runSelectB.value).then(updateCompareNote);
});

runSelectA.value = AVAILABLE_RUNS[0]?.path ?? "";
void panelA.load(runSelectA.value).then(updateCompareNote);
