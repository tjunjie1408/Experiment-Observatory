/**
 * Playback state machine for a single loaded run:
 *
 *   empty -> loading -> ready -> playing -> paused -> playing -> ... -> ended
 *                     \-> error
 *
 * Only real recorded steps are played; dragging always lands on a valid
 * step and pauses (no silent continued playback after a scrub). Re-playing
 * from `ended` restarts at step 0.
 */

import type { RunBundle } from "./schema";

export type PlaybackPhase = "empty" | "loading" | "ready" | "playing" | "paused" | "ended" | "error";

export interface PlaybackState {
  phase: PlaybackPhase;
  bundle: RunBundle | null;
  currentStep: number;
  selectedSampleId: string | null;
  errorMessage: string | null;
}

export type PlaybackListener = (state: PlaybackState) => void;

const STEP_INTERVAL_MS = 120;

export class PlaybackController {
  private state: PlaybackState = {
    phase: "empty",
    bundle: null,
    currentStep: 0,
    selectedSampleId: null,
    errorMessage: null,
  };

  private listeners: Set<PlaybackListener> = new Set();
  private timer: ReturnType<typeof setInterval> | null = null;

  getState(): PlaybackState {
    return this.state;
  }

  subscribe(listener: PlaybackListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private setState(partial: Partial<PlaybackState>): void {
    this.state = { ...this.state, ...partial };
    for (const listener of this.listeners) listener(this.state);
  }

  startLoading(): void {
    this.stopTimer();
    this.setState({
      phase: "loading",
      bundle: null,
      currentStep: 0,
      selectedSampleId: null,
      errorMessage: null,
    });
  }

  setReady(bundle: RunBundle): void {
    this.stopTimer();
    const firstSampleId = bundle.manifest.observedSampleIds[0] ?? null;
    this.setState({
      phase: "ready",
      bundle,
      currentStep: 0,
      selectedSampleId: firstSampleId,
      errorMessage: null,
    });
  }

  setError(message: string): void {
    this.stopTimer();
    this.setState({ phase: "error", bundle: null, errorMessage: message });
  }

  play(): void {
    if (this.state.bundle === null) return;
    if (this.state.phase === "ended") {
      this.setState({ currentStep: 0 });
    }
    if (this.state.phase !== "ready" && this.state.phase !== "paused" && this.state.phase !== "ended") {
      return;
    }
    this.setState({ phase: "playing" });
    this.stopTimer();
    this.timer = setInterval(() => this.advance(), STEP_INTERVAL_MS);
  }

  pause(): void {
    if (this.state.phase !== "playing") return;
    this.stopTimer();
    this.setState({ phase: "paused" });
  }

  private advance(): void {
    const bundle = this.state.bundle;
    if (bundle === null) return;
    const lastStep = bundle.snapshots.length - 1;
    const nextStep = this.state.currentStep + 1;
    if (nextStep > lastStep) {
      this.stopTimer();
      this.setState({ phase: "ended", currentStep: lastStep });
      return;
    }
    this.setState({ currentStep: nextStep });
  }

  /** Scrub to an arbitrary recorded step; always pauses. */
  seekToStep(step: number): void {
    const bundle = this.state.bundle;
    if (bundle === null) return;
    const lastStep = bundle.snapshots.length - 1;
    const clamped = Math.max(0, Math.min(step, lastStep));
    this.stopTimer();
    const phase: PlaybackPhase = clamped === lastStep ? "ended" : "paused";
    this.setState({ phase, currentStep: clamped });
  }

  stepBy(delta: number): void {
    this.seekToStep(this.state.currentStep + delta);
  }

  selectSample(sampleId: string | null): void {
    this.setState({ selectedSampleId: sampleId });
  }

  private stopTimer(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  dispose(): void {
    this.stopTimer();
    this.listeners.clear();
  }
}
