import { afterEach, describe, expect, it, vi } from "vitest";
import { PlaybackController } from "../lib/playback";
import { makeBundle } from "./helpers/bundle";

afterEach(() => {
  vi.useRealTimers();
});

describe("PlaybackController", () => {
  it("becomes ready at the first recorded step and selects the first observed sample", () => {
    const controller = new PlaybackController();
    controller.setReady(makeBundle());

    expect(controller.getState()).toMatchObject({
      phase: "ready",
      currentStep: 0,
      selectedSampleId: "sample-1",
    });
  });

  it("plays, pauses, and advances only through recorded snapshots", () => {
    vi.useFakeTimers();
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 3 }));

    controller.play();
    expect(controller.getState().phase).toBe("playing");
    vi.advanceTimersByTime(120);
    expect(controller.getState().currentStep).toBe(1);
    controller.pause();
    vi.advanceTimersByTime(500);
    expect(controller.getState()).toMatchObject({ phase: "paused", currentStep: 1 });
  });

  it("clamps stepping at both bounds and marks the final step ended", () => {
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 3 }));

    controller.stepBy(-1);
    expect(controller.getState()).toMatchObject({ phase: "paused", currentStep: 0 });
    controller.stepBy(99);
    expect(controller.getState()).toMatchObject({ phase: "ended", currentStep: 2 });
    controller.stepBy(1);
    expect(controller.getState().currentStep).toBe(2);
  });

  it("scrubbing pauses active playback", () => {
    vi.useFakeTimers();
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 4 }));
    controller.play();

    controller.seekToStep(2);

    expect(controller.getState()).toMatchObject({ phase: "paused", currentStep: 2 });
    vi.advanceTimersByTime(500);
    expect(controller.getState().currentStep).toBe(2);
  });

  it("restarts from step zero when replaying an ended run", () => {
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 3 }));
    controller.seekToStep(2);

    controller.play();

    expect(controller.getState()).toMatchObject({ phase: "playing", currentStep: 0 });
    controller.dispose();
  });

  it("clears its timer and listeners when disposed", () => {
    vi.useFakeTimers();
    const listener = vi.fn();
    const controller = new PlaybackController();
    controller.subscribe(listener);
    controller.setReady(makeBundle());
    controller.play();
    const callsBeforeDispose = listener.mock.calls.length;

    controller.dispose();
    vi.advanceTimersByTime(500);

    expect(listener).toHaveBeenCalledTimes(callsBeforeDispose);
  });
});
