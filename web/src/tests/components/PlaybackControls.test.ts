// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import PlaybackControls from "../../components/PlaybackControls.svelte";
import { PlaybackController } from "../../lib/playback";
import { makeBundle } from "../helpers/bundle";

afterEach(cleanup);

describe("PlaybackControls", () => {
  it("disables every control until a run is ready", () => {
    const controller = new PlaybackController();

    render(PlaybackControls, { controller, state: controller.getState() });

    for (const control of screen.getAllByRole("button")) {
      expect((control as HTMLButtonElement).disabled).toBe(true);
    }
    expect(
      (
        screen.getByRole("slider", {
          name: "Step scrubber",
        }) as HTMLInputElement
      ).disabled,
    ).toBe(true);
  });

  it("enforces the first and last recorded-step boundaries", async () => {
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 3 }));
    const view = render(PlaybackControls, {
      controller,
      state: controller.getState(),
    });

    expect(
      (screen.getByRole("button", { name: "First step" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Step back" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (
        screen.getByRole("button", {
          name: "Step forward",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);

    await userEvent.click(screen.getByRole("button", { name: "Last step" }));
    await view.rerender({ controller, state: controller.getState() });

    expect(
      (
        screen.getByRole("slider", {
          name: "Step scrubber",
        }) as HTMLInputElement
      ).value,
    ).toBe("2");
    expect(
      (
        screen.getByRole("button", {
          name: "Step forward",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Last step" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("plays, pauses, and pauses playback when the user scrubs", async () => {
    const controller = new PlaybackController();
    controller.setReady(makeBundle({ snapshotCount: 3 }));
    const view = render(PlaybackControls, {
      controller,
      state: controller.getState(),
    });
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Play" }));
    await view.rerender({ controller, state: controller.getState() });
    expect(
      screen
        .getByRole("button", { name: "Pause" })
        .getAttribute("aria-pressed"),
    ).toBe("true");

    await user.click(screen.getByRole("button", { name: "Pause" }));
    await view.rerender({ controller, state: controller.getState() });
    expect(
      screen.getByRole("button", { name: "Play" }).getAttribute("aria-pressed"),
    ).toBe("false");

    await user.click(screen.getByRole("button", { name: "Play" }));
    await fireEvent.input(
      screen.getByRole("slider", { name: "Step scrubber" }),
      {
        target: { value: "1" },
      },
    );
    await view.rerender({ controller, state: controller.getState() });

    expect(
      screen.getByRole("button", { name: "Play" }).getAttribute("aria-pressed"),
    ).toBe("false");
    expect(
      (
        screen.getByRole("slider", {
          name: "Step scrubber",
        }) as HTMLInputElement
      ).value,
    ).toBe("1");
    expect(screen.getByText("paused")).toBeTruthy();
    controller.dispose();
  });
});
