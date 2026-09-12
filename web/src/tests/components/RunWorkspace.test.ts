// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import RunWorkspace from "../../components/RunWorkspace.svelte";
import { makeBundle } from "../helpers/bundle";
import { deferred, mockBundleFetch, responseForBundle } from "./fetchBundles";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("RunWorkspace", () => {
  it("steps left and right and toggles playback from its keyboard toolbar", async () => {
    const bundle = makeBundle({ runId: "keyboard-run", snapshotCount: 3 });
    mockBundleFetch(() => bundle);
    const user = userEvent.setup();

    render(RunWorkspace, { path: "/runs/keyboard", label: "Run A" });
    const toolbar = await screen.findByRole("toolbar", {
      name: "Replay toolbar for Run A",
    });
    await screen.findByText("keyboard-run loaded with 3 recorded snapshots");
    toolbar.focus();

    const scrubber = screen.getByRole("slider", {
      name: "Step scrubber",
    }) as HTMLInputElement;
    await user.keyboard("{ArrowRight}");
    expect(scrubber.value).toBe("1");
    await user.keyboard("{ArrowLeft}");
    expect(scrubber.value).toBe("0");

    await user.keyboard(" ");
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    await user.keyboard(" ");
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
    expect(screen.getByText("paused")).toBeTruthy();
  });

  it("shows a load error and retries the same selected run", async () => {
    const bundle = makeBundle({ runId: "recovered-run" });
    let failing = true;
    const fetchMock = vi.fn<typeof fetch>(async (input) =>
      failing
        ? new Response("Unavailable", { status: 503 })
        : responseForBundle(String(input), bundle),
    );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();

    render(RunWorkspace, { path: "/runs/retry", label: "Run A" });

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Run artifact unavailable");
    expect(alert.textContent).toContain("HTTP 503");

    failing = false;
    await user.click(screen.getByRole("button", { name: "Retry load" }));

    expect(
      await screen.findByText("recovered-run loaded with 3 recorded snapshots"),
    ).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps the newest run when a superseded load resolves late", async () => {
    const oldBundle = makeBundle({ runId: "old-run" });
    const newBundle = makeBundle({ runId: "new-run" });
    const oldResponses = [
      deferred<Response>(),
      deferred<Response>(),
      deferred<Response>(),
    ];
    let oldResponseIndex = 0;
    const fetchMock = vi.fn<typeof fetch>((input) => {
      const url = String(input);
      if (url.startsWith("/runs/old/")) {
        const pending = oldResponses[oldResponseIndex++];
        if (!pending) throw new Error("Unexpected old-run request");
        return pending.promise;
      }
      return Promise.resolve(responseForBundle(url, newBundle));
    });
    vi.stubGlobal("fetch", fetchMock);

    const view = render(RunWorkspace, { path: "/runs/old", label: "Run A" });
    await waitFor(() => expect(oldResponseIndex).toBe(3));
    await view.rerender({ path: "/runs/new", label: "Run A" });

    expect(
      await screen.findByText("new-run loaded with 3 recorded snapshots"),
    ).toBeTruthy();
    oldResponses.forEach((pending, index) => {
      const suffix = ["manifest.json", "events.jsonl", "snapshots.json"][
        index
      ] as string;
      pending.resolve(responseForBundle(`/runs/old/${suffix}`, oldBundle));
    });

    await waitFor(() => {
      expect(
        screen.getByText("new-run loaded with 3 recorded snapshots"),
      ).toBeTruthy();
      expect(screen.queryByText(/old-run loaded/)).toBeNull();
    });
  });

  it("lets the user select an observed sample and updates its visible readout", async () => {
    const bundle = makeBundle({ runId: "samples-run" });
    bundle.manifest.observedSampleIds = ["sample-1", "sample-2"];
    bundle.snapshots.forEach((snapshot) => {
      snapshot.observedPredictions["sample-2"] = 2;
    });
    mockBundleFetch(() => bundle);
    const user = userEvent.setup();

    render(RunWorkspace, { path: "/runs/samples", label: "Run A" });
    const samplePicker = await screen.findByRole("combobox", {
      name: "Selected sample",
    });
    await user.selectOptions(samplePicker, "sample-2");

    expect(
      screen.getByRole("option", { name: "sample-2", selected: true }),
    ).toBeTruthy();
    expect(screen.getByText("1.0000")).toBeTruthy();
    expect(screen.getByText("2.5000")).toBeTruthy();
    expect(screen.getByText("2.0000")).toBeTruthy();
    expect(screen.getByText("-0.5000")).toBeTruthy();
  });
});
