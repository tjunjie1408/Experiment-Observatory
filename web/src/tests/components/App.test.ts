// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../../App.svelte";
import { makeBundle } from "../helpers/bundle";
import { mockBundleFetch } from "./fetchBundles";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App comparison", () => {
  it("activates Run B and announces the comparison contract", async () => {
    const runA = makeBundle({ runId: "converge", learningRate: 0.25 });
    const runB = makeBundle({ runId: "slow", learningRate: 0.001 });
    mockBundleFetch((url) => (url.startsWith("/runs/slow/") ? runB : runA));
    const user = userEvent.setup();

    render(App);
    await screen.findByText("converge loaded with 3 recorded snapshots");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Compare with Run B" }),
      "/runs/slow",
    );

    expect(
      await screen.findByRole("region", { name: "Run B workspace" }),
    ).toBeTruthy();
    await waitFor(() => {
      const notice = screen
        .getAllByRole("status")
        .find((element) =>
          element.textContent?.includes("Comparison contract"),
        );
      expect(notice?.textContent).toContain("Learning rates: A=0.25, B=0.001");
    });
  });
});
