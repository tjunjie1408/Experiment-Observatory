// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it } from "vitest";
import TelemetryPanel from "../../components/TelemetryPanel.svelte";
import { makeBundle, makeExternalBundle } from "../helpers/bundle";

afterEach(cleanup);

describe("TelemetryPanel dataset metadata", () => {
  it("shows the seed only for synthetic runs", () => {
    const bundle = makeBundle({ seed: 17 });

    render(TelemetryPanel, { bundle, snapshot: bundle.snapshots[0]! });

    expect(screen.getByText("Dataset seed")).toBeTruthy();
    expect(screen.getByText("17")).toBeTruthy();
    expect(screen.queryByText("Dataset version")).toBeNull();
  });

  it("shows external dataset version and feature-to-target identity with units", () => {
    const bundle = makeExternalBundle();

    render(TelemetryPanel, { bundle, snapshot: bundle.snapshots[0]! });

    expect(screen.queryByText("Dataset seed")).toBeNull();
    expect(screen.getByText("Dataset version")).toBeTruthy();
    expect(screen.getByText("auto-mpg 1.0.0")).toBeTruthy();
    expect(screen.getByText("Feature → target")).toBeTruthy();
    expect(
      screen.getByText(
        "weight_standardized (population standard deviations) → mpg (miles per gallon)",
      ),
    ).toBeTruthy();
  });
});
