// @vitest-environment jsdom

import { render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import KMeansPanel from "../../components/KMeansPanel.svelte";
import { makeKMeansBundle } from "../helpers/bundle";

describe("KMeansPanel", () => {
  it("renders recorded assignments, centers, phase, and inertia", () => {
    const bundle = makeKMeansBundle();

    const { container } = render(KMeansPanel, {
      bundle,
      snapshot: bundle.snapshots[0]!,
    });

    expect(screen.getByText("Recorded assignment phase")).toBeTruthy();
    expect(screen.getByText("Inertia 4.0000")).toBeTruthy();
    expect(container.querySelectorAll("circle")).toHaveLength(6);
  });
});
