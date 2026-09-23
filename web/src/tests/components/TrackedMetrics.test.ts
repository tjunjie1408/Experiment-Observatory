// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/svelte";
import { afterEach, expect, it, vi } from "vitest";
import TrackedMetrics from "../../components/TrackedMetrics.svelte";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("does not draw a metric line across steps with no recorded value", async () => {
  vi.stubGlobal("fetch", vi.fn<typeof fetch>(async () => new Response(JSON.stringify({
    runId: "tree-1", source: "mlflow", state: "verified",
    series: [{
      key: "gini_decrease.all", unit: "fraction", aggregation: "weighted_impurity_decrease",
      points: [{ step: 0, value: 0.3 }, { step: 1, value: 0.2 }, { step: 4, value: 0.1 }],
    }],
  }), { status: 200 })));

  render(TrackedMetrics, { runId: "tree-1" });
  const chart = await screen.findByRole("img", { name: /gini_decrease\.all metric history/ });
  expect(chart.querySelectorAll("circle")).toHaveLength(3);
  expect(chart.querySelectorAll("polyline")).toHaveLength(1);
  expect(chart.querySelector("polyline")?.getAttribute("data-recorded-steps")).toBe("0 1");
});
