import { render } from "svelte/server";
import { describe, expect, it } from "vitest";
import LossCurve from "../components/charts/LossCurve.svelte";
import ParameterContour from "../components/charts/ParameterContour.svelte";
import ResidualPlot from "../components/charts/ResidualPlot.svelte";
import ScatterPlot from "../components/charts/ScatterPlot.svelte";
import { makeBundle } from "./helpers/bundle";

describe("declarative charts", () => {
  it("renders deterministic scatter marks with sample IDs and finite coordinates", () => {
    const bundle = makeBundle();
    const snapshot = bundle.snapshots[1]!;
    bundle.manifest.dataset.x[1] = Number.POSITIVE_INFINITY;

    const first = render(ScatterPlot, { props: { bundle, snapshot } }).body;
    const second = render(ScatterPlot, { props: { bundle, snapshot } }).body;

    expect(first).toBe(second);
    expect(first).toContain('class="chart-viewport"');
    expect(first).toContain('class="sample-point"');
    expect(first).toContain('data-sample-id="sample-1"');
    expect(first).not.toContain('data-sample-id="sample-2"');
    expect(first).toContain('class="regression-line"');
    expect(first).not.toMatch(/NaN|Infinity/);
  });

  it("keeps residual marks associated with their source samples", () => {
    const bundle = makeBundle();
    const snapshot = bundle.snapshots[2]!;
    bundle.manifest.dataset.y[1] = Number.NaN;

    const html = render(ResidualPlot, { props: { bundle, snapshot } }).body;

    expect(html).toContain('class="zero-line"');
    expect(html).toContain('class="residual-point"');
    expect(html).toContain('data-sample-id="sample-1"');
    expect(html).not.toContain('data-sample-id="sample-2"');
    expect(html).not.toMatch(/NaN|Infinity/);
  });

  it("marks the selected recorded loss snapshot without filling sparse steps", () => {
    const bundle = makeBundle({ snapshotCount: 3 });
    bundle.snapshots[0]!.step = 0;
    bundle.snapshots[1]!.step = 7;
    bundle.snapshots[2]!.step = 20;

    const html = render(LossCurve, { props: { bundle, currentStep: 1 } }).body;

    expect(html).toContain('class="loss-curve-line"');
    expect(html).toContain('data-recorded-steps="0 7 20"');
    expect(html).toContain('class="current-step-marker"');
    expect(html).toContain('data-step="7"');
    expect(html).not.toContain('data-step="1"');
    expect(html).not.toMatch(/NaN|Infinity/);
  });

  it("renders a finite contour and marks the exact recorded trajectory snapshot", () => {
    const bundle = makeBundle({ snapshotCount: 3 });
    bundle.snapshots[0]!.step = 0;
    bundle.snapshots[1]!.step = 7;
    bundle.snapshots[2]!.step = 20;
    bundle.snapshots[2]!.b = Number.NaN;

    const html = render(ParameterContour, {
      props: { bundle, currentStep: 1 },
    }).body;

    expect(html).toContain("<rect");
    expect(html).toContain('class="trajectory-line"');
    expect(html).toContain('data-recorded-steps="0 7"');
    expect(html).toContain('class="current-step-marker"');
    expect(html).toContain('data-step="7"');
    expect(html).not.toMatch(/NaN|Infinity/);
  });
});
