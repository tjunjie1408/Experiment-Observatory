/**
 * SVG-based views: scatter + regression line, residuals, loss curve,
 * parameter trajectory over a loss contour. Plain DOM/SVG, no charting
 * library, since V0's view set is small and fixed.
 */

import type { RunBundle, Snapshot } from "./schema";
import { computeContourGrid, predict, residuals } from "./math";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl<K extends keyof SVGElementTagNameMap>(tag: K): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, tag);
}

function linearScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

function niceDomain(values: number[], pad = 0.1): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return [0, 1];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || Math.abs(max) || 1;
  return [min - span * pad, max + span * pad];
}

const WIDTH = 360;
const HEIGHT = 280;
const MARGIN = { top: 16, right: 16, bottom: 32, left: 44 };

function baseSvg(): SVGSVGElement {
  const svg = svgEl("svg");
  svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.setAttribute("role", "img");
  return svg;
}

function addAxisLabel(svg: SVGSVGElement, text: string, x: number, y: number, rotate = false): void {
  const label = svgEl("text");
  label.setAttribute("x", String(x));
  label.setAttribute("y", String(y));
  label.setAttribute("class", "axis-label");
  if (rotate) label.setAttribute("transform", `rotate(-90 ${x} ${y})`);
  label.textContent = text;
  svg.appendChild(label);
}

export function renderScatterView(
  container: HTMLElement,
  bundle: RunBundle,
  snapshot: Snapshot,
): void {
  container.innerHTML = "";
  const { x, y } = bundle.manifest.dataset;
  const svg = baseSvg();

  const xDomain = niceDomain(x);
  const yDomain = niceDomain([...y, ...predict(snapshot.b, snapshot.w, x)]);
  const xScale = linearScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const yScale = linearScale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);

  for (let i = 0; i < x.length; i++) {
    const circle = svgEl("circle");
    circle.setAttribute("cx", String(xScale(x[i] as number)));
    circle.setAttribute("cy", String(yScale(y[i] as number)));
    circle.setAttribute("r", "3");
    const sampleId = bundle.manifest.dataset.sampleIds[i];
    circle.setAttribute("class", "sample-point");
    if (sampleId !== undefined) circle.setAttribute("data-sample-id", sampleId);
    svg.appendChild(circle);
  }

  const line = svgEl("line");
  const xMin = xDomain[0];
  const xMax = xDomain[1];
  line.setAttribute("x1", String(xScale(xMin)));
  line.setAttribute("y1", String(yScale(snapshot.b + snapshot.w * xMin)));
  line.setAttribute("x2", String(xScale(xMax)));
  line.setAttribute("y2", String(yScale(snapshot.b + snapshot.w * xMax)));
  line.setAttribute("class", "regression-line");
  svg.appendChild(line);

  addAxisLabel(svg, "x", WIDTH / 2, HEIGHT - 6);
  addAxisLabel(svg, "y", 12, HEIGHT / 2, true);

  container.appendChild(svg);
}

export function renderResidualsView(
  container: HTMLElement,
  bundle: RunBundle,
  snapshot: Snapshot,
): void {
  container.innerHTML = "";
  const { x, y } = bundle.manifest.dataset;
  const r = residuals(snapshot.b, snapshot.w, x, y);
  const svg = baseSvg();

  const xDomain = niceDomain(x);
  const rDomain = niceDomain(r);
  const xScale = linearScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const rScale = linearScale(rDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);

  const zeroLine = svgEl("line");
  zeroLine.setAttribute("x1", String(MARGIN.left));
  zeroLine.setAttribute("y1", String(rScale(0)));
  zeroLine.setAttribute("x2", String(WIDTH - MARGIN.right));
  zeroLine.setAttribute("y2", String(rScale(0)));
  zeroLine.setAttribute("class", "zero-line");
  svg.appendChild(zeroLine);

  for (let i = 0; i < x.length; i++) {
    const circle = svgEl("circle");
    circle.setAttribute("cx", String(xScale(x[i] as number)));
    circle.setAttribute("cy", String(rScale(r[i] as number)));
    circle.setAttribute("r", "3");
    circle.setAttribute("class", "residual-point");
    svg.appendChild(circle);
  }

  addAxisLabel(svg, "x", WIDTH / 2, HEIGHT - 6);
  addAxisLabel(svg, "residual", 12, HEIGHT / 2, true);

  container.appendChild(svg);
}

export function renderLossCurveView(
  container: HTMLElement,
  bundle: RunBundle,
  currentStep: number,
): void {
  container.innerHTML = "";
  const svg = baseSvg();

  const steps = bundle.snapshots.map((s) => s.step);
  const mseValues = bundle.snapshots.map((s) => s.trainMse);
  const xScale = linearScale(niceDomain(steps, 0.02), [MARGIN.left, WIDTH - MARGIN.right]);
  const yScale = linearScale(niceDomain(mseValues), [HEIGHT - MARGIN.bottom, MARGIN.top]);

  const points = bundle.snapshots
    .map((s) => `${xScale(s.step)},${yScale(s.trainMse)}`)
    .join(" ");
  const polyline = svgEl("polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("class", "loss-curve-line");
  svg.appendChild(polyline);

  const current = bundle.snapshots[currentStep];
  if (current !== undefined) {
    const marker = svgEl("circle");
    marker.setAttribute("cx", String(xScale(current.step)));
    marker.setAttribute("cy", String(yScale(current.trainMse)));
    marker.setAttribute("r", "4");
    marker.setAttribute("class", "current-step-marker");
    svg.appendChild(marker);
  }

  addAxisLabel(svg, "step", WIDTH / 2, HEIGHT - 6);
  addAxisLabel(svg, "train MSE", 12, HEIGHT / 2, true);

  container.appendChild(svg);
}

export function renderContourView(
  container: HTMLElement,
  bundle: RunBundle,
  currentStep: number,
): void {
  container.innerHTML = "";
  const { x, y } = bundle.manifest.dataset;
  const svg = baseSvg();

  const bs = bundle.snapshots.map((s) => s.b);
  const ws = bundle.snapshots.map((s) => s.w);
  const bDomain = niceDomain(bs, 0.3);
  const wDomain = niceDomain(ws, 0.3);

  const resolution = 24;
  const contour = computeContourGrid(x, y, bDomain, wDomain, resolution);
  const maxMse = Math.max(...contour.grid.flat().filter((v) => Number.isFinite(v)));

  const bScale = linearScale(bDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const wScale = linearScale(wDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const cellW = (WIDTH - MARGIN.left - MARGIN.right) / resolution;
  const cellH = (HEIGHT - MARGIN.bottom - MARGIN.top) / resolution;

  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const value = contour.grid[i]?.[j] ?? NaN;
      if (!Number.isFinite(value)) continue;
      const intensity = maxMse > 0 ? Math.min(1, value / maxMse) : 0;
      const rect = svgEl("rect");
      rect.setAttribute("x", String(bScale(contour.bValues[i] as number) - cellW / 2));
      rect.setAttribute("y", String(wScale(contour.wValues[j] as number) - cellH / 2));
      rect.setAttribute("width", String(cellW));
      rect.setAttribute("height", String(cellH));
      const lightness = 92 - intensity * 55;
      rect.setAttribute("fill", `hsl(220 60% ${lightness}%)`);
      rect.setAttribute("stroke", "none");
      svg.appendChild(rect);
    }
  }

  const trajectoryPoints = bundle.snapshots
    .map((s) => `${bScale(s.b)},${wScale(s.w)}`)
    .join(" ");
  const trajectory = svgEl("polyline");
  trajectory.setAttribute("points", trajectoryPoints);
  trajectory.setAttribute("class", "trajectory-line");
  svg.appendChild(trajectory);

  const current = bundle.snapshots[currentStep];
  if (current !== undefined) {
    const marker = svgEl("circle");
    marker.setAttribute("cx", String(bScale(current.b)));
    marker.setAttribute("cy", String(wScale(current.w)));
    marker.setAttribute("r", "4");
    marker.setAttribute("class", "current-step-marker");
    svg.appendChild(marker);
  }

  addAxisLabel(svg, "b", WIDTH / 2, HEIGHT - 6);
  addAxisLabel(svg, "w", 12, HEIGHT / 2, true);

  container.appendChild(svg);
}
