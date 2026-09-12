export type Domain = [number, number];
export type Range = [number, number];

export interface ChartPoint {
  x: number;
  y: number;
}

export const CHART_WIDTH = 360;
export const CHART_HEIGHT = 280;
export const CHART_MARGIN = {
  top: 16,
  right: 16,
  bottom: 32,
  left: 44,
} as const;

export function linearScale(
  domain: Domain,
  range: Range,
): (value: number) => number {
  const [domainStart, domainEnd] = domain;
  const [rangeStart, rangeEnd] = range;
  const span = domainEnd - domainStart || 1;
  return (value: number) =>
    rangeStart + ((value - domainStart) / span) * (rangeEnd - rangeStart);
}

export function niceDomain(values: readonly number[], pad = 0.1): Domain {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return [0, 1];

  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || Math.abs(max) || 1;
  return [min - span * pad, max + span * pad];
}

export function finitePoints<T extends ChartPoint>(points: readonly T[]): T[] {
  return points.filter(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
  );
}

export function segmentConsecutivePoints<
  T extends ChartPoint & { step: number },
>(points: readonly T[]): T[][] {
  const segments: T[][] = [];
  for (const point of points) {
    const current = segments.at(-1);
    const previous = current?.at(-1);
    if (previous === undefined || point.step !== previous.step + 1) {
      segments.push([point]);
    } else {
      current.push(point);
    }
  }
  return segments;
}

export function serializePoints(
  points: readonly ChartPoint[],
  xScale: (value: number) => number,
  yScale: (value: number) => number,
): string {
  return points
    .map((point) => `${xScale(point.x)},${yScale(point.y)}`)
    .join(" ");
}
