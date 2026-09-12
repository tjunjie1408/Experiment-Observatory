/**
 * Independent recomputation of predictions / residuals / MSE / loss contour
 * from recorded (b, w) and the embedded dataset, implemented separately
 * from the Python side so displayed values can be cross-checked rather
 * than only trusting the recorded numbers.
 */

export function predict(b: number, w: number, x: number[]): number[] {
  return x.map((xi) => b + w * xi);
}

export function residuals(
  b: number,
  w: number,
  x: number[],
  y: number[],
): number[] {
  const preds = predict(b, w, x);
  return preds.map((p, i) => p - (y[i] as number));
}

export function mse(b: number, w: number, x: number[], y: number[]): number {
  const r = residuals(b, w, x, y);
  const sumSq = r.reduce((acc, ri) => acc + ri * ri, 0);
  return sumSq / r.length;
}

export interface LeastSquaresFit {
  b: number;
  w: number;
  mse: number;
}

/** Derive the one-feature ordinary least-squares optimum from the embedded dataset. */
export function leastSquares1d(
  x: number[],
  y: number[],
): LeastSquaresFit | null {
  if (x.length < 2 || x.length !== y.length) return null;
  if (!x.every(Number.isFinite) || !y.every(Number.isFinite)) return null;

  const xMean = x.reduce((sum, value) => sum + value, 0) / x.length;
  const yMean = y.reduce((sum, value) => sum + value, 0) / y.length;
  let covariance = 0;
  let variance = 0;
  for (let index = 0; index < x.length; index += 1) {
    const centeredX = (x[index] as number) - xMean;
    covariance += centeredX * ((y[index] as number) - yMean);
    variance += centeredX * centeredX;
  }
  if (!Number.isFinite(variance) || variance <= 0) return null;

  const w = covariance / variance;
  const b = yMean - w * xMean;
  const optimumMse = mse(b, w, x, y);
  if (![b, w, optimumMse].every(Number.isFinite)) return null;
  return { b, w, mse: optimumMse };
}

export interface ContourGrid {
  bValues: number[];
  wValues: number[];
  /** grid[i][j] = MSE at (bValues[i], wValues[j]) */
  grid: number[][];
}

/**
 * Compute an MSE contour grid over a (b, w) range. This is explicitly
 * derived data: it is recomputed from the dataset for display, not a
 * recorded quantity from any snapshot.
 */
export function computeContourGrid(
  x: number[],
  y: number[],
  bRange: [number, number],
  wRange: [number, number],
  resolution: number,
): ContourGrid {
  const bValues: number[] = [];
  const wValues: number[] = [];
  for (let i = 0; i < resolution; i++) {
    const t = i / (resolution - 1);
    bValues.push(bRange[0] + t * (bRange[1] - bRange[0]));
    wValues.push(wRange[0] + t * (wRange[1] - wRange[0]));
  }
  const grid: number[][] = bValues.map((b) =>
    wValues.map((w) => mse(b, w, x, y)),
  );
  return { bValues, wValues, grid };
}
