import { describe, expect, it } from "vitest";
import { leastSquares1d } from "../lib/math";

describe("leastSquares1d", () => {
  it("derives the exact affine fit independently from recorded snapshots", () => {
    expect(leastSquares1d([-1, 0, 1], [-1, 1, 3])).toEqual({
      b: 1,
      w: 2,
      mse: 0,
    });
  });

  it("returns null for mismatched, non-finite, or rank-deficient data", () => {
    expect(leastSquares1d([0], [1])).toBeNull();
    expect(leastSquares1d([0, 1], [1])).toBeNull();
    expect(leastSquares1d([0, Number.NaN], [1, 2])).toBeNull();
    expect(leastSquares1d([2, 2], [1, 2])).toBeNull();
  });
});
