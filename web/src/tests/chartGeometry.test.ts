import { describe, expect, it } from "vitest";
import {
  finitePoints,
  linearScale,
  niceDomain,
  serializePoints,
} from "../lib/chartGeometry";

describe("chart geometry", () => {
  it("builds padded domains from finite values only", () => {
    expect(niceDomain([Number.NaN, 2, 6, Number.POSITIVE_INFINITY])).toEqual([
      1.6, 6.4,
    ]);
    expect(niceDomain([Number.NaN, Number.NEGATIVE_INFINITY])).toEqual([0, 1]);
  });

  it("expands a constant finite domain deterministically", () => {
    expect(niceDomain([5, 5])).toEqual([4.5, 5.5]);
    expect(niceDomain([0, 0])).toEqual([-0.1, 0.1]);
  });

  it("maps values linearly and handles a degenerate domain", () => {
    expect(linearScale([0, 10], [20, 120])(2.5)).toBe(45);
    expect(linearScale([3, 3], [10, 30])(3)).toBe(10);
  });

  it("keeps only finite recorded points without interpolating gaps", () => {
    const points = finitePoints([
      { x: 0, y: 4, id: "step-0" },
      { x: 7, y: Number.NaN, id: "step-7" },
      { x: 20, y: 1, id: "step-20" },
    ]);

    expect(points).toEqual([
      { x: 0, y: 4, id: "step-0" },
      { x: 20, y: 1, id: "step-20" },
    ]);
    expect(serializePoints(points, (value) => value * 2, (value) => value + 1)).toBe(
      "0,5 40,2",
    );
  });
});
