import { describe, expect, it } from "vitest";
import { getComparisonNotice } from "../lib/comparison";
import { makeBundle } from "./helpers/bundle";

describe("getComparisonNotice", () => {
  it("returns no notice when comparison is disabled", () => {
    expect(getComparisonNotice(makeBundle(), null)).toBeNull();
  });

  it("waits for Run A when Run B has loaded first", () => {
    expect(getComparisonNotice(null, makeBundle())?.kind).toBe("waiting");
  });

  it("allows the same run on both sides without inventing alignment", () => {
    const bundle = makeBundle({ learningRate: 0.25 });
    const notice = getComparisonNotice(bundle, bundle);

    expect(notice).toEqual({
      kind: "compatible",
      message: "Same data and initialization. Learning rates: A=0.25, B=0.25.",
    });
  });

  it("marks different data or initialization as side-by-side only", () => {
    const notice = getComparisonNotice(makeBundle(), makeBundle({ seed: 8 }));

    expect(notice?.kind).toBe("incompatible");
    expect(notice?.message).toContain("not aligned or interpolated");
  });

  it("describes unequal step budgets without interpolation or extrapolation", () => {
    const notice = getComparisonNotice(
      makeBundle({ snapshotCount: 4 }),
      makeBundle({ snapshotCount: 2 }),
    );

    expect(notice?.kind).toBe("compatible");
    expect(notice?.message).toContain("A has 4 snapshots, B has 2 snapshots");
    expect(notice?.message).toContain("not extrapolated");
  });
});
