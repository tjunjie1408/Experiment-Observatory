import { describe, expect, it } from "vitest";
import {
  AVAILABLE_RUNS,
  groupAvailableRuns,
  studyLabelForPath,
  type AvailableRun,
} from "../lib/availableRuns";

describe("groupAvailableRuns", () => {
  it("keeps every run exactly once, grouped in first-appearance order", () => {
    const groups = groupAvailableRuns();

    expect(groups.map((group) => group.group)).toEqual([
      "synthetic-learning-rate",
      "auto-mpg-weight",
      "kmeans-initialization",
      "tree-depth",
    ]);
    expect(groups.flatMap((group) => group.runs)).toEqual(AVAILABLE_RUNS);
  });

  it("does not merge non-adjacent runs of the same group out of order", () => {
    const run = (id: string, comparisonGroup: AvailableRun["comparisonGroup"]): AvailableRun => ({
      id,
      runId: `${id}-run`,
      label: id,
      path: `/runs/${id}`,
      comparisonGroup,
    });
    const groups = groupAvailableRuns([
      run("a", "tree-depth"),
      run("b", "kmeans-initialization"),
      run("c", "tree-depth"),
    ]);

    expect(groups.map((group) => [group.group, group.runs.map((item) => item.id)])).toEqual([
      ["tree-depth", ["a", "c"]],
      ["kmeans-initialization", ["b"]],
    ]);
  });
});

describe("studyLabelForPath", () => {
  it("names the study of a shipped run and returns null for local replay paths", () => {
    expect(studyLabelForPath("/runs/tree-depth-3")).toBe("Decision tree · WDBC depth study");
    expect(studyLabelForPath("/api/replay/some-run")).toBeNull();
  });
});
