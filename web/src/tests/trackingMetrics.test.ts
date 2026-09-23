import { describe, expect, it } from "vitest";
import { parseTrackingHistory } from "../lib/trackingMetrics";

const history = {
  runId: "tree-1",
  source: "mlflow",
  state: "verified",
  series: [{
    key: "nodes_recorded.all",
    unit: "count",
    aggregation: "count",
    points: [{ step: 0, value: 1 }, { step: 1, value: 2 }],
  }],
};

describe("MLflow metric history contract", () => {
  it("accepts ordered finite points for the selected Observatory run", () => {
    expect(parseTrackingHistory(history, "tree-1")).toEqual(history);
  });

  it("rejects another run, duplicate steps, and non-finite values", () => {
    expect(() => parseTrackingHistory(history, "tree-2")).toThrow(/run identity/);
    expect(() => parseTrackingHistory({ ...history, series: [{ ...history.series[0], points: [{ step: 0, value: 1 }, { step: 0, value: 2 }] }] }, "tree-1")).toThrow(/step order/);
    expect(() => parseTrackingHistory({ ...history, series: [{ ...history.series[0], points: [{ step: 0, value: Infinity }] }] }, "tree-1")).toThrow(/metric point/);
  });
});
