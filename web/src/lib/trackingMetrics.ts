/** Local MLflow metric history; never a second persisted replay format. */
export interface MetricPoint { step: number; value: number }
export interface MetricSeries { key: string; unit: string; aggregation: string; points: MetricPoint[] }
export interface TrackingHistory {
  runId: string;
  source: "mlflow";
  state: "verified";
  series: MetricSeries[];
}

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("tracking response object expected");
  }
  return value as Record<string, unknown>;
}

function nonempty(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) throw new Error("tracking string expected");
  return value;
}

export function parseTrackingHistory(raw: unknown, expectedRunId: string): TrackingHistory {
  const input = record(raw);
  if (input.runId !== expectedRunId || input.source !== "mlflow" || input.state !== "verified") {
    throw new Error("tracking run identity or verification state mismatch");
  }
  if (!Array.isArray(input.series)) throw new Error("tracking series expected");
  const keys = new Set<string>();
  const series = input.series.map((item): MetricSeries => {
    const row = record(item);
    const key = nonempty(row.key);
    if (keys.has(key)) throw new Error("duplicate tracking metric key");
    keys.add(key);
    if (!Array.isArray(row.points)) throw new Error("tracking points expected");
    let previous = -1;
    const points = row.points.map((value): MetricPoint => {
      const point = record(value);
      if (typeof point.step !== "number" || !Number.isSafeInteger(point.step) || point.step < 0 ||
          typeof point.value !== "number" || !Number.isFinite(point.value)) {
        throw new Error("invalid tracking metric point");
      }
      if (point.step <= previous) throw new Error("tracking step order is invalid");
      previous = point.step;
      return { step: point.step, value: point.value };
    });
    return { key, unit: nonempty(row.unit), aggregation: nonempty(row.aggregation), points };
  });
  return { runId: expectedRunId, source: "mlflow", state: "verified", series };
}

export async function loadTrackingHistory(runId: string, signal?: AbortSignal): Promise<TrackingHistory> {
  const response = await fetch(`/api/tracking/${encodeURIComponent(runId)}/metrics`, { signal });
  if (!response.ok) throw new Error(`MLflow metric history unavailable (HTTP ${response.status})`);
  return parseTrackingHistory(await response.json(), runId);
}
