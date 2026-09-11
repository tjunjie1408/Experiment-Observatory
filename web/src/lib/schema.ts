/**
 * Hand-written TS types + runtime validation mirroring
 * src/observatory/runtime/schema.py. This is the frontend half of the
 * cross-language contract: a small amount of duplication across the
 * Python/TS boundary is acceptable for V0 instead of generating types
 * from the Python models.
 *
 * validateBundle() performs the same checks export.py performs on the
 * Python side before exporting: schemaVersion, required shape, finite
 * numeric values, and cross-references between manifest / events /
 * snapshots. A bundle that fails validation is rejected in full: nothing
 * partially valid is displayed.
 */

export const SCHEMA_VERSION = 1;

export type RunStatus = "created" | "running" | "completed" | "failed";
export type StopReason =
  | "max_steps"
  | "numerical_error"
  | "io_error"
  | "user_cancelled"
  | "runtime_error";

export interface DataConfig {
  generator: string;
  nSamples: number;
  trueBias: number;
  trueWeight: number;
  noiseStd: number;
  seed: number;
}

export interface ModelConfig {
  algorithm: "linear_regression_gradient_descent";
  initialBias: number;
  initialWeight: number;
  learningRate: number;
  nUpdates: number;
}

export interface DatasetSummary {
  generatorId: string;
  sampleIds: string[];
  x: number[];
  y: number[];
}

export interface CodeProvenance {
  gitCommit: string | null;
  gitDirty: boolean | null;
  unavailableReason: string | null;
}

export interface RunManifest {
  schemaVersion: number;
  runId: string;
  experimentId: string;
  createdAt: string;
  status: RunStatus;
  stopReason: StopReason | null;
  lastValidStep: number | null;
  errorMessage: string | null;
  dataConfig: DataConfig;
  dataset: DatasetSummary;
  trainingConfig: ModelConfig;
  codeProvenance: CodeProvenance;
  observedSampleIds: string[];
  nSnapshotsWritten: number;
}

export interface Snapshot {
  step: number;
  b: number;
  w: number;
  gradientB: number;
  gradientW: number;
  trainMse: number;
  observedPredictions: Record<string, number>;
}

export type EventKind =
  | "run.created"
  | "run.started"
  | "step.recorded"
  | "run.completed"
  | "run.failed";

export interface RunEvent {
  schemaVersion: number;
  runId: string;
  seq: number;
  kind: EventKind;
  step: number | null;
  message: string | null;
}

export interface RunBundle {
  manifest: RunManifest;
  events: RunEvent[];
  snapshots: Snapshot[];
}

/** Raised for any validation failure; `reason` is a human-readable, specific message. */
export class BundleValidationError extends Error {
  constructor(reason: string) {
    super(reason);
    this.name = "BundleValidationError";
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function assertFiniteNumber(value: unknown, path: string): number {
  if (!isFiniteNumber(value)) {
    throw new BundleValidationError(`non-finite or non-numeric value at ${path}: ${JSON.stringify(value)}`);
  }
  return value;
}

function assertString(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new BundleValidationError(`expected string at ${path}, got ${JSON.stringify(value)}`);
  }
  return value;
}

function assertNullableString(value: unknown, path: string): string | null {
  if (value === null) return null;
  return assertString(value, path);
}

function assertArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new BundleValidationError(`expected array at ${path}, got ${JSON.stringify(value)}`);
  }
  return value;
}

function assertObject(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new BundleValidationError(`expected object at ${path}, got ${JSON.stringify(value)}`);
  }
  return value as Record<string, unknown>;
}

const RUN_STATUSES: RunStatus[] = ["created", "running", "completed", "failed"];
const STOP_REASONS: StopReason[] = [
  "max_steps",
  "numerical_error",
  "io_error",
  "user_cancelled",
  "runtime_error",
];
const EVENT_KINDS: EventKind[] = [
  "run.created",
  "run.started",
  "step.recorded",
  "run.completed",
  "run.failed",
];

function parseDataConfig(raw: unknown, path: string): DataConfig {
  const obj = assertObject(raw, path);
  return {
    generator: assertString(obj.generator, `${path}.generator`),
    nSamples: assertFiniteNumber(obj.nSamples, `${path}.nSamples`),
    trueBias: assertFiniteNumber(obj.trueBias, `${path}.trueBias`),
    trueWeight: assertFiniteNumber(obj.trueWeight, `${path}.trueWeight`),
    noiseStd: assertFiniteNumber(obj.noiseStd, `${path}.noiseStd`),
    seed: assertFiniteNumber(obj.seed, `${path}.seed`),
  };
}

function parseModelConfig(raw: unknown, path: string): ModelConfig {
  const obj = assertObject(raw, path);
  if (obj.algorithm !== "linear_regression_gradient_descent") {
    throw new BundleValidationError(`unsupported algorithm at ${path}.algorithm: ${JSON.stringify(obj.algorithm)}`);
  }
  return {
    algorithm: "linear_regression_gradient_descent",
    initialBias: assertFiniteNumber(obj.initialBias, `${path}.initialBias`),
    initialWeight: assertFiniteNumber(obj.initialWeight, `${path}.initialWeight`),
    learningRate: assertFiniteNumber(obj.learningRate, `${path}.learningRate`),
    nUpdates: assertFiniteNumber(obj.nUpdates, `${path}.nUpdates`),
  };
}

function parseDatasetSummary(raw: unknown, path: string): DatasetSummary {
  const obj = assertObject(raw, path);
  const sampleIds = assertArray(obj.sampleIds, `${path}.sampleIds`).map((v, i) =>
    assertString(v, `${path}.sampleIds[${i}]`),
  );
  const x = assertArray(obj.x, `${path}.x`).map((v, i) => assertFiniteNumber(v, `${path}.x[${i}]`));
  const y = assertArray(obj.y, `${path}.y`).map((v, i) => assertFiniteNumber(v, `${path}.y[${i}]`));
  if (x.length !== sampleIds.length || y.length !== sampleIds.length) {
    throw new BundleValidationError(`${path}: sampleIds, x, and y must have the same length`);
  }
  return {
    generatorId: assertString(obj.generatorId, `${path}.generatorId`),
    sampleIds,
    x,
    y,
  };
}

function assertNullableBoolean(value: unknown, path: string): boolean | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "boolean") {
    throw new BundleValidationError(`expected boolean or null at ${path}, got ${JSON.stringify(value)}`);
  }
  return value;
}

function parseCodeProvenance(raw: unknown, path: string): CodeProvenance {
  const obj = assertObject(raw, path);
  return {
    gitCommit: assertNullableString(obj.gitCommit ?? null, `${path}.gitCommit`),
    gitDirty: assertNullableBoolean(obj.gitDirty, `${path}.gitDirty`),
    unavailableReason: assertNullableString(obj.unavailableReason ?? null, `${path}.unavailableReason`),
  };
}

function parseManifest(raw: unknown): RunManifest {
  const obj = assertObject(raw, "manifest");

  const schemaVersion = assertFiniteNumber(obj.schemaVersion, "manifest.schemaVersion");
  if (schemaVersion !== SCHEMA_VERSION) {
    throw new BundleValidationError(
      `unsupported schemaVersion ${schemaVersion}; this viewer supports ${SCHEMA_VERSION}`,
    );
  }

  const status = assertString(obj.status, "manifest.status");
  if (!RUN_STATUSES.includes(status as RunStatus)) {
    throw new BundleValidationError(`manifest.status has an unknown value: ${status}`);
  }

  const stopReasonRaw = obj.stopReason;
  let stopReason: StopReason | null = null;
  if (stopReasonRaw !== null && stopReasonRaw !== undefined) {
    const s = assertString(stopReasonRaw, "manifest.stopReason");
    if (!STOP_REASONS.includes(s as StopReason)) {
      throw new BundleValidationError(`manifest.stopReason has an unknown value: ${s}`);
    }
    stopReason = s as StopReason;
  }

  if (status !== "completed") {
    throw new BundleValidationError(
      `manifest.status is "${status}", not "completed"; only completed runs can be replayed`,
    );
  }

  return {
    schemaVersion,
    runId: assertString(obj.runId, "manifest.runId"),
    experimentId: assertString(obj.experimentId, "manifest.experimentId"),
    createdAt: assertString(obj.createdAt, "manifest.createdAt"),
    status: status as RunStatus,
    stopReason,
    lastValidStep:
      obj.lastValidStep === null || obj.lastValidStep === undefined
        ? null
        : assertFiniteNumber(obj.lastValidStep, "manifest.lastValidStep"),
    errorMessage: assertNullableString(obj.errorMessage ?? null, "manifest.errorMessage"),
    dataConfig: parseDataConfig(obj.dataConfig, "manifest.dataConfig"),
    dataset: parseDatasetSummary(obj.dataset, "manifest.dataset"),
    trainingConfig: parseModelConfig(obj.trainingConfig, "manifest.trainingConfig"),
    codeProvenance: parseCodeProvenance(obj.codeProvenance, "manifest.codeProvenance"),
    observedSampleIds: assertArray(obj.observedSampleIds, "manifest.observedSampleIds").map((v, i) =>
      assertString(v, `manifest.observedSampleIds[${i}]`),
    ),
    nSnapshotsWritten: assertFiniteNumber(obj.nSnapshotsWritten, "manifest.nSnapshotsWritten"),
  };
}

function parseSnapshot(raw: unknown, index: number): Snapshot {
  const path = `snapshots[${index}]`;
  const obj = assertObject(raw, path);
  const predictionsObj = assertObject(obj.observedPredictions, `${path}.observedPredictions`);
  const observedPredictions: Record<string, number> = {};
  for (const [sampleId, value] of Object.entries(predictionsObj)) {
    observedPredictions[sampleId] = assertFiniteNumber(
      value,
      `${path}.observedPredictions[${sampleId}]`,
    );
  }
  return {
    step: assertFiniteNumber(obj.step, `${path}.step`),
    b: assertFiniteNumber(obj.b, `${path}.b`),
    w: assertFiniteNumber(obj.w, `${path}.w`),
    gradientB: assertFiniteNumber(obj.gradientB, `${path}.gradientB`),
    gradientW: assertFiniteNumber(obj.gradientW, `${path}.gradientW`),
    trainMse: assertFiniteNumber(obj.trainMse, `${path}.trainMse`),
    observedPredictions,
  };
}

function parseEvent(raw: unknown, index: number): RunEvent {
  const path = `events[${index}]`;
  const obj = assertObject(raw, path);
  const kind = assertString(obj.kind, `${path}.kind`);
  if (!EVENT_KINDS.includes(kind as EventKind)) {
    throw new BundleValidationError(`${path}.kind has an unknown value: ${kind}`);
  }
  return {
    schemaVersion: assertFiniteNumber(obj.schemaVersion, `${path}.schemaVersion`),
    runId: assertString(obj.runId, `${path}.runId`),
    seq: assertFiniteNumber(obj.seq, `${path}.seq`),
    kind: kind as EventKind,
    step: obj.step === null || obj.step === undefined ? null : assertFiniteNumber(obj.step, `${path}.step`),
    message: assertNullableString(obj.message ?? null, `${path}.message`),
  };
}

/**
 * Validate raw parsed JSON for manifest.json / events.jsonl / snapshots.json
 * into a fully-typed, cross-checked RunBundle. Throws BundleValidationError
 * with a specific reason for any violation; never returns a partially-valid
 * bundle.
 */
export function validateBundle(
  manifestRaw: unknown,
  eventsRaw: unknown[],
  snapshotsRaw: unknown[],
): RunBundle {
  const manifest = parseManifest(manifestRaw);
  const snapshots = snapshotsRaw.map((raw, i) => parseSnapshot(raw, i));
  const events = eventsRaw.map((raw, i) => parseEvent(raw, i));

  // Cross-reference checks, mirroring export.py's validate_run_for_export.
  const steps = snapshots.map((s) => s.step);
  const sortedSteps = [...steps].sort((a, b) => a - b);
  const uniqueStepCount = new Set(steps).size;
  if (steps.some((s, i) => s !== sortedSteps[i]) || uniqueStepCount !== steps.length) {
    throw new BundleValidationError("snapshots steps are not strictly increasing and unique");
  }

  if (manifest.nSnapshotsWritten !== snapshots.length) {
    throw new BundleValidationError(
      `manifest declares ${manifest.nSnapshotsWritten} snapshots but the bundle has ${snapshots.length}`,
    );
  }

  for (const event of events) {
    if (event.runId !== manifest.runId) {
      throw new BundleValidationError(
        `events contain an event for a different run (${event.runId} != ${manifest.runId})`,
      );
    }
    if (event.schemaVersion !== SCHEMA_VERSION) {
      throw new BundleValidationError(`event has unsupported schemaVersion ${event.schemaVersion}`);
    }
  }
  const seqs = events.map((e) => e.seq);
  const sortedSeqs = [...seqs].sort((a, b) => a - b);
  if (seqs.some((s, i) => s !== sortedSeqs[i]) || new Set(seqs).size !== seqs.length) {
    throw new BundleValidationError("event seq values are not strictly increasing and unique");
  }

  const knownSampleIds = new Set(manifest.dataset.sampleIds);
  for (const sampleId of manifest.observedSampleIds) {
    if (!knownSampleIds.has(sampleId)) {
      throw new BundleValidationError(`observed sample id "${sampleId}" is not in the dataset`);
    }
  }
  for (const snapshot of snapshots) {
    for (const sampleId of Object.keys(snapshot.observedPredictions)) {
      if (!knownSampleIds.has(sampleId)) {
        throw new BundleValidationError(
          `snapshot step ${snapshot.step} predicts unknown sample id "${sampleId}"`,
        );
      }
    }
  }

  return { manifest, events, snapshots };
}
