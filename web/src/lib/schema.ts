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

import { validateTreeBundle, type TreeRunBundle } from "./treeSchema";
export const SCHEMA_VERSION = 1;
export const SUPPORTED_SCHEMA_VERSIONS = [1, 2, 3, 4] as const;
export type SchemaVersion = (typeof SUPPORTED_SCHEMA_VERSIONS)[number];

export type RunStatus =
  | "created"
  | "running"
  | "cancelling"
  | "completed"
  | "cancelled"
  | "interrupted"
  | "failed";
export type StopReason =
  | "max_steps"
  | "numerical_error"
  | "io_error"
  | "user_cancelled"
  | "runtime_error"
  | "forced_termination"
  | "worker_lost"
  | "service_restart"
  | "service_shutdown";

export interface SyntheticDataConfig {
  generator: string;
  nSamples: number;
  trueBias: number;
  trueWeight: number;
  noiseStd: number;
  seed: number;
}

export interface ExternalDataConfig {
  source: "external_dataset";
  datasetId: string;
  datasetVersion: string;
  versionManifestSha256: string;
  processedArtifactSha256: string;
  sourceFeature: "weight";
  feature: "weight_standardized";
  featureUnit: "population standard deviations";
  target: "mpg";
  targetUnit: "miles per gallon";
  preprocessing: "population_standardization";
  split: "all-398-rows";
}

export type DataConfig = SyntheticDataConfig | ExternalDataConfig;

export interface ModelConfig {
  algorithm: "linear_regression_gradient_descent";
  initialBias: number;
  initialWeight: number;
  learningRate: number;
  nUpdates: number;
}

interface DatasetSummaryBase {
  sampleIds: string[];
  x: number[];
  y: number[];
}

export interface SyntheticDatasetSummary extends DatasetSummaryBase {
  generatorId: string;
}

export interface ExternalDatasetSummary extends DatasetSummaryBase {
  sourceId: string;
}

export type DatasetSummary = SyntheticDatasetSummary | ExternalDatasetSummary;

export interface CodeProvenance {
  gitCommit: string | null;
  gitDirty: boolean | null;
  unavailableReason: string | null;
}

interface RunManifestBase {
  runId: string;
  experimentId: string;
  createdAt: string;
  status: RunStatus;
  stopReason: StopReason | null;
  lastValidStep: number | null;
  errorMessage: string | null;
  trainingConfig: ModelConfig;
  codeProvenance: CodeProvenance;
  observedSampleIds: string[];
  nSnapshotsWritten: number;
}

export interface SyntheticRunManifest extends RunManifestBase {
  schemaVersion: 1;
  dataConfig: SyntheticDataConfig;
  dataset: SyntheticDatasetSummary;
}

export interface ExternalRunManifest extends RunManifestBase {
  schemaVersion: 2;
  dataConfig: ExternalDataConfig;
  dataset: ExternalDatasetSummary;
}

export type RunManifest = SyntheticRunManifest | ExternalRunManifest;

export interface Snapshot {
  step: number;
  b: number;
  w: number;
  gradientB: number;
  gradientW: number;
  trainMse: number;
  observedPredictions: Record<string, number>;
}

export interface KMeansDataConfig {
  generator: "synthetic_kmeans_v1";
  blobCenters: [number, number][];
  blobSizes: number[];
  clusterStd: number;
  seed: number;
}

export interface KMeansConfig {
  algorithm: "kmeans_lloyd";
  nClusters: number;
  initSeed: number;
  maxIterations: number;
}

export interface KMeansDatasetSummary {
  generatorId: "synthetic_kmeans_v1";
  sampleIds: string[];
  points: [number, number][];
}

export interface KMeansRunManifest {
  schemaVersion: 3;
  runId: string;
  experimentId: string;
  createdAt: string;
  status: "completed";
  stopReason: "assignments_stable" | "max_iterations";
  lastValidStep: number;
  errorMessage: string | null;
  dataConfig: KMeansDataConfig;
  dataset: KMeansDatasetSummary;
  trainingConfig: KMeansConfig;
  codeProvenance: CodeProvenance;
  observedSampleIds: string[];
  nSnapshotsWritten: number;
}

export interface KMeansSnapshot {
  step: number;
  iteration: number;
  phase: "assignment" | "update";
  centers: [number, number][];
  assignments: number[];
  inertia: number;
  emptyClusters: number[];
}

export interface KMeansEvent {
  schemaVersion: 3;
  runId: string;
  seq: number;
  kind:
    | "run.created"
    | "run.started"
    | "iteration.assigned"
    | "iteration.updated"
    | "cluster.empty"
    | "run.completed"
    | "run.failed";
  step: number | null;
  iteration: number | null;
  message: string | null;
}

export type EventKind =
  | "run.created"
  | "run.started"
  | "run.cancelling"
  | "step.recorded"
  | "run.completed"
  | "run.cancelled"
  | "run.interrupted"
  | "run.failed";

export interface RunEvent {
  schemaVersion: SchemaVersion;
  runId: string;
  seq: number;
  kind: EventKind;
  step: number | null;
  message: string | null;
}

export interface LinearRunBundle {
  manifest: RunManifest;
  events: RunEvent[];
  snapshots: Snapshot[];
}

export interface KMeansRunBundle {
  manifest: KMeansRunManifest;
  events: KMeansEvent[];
  snapshots: KMeansSnapshot[];
}

export type RunBundle = LinearRunBundle | KMeansRunBundle | TreeRunBundle;

export function isTreeBundle(bundle: RunBundle): bundle is TreeRunBundle {
  return bundle.manifest.schemaVersion === 4;
}

export function isKMeansBundle(bundle: RunBundle): bundle is KMeansRunBundle {
  return bundle.manifest.schemaVersion === 3;
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
    throw new BundleValidationError(
      `non-finite or non-numeric value at ${path}: ${JSON.stringify(value)}`,
    );
  }
  return value;
}

function assertString(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new BundleValidationError(
      `expected string at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return value;
}

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function assertSha256(value: unknown, path: string): string {
  const digest = assertString(value, path);
  if (!SHA256_PATTERN.test(digest)) {
    throw new BundleValidationError(
      `expected a lowercase 64-character SHA-256 hex digest at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return digest;
}

function assertLiteral<T extends string>(
  value: unknown,
  expected: T,
  path: string,
): T {
  if (value !== expected) {
    throw new BundleValidationError(
      `expected ${JSON.stringify(expected)} at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return expected;
}

function assertSchemaVersion(value: unknown, path: string): SchemaVersion {
  const version = assertFiniteNumber(value, path);
  if (!SUPPORTED_SCHEMA_VERSIONS.includes(version as SchemaVersion)) {
    throw new BundleValidationError(
      `unsupported schemaVersion ${version}; this viewer supports schema versions 1, 2, 3, and 4`,
    );
  }
  return version as SchemaVersion;
}

function assertNoKeys(
  obj: Record<string, unknown>,
  forbiddenKeys: readonly string[],
  path: string,
): void {
  const presentKey = forbiddenKeys.find((key) => key in obj);
  if (presentKey !== undefined) {
    throw new BundleValidationError(
      `${path} contains ${presentKey}, which does not match its schema version`,
    );
  }
}

function assertNullableString(value: unknown, path: string): string | null {
  if (value === null) return null;
  return assertString(value, path);
}

function assertArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new BundleValidationError(
      `expected array at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return value;
}

function assertObject(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new BundleValidationError(
      `expected object at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return value as Record<string, unknown>;
}

const RUN_STATUSES: RunStatus[] = [
  "created",
  "running",
  "cancelling",
  "completed",
  "cancelled",
  "interrupted",
  "failed",
];
const STOP_REASONS: StopReason[] = [
  "max_steps",
  "numerical_error",
  "io_error",
  "user_cancelled",
  "runtime_error",
  "forced_termination",
  "worker_lost",
  "service_restart",
  "service_shutdown",
];
const EVENT_KINDS: EventKind[] = [
  "run.created",
  "run.started",
  "run.cancelling",
  "step.recorded",
  "run.completed",
  "run.cancelled",
  "run.interrupted",
  "run.failed",
];

const SYNTHETIC_DATA_KEYS = [
  "generator",
  "nSamples",
  "trueBias",
  "trueWeight",
  "noiseStd",
  "seed",
] as const;
const EXTERNAL_DATA_KEYS = [
  "source",
  "datasetId",
  "datasetVersion",
  "versionManifestSha256",
  "processedArtifactSha256",
  "sourceFeature",
  "feature",
  "featureUnit",
  "target",
  "targetUnit",
  "preprocessing",
  "split",
] as const;

function parseSyntheticDataConfig(
  raw: unknown,
  path: string,
): SyntheticDataConfig {
  const obj = assertObject(raw, path);
  assertNoKeys(obj, EXTERNAL_DATA_KEYS, path);
  return {
    generator: assertString(obj.generator, `${path}.generator`),
    nSamples: assertFiniteNumber(obj.nSamples, `${path}.nSamples`),
    trueBias: assertFiniteNumber(obj.trueBias, `${path}.trueBias`),
    trueWeight: assertFiniteNumber(obj.trueWeight, `${path}.trueWeight`),
    noiseStd: assertFiniteNumber(obj.noiseStd, `${path}.noiseStd`),
    seed: assertFiniteNumber(obj.seed, `${path}.seed`),
  };
}

function parseExternalDataConfig(
  raw: unknown,
  path: string,
): ExternalDataConfig {
  const obj = assertObject(raw, path);
  assertNoKeys(obj, SYNTHETIC_DATA_KEYS, path);
  return {
    source: assertLiteral(obj.source, "external_dataset", `${path}.source`),
    datasetId: assertString(obj.datasetId, `${path}.datasetId`),
    datasetVersion: assertString(obj.datasetVersion, `${path}.datasetVersion`),
    versionManifestSha256: assertSha256(
      obj.versionManifestSha256,
      `${path}.versionManifestSha256`,
    ),
    processedArtifactSha256: assertSha256(
      obj.processedArtifactSha256,
      `${path}.processedArtifactSha256`,
    ),
    sourceFeature: assertLiteral(
      obj.sourceFeature,
      "weight",
      `${path}.sourceFeature`,
    ),
    feature: assertLiteral(
      obj.feature,
      "weight_standardized",
      `${path}.feature`,
    ),
    featureUnit: assertLiteral(
      obj.featureUnit,
      "population standard deviations",
      `${path}.featureUnit`,
    ),
    target: assertLiteral(obj.target, "mpg", `${path}.target`),
    targetUnit: assertLiteral(
      obj.targetUnit,
      "miles per gallon",
      `${path}.targetUnit`,
    ),
    preprocessing: assertLiteral(
      obj.preprocessing,
      "population_standardization",
      `${path}.preprocessing`,
    ),
    split: assertLiteral(obj.split, "all-398-rows", `${path}.split`),
  };
}

export function isExternalDataConfig(
  dataConfig: DataConfig,
): dataConfig is ExternalDataConfig {
  return "source" in dataConfig && dataConfig.source === "external_dataset";
}

export function featureLabel(dataConfig: DataConfig): string {
  return isExternalDataConfig(dataConfig)
    ? `${dataConfig.feature} (${dataConfig.featureUnit})`
    : "x";
}

export function targetLabel(dataConfig: DataConfig): string {
  return isExternalDataConfig(dataConfig)
    ? `${dataConfig.target} (${dataConfig.targetUnit})`
    : "y";
}

function parseModelConfig(raw: unknown, path: string): ModelConfig {
  const obj = assertObject(raw, path);
  if (obj.algorithm !== "linear_regression_gradient_descent") {
    throw new BundleValidationError(
      `unsupported algorithm at ${path}.algorithm: ${JSON.stringify(obj.algorithm)}`,
    );
  }
  return {
    algorithm: "linear_regression_gradient_descent",
    initialBias: assertFiniteNumber(obj.initialBias, `${path}.initialBias`),
    initialWeight: assertFiniteNumber(
      obj.initialWeight,
      `${path}.initialWeight`,
    ),
    learningRate: assertFiniteNumber(obj.learningRate, `${path}.learningRate`),
    nUpdates: assertFiniteNumber(obj.nUpdates, `${path}.nUpdates`),
  };
}

function parseDatasetValues(
  obj: Record<string, unknown>,
  path: string,
): DatasetSummaryBase {
  const sampleIds = assertArray(obj.sampleIds, `${path}.sampleIds`).map(
    (v, i) => assertString(v, `${path}.sampleIds[${i}]`),
  );
  const x = assertArray(obj.x, `${path}.x`).map((v, i) =>
    assertFiniteNumber(v, `${path}.x[${i}]`),
  );
  const y = assertArray(obj.y, `${path}.y`).map((v, i) =>
    assertFiniteNumber(v, `${path}.y[${i}]`),
  );
  if (x.length !== sampleIds.length || y.length !== sampleIds.length) {
    throw new BundleValidationError(
      `${path}: sampleIds, x, and y must have the same length`,
    );
  }
  return { sampleIds, x, y };
}

function parseSyntheticDatasetSummary(
  raw: unknown,
  path: string,
): SyntheticDatasetSummary {
  const obj = assertObject(raw, path);
  assertNoKeys(obj, ["sourceId"], path);
  return {
    generatorId: assertString(obj.generatorId, `${path}.generatorId`),
    ...parseDatasetValues(obj, path),
  };
}

function parseExternalDatasetSummary(
  raw: unknown,
  path: string,
): ExternalDatasetSummary {
  const obj = assertObject(raw, path);
  assertNoKeys(obj, ["generatorId"], path);
  return {
    sourceId: assertString(obj.sourceId, `${path}.sourceId`),
    ...parseDatasetValues(obj, path),
  };
}

function assertNullableBoolean(value: unknown, path: string): boolean | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "boolean") {
    throw new BundleValidationError(
      `expected boolean or null at ${path}, got ${JSON.stringify(value)}`,
    );
  }
  return value;
}

function parseCodeProvenance(raw: unknown, path: string): CodeProvenance {
  const obj = assertObject(raw, path);
  return {
    gitCommit: assertNullableString(obj.gitCommit ?? null, `${path}.gitCommit`),
    gitDirty: assertNullableBoolean(obj.gitDirty, `${path}.gitDirty`),
    unavailableReason: assertNullableString(
      obj.unavailableReason ?? null,
      `${path}.unavailableReason`,
    ),
  };
}

function parseManifest(raw: unknown): RunManifest {
  const obj = assertObject(raw, "manifest");

  const schemaVersion = assertSchemaVersion(
    obj.schemaVersion,
    "manifest.schemaVersion",
  );
  if (schemaVersion === 3) {
    throw new BundleValidationError("schemaVersion 3 requires the K-means parser");
  }
  if (schemaVersion === 4) {
    throw new BundleValidationError("schemaVersion 4 requires the tree parser");
  }

  const status = assertString(obj.status, "manifest.status");
  if (!RUN_STATUSES.includes(status as RunStatus)) {
    throw new BundleValidationError(
      `manifest.status has an unknown value: ${status}`,
    );
  }

  const stopReasonRaw = obj.stopReason;
  let stopReason: StopReason | null = null;
  if (stopReasonRaw !== null && stopReasonRaw !== undefined) {
    const s = assertString(stopReasonRaw, "manifest.stopReason");
    if (!STOP_REASONS.includes(s as StopReason)) {
      throw new BundleValidationError(
        `manifest.stopReason has an unknown value: ${s}`,
      );
    }
    stopReason = s as StopReason;
  }

  if (status !== "completed") {
    throw new BundleValidationError(
      `manifest.status is "${status}", not "completed"; only completed runs can be replayed`,
    );
  }

  const common: RunManifestBase = {
    runId: assertString(obj.runId, "manifest.runId"),
    experimentId: assertString(obj.experimentId, "manifest.experimentId"),
    createdAt: assertString(obj.createdAt, "manifest.createdAt"),
    status: status as RunStatus,
    stopReason,
    lastValidStep:
      obj.lastValidStep === null || obj.lastValidStep === undefined
        ? null
        : assertFiniteNumber(obj.lastValidStep, "manifest.lastValidStep"),
    errorMessage: assertNullableString(
      obj.errorMessage ?? null,
      "manifest.errorMessage",
    ),
    trainingConfig: parseModelConfig(
      obj.trainingConfig,
      "manifest.trainingConfig",
    ),
    codeProvenance: parseCodeProvenance(
      obj.codeProvenance,
      "manifest.codeProvenance",
    ),
    observedSampleIds: assertArray(
      obj.observedSampleIds,
      "manifest.observedSampleIds",
    ).map((v, i) => assertString(v, `manifest.observedSampleIds[${i}]`)),
    nSnapshotsWritten: assertFiniteNumber(
      obj.nSnapshotsWritten,
      "manifest.nSnapshotsWritten",
    ),
  };

  return schemaVersion === 1
    ? {
        ...common,
        schemaVersion,
        dataConfig: parseSyntheticDataConfig(
          obj.dataConfig,
          "manifest.dataConfig",
        ),
        dataset: parseSyntheticDatasetSummary(obj.dataset, "manifest.dataset"),
      }
    : {
        ...common,
        schemaVersion: 2,
        dataConfig: parseExternalDataConfig(
          obj.dataConfig,
          "manifest.dataConfig",
        ),
        dataset: parseExternalDatasetSummary(obj.dataset, "manifest.dataset"),
      };
}

function parseSnapshot(raw: unknown, index: number): Snapshot {
  const path = `snapshots[${index}]`;
  const obj = assertObject(raw, path);
  const predictionsObj = assertObject(
    obj.observedPredictions,
    `${path}.observedPredictions`,
  );
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
    throw new BundleValidationError(
      `${path}.kind has an unknown value: ${kind}`,
    );
  }
  return {
    schemaVersion: assertSchemaVersion(
      obj.schemaVersion,
      `${path}.schemaVersion`,
    ),
    runId: assertString(obj.runId, `${path}.runId`),
    seq: assertFiniteNumber(obj.seq, `${path}.seq`),
    kind: kind as EventKind,
    step:
      obj.step === null || obj.step === undefined
        ? null
        : assertFiniteNumber(obj.step, `${path}.step`),
    message: assertNullableString(obj.message ?? null, `${path}.message`),
  };
}

/**
 * Validate raw parsed JSON for manifest.json / events.jsonl / snapshots.json
 * into a fully-typed, cross-checked RunBundle. Throws BundleValidationError
 * with a specific reason for any violation; never returns a partially-valid
 * bundle.
 */
function validateLinearBundle(
  manifestRaw: unknown,
  eventsRaw: unknown[],
  snapshotsRaw: unknown[],
): LinearRunBundle {
  const manifest = parseManifest(manifestRaw);
  const snapshots = snapshotsRaw.map((raw, i) => parseSnapshot(raw, i));
  const events = eventsRaw.map((raw, i) => parseEvent(raw, i));

  // Cross-reference checks, mirroring export.py's validate_run_for_export.
  const steps = snapshots.map((s) => s.step);
  const sortedSteps = [...steps].sort((a, b) => a - b);
  const uniqueStepCount = new Set(steps).size;
  if (
    steps.some((s, i) => s !== sortedSteps[i]) ||
    uniqueStepCount !== steps.length
  ) {
    throw new BundleValidationError(
      "snapshots steps are not strictly increasing and unique",
    );
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
    if (event.schemaVersion !== manifest.schemaVersion) {
      throw new BundleValidationError(
        `event schemaVersion ${event.schemaVersion} does not match manifest schemaVersion ${manifest.schemaVersion}`,
      );
    }
  }
  const seqs = events.map((e) => e.seq);
  const sortedSeqs = [...seqs].sort((a, b) => a - b);
  if (
    seqs.some((s, i) => s !== sortedSeqs[i]) ||
    new Set(seqs).size !== seqs.length
  ) {
    throw new BundleValidationError(
      "event seq values are not strictly increasing and unique",
    );
  }

  const knownSampleIds = new Set(manifest.dataset.sampleIds);
  for (const sampleId of manifest.observedSampleIds) {
    if (!knownSampleIds.has(sampleId)) {
      throw new BundleValidationError(
        `observed sample id "${sampleId}" is not in the dataset`,
      );
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

function assertInteger(value: unknown, path: string): number {
  const parsed = assertFiniteNumber(value, path);
  if (!Number.isInteger(parsed)) {
    throw new BundleValidationError(`expected integer at ${path}`);
  }
  return parsed;
}

function parsePoint(value: unknown, path: string): [number, number] {
  const point = assertArray(value, path);
  if (point.length !== 2) {
    throw new BundleValidationError(`${path} must contain exactly two coordinates`);
  }
  return [
    assertFiniteNumber(point[0], `${path}[0]`),
    assertFiniteNumber(point[1], `${path}[1]`),
  ];
}

function validateKMeansBundle(
  manifestRaw: unknown,
  eventsRaw: unknown[],
  snapshotsRaw: unknown[],
): KMeansRunBundle {
  const obj = assertObject(manifestRaw, "manifest");
  if (obj.schemaVersion !== 3) {
    throw new BundleValidationError("K-means manifest must use schemaVersion 3");
  }
  if (obj.status !== "completed") {
    throw new BundleValidationError(
      `manifest.status is ${JSON.stringify(obj.status)}, not "completed"; only completed runs can be replayed`,
    );
  }
  if (obj.stopReason !== "assignments_stable" && obj.stopReason !== "max_iterations") {
    throw new BundleValidationError("manifest.stopReason is invalid for K-means");
  }
  const dataObj = assertObject(obj.dataConfig, "manifest.dataConfig");
  const datasetObj = assertObject(obj.dataset, "manifest.dataset");
  const trainingObj = assertObject(obj.trainingConfig, "manifest.trainingConfig");
  const sampleIds = assertArray(datasetObj.sampleIds, "manifest.dataset.sampleIds").map(
    (value, index) => assertString(value, `manifest.dataset.sampleIds[${index}]`),
  );
  const points = assertArray(datasetObj.points, "manifest.dataset.points").map(
    (value, index) => parsePoint(value, `manifest.dataset.points[${index}]`),
  );
  if (points.length !== sampleIds.length || new Set(sampleIds).size !== sampleIds.length) {
    throw new BundleValidationError("K-means sampleIds and points must align and be unique");
  }
  const nClusters = assertInteger(trainingObj.nClusters, "manifest.trainingConfig.nClusters");
  const manifest: KMeansRunManifest = {
    schemaVersion: 3,
    runId: assertString(obj.runId, "manifest.runId"),
    experimentId: assertString(obj.experimentId, "manifest.experimentId"),
    createdAt: assertString(obj.createdAt, "manifest.createdAt"),
    status: "completed",
    stopReason: obj.stopReason,
    lastValidStep: assertInteger(obj.lastValidStep, "manifest.lastValidStep"),
    errorMessage: assertNullableString(obj.errorMessage ?? null, "manifest.errorMessage"),
    dataConfig: {
      generator: assertLiteral(
        dataObj.generator,
        "synthetic_kmeans_v1",
        "manifest.dataConfig.generator",
      ),
      blobCenters: assertArray(dataObj.blobCenters, "manifest.dataConfig.blobCenters").map(
        (value, index) => parsePoint(value, `manifest.dataConfig.blobCenters[${index}]`),
      ),
      blobSizes: assertArray(dataObj.blobSizes, "manifest.dataConfig.blobSizes").map(
        (value, index) => assertInteger(value, `manifest.dataConfig.blobSizes[${index}]`),
      ),
      clusterStd: assertFiniteNumber(dataObj.clusterStd, "manifest.dataConfig.clusterStd"),
      seed: assertInteger(dataObj.seed, "manifest.dataConfig.seed"),
    },
    dataset: {
      generatorId: assertLiteral(
        datasetObj.generatorId,
        "synthetic_kmeans_v1",
        "manifest.dataset.generatorId",
      ),
      sampleIds,
      points,
    },
    trainingConfig: {
      algorithm: assertLiteral(
        trainingObj.algorithm,
        "kmeans_lloyd",
        "manifest.trainingConfig.algorithm",
      ),
      nClusters,
      initSeed: assertInteger(trainingObj.initSeed, "manifest.trainingConfig.initSeed"),
      maxIterations: assertInteger(
        trainingObj.maxIterations,
        "manifest.trainingConfig.maxIterations",
      ),
    },
    codeProvenance: parseCodeProvenance(obj.codeProvenance, "manifest.codeProvenance"),
    observedSampleIds: assertArray(
      obj.observedSampleIds,
      "manifest.observedSampleIds",
    ).map((value, index) =>
      assertString(value, `manifest.observedSampleIds[${index}]`),
    ),
    nSnapshotsWritten: assertInteger(
      obj.nSnapshotsWritten,
      "manifest.nSnapshotsWritten",
    ),
  };

  const snapshots: KMeansSnapshot[] = snapshotsRaw.map((raw, index) => {
    const path = `snapshots[${index}]`;
    const snapshot = assertObject(raw, path);
    const phase = assertString(snapshot.phase, `${path}.phase`);
    if (phase !== "assignment" && phase !== "update") {
      throw new BundleValidationError(`${path}.phase is invalid`);
    }
    const centers = assertArray(snapshot.centers, `${path}.centers`).map((value, center) =>
      parsePoint(value, `${path}.centers[${center}]`),
    );
    const assignments = assertArray(snapshot.assignments, `${path}.assignments`).map(
      (value, sample) => assertInteger(value, `${path}.assignments[${sample}]`),
    );
    if (centers.length !== nClusters || assignments.length !== points.length) {
      throw new BundleValidationError(`${path} has inconsistent K-means shapes`);
    }
    const inertia = assertFiniteNumber(snapshot.inertia, `${path}.inertia`);
    let reconstructed = 0;
    assignments.forEach((cluster, sample) => {
      if (cluster < 0 || cluster >= nClusters) {
        throw new BundleValidationError(`${path}.assignments[${sample}] is out of range`);
      }
      const point = points[sample];
      const center = centers[cluster];
      if (!point || !center) throw new BundleValidationError(`${path} has missing geometry`);
      reconstructed += point.reduce(
        (sum, coordinate, axis) => sum + (coordinate - center[axis]!) ** 2,
        0,
      );
    });
    if (Math.abs(inertia - reconstructed) > 1e-9 * Math.max(1, reconstructed)) {
      throw new BundleValidationError(`${path}.inertia does not match points and centers`);
    }
    return {
      step: assertInteger(snapshot.step, `${path}.step`),
      iteration: assertInteger(snapshot.iteration, `${path}.iteration`),
      phase,
      centers,
      assignments,
      inertia,
      emptyClusters: assertArray(snapshot.emptyClusters, `${path}.emptyClusters`).map(
        (value, emptyIndex) =>
          assertInteger(value, `${path}.emptyClusters[${emptyIndex}]`),
      ),
    };
  });
  if (
    manifest.nSnapshotsWritten !== snapshots.length ||
    snapshots.some((snapshot, index) => snapshot.step !== index)
  ) {
    throw new BundleValidationError("K-means snapshot count or contiguous steps are invalid");
  }

  const allowedKinds = new Set([
    "run.created",
    "run.started",
    "iteration.assigned",
    "iteration.updated",
    "cluster.empty",
    "run.completed",
    "run.failed",
  ]);
  const events: KMeansEvent[] = eventsRaw.map((raw, index) => {
    const path = `events[${index}]`;
    const event = assertObject(raw, path);
    const kind = assertString(event.kind, `${path}.kind`);
    if (!allowedKinds.has(kind)) throw new BundleValidationError(`${path}.kind is invalid`);
    if (event.runId !== manifest.runId || event.schemaVersion !== 3) {
      throw new BundleValidationError(`${path} does not belong to this schema-v3 run`);
    }
    return {
      schemaVersion: 3,
      runId: manifest.runId,
      seq: assertInteger(event.seq, `${path}.seq`),
      kind: kind as KMeansEvent["kind"],
      step:
        event.step == null ? null : assertInteger(event.step, `${path}.step`),
      iteration:
        event.iteration == null
          ? null
          : assertInteger(event.iteration, `${path}.iteration`),
      message: assertNullableString(event.message ?? null, `${path}.message`),
    };
  });
  if (events.some((event, index) => event.seq !== index + 1)) {
    throw new BundleValidationError("K-means event seq values must be contiguous from one");
  }
  return { manifest, events, snapshots };
}

export function validateBundle(
  manifestRaw: unknown,
  eventsRaw: unknown[],
  snapshotsRaw: unknown[],
): RunBundle {
  const manifest = assertObject(manifestRaw, "manifest");
  if (manifest.schemaVersion === 4) {
    return validateTreeBundle(manifestRaw, eventsRaw, snapshotsRaw);
  }
  return manifest.schemaVersion === 3
    ? validateKMeansBundle(manifestRaw, eventsRaw, snapshotsRaw)
    : validateLinearBundle(manifestRaw, eventsRaw, snapshotsRaw);
}
