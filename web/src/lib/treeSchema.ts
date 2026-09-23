/** Schema-v4 CART replay contract. Older schemas stay in schema.ts. */

import { BundleValidationError, type CodeProvenance } from "./schema";

const FEATURE_NAMES = ["mean", "se", "worst"].flatMap((group) =>
  ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"].map((name) => `${name}_${group}`),
);

export interface TreeRosterRow {
  sampleId: string;
  target: number;
  split: "train" | "validation";
}

export interface TreeObservedRow extends TreeRosterRow {
  features: number[];
}

export interface TreeSnapshot {
  step: number;
  nodeId: string;
  parentId: string | null;
  side: "left" | "right" | null;
  depth: number;
  sampleIds: string[];
  nSamples: number;
  classCounts: [number, number];
  giniParent: number;
  isLeaf: boolean;
  splitFeatureIndex: number | null;
  splitFeatureName: string | null;
  splitThreshold: number | null;
  leftChildId: string | null;
  rightChildId: string | null;
  leftCount: number | null;
  rightCount: number | null;
  giniLeft: number | null;
  giniRight: number | null;
  weightedGiniDecrease: number | null;
  predictedClass: number | null;
  predictedLabel: "Benign" | "Malignant" | null;
  leafReason: string | null;
}

export interface TreePrediction {
  sampleId: string;
  leafId: string;
  predictedClass: number;
}

export interface TreeEvaluation {
  correct: number;
  total: number;
  accuracy: number;
}

export interface TreeRunManifest {
  schemaVersion: 4;
  runId: string;
  experimentId: string;
  createdAt: string;
  status: "completed";
  stopReason: "tree_complete";
  lastValidStep: number;
  errorMessage: string | null;
  dataConfig: {
    source: "external_dataset";
    datasetId: "uci-wdbc";
    datasetVersion: "1.0.0";
    versionManifestSha256: string;
    processedArtifactSha256: string;
    splitSha256: string;
    splitStrategy: "stratified-sha256-2026-v1";
    featureNames: string[];
    targetMapping: { B: number; M: number };
    preprocessing: "none";
    trainCount: 397;
    validationCount: 172;
  };
  dataset: { roster: TreeRosterRow[]; observedRows: TreeObservedRow[] };
  trainingConfig: { algorithm: "cart_gini"; maxDepth: number; minSamplesSplit: 2 };
  codeProvenance: CodeProvenance;
  observedSampleIds: string[];
  nSnapshotsWritten: number;
  trainEvaluation: TreeEvaluation;
  validationEvaluation: TreeEvaluation;
  predictions: TreePrediction[];
}

export interface TreeEvent {
  schemaVersion: 4;
  runId: string;
  seq: number;
  kind: string;
  step: number | null;
  nodeId: string | null;
  message: string | null;
}

export interface TreeRunBundle {
  manifest: TreeRunManifest;
  snapshots: TreeSnapshot[];
  events: TreeEvent[];
}

function fail(message: string): never {
  throw new BundleValidationError(`schema-v4: ${message}`);
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${path} must be an object`);
  return value as Record<string, unknown>;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(`${path} must be an array`);
  return value as unknown[];
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string" || !value) fail(`${path} must be a nonempty string`);
  return value as string;
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${path} must be finite`);
  return value as number;
}

function integer(value: unknown, path: string): number {
  const parsed = number(value, path);
  if (!Number.isInteger(parsed)) fail(`${path} must be an integer`);
  return parsed;
}

function sha(value: unknown, path: string): string {
  const parsed = string(value, path);
  if (!/^[0-9a-f]{64}$/.test(parsed)) fail(`${path} must be SHA-256`);
  return parsed;
}

function nullableString(value: unknown, path: string): string | null {
  return value === null ? null : string(value, path);
}

function nullableNumber(value: unknown, path: string): number | null {
  return value === null ? null : number(value, path);
}

function literal<T extends string | number | boolean>(value: unknown, expected: T, path: string): T {
  if (value !== expected) fail(`${path} must be ${expected}`);
  return expected;
}

function label(value: unknown, path: string): number {
  const parsed = integer(value, path);
  if (parsed !== 0 && parsed !== 1) fail(`${path} must be class 0 or 1`);
  return parsed;
}

function split(value: unknown, path: string): "train" | "validation" {
  if (value !== "train" && value !== "validation") fail(`${path} must be train/validation`);
  return value as "train" | "validation";
}

function evaluation(value: unknown, path: string): TreeEvaluation {
  const source = object(value, path);
  return {
    correct: integer(source.correct, `${path}.correct`),
    total: integer(source.total, `${path}.total`),
    accuracy: number(source.accuracy, `${path}.accuracy`),
  };
}

function gini(n0: number, n1: number): number {
  const n = n0 + n1;
  if (n < 1) fail("empty node");
  return (2 * n0 * n1) / (n * n);
}

function close(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-12;
}

export function validateTreeBundle(
  manifestRaw: unknown,
  eventsRaw: unknown[],
  snapshotsRaw: unknown[],
): TreeRunBundle {
  const raw = object(manifestRaw, "manifest");
  literal(raw.schemaVersion, 4, "schemaVersion");
  literal(raw.status, "completed", "manifest.status");
  literal(raw.stopReason, "tree_complete", "manifest.stopReason");
  const data = object(raw.dataConfig, "manifest.dataConfig");
  const config = object(raw.trainingConfig, "manifest.trainingConfig");
  const source = object(raw.dataset, "manifest.dataset");
  const mapping = object(data.targetMapping, "manifest.dataConfig.targetMapping");
  if (integer(mapping.B, "targetMapping.B") !== 0 || integer(mapping.M, "targetMapping.M") !== 1) fail("target mapping invalid");
  const featureNames = array(data.featureNames, "featureNames").map((name, i) => string(name, `featureNames[${i}]`));
  if (featureNames.length !== 30 || featureNames.some((name, i) => name !== FEATURE_NAMES[i])) fail("feature order invalid");
  const roster = array(source.roster, "dataset.roster").map((item, i): TreeRosterRow => {
    const row = object(item, `roster[${i}]`);
    return { sampleId: string(row.sampleId, "sampleId"), target: label(row.target, "target"), split: split(row.split, "split") };
  });
  const keys = roster.map((row) => row.sampleId);
  const sortedKeys = [...keys].sort();
  if (roster.length !== 569 || keys.some((key, i) => key !== sortedKeys[i]) || new Set(keys).size !== keys.length) fail("roster IDs invalid");
  const observedRows = array(source.observedRows, "dataset.observedRows").map((item, i): TreeObservedRow => {
    const row = object(item, `observedRows[${i}]`);
    const features = array(row.features, "features").map((value, j) => number(value, `features[${j}]`));
    if (features.length !== 30) fail("observed feature count invalid");
    return { sampleId: string(row.sampleId, "sampleId"), target: label(row.target, "target"), split: split(row.split, "split"), features };
  });
  const observedSampleIds = array(raw.observedSampleIds, "observedSampleIds").map((key) => string(key, "observedSampleId"));
  if (observedRows.length !== 20 || observedRows.some((row, i) => row.sampleId !== observedSampleIds[i])) fail("observed row IDs invalid");
  const byId = new Map(roster.map((row) => [row.sampleId, row]));
  for (const row of observedRows) {
    const reference = byId.get(row.sampleId);
    if (!reference || reference.target !== row.target || reference.split !== row.split) fail("observed row roster mismatch");
  }
  for (const splitName of ["train", "validation"] as const) {
    for (const target of [0, 1]) {
      const expected = roster.filter((row) => row.split === splitName && row.target === target).slice(0, 5).map((row) => row.sampleId);
      const actual = observedRows.filter((row) => row.split === splitName && row.target === target).map((row) => row.sampleId);
      if (JSON.stringify(actual) !== JSON.stringify(expected)) fail("observed stratum selection invalid");
    }
  }
  const train = new Set(roster.filter((row) => row.split === "train").map((row) => row.sampleId));
  if (train.size !== 397) fail("train split count invalid");
  const provenance = object(raw.codeProvenance, "codeProvenance");
  const codeProvenance: CodeProvenance = {
    gitCommit: provenance.gitCommit === null ? null : string(provenance.gitCommit, "gitCommit"),
    gitDirty: provenance.gitDirty === null ? null : literalBoolean(provenance.gitDirty, "gitDirty"),
    unavailableReason: nullableString(provenance.unavailableReason, "unavailableReason"),
  };
  const depth = integer(config.maxDepth, "maxDepth");
  if (depth < 1 || depth > 5) fail("maxDepth invalid");
  const manifest: TreeRunManifest = {
    schemaVersion: 4,
    runId: string(raw.runId, "runId"),
    experimentId: string(raw.experimentId, "experimentId"),
    createdAt: string(raw.createdAt, "createdAt"),
    status: "completed",
    stopReason: "tree_complete",
    lastValidStep: integer(raw.lastValidStep, "lastValidStep"),
    errorMessage: nullableString(raw.errorMessage, "errorMessage"),
    dataConfig: {
      source: literal(data.source, "external_dataset", "source"),
      datasetId: literal(data.datasetId, "uci-wdbc", "datasetId"),
      datasetVersion: literal(data.datasetVersion, "1.0.0", "datasetVersion"),
      versionManifestSha256: sha(data.versionManifestSha256, "versionManifestSha256"),
      processedArtifactSha256: sha(data.processedArtifactSha256, "processedArtifactSha256"),
      splitSha256: sha(data.splitSha256, "splitSha256"),
      splitStrategy: literal(data.splitStrategy, "stratified-sha256-2026-v1", "splitStrategy"),
      featureNames,
      targetMapping: { B: 0, M: 1 },
      preprocessing: literal(data.preprocessing, "none", "preprocessing"),
      trainCount: literal(data.trainCount, 397, "trainCount"),
      validationCount: literal(data.validationCount, 172, "validationCount"),
    },
    dataset: { roster, observedRows },
    trainingConfig: {
      algorithm: literal(config.algorithm, "cart_gini", "algorithm"),
      maxDepth: depth,
      minSamplesSplit: literal(config.minSamplesSplit, 2, "minSamplesSplit"),
    },
    codeProvenance,
    observedSampleIds,
    nSnapshotsWritten: integer(raw.nSnapshotsWritten, "nSnapshotsWritten"),
    trainEvaluation: evaluation(raw.trainEvaluation, "trainEvaluation"),
    validationEvaluation: evaluation(raw.validationEvaluation, "validationEvaluation"),
    predictions: array(raw.predictions, "predictions").map((item): TreePrediction => {
      const prediction = object(item, "prediction");
      return { sampleId: string(prediction.sampleId, "sampleId"), leafId: string(prediction.leafId, "leafId"), predictedClass: label(prediction.predictedClass, "predictedClass") };
    }),
  };
  const snapshots = snapshotsRaw.map((item, i): TreeSnapshot => {
    const node = object(item, `snapshots[${i}]`);
    const counts = array(node.classCounts, "classCounts").map((value) => integer(value, "classCount"));
    if (counts.length !== 2 || counts.some((value) => value < 0)) fail("classCounts invalid");
    const sampleIds = array(node.sampleIds, "sampleIds").map((key) => string(key, "sampleId"));
    const leaf = literalBoolean(node.isLeaf, "isLeaf");
    const reason = nullableString(node.leafReason, "leafReason");
    if (leaf && !["pure", "max_depth", "insufficient_samples", "no_positive_gain"].includes(reason ?? "")) fail("leaf reason invalid");
    const snapshot: TreeSnapshot = {
      step: integer(node.step, "step"), nodeId: string(node.nodeId, "nodeId"),
      parentId: nullableString(node.parentId, "parentId"),
      side: node.side === null ? null : splitSide(node.side),
      depth: integer(node.depth, "depth"), sampleIds,
      nSamples: integer(node.nSamples, "nSamples"), classCounts: [counts[0]!, counts[1]!],
      giniParent: number(node.giniParent, "giniParent"), isLeaf: leaf,
      splitFeatureIndex: node.splitFeatureIndex === null ? null : integer(node.splitFeatureIndex, "splitFeatureIndex"),
      splitFeatureName: nullableString(node.splitFeatureName, "splitFeatureName"),
      splitThreshold: nullableNumber(node.splitThreshold, "splitThreshold"),
      leftChildId: nullableString(node.leftChildId, "leftChildId"),
      rightChildId: nullableString(node.rightChildId, "rightChildId"),
      leftCount: node.leftCount === null ? null : integer(node.leftCount, "leftCount"),
      rightCount: node.rightCount === null ? null : integer(node.rightCount, "rightCount"),
      giniLeft: nullableNumber(node.giniLeft, "giniLeft"), giniRight: nullableNumber(node.giniRight, "giniRight"),
      weightedGiniDecrease: nullableNumber(node.weightedGiniDecrease, "weightedGiniDecrease"),
      predictedClass: node.predictedClass === null ? null : label(node.predictedClass, "predictedClass"),
      predictedLabel: node.predictedLabel === null ? null : displayLabel(node.predictedLabel),
      leafReason: reason,
    };
    const splitFields = [snapshot.splitFeatureIndex, snapshot.splitFeatureName, snapshot.splitThreshold, snapshot.leftChildId, snapshot.rightChildId, snapshot.leftCount, snapshot.rightCount, snapshot.giniLeft, snapshot.giniRight, snapshot.weightedGiniDecrease];
    if (leaf ? splitFields.some((v) => v !== null) || snapshot.predictedClass === null || snapshot.predictedLabel === null : splitFields.some((v) => v === null) || snapshot.predictedClass !== null || snapshot.predictedLabel !== null || reason !== null) fail("split/leaf variant invalid");
    return snapshot;
  });
  if (!snapshots.length || snapshots.length !== manifest.nSnapshotsWritten || manifest.lastValidStep !== snapshots.length - 1) fail("snapshot count invalid");
  const pending = ["r"];
  const nodes = new Map<string, TreeSnapshot>();
  snapshots.forEach((node, index) => {
    if (node.step !== index || pending.pop() !== node.nodeId || nodes.has(node.nodeId)) fail("DFS node order invalid");
    const parent = node.parentId === null ? null : nodes.get(node.parentId);
    if (node.nodeId === "r" ? parent !== null || node.depth !== 0 || node.side !== null : !parent || node.depth !== parent.depth + 1 || node.nodeId !== `${parent.nodeId}${node.side === "left" ? "L" : "R"}`) fail("node parent/depth invalid");
    if (sampleSet(node.sampleIds).size !== node.sampleIds.length || node.sampleIds.some((key, i) => key !== [...node.sampleIds].sort()[i] || !train.has(key))) fail("node membership invalid");
    const n0 = node.sampleIds.filter((key) => byId.get(key)?.target === 0).length;
    const n1 = node.sampleIds.filter((key) => byId.get(key)?.target === 1).length;
    if (node.nSamples !== node.sampleIds.length || node.classCounts[0] !== n0 || node.classCounts[1] !== n1 || !close(node.giniParent, gini(n0, n1))) fail("node Gini/count invalid");
    if (node.isLeaf) {
      if (node.predictedClass !== Number(n1 > node.nSamples / 2) || node.predictedLabel !== (node.predictedClass === 1 ? "Malignant" : "Benign")) fail("leaf prediction/label invalid");
    } else {
      if (node.depth >= depth || node.splitFeatureIndex === null || node.splitFeatureIndex < 0 || node.splitFeatureIndex >= 30 || node.splitFeatureName !== featureNames[node.splitFeatureIndex] || node.leftChildId !== `${node.nodeId}L` || node.rightChildId !== `${node.nodeId}R`) fail("split child/feature invalid");
      pending.push(node.rightChildId!, node.leftChildId!);
    }
    nodes.set(node.nodeId, node);
  });
  if (pending.length || sampleSet(snapshots[0]!.sampleIds).size !== train.size) fail("missing nodes/root membership invalid");
  for (const node of snapshots) {
    if (node.isLeaf) continue;
    const left = nodes.get(node.leftChildId!)!, right = nodes.get(node.rightChildId!)!;
    if (left.sampleIds.some((key) => right.sampleIds.includes(key)) || sampleSet([...left.sampleIds, ...right.sampleIds]).size !== node.nSamples || left.sampleIds.some((key) => !node.sampleIds.includes(key))) fail("child membership invalid");
    const gain = node.giniParent - (left.nSamples * left.giniParent + right.nSamples * right.giniParent) / node.nSamples;
    if (node.leftCount !== left.nSamples || node.rightCount !== right.nSamples || !close(node.giniLeft!, left.giniParent) || !close(node.giniRight!, right.giniParent) || !close(node.weightedGiniDecrease!, gain) || gain <= 0) fail("split Gini decrease invalid");
  }
  const events = eventsRaw.map((item): TreeEvent => {
    const event = object(item, "event");
    return { schemaVersion: literal(event.schemaVersion, 4, "event.schemaVersion"), runId: string(event.runId, "event.runId"), seq: integer(event.seq, "event.seq"), kind: string(event.kind, "event.kind"), step: event.step === null ? null : integer(event.step, "event.step"), nodeId: nullableString(event.nodeId, "event.nodeId"), message: nullableString(event.message, "event.message") };
  });
  if (events.length !== snapshots.length + 3 || events.some((event, i) => event.seq !== i + 1 || event.runId !== manifest.runId)) fail("event sequence invalid");
  if (events[0]?.kind !== "run.created" || events[1]?.kind !== "run.started" || events.at(-1)?.kind !== "run.completed") fail("lifecycle events invalid");
  for (const node of snapshots) {
    const event = events[node.step + 2];
    if (event?.step !== node.step || event.nodeId !== node.nodeId || event.kind !== (node.isLeaf ? "tree.node_leaf" : "tree.node_split")) fail("node event mismatch");
  }
  if (manifest.predictions.length !== roster.length || manifest.predictions.some((p, i) => p.sampleId !== keys[i])) fail("prediction roster invalid");
  const predictionById = new Map(manifest.predictions.map((item) => [item.sampleId, item]));
  for (const row of observedRows) {
    let node = nodes.get("r")!;
    while (!node.isLeaf) {
      const childId: string = row.features[node.splitFeatureIndex!]! <= node.splitThreshold! ? node.leftChildId! : node.rightChildId!;
      node = nodes.get(childId)!;
    }
    if (predictionById.get(row.sampleId)?.leafId !== node.nodeId) fail("observed sample route invalid");
  }
  for (const splitName of ["train", "validation"] as const) {
    const rows = roster.filter((row) => row.split === splitName);
    const correct = rows.filter((row) => {
      const prediction = manifest.predictions.find((p) => p.sampleId === row.sampleId)!;
      const leaf = nodes.get(prediction.leafId);
      if (!leaf?.isLeaf || leaf.predictedClass !== prediction.predictedClass || (splitName === "train" && !leaf.sampleIds.includes(row.sampleId))) fail("prediction leaf invalid");
      return row.target === prediction.predictedClass;
    }).length;
    const result = splitName === "train" ? manifest.trainEvaluation : manifest.validationEvaluation;
    if (result.total !== rows.length || result.correct !== correct || !close(result.accuracy, correct / rows.length)) fail("evaluation invalid");
  }
  return { manifest, events, snapshots };
}

function literalBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(`${path} must be boolean`);
  return value as boolean;
}

function displayLabel(value: unknown): "Benign" | "Malignant" {
  if (value !== "Benign" && value !== "Malignant") fail("predicted label invalid");
  return value as "Benign" | "Malignant";
}

function splitSide(value: unknown): "left" | "right" {
  if (value !== "left" && value !== "right") fail("side invalid");
  return value as "left" | "right";
}

function sampleSet(values: string[]): Set<string> {
  return new Set(values);
}
