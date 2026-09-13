import type {
  ExternalDataConfig,
  KMeansRunBundle,
  LinearRunBundle,
  Snapshot,
} from "../../lib/schema";

interface BundleOptions {
  runId?: string;
  snapshotCount?: number;
  seed?: number;
  initialBias?: number;
  initialWeight?: number;
  learningRate?: number;
}

export function makeKMeansBundle(): KMeansRunBundle {
  return {
    manifest: {
      schemaVersion: 3,
      runId: "kmeans-test",
      experimentId: "kmeans-study-seed-0",
      createdAt: "2026-09-13T00:00:00Z",
      status: "completed",
      stopReason: "assignments_stable",
      lastValidStep: 0,
      errorMessage: null,
      dataConfig: {
        generator: "synthetic_kmeans_v1",
        blobCenters: [[0, 0], [10, 10]],
        blobSizes: [2, 2],
        clusterStd: 0,
        seed: 2026,
      },
      dataset: {
        generatorId: "synthetic_kmeans_v1",
        sampleIds: ["p0", "p1", "p2", "p3"],
        points: [[0, 0], [0, 2], [10, 10], [10, 12]],
      },
      trainingConfig: {
        algorithm: "kmeans_lloyd",
        nClusters: 2,
        initSeed: 0,
        maxIterations: 10,
      },
      codeProvenance: {
        gitCommit: null,
        gitDirty: null,
        unavailableReason: "test fixture",
      },
      observedSampleIds: ["p0"],
      nSnapshotsWritten: 1,
    },
    events: [
      {
        schemaVersion: 3,
        runId: "kmeans-test",
        seq: 1,
        kind: "run.created",
        step: null,
        iteration: null,
        message: null,
      },
    ],
    snapshots: [
      {
        step: 0,
        iteration: 0,
        phase: "assignment",
        centers: [[0, 1], [10, 11]],
        assignments: [0, 0, 1, 1],
        inertia: 4,
        emptyClusters: [],
      },
    ],
  };
}

export function makeBundle(options: BundleOptions = {}): LinearRunBundle {
  const snapshotCount = options.snapshotCount ?? 3;
  const snapshots: Snapshot[] = Array.from(
    { length: snapshotCount },
    (_, step) => ({
      step,
      b: step * 0.1,
      w: step * 0.2,
      gradientB: -0.1,
      gradientW: -0.2,
      trainMse: 10 / (step + 1),
      observedPredictions: { "sample-1": step * 0.2 },
    }),
  );

  return {
    manifest: {
      schemaVersion: 1,
      runId: options.runId ?? "test-run",
      experimentId: "linear-test",
      createdAt: "2026-01-01T00:00:00Z",
      status: "completed",
      stopReason: "max_steps",
      lastValidStep: snapshotCount - 1,
      errorMessage: null,
      dataConfig: {
        generator: "synthetic_linear",
        nSamples: 2,
        trueBias: 0.5,
        trueWeight: 2,
        noiseStd: 0.1,
        seed: options.seed ?? 7,
      },
      dataset: {
        generatorId: "synthetic_linear",
        sampleIds: ["sample-1", "sample-2"],
        x: [0, 1],
        y: [0.5, 2.5],
      },
      trainingConfig: {
        algorithm: "linear_regression_gradient_descent",
        initialBias: options.initialBias ?? 0,
        initialWeight: options.initialWeight ?? 0,
        learningRate: options.learningRate ?? 0.1,
        nUpdates: Math.max(0, snapshotCount - 1),
      },
      codeProvenance: {
        gitCommit: null,
        gitDirty: null,
        unavailableReason: "test fixture",
      },
      observedSampleIds: ["sample-1"],
      nSnapshotsWritten: snapshotCount,
    },
    events: [],
    snapshots,
  };
}

export function makeExternalBundle(
  dataConfigOverrides: Partial<ExternalDataConfig> = {},
  options: BundleOptions = {},
): LinearRunBundle {
  const bundle = makeBundle(options);
  bundle.manifest.schemaVersion = 2;
  bundle.manifest.dataConfig = {
    source: "external_dataset",
    datasetId: "auto-mpg",
    datasetVersion: "1.0.0",
    versionManifestSha256: "a".repeat(64),
    processedArtifactSha256: "b".repeat(64),
    sourceFeature: "weight",
    feature: "weight_standardized",
    featureUnit: "population standard deviations",
    target: "mpg",
    targetUnit: "miles per gallon",
    preprocessing: "population_standardization",
    split: "all-398-rows",
    ...dataConfigOverrides,
  };
  bundle.manifest.dataset = {
    sourceId: "auto-mpg",
    sampleIds: bundle.manifest.dataset.sampleIds,
    x: bundle.manifest.dataset.x,
    y: bundle.manifest.dataset.y,
  };
  bundle.events.forEach((event) => {
    event.schemaVersion = 2;
  });
  return bundle;
}
