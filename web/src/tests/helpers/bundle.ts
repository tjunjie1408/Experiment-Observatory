import type { ExternalDataConfig, RunBundle, Snapshot } from "../../lib/schema";

interface BundleOptions {
  runId?: string;
  snapshotCount?: number;
  seed?: number;
  initialBias?: number;
  initialWeight?: number;
  learningRate?: number;
}

export function makeBundle(options: BundleOptions = {}): RunBundle {
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
): RunBundle {
  const bundle = makeBundle(options);
  bundle.manifest.schemaVersion = 2;
  bundle.manifest.dataConfig = {
    source: "external_dataset",
    datasetId: "auto-mpg",
    datasetVersion: "1.0.0",
    versionManifestSha256: "version-manifest-sha256",
    processedArtifactSha256: "processed-artifact-sha256",
    sourceFeature: "weight",
    feature: "weight_standardized",
    featureUnit: "population standard deviations",
    target: "mpg",
    targetUnit: "miles per gallon",
    preprocessing: "population_standardization",
    split: "all-398-rows",
    ...dataConfigOverrides,
  };
  bundle.manifest.dataset.generatorId = "auto-mpg";
  bundle.events.forEach((event) => {
    event.schemaVersion = 2;
  });
  return bundle;
}
