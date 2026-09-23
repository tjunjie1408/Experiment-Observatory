import {
  isExternalDataConfig,
  isKMeansBundle,
  isTreeBundle,
  type DataConfig,
  type RunBundle,
} from "./schema";

export type ComparisonNoticeKind = "waiting" | "incompatible" | "compatible";

export interface ComparisonNotice {
  kind: ComparisonNoticeKind;
  message: string;
}

function hasSameDataIdentity(dataA: DataConfig, dataB: DataConfig): boolean {
  const externalA = isExternalDataConfig(dataA);
  const externalB = isExternalDataConfig(dataB);
  if (externalA !== externalB) return false;

  if (externalA && externalB) {
    return (
      dataA.source === dataB.source &&
      dataA.datasetId === dataB.datasetId &&
      dataA.datasetVersion === dataB.datasetVersion &&
      dataA.versionManifestSha256 === dataB.versionManifestSha256 &&
      dataA.processedArtifactSha256 === dataB.processedArtifactSha256 &&
      dataA.sourceFeature === dataB.sourceFeature &&
      dataA.feature === dataB.feature &&
      dataA.target === dataB.target &&
      dataA.preprocessing === dataB.preprocessing &&
      dataA.split === dataB.split
    );
  }

  if (!externalA && !externalB) {
    return (
      dataA.seed === dataB.seed &&
      dataA.nSamples === dataB.nSamples &&
      dataA.trueBias === dataB.trueBias &&
      dataA.trueWeight === dataB.trueWeight &&
      dataA.noiseStd === dataB.noiseStd
    );
  }

  return false;
}

export function getComparisonNotice(
  bundleA: RunBundle | null,
  bundleB: RunBundle | null,
): ComparisonNotice | null {
  if (bundleB === null) return null;
  if (bundleA === null) {
    return {
      kind: "waiting",
      message: "Run A has not finished loading; comparison is unavailable.",
    };
  }

  const treeA = isTreeBundle(bundleA);
  const treeB = isTreeBundle(bundleB);
  if (treeA !== treeB) {
    return { kind: "incompatible", message: "Different model families are shown side by side; their metrics are not aligned." };
  }
  if (isTreeBundle(bundleA) && isTreeBundle(bundleB)) {
    const a = bundleA.manifest.dataConfig;
    const b = bundleB.manifest.dataConfig;
    const sameData = a.datasetId === b.datasetId && a.datasetVersion === b.datasetVersion &&
      a.versionManifestSha256 === b.versionManifestSha256 &&
      a.processedArtifactSha256 === b.processedArtifactSha256 && a.splitSha256 === b.splitSha256 &&
      a.splitStrategy === b.splitStrategy && JSON.stringify(a.featureNames) === JSON.stringify(b.featureNames) &&
      JSON.stringify(a.targetMapping) === JSON.stringify(b.targetMapping) && a.preprocessing === b.preprocessing;
    if (!sameData) return { kind: "incompatible", message: "These CART runs use different data or split identities and are not directly comparable." };
    return { kind: "compatible", message: `Same WDBC data and split. Depths: A=${bundleA.manifest.trainingConfig.maxDepth}, B=${bundleB.manifest.trainingConfig.maxDepth}. Final validation accuracy: A=${bundleA.manifest.validationEvaluation.accuracy.toFixed(4)}, B=${bundleB.manifest.validationEvaluation.accuracy.toFixed(4)}. Construction steps are not equivalent tree states.` };
  }

  const kmeansA = isKMeansBundle(bundleA);
  const kmeansB = isKMeansBundle(bundleB);
  if (kmeansA !== kmeansB) {
    return {
      kind: "incompatible",
      message: "Different model families are shown side by side; their metrics are not aligned.",
    };
  }
  if (isKMeansBundle(bundleA) && isKMeansBundle(bundleB)) {
    const sameData =
      JSON.stringify(bundleA.manifest.dataConfig) ===
      JSON.stringify(bundleB.manifest.dataConfig);
    if (!sameData) {
      return {
        kind: "incompatible",
        message: "These K-means runs use different datasets and are not directly comparable.",
      };
    }
    if (
      bundleA.manifest.trainingConfig.nClusters !==
      bundleB.manifest.trainingConfig.nClusters
    ) {
      return {
        kind: "incompatible",
        message: "These K-means runs use different K values and are not directly comparable.",
      };
    }
    const finalA = bundleA.snapshots.at(-1)?.inertia;
    const finalB = bundleB.snapshots.at(-1)?.inertia;
    return {
      kind: "compatible",
      message: `Same data and K. Initialization seeds: A=${bundleA.manifest.trainingConfig.initSeed}, B=${bundleB.manifest.trainingConfig.initSeed}. Final inertia: A=${finalA?.toFixed(4)}, B=${finalB?.toFixed(4)}. Cluster labels are arbitrary.`,
    };
  }
  if (isKMeansBundle(bundleA) || isKMeansBundle(bundleB) || isTreeBundle(bundleA) || isTreeBundle(bundleB)) {
    return { kind: "incompatible", message: "Model families cannot be aligned." };
  }

  const dataA = bundleA.manifest.dataConfig;
  const dataB = bundleB.manifest.dataConfig;
  const sameData = hasSameDataIdentity(dataA, dataB);
  const sameInitialization =
    bundleA.manifest.trainingConfig.initialBias ===
      bundleB.manifest.trainingConfig.initialBias &&
    bundleA.manifest.trainingConfig.initialWeight ===
      bundleB.manifest.trainingConfig.initialWeight;

  if (!sameData || !sameInitialization) {
    return {
      kind: "incompatible",
      message:
        "These runs use different data or initialization. They are shown side by side only; values are not aligned or interpolated.",
    };
  }

  const lengthA = bundleA.snapshots.length;
  const lengthB = bundleB.snapshots.length;
  const rateA = bundleA.manifest.trainingConfig.learningRate;
  const rateB = bundleB.manifest.trainingConfig.learningRate;
  let message = `Same data and initialization. Learning rates: A=${rateA}, B=${rateB}.`;
  if (lengthA !== lengthB) {
    message += ` Different step budgets (A has ${lengthA} snapshots, B has ${lengthB} snapshots). Steps beyond the shorter run have no recorded data and are not extrapolated.`;
  }

  return { kind: "compatible", message };
}
