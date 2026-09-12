import type { RunBundle } from "./schema";

export type ComparisonNoticeKind = "waiting" | "incompatible" | "compatible";

export interface ComparisonNotice {
  kind: ComparisonNoticeKind;
  message: string;
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

  const dataA = bundleA.manifest.dataConfig;
  const dataB = bundleB.manifest.dataConfig;
  const sameData =
    dataA.seed === dataB.seed &&
    dataA.nSamples === dataB.nSamples &&
    dataA.trueBias === dataB.trueBias &&
    dataA.trueWeight === dataB.trueWeight &&
    dataA.noiseStd === dataB.noiseStd;
  const sameInitialization =
    bundleA.manifest.trainingConfig.initialBias === bundleB.manifest.trainingConfig.initialBias &&
    bundleA.manifest.trainingConfig.initialWeight === bundleB.manifest.trainingConfig.initialWeight;

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
