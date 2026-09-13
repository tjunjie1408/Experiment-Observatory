/**
 * Canonical list of demo run bundles shipped under public/runs/.
 *
 * This is the single source of truth for which run IDs/paths the app
 * exposes in the UI (main.ts) AND which bundles the contract test
 * (src/tests/public-runs.contract.test.ts) validates. Previously these
 * were two independently maintained lists; adding a new shipped bundle
 * without updating both meant the test could silently stop covering the
 * new run. Keeping this module free of DOM/document access lets it be
 * imported from both browser code and Node-based tests.
 */

export interface AvailableRun {
  id: string;
  label: string;
  path: string;
  comparisonGroup: "synthetic-learning-rate" | "auto-mpg-weight" | "kmeans-initialization";
}

export const AVAILABLE_RUNS: AvailableRun[] = [
  {
    id: "converge",
    label: "Converge (lr=0.25)",
    path: "/runs/converge",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "slow",
    label: "Slow (lr=0.001)",
    path: "/runs/slow",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "diverge",
    label: "Diverge (lr=1.5)",
    path: "/runs/diverge",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "auto-mpg",
    label: "Auto MPG: weight to mpg (lr=0.1)",
    path: "/runs/auto-mpg",
    comparisonGroup: "auto-mpg-weight",
  },
];
