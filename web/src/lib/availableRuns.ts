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
  comparisonGroup:
    | "synthetic-learning-rate"
    | "auto-mpg-weight"
    | "kmeans-initialization"
    | "tree-depth";
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
  {
    id: "kmeans-seed-0",
    label: "K-means init seed 0",
    path: "/runs/kmeans-seed-0",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-1",
    label: "K-means init seed 1",
    path: "/runs/kmeans-seed-1",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-2",
    label: "K-means init seed 2",
    path: "/runs/kmeans-seed-2",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-3",
    label: "K-means init seed 3",
    path: "/runs/kmeans-seed-3",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-4",
    label: "K-means init seed 4",
    path: "/runs/kmeans-seed-4",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "tree-depth-1",
    label: "WDBC decision tree depth 1",
    path: "/runs/tree-depth-1",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-2",
    label: "WDBC decision tree depth 2",
    path: "/runs/tree-depth-2",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-3",
    label: "WDBC decision tree depth 3",
    path: "/runs/tree-depth-3",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-4",
    label: "WDBC decision tree depth 4",
    path: "/runs/tree-depth-4",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-5",
    label: "WDBC decision tree depth 5",
    path: "/runs/tree-depth-5",
    comparisonGroup: "tree-depth",
  },
];

export type ComparisonGroup = AvailableRun["comparisonGroup"];

export const COMPARISON_GROUP_LABELS: Record<ComparisonGroup, string> = {
  "synthetic-learning-rate": "Linear regression · synthetic learning rates",
  "auto-mpg-weight": "Linear regression · Auto MPG",
  "kmeans-initialization": "K-means · initialization seeds",
  "tree-depth": "Decision tree · WDBC depth study",
};

export interface RunGroup {
  group: ComparisonGroup;
  label: string;
  runs: AvailableRun[];
}

/** Groups runs by comparison group, keeping first-appearance order. */
export function groupAvailableRuns(runs: AvailableRun[] = AVAILABLE_RUNS): RunGroup[] {
  const groups = new Map<ComparisonGroup, RunGroup>();
  for (const run of runs) {
    let entry = groups.get(run.comparisonGroup);
    if (entry === undefined) {
      entry = {
        group: run.comparisonGroup,
        label: COMPARISON_GROUP_LABELS[run.comparisonGroup],
        runs: [],
      };
      groups.set(run.comparisonGroup, entry);
    }
    entry.runs.push(run);
  }
  return [...groups.values()];
}

/** Study label for a replay path, or null for paths outside the static registry. */
export function studyLabelForPath(path: string, runs: AvailableRun[] = AVAILABLE_RUNS): string | null {
  const run = runs.find((item) => item.path === path);
  return run ? COMPARISON_GROUP_LABELS[run.comparisonGroup] : null;
}
