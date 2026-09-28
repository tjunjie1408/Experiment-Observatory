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
  /** The shipped manifest's runId; links catalog rows to this bundle. */
  runId: string;
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
    runId: "linear-lr-converge-20260912T110646751677-d8d354d9",
    label: "Converge (lr=0.25)",
    path: "runs/converge",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "slow",
    runId: "linear-lr-slow-20260912T110647335452-0a965ade",
    label: "Slow (lr=0.001)",
    path: "runs/slow",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "diverge",
    runId: "linear-lr-diverge-20260912T110647965164-a3c75e77",
    label: "Diverge (lr=1.5)",
    path: "runs/diverge",
    comparisonGroup: "synthetic-learning-rate",
  },
  {
    id: "auto-mpg",
    runId: "auto-mpg-weight-20260924T154906116967-25e375c6",
    label: "Auto MPG: weight to mpg (lr=0.1)",
    path: "runs/auto-mpg",
    comparisonGroup: "auto-mpg-weight",
  },
  {
    id: "kmeans-seed-0",
    runId: "kmeans-init-study-seed-0-20260914T125732443370-f9739585",
    label: "K-means init seed 0",
    path: "runs/kmeans-seed-0",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-1",
    runId: "kmeans-init-study-seed-1-20260914T125732560908-82ce95f4",
    label: "K-means init seed 1",
    path: "runs/kmeans-seed-1",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-2",
    runId: "kmeans-init-study-seed-2-20260914T125732676185-16c8c0fa",
    label: "K-means init seed 2",
    path: "runs/kmeans-seed-2",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-3",
    runId: "kmeans-init-study-seed-3-20260914T125732770755-832ec7ae",
    label: "K-means init seed 3",
    path: "runs/kmeans-seed-3",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "kmeans-seed-4",
    runId: "kmeans-init-study-seed-4-20260914T125732896281-4c0a8d22",
    label: "K-means init seed 4",
    path: "runs/kmeans-seed-4",
    comparisonGroup: "kmeans-initialization",
  },
  {
    id: "tree-depth-1",
    runId: "wdbc-tree-depth-1-20260924T122846689871-b4851fed",
    label: "WDBC decision tree depth 1",
    path: "runs/tree-depth-1",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-2",
    runId: "wdbc-tree-depth-2-20260924T122847042442-7fb34483",
    label: "WDBC decision tree depth 2",
    path: "runs/tree-depth-2",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-3",
    runId: "wdbc-tree-depth-3-20260924T122847483857-4ae8adcf",
    label: "WDBC decision tree depth 3",
    path: "runs/tree-depth-3",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-4",
    runId: "wdbc-tree-depth-4-20260924T122847969809-8f4d0ce8",
    label: "WDBC decision tree depth 4",
    path: "runs/tree-depth-4",
    comparisonGroup: "tree-depth",
  },
  {
    id: "tree-depth-5",
    runId: "wdbc-tree-depth-5-20260924T122848526068-0b2fdbc7",
    label: "WDBC decision tree depth 5",
    path: "runs/tree-depth-5",
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
