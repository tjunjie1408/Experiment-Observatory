/**
 * Contract test for the shipped demo bundles under public/runs/.
 *
 * These files are the frontend's real data (not test fixtures): the app's
 * AVAILABLE_RUNS list in main.ts points directly at them. Previously
 * nothing in the test/build/typecheck pipeline ever parsed or validated
 * them, so a truncated, stale, or cross-referenced-incorrectly bundle
 * would still pass `tsc --noEmit` and `vite build` (which only copies
 * files from public/ without inspecting their contents) and only fail
 * once loaded in a real browser.
 *
 * This test loads and validates every bundle the app actually ships with,
 * the same way the running app does (fs reads instead of fetch(), same
 * validateBundle() call), plus checks the reproducibility-relevant
 * provenance fields that validateBundle() does not itself enforce.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { validateBundle } from "../lib/schema";
import { AVAILABLE_RUNS } from "../lib/availableRuns";

const RUNS_DIR = fileURLToPath(new URL("../../public/runs", import.meta.url));
const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));

// Single source of truth shared with main.ts (web/src/lib/availableRuns.ts),
// so a bundle added to the UI without updating this test (or vice versa)
// is no longer possible.
const SHIPPED_RUN_IDS = AVAILABLE_RUNS.map((run) => run.id);
const SYNTHETIC_COMPARISON_RUN_IDS = AVAILABLE_RUNS.filter(
  (run) => run.comparisonGroup === "synthetic-learning-rate",
).map((run) => run.id);

/**
 * True if `commit` resolves to an actual object in this repository's git
 * history. Uses `git cat-file -e`, which exits 0 iff the object exists
 * and is readable, without printing its contents. This is the check that
 * closes the gap the previous version of this test left open: a
 * plausible-looking but fabricated or dangling `gitCommit` value (e.g.
 * copy-pasted from a different clone, or from a rebased/deleted branch)
 * would pass the old `gitCommit !== null` assertion but fail this one.
 */
function commitExistsInRepo(commit: string): boolean {
  try {
    execFileSync("git", ["cat-file", "-e", `${commit}^{commit}`], {
      cwd: REPO_ROOT,
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

function loadBundleFromDisk(runId: string): {
  manifestRaw: unknown;
  eventsRaw: unknown[];
  snapshotsRaw: unknown[];
} {
  const dir = `${RUNS_DIR}/${runId}`;
  const manifestRaw = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf-8"));
  const eventsRaw = readFileSync(`${dir}/events.jsonl`, "utf-8")
    .split("\n")
    .filter((line: string) => line.trim().length > 0)
    .map((line: string) => JSON.parse(line));
  const snapshotsRaw = JSON.parse(
    readFileSync(`${dir}/snapshots.json`, "utf-8"),
  );
  return { manifestRaw, eventsRaw, snapshotsRaw };
}

describe("shipped demo bundles (public/runs)", () => {
  it.each(SHIPPED_RUN_IDS)(
    "%s passes the same validation the app performs at load time",
    (runId) => {
      const { manifestRaw, eventsRaw, snapshotsRaw } =
        loadBundleFromDisk(runId);
      const bundle = validateBundle(manifestRaw, eventsRaw, snapshotsRaw);

      expect(bundle.manifest.status).toBe("completed");
      expect(bundle.snapshots.length).toBeGreaterThan(0);
    },
  );

  it.each(SHIPPED_RUN_IDS)(
    "%s has non-dirty, resolvable code provenance",
    (runId) => {
      const { manifestRaw } = loadBundleFromDisk(runId);
      const manifest = manifestRaw as {
        codeProvenance: {
          gitCommit: string | null;
          gitDirty: boolean | null;
          unavailableReason: string | null;
        };
      };

      // A bundle's provenance must either point at a real, clean, resolvable
      // commit, or explicitly say why it can't. "Resolvable" is verified
      // here (not deferred to a separate CI step) via `git cat-file -e`
      // against this checkout's object database: a dirty-tree export
      // (gitDirty === true) or a fabricated/dangling commit hash would both
      // fail this test, which is exactly the defect class this test exists
      // to prevent from recurring.
      if (manifest.codeProvenance.unavailableReason === null) {
        const { gitCommit, gitDirty } = manifest.codeProvenance;
        expect(gitCommit).not.toBeNull();
        expect(gitDirty).toBe(false);
        expect(commitExistsInRepo(gitCommit as string)).toBe(true);
      }
    },
  );

  it("the synthetic learning-rate cohort shares data, initialization, and budget", () => {
    const manifests = SYNTHETIC_COMPARISON_RUN_IDS.map((runId) => {
      const { manifestRaw } = loadBundleFromDisk(runId);
      return manifestRaw as {
        dataConfig: Record<string, unknown>;
        trainingConfig: {
          initialBias: number;
          initialWeight: number;
          nUpdates: number;
        };
      };
    });

    const dataConfigsJson = new Set(
      manifests.map((m) => JSON.stringify(m.dataConfig)),
    );
    expect(dataConfigsJson.size).toBe(1);

    const initPairs = new Set(
      manifests.map(
        (m) =>
          `${m.trainingConfig.initialBias},${m.trainingConfig.initialWeight}`,
      ),
    );
    expect(initPairs.size).toBe(1);

    const budgets = new Set(manifests.map((m) => m.trainingConfig.nUpdates));
    expect(budgets.size).toBe(1);
  });

  it("ships Auto MPG as an external schema-v2 run", () => {
    const { manifestRaw, eventsRaw, snapshotsRaw } =
      loadBundleFromDisk("auto-mpg");
    const bundle = validateBundle(manifestRaw, eventsRaw, snapshotsRaw);

    expect(bundle.manifest.schemaVersion).toBe(2);
    expect(bundle.manifest.dataConfig).toMatchObject({
      source: "external_dataset",
      datasetId: "uci-auto-mpg",
      datasetVersion: "1.0.0",
    });
  });
});
