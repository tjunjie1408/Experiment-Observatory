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
import { describe, expect, it } from "vitest";
import { validateBundle } from "../lib/schema";

const RUNS_DIR = fileURLToPath(new URL("../../public/runs", import.meta.url));

// Mirrors AVAILABLE_RUNS in main.ts. Kept as a separate literal (not an
// import from main.ts) because main.ts has module-level side effects that
// assume a DOM/document is present.
const SHIPPED_RUN_IDS = ["converge", "slow", "diverge"];

function loadBundleFromDisk(runId: string): {
  manifestRaw: unknown;
  eventsRaw: unknown[];
  snapshotsRaw: unknown[];
} {
  const dir = `${RUNS_DIR}/${runId}`;
  const manifestRaw = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf-8"));
  const eventsRaw = readFileSync(`${dir}/events.jsonl`, "utf-8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
  const snapshotsRaw = JSON.parse(readFileSync(`${dir}/snapshots.json`, "utf-8"));
  return { manifestRaw, eventsRaw, snapshotsRaw };
}

describe("shipped demo bundles (public/runs)", () => {
  it.each(SHIPPED_RUN_IDS)("%s passes the same validation the app performs at load time", (runId) => {
    const { manifestRaw, eventsRaw, snapshotsRaw } = loadBundleFromDisk(runId);
    const bundle = validateBundle(manifestRaw, eventsRaw, snapshotsRaw);

    expect(bundle.manifest.status).toBe("completed");
    expect(bundle.snapshots.length).toBeGreaterThan(0);
  });

  it.each(SHIPPED_RUN_IDS)("%s has non-dirty, resolvable code provenance", (runId) => {
    const { manifestRaw } = loadBundleFromDisk(runId);
    const manifest = manifestRaw as {
      codeProvenance: { gitCommit: string | null; gitDirty: boolean | null; unavailableReason: string | null };
    };

    // A bundle's provenance must either point at a real, clean commit, or
    // explicitly say why it can't (this test does not check that the
    // commit actually exists in the repo -- that requires git and is
    // exercised by the regeneration procedure in HANDOFF.md, not by this
    // fast unit test). What it does catch: a dirty-tree export (gitDirty
    // === true) silently shipped as if it were reproducible, which is
    // exactly the defect this test was added to prevent from recurring.
    if (manifest.codeProvenance.unavailableReason === null) {
      expect(manifest.codeProvenance.gitCommit).not.toBeNull();
      expect(manifest.codeProvenance.gitDirty).toBe(false);
    }
  });

  it("all shipped runs share the same data configuration and initialization", () => {
    const manifests = SHIPPED_RUN_IDS.map((runId) => {
      const { manifestRaw } = loadBundleFromDisk(runId);
      return manifestRaw as {
        dataConfig: Record<string, unknown>;
        trainingConfig: { initialBias: number; initialWeight: number; nUpdates: number };
      };
    });

    const dataConfigsJson = new Set(manifests.map((m) => JSON.stringify(m.dataConfig)));
    expect(dataConfigsJson.size).toBe(1);

    const initPairs = new Set(
      manifests.map((m) => `${m.trainingConfig.initialBias},${m.trainingConfig.initialWeight}`),
    );
    expect(initPairs.size).toBe(1);

    const budgets = new Set(manifests.map((m) => m.trainingConfig.nUpdates));
    expect(budgets.size).toBe(1);
  });
});
