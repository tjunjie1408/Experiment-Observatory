/**
 * Contract test for the shipped static catalog (public/catalog/catalog.json).
 *
 * Without the local service, this file is the only index the site has. It
 * must describe exactly the bundles shipped under public/runs/, or the catalog
 * would advertise runs the static site cannot open and hide ones it can.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AVAILABLE_RUNS } from "../lib/availableRuns";
import { parseCatalog } from "../lib/catalog";

const PUBLIC_DIR = fileURLToPath(new URL("../../public", import.meta.url));

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(`${PUBLIC_DIR}/${path}`, "utf-8"));
}

const catalog = parseCatalog(readJson("catalog/catalog.json"));

describe("shipped static catalog (public/catalog)", () => {
  it("indexes exactly the registered replay bundles", () => {
    expect(catalog.runs.map((run) => run.run_id).sort()).toEqual(
      AVAILABLE_RUNS.map((run) => run.runId).sort(),
    );
  });

  it.each(AVAILABLE_RUNS.map((run) => [run.id, run] as const))(
    "%s catalog row agrees with its shipped manifest",
    (id, run) => {
      const manifest = readJson(`runs/${id}/manifest.json`) as {
        experimentId: string;
        schemaVersion: number;
        status: string;
      };
      const row = catalog.runs.find((item) => item.run_id === run.runId);
      expect(row).toMatchObject({
        experiment_id: manifest.experimentId,
        schema_version: manifest.schemaVersion,
        status: manifest.status,
      });
    },
  );

  it.each(AVAILABLE_RUNS.map((run) => [run.id, run] as const))(
    "%s catalog artifact hashes match the shipped bytes",
    (id, run) => {
      const artifacts = catalog.artifacts.filter((item) => item.run_id === run.runId);
      expect(artifacts.map((item) => item.relative_path).sort()).toEqual(
        ["events.jsonl", "manifest.json", "snapshots.json"].map((name) => `${run.runId}/${name}`),
      );
      for (const artifact of artifacts) {
        const name = artifact.relative_path.slice(run.runId.length + 1);
        const bytes = readFileSync(`${PUBLIC_DIR}/runs/${id}/${name}`);
        expect(createHash("sha256").update(bytes).digest("hex"), artifact.relative_path).toBe(
          artifact.sha256,
        );
      }
    },
  );
});
