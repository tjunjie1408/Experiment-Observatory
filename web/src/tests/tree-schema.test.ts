import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { validateBundle, isTreeBundle } from "../lib/schema";
import { getComparisonNotice } from "../lib/comparison";

const fixture = fileURLToPath(new URL("./fixtures/tree-depth-2-v4/", import.meta.url));

function rawBundle() {
  const manifest = JSON.parse(readFileSync(`${fixture}/manifest.json`, "utf8"));
  const snapshots = JSON.parse(readFileSync(`${fixture}/snapshots.json`, "utf8"));
  const events = readFileSync(`${fixture}/events.jsonl`, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  return { manifest, snapshots, events };
}

describe("real WDBC schema-v4 export", () => {
  it("loads all recorded nodes and final evaluations", () => {
    const raw = rawBundle();
    const bundle = validateBundle(raw.manifest, raw.events, raw.snapshots);
    expect(isTreeBundle(bundle)).toBe(true);
    if (!isTreeBundle(bundle)) return;
    expect(bundle.manifest.dataset.roster).toHaveLength(569);
    expect(bundle.manifest.dataset.observedRows).toHaveLength(20);
    expect(bundle.snapshots[0]?.nodeId).toBe("r");
    expect(bundle.manifest.validationEvaluation.total).toBe(172);
    expect(getComparisonNotice(bundle, bundle)?.message).toContain("Construction steps are not equivalent");
  });

  it("rejects tampered node membership", () => {
    const raw = rawBundle();
    raw.snapshots[1].sampleIds = raw.snapshots[0].sampleIds;
    expect(() => validateBundle(raw.manifest, raw.events, raw.snapshots)).toThrow();
  });

  it("rejects a changed split identity in comparison", () => {
    const raw = rawBundle();
    const bundle = validateBundle(raw.manifest, raw.events, raw.snapshots);
    if (!isTreeBundle(bundle)) throw new Error("expected tree");
    const changed = structuredClone(bundle);
    changed.manifest.dataConfig.splitSha256 = "a".repeat(64);
    expect(getComparisonNotice(bundle, changed)?.kind).toBe("incompatible");
  });
});
