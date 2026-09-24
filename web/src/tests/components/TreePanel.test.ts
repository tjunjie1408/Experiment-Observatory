// @vitest-environment jsdom

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import TreePanel from "../../components/TreePanel.svelte";
import { isTreeBundle, validateBundle } from "../../lib/schema";
import type { TreeRunBundle } from "../../lib/treeSchema";

// jsdom replaces import.meta.url with an http URL, so resolve from this file's directory.
const fixture = resolve(__dirname, "../fixtures/tree-depth-2-v4");

function loadBundle(): TreeRunBundle {
  const manifest = JSON.parse(readFileSync(`${fixture}/manifest.json`, "utf8"));
  const snapshots = JSON.parse(readFileSync(`${fixture}/snapshots.json`, "utf8"));
  const events = readFileSync(`${fixture}/events.jsonl`, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  const bundle = validateBundle(manifest, events, snapshots);
  if (!isTreeBundle(bundle)) throw new Error("expected tree bundle");
  return bundle;
}

// Independent of the component: walk the recorded rules to the leaf.
function expectedRoute(bundle: TreeRunBundle, features: number[]): string[] {
  const nodes = new Map(bundle.snapshots.map((node) => [node.nodeId, node]));
  const route: string[] = [];
  let node = nodes.get("r");
  while (node) {
    route.push(node.nodeId);
    if (node.isLeaf) break;
    const value = features[node.splitFeatureIndex!]!;
    node = nodes.get(value <= node.splitThreshold! ? node.leftChildId! : node.rightChildId!);
  }
  return route;
}

afterEach(() => {
  cleanup();
});

function renderPanel(bundle: TreeRunBundle, currentStep: number, selectedSampleId: string | null = null) {
  const onSelectSample = vi.fn();
  render(TreePanel, { bundle, currentStep, selectedSampleId, onSelectSample });
  return onSelectSample;
}

describe("TreePanel", () => {
  it("shows only nodes recorded through the current step", () => {
    const bundle = loadBundle();
    const root = bundle.snapshots[0]!;
    renderPanel(bundle, 0);

    expect(screen.getAllByRole("treeitem")).toHaveLength(1);
    expect(screen.getByText(`${root.leftChildId} · pending`)).toBeTruthy();
    expect(screen.getByText(`${root.rightChildId} · pending`)).toBeTruthy();
    expect(screen.getByText(`Recorded node 1 / ${bundle.snapshots.length}`)).toBeTruthy();
  });

  it("renders the full tree and inspects a selected leaf", async () => {
    const bundle = loadBundle();
    const leaf = bundle.snapshots.find((node) => node.isLeaf)!;
    renderPanel(bundle, bundle.snapshots.length - 1);

    const items = screen.getAllByRole("treeitem");
    expect(items).toHaveLength(bundle.snapshots.length);
    expect(screen.queryByText(/· pending$/)).toBeNull();

    const leafItem = items.find((item) => item.querySelector(".node-id")?.textContent === leaf.nodeId)!;
    await fireEvent.click(leafItem);

    expect(leafItem.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText(`${leaf.nodeId} · depth ${leaf.depth}`)).toBeTruthy();
    expect(screen.getByText(leaf.leafReason!)).toBeTruthy();
    expect(screen.getByText(leaf.classCounts.join(" / "))).toBeTruthy();
  });

  it("routes an observed sample to the independently traced leaf", () => {
    const bundle = loadBundle();
    const row = bundle.manifest.dataset.observedRows.find((item) => item.split === "validation")!;
    const route = expectedRoute(bundle, row.features);
    renderPanel(bundle, bundle.snapshots.length - 1, row.sampleId);

    expect(screen.getByText(route.join(" → "))).toBeTruthy();
    expect(screen.getByText("Route reaches a recorded leaf.")).toBeTruthy();
    for (const item of screen.getAllByRole("treeitem")) {
      const id = item.querySelector(".node-id")?.textContent ?? "";
      expect(item.classList.contains("route")).toBe(route.includes(id));
    }
  });

  it("stops a sample route at an unrecorded child during playback", () => {
    const bundle = loadBundle();
    const row = bundle.manifest.dataset.observedRows[0]!;
    renderPanel(bundle, 0, row.sampleId);

    expect(screen.getByText(/^r → r[LR] \(pending\)$/)).toBeTruthy();
    expect(screen.getByText("Route stops at an unrecorded child.")).toBeTruthy();
  });

  it("reports sample selection without changing the recorded step", async () => {
    const bundle = loadBundle();
    const target = bundle.manifest.dataset.observedRows[1]!;
    const onSelectSample = renderPanel(bundle, 0);

    await fireEvent.change(screen.getByLabelText("Sample"), { target: { value: target.sampleId } });

    expect(onSelectSample).toHaveBeenCalledWith(target.sampleId);
    expect(screen.getAllByRole("treeitem")).toHaveLength(1);
  });
});
