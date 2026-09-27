<script lang="ts">
  import type { TreeRunBundle, TreeSnapshot } from "../lib/treeSchema";

  export let bundle: TreeRunBundle;
  export let currentStep: number;
  export let selectedSampleId: string | null;
  export let onSelectSample: (id: string) => void;

  let selectedNodeId = "r";
  $: visible = bundle.snapshots.slice(0, currentStep + 1);
  $: byId = new Map(visible.map((node) => [node.nodeId, node]));
  $: selectedNode = byId.get(selectedNodeId) ?? visible[visible.length - 1];
  $: observed = bundle.manifest.dataset.observedRows.find((row) => row.sampleId === selectedSampleId);
  $: route = observed ? trace(observed.features, byId) : [];
  $: pending = visible.flatMap((node) => node.isLeaf ? [] : [node.leftChildId!, node.rightChildId!].filter((id) => !byId.has(id)));

  function trace(features: number[], nodes: Map<string, TreeSnapshot>): string[] {
    const path: string[] = [];
    let id = "r";
    while (nodes.has(id)) {
      const node = nodes.get(id)!;
      path.push(id);
      if (node.isLeaf) break;
      id = features[node.splitFeatureIndex!]! <= node.splitThreshold! ? node.leftChildId! : node.rightChildId!;
    }
    if (!nodes.has(id)) path.push(`${id} (pending)`);
    return path;
  }
</script>

<div class="tree-layout">
  <article class="chart-card tree-card">
    <header><h3>Decision-tree construction</h3><span>Recorded node {currentStep + 1} / {bundle.snapshots.length}</span></header>
    <div class="tree-scroll" role="tree" aria-label="Recorded decision tree">
      {#each visible as node (node.nodeId)}
        <button class:selected={selectedNode?.nodeId === node.nodeId} class:route={route.includes(node.nodeId)}
          type="button" role="treeitem" aria-selected={selectedNode?.nodeId === node.nodeId} aria-level={node.depth + 1} style={`--indent: ${node.depth * 1.5}rem`}
          on:click={() => selectedNodeId = node.nodeId}>
          <span class="node-id">{node.nodeId}</span>
          {#if node.isLeaf}
            <span>Leaf → {node.predictedLabel}</span>
          {:else}
            <span>{node.splitFeatureName?.replaceAll("_", " ")} ≤ {node.splitThreshold?.toPrecision(5)}</span>
          {/if}
          <small>n={node.nSamples} · Gini {node.giniParent.toFixed(3)}</small>
        </button>
      {/each}
      {#each pending as id (id)}
        <div class="pending" style={`margin-left: ${(id.length - 1) * 1.5}rem`}>{id} · pending</div>
      {/each}
    </div>
    <footer><span>Train: {bundle.manifest.dataConfig.trainCount} · Validation: {bundle.manifest.dataConfig.validationCount}</span><strong>Max depth {bundle.manifest.trainingConfig.maxDepth}</strong></footer>
  </article>

  <aside class="tree-inspector">
    <section>
      <span class="eyebrow">Selected node</span>
      {#if selectedNode}
        <h3>{selectedNode.nodeId} · depth {selectedNode.depth}</h3>
        <dl>
          <div><dt>Class counts (B/M)</dt><dd>{selectedNode.classCounts.join(" / ")}</dd></div>
          <div><dt>Gini</dt><dd>{selectedNode.giniParent.toFixed(4)}</dd></div>
          {#if selectedNode.isLeaf}
            <div><dt>Leaf reason</dt><dd>{selectedNode.leafReason}</dd></div>
          {:else}
            <div><dt>Threshold</dt><dd>{selectedNode.splitThreshold?.toPrecision(7)}</dd></div>
            <div><dt>Gini decrease</dt><dd>{selectedNode.weightedGiniDecrease?.toFixed(4)}</dd></div>
          {/if}
        </dl>
      {/if}
    </section>
    <section>
      <span class="eyebrow">Observed sample route</span>
      <label for="tree-sample">Sample</label>
      <select id="tree-sample" value={selectedSampleId ?? ""} on:change={(event) => onSelectSample(event.currentTarget.value)}>
        {#each bundle.manifest.dataset.observedRows as row}
          <option value={row.sampleId}>{row.sampleId} · {row.target === 1 ? "M" : "B"} · {row.split}</option>
        {/each}
      </select>
      {#if observed}
        <p>{route.join(" → ")}</p>
        <p>{route.at(-1)?.includes("pending") ? "Route stops at an unrecorded child." : "Route reaches a recorded leaf."}</p>
      {/if}
    </section>
    <section>
      <span class="eyebrow">Final model evaluation</span>
      <p>Train {bundle.manifest.trainEvaluation.correct}/{bundle.manifest.trainEvaluation.total} ({(100 * bundle.manifest.trainEvaluation.accuracy).toFixed(1)}%)</p>
      <p>Validation {bundle.manifest.validationEvaluation.correct}/{bundle.manifest.validationEvaluation.total} ({(100 * bundle.manifest.validationEvaluation.accuracy).toFixed(1)}%)</p>
      <small>These are final-tree metrics, not metrics of the partial replay above.</small>
    </section>
  </aside>
</div>

<style>
  .tree-layout { display: grid; grid-template-columns: minmax(0, 3fr) minmax(260px, 1fr); gap: 16px; padding: 16px; }
  .tree-card { min-height: 0; }
  .tree-scroll { max-height: 590px; margin: 10px 0; overflow: auto; }
  button { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; width: calc(100% - var(--indent, 0rem)); min-height: 0; margin: 4px 0 0 var(--indent, 0rem); padding: 7px 10px; border: 1px solid var(--border); border-radius: 7px; background: var(--surface); color: var(--text); text-align: left; }
  button:hover:not(:disabled) { border-color: #aeb4bf; background: var(--surface-muted); }
  button.route { border-color: #b7c0eb; background: #f1f3ff; }
  button.selected { border-color: var(--accent); background: var(--surface-selected); box-shadow: inset 3px 0 0 var(--accent); }
  .node-id { min-width: 3rem; color: var(--run-a); font: 700 11px/1.2 var(--font-data); }
  small { color: var(--text-muted); font: 500 10px/1.3 var(--font-data); }
  .pending { padding: 5px 10px; color: var(--text-muted); font: italic 500 11px/1.3 var(--font-data); }
  .tree-inspector { display: grid; gap: 12px; align-content: start; min-width: 0; }
  .tree-inspector section { display: grid; gap: 6px; min-width: 0; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }
  .eyebrow, label { color: var(--text-muted); font: 600 10px/1 var(--font-data); letter-spacing: .07em; text-transform: uppercase; }
  h3 { margin: 0; font-size: 13px; }
  dl { margin: 0; font: 500 11px/1.3 var(--font-data); }
  dl div { display: flex; justify-content: space-between; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--grid); }
  dt { color: var(--text-muted); }
  dd { margin: 0; text-align: right; }
  select { display: block; width: 100%; }
  p { margin: 0; font-size: 12px; overflow-wrap: anywhere; }
  @media (max-width: 1080px) { .tree-layout { grid-template-columns: 1fr; } }
  @media (max-width: 760px) { .tree-layout { gap: 12px; padding: 10px; } }
</style>
