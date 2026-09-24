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
          type="button" role="treeitem" aria-selected={selectedNode?.nodeId === node.nodeId} aria-level={node.depth + 1} style={`margin-left: ${node.depth * 1.5}rem`}
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
  .tree-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 1rem; padding: 1rem; }
  .tree-scroll { max-height: 590px; overflow: auto; padding: 0.75rem; }
  button { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; width: calc(100% - var(--indent, 0px)); max-width: 100%; margin-top: .25rem; padding: .55rem .7rem; border: 1px solid #334155; border-radius: .5rem; background: #172334; color: #e2e8f0; text-align: left; cursor: pointer; }
  button.selected { border-color: #38bdf8; }
  button.route { background: #1e3a4b; }
  .node-id { min-width: 3rem; font-weight: 700; color: #7dd3fc; }
  small { color: #94a3b8; }
  .pending { padding: .35rem .7rem; color: #94a3b8; font-style: italic; }
  .tree-inspector { display: grid; gap: 1rem; align-content: start; }
  .tree-inspector section { padding: 1rem; border: 1px solid #334155; border-radius: .75rem; background: #111e2e; color: #e2e8f0; }
  .eyebrow { color: #38bdf8; font-size: .75rem; letter-spacing: .1em; text-transform: uppercase; }
  h3 { margin: .4rem 0; }
  dl div { display: flex; justify-content: space-between; gap: .5rem; padding: .35rem 0; border-bottom: 1px solid #334155; }
  dt { color: #94a3b8; }
  dd { margin: 0; text-align: right; }
  select { display: block; width: 100%; margin: .5rem 0; padding: .4rem; background: #172334; color: #e2e8f0; border: 1px solid #475569; }
  p { overflow-wrap: anywhere; }
  @media (max-width: 780px) { .tree-layout { grid-template-columns: 1fr; } }
</style>
