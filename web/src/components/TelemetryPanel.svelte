<script lang="ts">
  import type { RunBundle, Snapshot } from "../lib/schema";

  export let bundle: RunBundle;
  export let snapshot: Snapshot;

  function value(number: number): string {
    return Number.isFinite(number) ? number.toFixed(4) : "unavailable";
  }
</script>

<aside class="telemetry-panel" aria-labelledby="telemetry-title">
  <div class="panel-heading">
    <h3 id="telemetry-title">Telemetry S{String(snapshot.step).padStart(3, "0")}</h3>
    <span class:success-tag={bundle.manifest.status === "completed"} class="status-tag">{bundle.manifest.status}</span>
  </div>
  <div class="metric-grid">
    <div class="metric"><span>Weight (w)</span><strong>{value(snapshot.w)}</strong><small>∂L/∂w {value(snapshot.gradientW)}</small></div>
    <div class="metric"><span>Bias (b)</span><strong>{value(snapshot.b)}</strong><small>∂L/∂b {value(snapshot.gradientB)}</small></div>
    <div class="metric accent-metric"><span>MSE loss</span><strong>{value(snapshot.trainMse)}</strong><small>Recorded</small></div>
    <div class="metric"><span>Learning rate</span><strong>{value(bundle.manifest.trainingConfig.learningRate)}</strong><small>Fixed schedule</small></div>
  </div>
  <dl class="run-facts">
    <div><dt>Run ID</dt><dd>{bundle.manifest.runId}</dd></div>
    <div><dt>Snapshots</dt><dd>{bundle.snapshots.length}</dd></div>
    <div><dt>Stop reason</dt><dd>{bundle.manifest.stopReason ?? "not recorded"}</dd></div>
    <div><dt>Dataset seed</dt><dd>{bundle.manifest.dataConfig.seed}</dd></div>
  </dl>
</aside>
