<script lang="ts">
  export let comparisonActive = false;
</script>

<header class="app-header">
  <div class="brand-mark" aria-hidden="true">∿</div>
  <div class="brand-copy">
    <strong>AI Experiment Observatory</strong>
    <span aria-hidden="true">/</span>
    <span class="product-name">Linear Regression Replay</span>
  </div>
  <div class="artifact-note">
    <span class="status-indicator" aria-hidden="true"></span>
    Static artifact replay
  </div>
  <div class="header-mode" aria-label="Current view">
    {comparisonActive ? "Run A / Run B" : "Single run"}
  </div>
</header>
