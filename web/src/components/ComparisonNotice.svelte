<script lang="ts">
  import type { ComparisonNotice } from "../lib/comparison";
  export let notice: ComparisonNotice;
</script>

<section class="comparison-notice" data-kind={notice.kind} role="status">
  <strong>{notice.kind === "compatible" ? "Comparison contract" : "Comparison unavailable"}</strong>
  <span>{notice.message}</span>
</section>
