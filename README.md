# AI Experiment Observatory

A local, reproducible machine-learning lab. Each model is implemented by hand,
trained on a fixed dataset, and recorded step by step. A static web app then
replays the recording: how the fit, clusters, or tree evolve, what each sample is
predicted as, and where a run diverges or ends worse than another.

**Current status:** four experiment units are implemented and shipped as replay
bundles in the static site:

- gradient-descent linear regression on synthetic data (three learning rates)
- the same linear model on UCI Auto MPG (weight → mpg)
- a handwritten two-dimensional Lloyd K-means on synthetic blobs (five
  initialization seeds)
- a handwritten Gini CART classifier on UCI Breast Cancer Wisconsin
  (Diagnostic) (depths 1–5)

The static site needs no Python. Optional local extras are a FastAPI service
(one training worker, persisted SSE events, cancellation), MLflow tracking, and a
Parquet/DuckDB catalog. The results below come straight from the shipped bundles.
Planning notes and acceptance records are kept outside this repository.

## Shipped studies

Every number below can be recomputed from the bundle in `web/public/runs/<id>/`.
Each is a single fixed run or cohort, not a benchmark.

**Linear regression learning rate** (`converge`, `slow`, `diverge`). Same 50
synthetic samples, same start (b = w = 0) and 80 updates; only the learning rate
differs. The initial MSE is 13.119 for all three. After 80 updates:

| Bundle | Learning rate | Final train MSE |
| --- | --- | --- |
| `converge` | 0.25 | 0.05161, equal to the least-squares optimum |
| `slow` | 0.001 | 5.456, still descending |
| `diverge` | 1.5 | 3.25 × 10^141, finite but diverged |

**Auto MPG** (`auto-mpg`). Weight predicting mpg on all 398 rows. After 200 updates
at learning rate 0.1, train MSE is 18.781, matching the closed-form least-squares
fit. There is no held-out split, so this shows the mechanism and data lineage,
not generalization.

**K-means initialization** (`kmeans-seed-0` … `kmeans-seed-4`). Same 70 points and
k = 3; only the initialization seed differs. All five runs stop with stable
assignments, at two different final partitions:

| Seeds | Final inertia |
| --- | --- |
| 0, 1, 4 | 158.83 |
| 2, 3 | 195.98 |

On this data, the final partition depends on the initialization, and seeds 2
and 3 stop at a worse local optimum.

**Decision-tree depth** (`tree-depth-1` … `tree-depth-5`). WDBC, 397 training and
172 validation rows from one fixed stratified split, no scaling. Every depth
splits the root on `radius_worst` ≤ 16.790.

| Depth | Nodes | Train accuracy | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 3 | 370/397 (0.9320) | 155/172 (0.9012) |
| 2 | 7 | 381/397 (0.9597) | 160/172 (0.9302) |
| 3 | 13 | 389/397 (0.9798) | 162/172 (0.9419) |
| 4 | 17 | 393/397 (0.9899) | 164/172 (0.9535) |
| 5 | 25 | 396/397 (0.9975) | 161/172 (0.9360) |

Train accuracy rises with depth. Validation accuracy peaks at depth 4 and drops
by 3 rows at depth 5. That drop is within the spread caused by exact Gini ties,
so it is not evidence of overfitting. None of these results extend beyond this
one split.

## Layout

```text
datasets/auto-mpg/    immutable source, processed data, and version manifests
datasets/breast-cancer/ real WDBC source, canonical features and fixed split
src/observatory/
  datasets/           reusable synthetic and tabular data sources
    synthetic/        linear samples and two-dimensional blobs
    tabular/          verified external dataset loaders such as Auto MPG
  models/             model families, separated from datasets and orchestration
    linear_regression/ gradient-descent implementation
    kmeans/            handwritten Lloyd implementation
    tree/              handwritten Gini CART classifier
  experiments/        model-specific recording and study orchestration
    linear_regression/ linear run lifecycle and recording
    kmeans/            K-means recording and initialization study
    tree/              tree recording and depth study
  runtime/            cross-model schema, validation, storage, and static export
  analytics/          MLflow tracking, Parquet warehouse, and tracking recovery
  api/                local FastAPI service, worker coordinator, and request models
  cli.py              run, study, export, dataset, tracking, and warehouse commands
tests/                numerical, contract, data, and run tests
configs/linear/       synthetic replay and external dataset training configs
configs/kmeans/       controlled K-means initialization study config
configs/tree/         fixed WDBC depth-study config
web/                  Svelte 5 static replay application and frontend tests
  public/runs/        shipped replay bundles (generated; see AVAILABLE_RUNS)
```

## Environment and tooling

- Package management and dependency locking: [uv](https://docs.astral.sh/uv/), versions pinned in `uv.lock`
- Linting and formatting: [ruff](https://docs.astral.sh/ruff/)
- Static type checking: [mypy](https://mypy-lang.org/)
- Testing: [pytest](https://docs.pytest.org/)
- Frontend: Node.js and npm. No version is pinned; verified with Node 26.3.0 and
  npm 12.0.0; CI uses Node 26. Dependencies are locked in `web/package-lock.json`.

```bash
uv sync --locked
cd web && npm ci
```

`npm ci` may warn that the esbuild install script was blocked. The build does not
need it.

## Common commands

```bash
uv run observatory run-all configs/linear/converge.yaml configs/linear/slow.yaml configs/linear/diverge.yaml
uv run observatory train-dataset configs/linear/auto_mpg_weight.yaml
uv run --locked observatory prepare-wdbc datasets/breast-cancer/versions/1.0.0.yaml datasets/breast-cancer/processed
uv run observatory run-dataset configs/linear/auto_mpg_weight.yaml --runs-root runs
uv run observatory run-kmeans-study configs/kmeans/initialization_study.yaml --runs-root runs
uv run observatory run-tree-study configs/tree/depth_study.yaml --runs-root runs
uv run observatory export runs/<run-id> <new-export-directory>
uv run --locked observatory publish-static-catalog web/public/runs/*/
uv run observatory-service
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src

cd web
npm ci
npm test
npm run typecheck
npm run build
```

`export` refuses to overwrite an existing directory. To ship a bundle, record it
from a commit where `src`, `configs`, `datasets`, `pyproject.toml`, `uv.lock` and
`dvc.lock` are committed. That keeps `gitDirty: false`, which the public-run
contract test requires. Then export it under `web/public/runs/<id>/`, add it
to `web/src/lib/availableRuns.ts` with the manifest's `runId`, and rerun
`publish-static-catalog` so `web/public/catalog/catalog.json` indexes exactly the
shipped bundles. The public-run and public-catalog contract tests fail if the
registry, bundles, or catalog disagree, including catalog hashes that don't match
the shipped bytes.

### Static site

`npm run build` writes a self-contained site to `web/dist/`. Asset and data URLs
are relative, so the directory can be served from any path, such as
`https://<host>/<repo>/`. No Python service is needed. Offline, the catalog shows
"Local API offline" and opens only shipped bundles.

The service binds only to `127.0.0.1:8000`. `POST /api/runs` accepts an inline
synthetic linear-regression configuration, `GET /api/runs/{runId}` reads its
manifest, `POST /api/runs/{runId}/cancel` requests idempotent cancellation, and
`GET /api/runs/{runId}/events` streams persisted events. Use `after_seq` or
`Last-Event-ID` to resume an SSE subscription.
With the service running and `npm run dev` in `web/`, the catalog offers
**Open local recorded replay** for validated completed runs in `runs/`. This
needs a built local catalog, otherwise `/api/catalog` returns 503 "catalog has
not been built". Build it from the run directories you want indexed:

```bash
uv run --locked observatory build-warehouse runs/<run-id> [runs/<run-id> ...]
```

This writes a Parquet batch under `warehouse/` and the DuckDB catalog at
`database/observatory.duckdb`. Both are local and gitignored.
The service reads only `manifest.json`, `events.jsonl`, and `snapshots.json`
through `/api/replay/{run_id}/`; the static deployment continues to work
without Python and does not expose unshipped local runs.

All shipped bundles (synthetic linear, Auto MPG, K-means seeds, and tree depths) have clean Git provenance and are covered by the public-run contract test.

## MLflow tracking and portable local recovery

The recorder owns the exact model timeline (`manifest.json`, `events.jsonl`,
`snapshots.json`). The shared post-recording hook projects *validated* terminal
linear-regression, K-means, and tree runs to MLflow/SQLite. It logs recorded
step metrics and a downloadable `observatory-replay/` artifact bundle; MLflow
curves are a summary, while the web replay shows per-step predictions, centers,
and tree decisions. Tracking is opt-in so an MLflow outage cannot erase or
prevent the underlying run. A tracking failure leaves the run saved and returns
a nonzero CLI status. Future models should implement a validated analysis
projection in `analytics/records.py` and reuse `analytics/tracking.py:sync_run`,
not write directly to MLflow from the numerical training loop.

```bash
uv run --locked observatory run configs/linear/converge.yaml --tracking-database database/mlflow.db
uv run --locked observatory run-kmeans-study configs/kmeans/initialization_study.yaml --tracking-database database/mlflow.db
uv run --locked observatory run-tree-study configs/tree/depth_study.yaml --tracking-database database/mlflow.db
uv run --locked observatory sync-tracking runs/<run-id> --database database/mlflow.db
uv run --locked observatory backup-tracking database/backups/<new-name> --database database/mlflow.db
uv run --locked observatory restore-tracking-backup database/backups/<new-name> database/restored/<new-name>
```

Restoring to another path verifies the backup, recreates Observatory runs and
their artifacts through MLflow's public API, and writes `restore.json` with
old-to-new MLflow run IDs. The Observatory run ID remains stable; resync a
copied run against the restored database to refresh its `tracking.json` link.
This local relocation is not a Google Drive/DVC pull or evidence of remote
disaster recovery. Keep a verified backup plus selected runs, dataset versions,
and Parquet batches in the remote recovery set before making that claim.

## DVC data recovery

The [dataset catalog](datasets/README.md) describes dataset ownership and categories.
The [WDBC pipeline](datasets/breast-cancer/README.md) prepares 569 real observations
into 397 training and 172 validation rows. The decision-tree study consumes it.

Auto MPG and WDBC raw files and processed outputs are DVC-managed. Git stores the dataset/version manifests, `raw.dvc`, `dvc.yaml`, and `dvc.lock`; dataset bytes are pushed to the configured DVC remote.

The default remote is the project Google Drive folder, and `dvc pull` needs an
account with access to it. A fresh clone has no credentials. Configure them once
per machine in `.dvc/config.local`, which Git ignores, using the OAuth client
(desktop app) that the project owner created in Google Cloud:

```bash
uv run --locked dvc remote modify --local google-drive gdrive_client_id <client-id>
uv run --locked dvc remote modify --local google-drive gdrive_client_secret <client-secret>
uv run --locked dvc remote modify --local google-drive profile <profile-name>
```

The first `dvc pull` or `dvc status -c` opens Google's sign-in flow and caches the
token under that profile. Never commit `.dvc/config.local` or token files. Without
these settings, DVC falls back to its shared default OAuth client. That fallback
is not the tested path.

```bash
uv run dvc pull
uv run dvc repro
uv run pytest -q tests/test_external_dataset.py tests/test_breast_cancer_dataset.py
```

In a fresh clone, `dvc repro` should skip every stage. The repository enforces LF
line endings (`.gitattributes`) because DVC hashes stage dependencies byte by
byte, so a CRLF checkout would look like a code change. Google Drive recovery was
verified on 2026-09-24. A fresh clone with an empty cache restored both datasets
(10 objects), `dvc repro` skipped every stage, and the full test suite passed.

## Pre-commit checks

[pre-commit](https://pre-commit.com/) is configured to run ruff check/format and mypy before each commit:

```bash
uv run pre-commit install
```

## Limitations

- Development and every manual check ran on Windows 11. CI runs the web tests,
  typecheck, build and Python static checks on Linux, but not pytest, because
  the dataset tests need DVC data that CI cannot pull.
- Restoring the datasets from Google Drive needs access granted by the project
  owner. The raw files can also be downloaded from the UCI URLs in each version
  manifest; the pinned hashes reject changed bytes.
- Each study uses one fixed dataset version, split and configuration. There is
  no hyperparameter search, cross-validation or held-out test set, and the
  project makes no accuracy or speed claims beyond the tables above.
- No resource limits are enforced. Every step is recorded without sampling, so
  very large `n_samples` or `n_updates` produce large bundles and long runs.
- Ctrl+C cancellation of a CLI run is not covered by an automated test (only
  service cancellation and tree checkpoint cancellation are).
