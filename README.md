# AI Experiment Observatory

A continuously growing, runnable, inspectable, reproducible machine learning learning archive: each new model ships as a complete experiment unit with a real question, real computation, real validation, and a written explanation, replayed in a web UI that shows how the model fits data, forms predictions, and where it fails.

**Current status:** V0 linear regression training, recording, export, and static replay are implemented. M4 adds an optional local-only FastAPI service for one asynchronous training worker, persisted SSE events, and explicit cancellation recovery. M5 adds a handwritten two-dimensional Lloyd K-means experiment, schema-v3 replay, and a controlled five-seed initialization study. The static Svelte replay remains independent of the service. A first D1 dataset slice uses the openly licensed UCI Auto MPG dataset for verified one-feature training and backward-compatible schema-v2 replay export.

- [PROJECT_IMPLEMENTATION_PLAN_V0.1.md](docs/PROJECT_IMPLEMENTATION_PLAN_V0.1.md) — initial plan and technology choices
- [BEHAVIOR_SPECIFICATION_V0.2.md](docs/BEHAVIOR_SPECIFICATION_V0.2.md) — V0 behavior spec and acceptance conditions
- [PHASE_ACCEPTANCE_PLAN_V0.2.md](docs/PHASE_ACCEPTANCE_PLAN_V0.2.md) — phased tasks and evidence log
- [MODEL_BEHAVIOR_CATALOG_V0.3.md](docs/MODEL_BEHAVIOR_CATALOG_V0.3.md) / [VERSION_ACCEPTANCE_ROADMAP_V0.3.md](docs/VERSION_ACCEPTANCE_ROADMAP_V0.3.md) — model catalog and version roadmap

## Layout

```text
docs/                 project plans, acceptance evidence, model notes
datasets/auto-mpg/    immutable source, processed data, and version manifests
datasets/breast-cancer/ real WDBC source, canonical features and fixed split
src/observatory/
  datasets/           reusable synthetic and tabular data sources
    synthetic/        linear samples and two-dimensional blobs
    tabular/          verified external dataset loaders such as Auto MPG
  models/             model families, separated from datasets and orchestration
    linear_regression/ gradient-descent implementation
    kmeans/            handwritten Lloyd implementation
  experiments/        model-specific recording and study orchestration
    linear_regression/ linear run lifecycle and recording
    kmeans/            K-means recording and initialization study
  runtime/            cross-model schema, storage, and static export
  api/                local FastAPI service, worker coordinator, and request models
  cli.py              run, export, prepare-dataset, and dataset commands
tests/                numerical, contract, data, and run tests
configs/linear/       synthetic replay and external dataset training configs
configs/kmeans/       controlled K-means initialization study config
web/                   Svelte 5 static replay application and frontend tests
```

## Environment and tooling

- Package management and dependency locking: [uv](https://docs.astral.sh/uv/), versions pinned in `uv.lock`
- Linting and formatting: [ruff](https://docs.astral.sh/ruff/)
- Static type checking: [mypy](https://mypy-lang.org/)
- Testing: [pytest](https://docs.pytest.org/)

```bash
uv sync --locked
```

## Common commands

```bash
uv run observatory run-all configs/linear/converge.yaml configs/linear/slow.yaml configs/linear/diverge.yaml
uv run observatory train-dataset configs/linear/auto_mpg_weight.yaml
uv run --locked observatory prepare-wdbc datasets/breast-cancer/versions/1.0.0.yaml datasets/breast-cancer/processed
uv run observatory run-dataset configs/linear/auto_mpg_weight.yaml --runs-root runs
uv run observatory run-kmeans-study configs/kmeans/initialization_study.yaml --runs-root runs
uv run observatory export runs/<auto-mpg-run-id> <new-export-directory>
uv run observatory-service
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src

cd web
npm test
npm run typecheck
npm run build
```

The service binds only to `127.0.0.1:8000`. `POST /api/runs` accepts an inline
synthetic linear-regression configuration, `GET /api/runs/{runId}` reads its
manifest, `POST /api/runs/{runId}/cancel` requests idempotent cancellation, and
`GET /api/runs/{runId}/events` streams persisted events. Use `after_seq` or
`Last-Event-ID` to resume an SSE subscription.

The shipped synthetic and Auto MPG bundles have clean Git provenance and are covered by the public-run contract test.

## DVC data recovery

The [dataset catalog](datasets/README.md) describes dataset ownership and categories.
The [WDBC pipeline](datasets/breast-cancer/README.md) prepares 569 real observations
into 397 training and 172 validation rows. Its local data pipeline is implemented;
tree training and WDBC remote publication are separate pending tasks.

Auto MPG raw files and the processed training table are DVC-managed. Git stores the dataset/version manifests, `raw.dvc`, `dvc.yaml`, and `dvc.lock`; dataset bytes are pushed to the configured DVC remote.

```bash
uv sync --locked
uv run dvc pull
uv run dvc repro
uv run pytest -q tests/test_external_dataset.py
```

The default remote points to the project Google Drive folder. OAuth credentials are local-only and must never be committed. Google Drive recovery was verified by restoring both dataset outputs and four DVC objects through a fresh empty cache, confirming `dvc status` was up to date, and passing all 39 dataset tests.

## Pre-commit checks

[pre-commit](https://pre-commit.com/) is configured to run ruff check/format and mypy before each commit:

```bash
uv run pre-commit install
```
