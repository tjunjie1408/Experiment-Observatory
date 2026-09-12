# AI Experiment Observatory

A continuously growing, runnable, inspectable, reproducible machine learning learning archive: each new model ships as a complete experiment unit with a real question, real computation, real validation, and a written explanation, replayed in a web UI that shows how the model fits data, forms predictions, and where it fails.

**Current status:** V0 linear regression training, recording, export, and static replay are implemented. The frontend is migrating to Svelte 5 with declarative SVG charts and accessible component tests. A first D1 dataset slice uses the openly licensed UCI Auto MPG dataset for verified one-feature training and backward-compatible schema-v2 replay export. It is not yet included in the shipped public-run catalog. Live browser acceptance of the migrated UI and clean-provenance regeneration of the shipped demo bundles remain pending.

- [PROJECT_IMPLEMENTATION_PLAN_V0.1.md](docs/PROJECT_IMPLEMENTATION_PLAN_V0.1.md) — initial plan and technology choices
- [BEHAVIOR_SPECIFICATION_V0.2.md](docs/BEHAVIOR_SPECIFICATION_V0.2.md) — V0 behavior spec and acceptance conditions
- [PHASE_ACCEPTANCE_PLAN_V0.2.md](docs/PHASE_ACCEPTANCE_PLAN_V0.2.md) — phased tasks and evidence log
- [MODEL_BEHAVIOR_CATALOG_V0.3.md](docs/MODEL_BEHAVIOR_CATALOG_V0.3.md) / [VERSION_ACCEPTANCE_ROADMAP_V0.3.md](docs/VERSION_ACCEPTANCE_ROADMAP_V0.3.md) — model catalog and version roadmap

## Layout

```text
docs/                 project plans, acceptance evidence, model notes
datasets/auto-mpg/    immutable source, processed data, and version manifests
src/observatory/
  data/               synthetic and UCI Auto MPG data preparation
  models/             NumPy linear regression
  runtime/            run lifecycle, recording, schema, and static export
  api/                reserved for the deferred V1 local API
  cli.py              run, export, prepare-dataset, and dataset commands
tests/                numerical, contract, data, and run tests
configs/linear/       synthetic replay and external dataset training configs
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
uv run observatory run-dataset configs/linear/auto_mpg_weight.yaml --runs-root runs
uv run observatory export runs/<auto-mpg-run-id> <new-export-directory>
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src

cd web
npm test
npm run typecheck
npm run build
```

`npm test` currently retains three intentional failing provenance checks until the shipped bundles can be regenerated from a clean Git working tree. The other frontend suites pass.

## DVC data recovery

Auto MPG raw files and the processed training table are DVC-managed. Git stores the dataset/version manifests, `raw.dvc`, `dvc.yaml`, and `dvc.lock`; dataset bytes are pushed to the configured DVC remote.

```bash
uv sync --locked
uv run dvc pull
uv run dvc repro
uv run pytest -q tests/test_external_dataset.py
```

The default remote points to the project Google Drive folder. OAuth credentials are local-only and must never be committed. A local-remote push and empty-cache restore have been verified; Google Drive upload remains pending until a project-owned OAuth client is configured.

## Pre-commit checks

[pre-commit](https://pre-commit.com/) is configured to run ruff check/format and mypy before each commit:

```bash
uv run pre-commit install
```
