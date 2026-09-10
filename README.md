# AI Experiment Observatory

A continuously growing, runnable, inspectable, reproducible machine learning learning archive: each new model ships as a complete experiment unit with a real question, real computation, real validation, and a written explanation, replayed in a web UI that shows how the model fits data, forms predictions, and where it fails.

**Current status: M0 (environment and tooling setup).** No models or frontend are implemented yet; this repository currently contains only the project docs and the Python tooling skeleton. Full background, scope, and acceptance criteria live in the docs below, not in this README:

- [PROJECT_IMPLEMENTATION_PLAN_V0.1.md](docs/PROJECT_IMPLEMENTATION_PLAN_V0.1.md) — initial plan and technology choices
- [BEHAVIOR_SPECIFICATION_V0.2.md](docs/BEHAVIOR_SPECIFICATION_V0.2.md) — V0 behavior spec and acceptance conditions
- [PHASE_ACCEPTANCE_PLAN_V0.2.md](docs/PHASE_ACCEPTANCE_PLAN_V0.2.md) — phased tasks and evidence log
- [MODEL_BEHAVIOR_CATALOG_V0.3.md](docs/MODEL_BEHAVIOR_CATALOG_V0.3.md) / [VERSION_ACCEPTANCE_ROADMAP_V0.3.md](docs/VERSION_ACCEPTANCE_ROADMAP_V0.3.md) — model catalog and version roadmap

## Layout

```text
docs/                 project plans, protocols, model notes
src/observatory/
  data/               data validation, splitting, preprocessing (not yet implemented)
  models/             implemented models (not yet implemented)
  runtime/            config, run lifecycle, recording, export (not yet implemented)
  api/                V1 local API (not yet implemented)
  cli.py              CLI entry point (placeholder; subcommands land in M1)
tests/                numerical, contract, data, and run tests
configs/linear/       linear regression experiment configs (not yet populated)
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
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## Pre-commit checks

[pre-commit](https://pre-commit.com/) is configured to run ruff check/format and mypy before each commit:

```bash
uv run pre-commit install
```
