# Dataset catalog

Datasets are reusable inputs, independent of model families. Keep one directory
per dataset, with category/task metadata in `dataset.yaml`; do not duplicate the
same data under separate tree or K-means folders.

| Directory | Category | Task / target | Contents |
| --- | --- | --- | --- |
| `auto-mpg/` | Real tabular, automotive | Regression / mpg | 398 rows; current experiment uses weight |
| `breast-cancer/` | Real tabular, health | Binary classification / diagnosis | 569 rows, 30 numeric features, fixed train/validation split |

Each registered dataset separates immutable downloaded `raw/` files, reproducible
`processed/` outputs, and `versions/` manifests with artifact SHA-256 digests.
DVC stores the large artifact bytes; Git stores metadata, loader code and pipeline
definitions. A new dataset version is required when intentionally changing pinned
source, preparation semantics or split identity.

Synthetic generators belong in `src/observatory/datasets/synthetic/`; their current
run artifacts embed generated data. They are distinct from the real datasets here.
Model algorithms belong in `src/observatory/models/`, and model/dataset composition
belongs in `src/observatory/experiments/`.
