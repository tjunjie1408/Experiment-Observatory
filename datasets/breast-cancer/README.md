# UCI Breast Cancer Wisconsin (Diagnostic)

Real tabular dataset for binary classification: 569 observations, 30 numeric
features. Diagnosis is encoded B=0 (benign), M=1 (malignant). Source ID is retained
as identity only, never as a feature. Attribution and CC BY 4.0 license are in
`dataset.yaml`; original feature definitions are preserved in `raw/wdbc.names`.

```text
breast-cancer/
  dataset.yaml          classification, attribution and license
  versions/1.0.0.yaml    source URLs, feature order, split rules and SHA-256 pins
  raw.dvc               local DVC tracking for original downloads
  raw/
    wdbc.data           original UCI observations, unchanged
    wdbc.names          original UCI description, unchanged
  processed/            DVC output of prepare-wdbc in the root dvc.yaml
    wdbc.csv            ID, target, then 30 unscaled features; sorted by ID
    split.csv           ID -> train or validation; sorted by ID
```

## Prepare and verify

From the repository root, with the locked project environment:

```bash
uv run --locked observatory prepare-wdbc datasets/breast-cancer/versions/1.0.0.yaml datasets/breast-cancer/processed
uv run --locked dvc repro prepare-wdbc
uv run --locked pytest -q tests/test_breast_cancer_dataset.py
```

Preparation verifies the source hashes and expected generated hashes before
writing outputs. Each output file is replaced atomically; the pair is not a
multi-file transaction. The loader rejects missing, corrupted or inconsistent
outputs and recomputes canonical preparation even when supplied hashes agree.

The split uses SHA-256 ranks of `uci-wdbc:2026:<sampleId>` independently within
each class, taking floor(70%) as training. It yields **397 training** observations
(249 B, 148 M) and **172 validation** observations (108 B, 64 M). Input row order
does not affect it. Validation is for later evaluation, not fitting. No scaling
or imputation occurs; future K-means standardization must fit training rows only.

Load verified model inputs:

```python
from pathlib import Path
from observatory.datasets.tabular.breast_cancer import load_breast_cancer

data = load_breast_cancer(Path("datasets/breast-cancer/versions/1.0.0.yaml"))
X_train, y_train = data.train.features, data.train.targets
X_val, y_val = data.validation.features, data.validation.targets
```

Returned arrays are read-only. Feature names, ordered sample IDs, artifact/split
digests and 20 deterministic observed sample IDs remain available on the result.
This pipeline does not fit a model.

## Storage and recovery

Original files were downloaded from the URLs in the version manifest on
2026-09-22. Local DVC caching and reproduction are separate from remote backup.
The WDBC raw and processed objects were pushed to the configured Google Drive
remote on 2026-09-23 (`dvc push`: 6 files pushed; `dvc status -c` verifies the
current remote state). On 2026-09-24 a fresh `git clone` with an empty DVC cache
ran `dvc pull -r google-drive datasets/breast-cancer/raw.dvc prepare-wdbc`. That
fetched 6 objects from Google Drive and restored all four files, matching the
pinned SHA-256 values. Since `.gitattributes` enforces LF checkouts, `dvc status`
and `dvc repro` in a fresh clone report every stage up to date (re-verified
2026-09-24 on Windows with `core.autocrlf=true`).
With the local cache available, `dvc checkout datasets/breast-cancer/raw.dvc`
restores raw files, and `dvc repro prepare-wdbc` reconstructs processed outputs.
Alternatively download the exact official raw files to the listed paths and
prepare; any changed upstream bytes are rejected by the pins.

If DVC reports `Failed to authenticate GDrive` with `invalid_grant: Token has
been expired or revoked`, reauthorize its cached Google Drive credentials from
the repository root in PowerShell:

```powershell
uv run --locked dvc remote modify --local google-drive profile observatory-reauth
uv run --locked dvc status -c
```

The status command should open Google's authorization flow if that profile has
no valid cached credentials. Use an account with access to the configured
remote. The `--local` setting stays in `.dvc/config.local`; do not commit
credential files or paste tokens into issues. If the current profile is already
`observatory-reauth` and has also expired, choose a new profile name in the
first command before rerunning status. This is a recovery step, not something
required for every push.
