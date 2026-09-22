"""Shared run-artifact storage and provenance helpers."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from observatory.runtime.schema import CodeProvenance

_REPLACE_MAX_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_S = 0.05


class RunIOError(Exception):
    """Raised when run artifact I/O fails; wraps the underlying OSError."""


_PROVENANCE_PATHS: tuple[str, ...] = (
    "src",
    "configs",
    "datasets",
    "pyproject.toml",
    "uv.lock",
    "dvc.yaml",
    "dvc.lock",
)


def new_run_id(experiment_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    token = secrets.token_hex(4)
    return f"{experiment_id}-{stamp}-{token}"


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON through a temporary file and bounded atomic-replace retries."""
    tmp_path = path.with_suffix(path.suffix + f".tmp{secrets.token_hex(4)}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        last_exc: OSError | None = None
        for attempt in range(_REPLACE_MAX_ATTEMPTS):
            try:
                tmp_path.replace(path)
                return
            except OSError as exc:
                last_exc = exc
                if attempt < _REPLACE_MAX_ATTEMPTS - 1:
                    time.sleep(_REPLACE_RETRY_DELAY_S)
        assert last_exc is not None
        raise last_exc
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RunIOError(f"failed to write {path}: {exc}") from exc


def append_jsonl(path: Path, line_payload: object) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line_payload, allow_nan=False))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise RunIOError(f"failed to append to {path}: {exc}") from exc


def get_code_provenance(repo_root: Path) -> CodeProvenance:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *_PROVENANCE_PATHS],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return CodeProvenance(git_commit=commit, git_dirty=bool(dirty_output.strip()))
    except (OSError, subprocess.CalledProcessError):
        return CodeProvenance(unavailable_reason="git metadata unavailable in this environment")


__all__ = [
    "RunIOError",
    "append_jsonl",
    "atomic_write_json",
    "get_code_provenance",
    "new_run_id",
]
