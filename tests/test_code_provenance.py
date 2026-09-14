"""Unit tests for scoped git provenance in `get_code_provenance`.

`git_dirty` must reflect only the working-tree state of paths that affect a
run's computation (the package, configs, dataset pins, dependency/pipeline
locks). Local-only notes, the frontend, tests, and exported bundles must not
mark a run dirty.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from observatory.runtime.storage import get_code_provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _init_clean_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Provenance Test")
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")


def test_clean_scoped_tree_reports_resolvable_non_dirty_commit(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)

    prov = get_code_provenance(tmp_path)

    assert prov.unavailable_reason is None
    assert prov.git_commit is not None
    assert prov.git_dirty is False


def test_out_of_scope_untracked_files_do_not_mark_dirty(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    # Local-only planning notes, the frontend, and exported bundles are all
    # outside the provenance scope and must not dirty a run.
    (tmp_path / "HANDOFF.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    bundle_dir = tmp_path / "web" / "public" / "runs" / "r"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    prov = get_code_provenance(tmp_path)

    assert prov.git_dirty is False


def test_untracked_file_under_scoped_path_marks_dirty(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    (tmp_path / "src" / "new_module.py").write_text("y = 2\n", encoding="utf-8")

    prov = get_code_provenance(tmp_path)

    assert prov.git_dirty is True


def test_modified_scoped_file_marks_dirty(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    (tmp_path / "src" / "module.py").write_text("x = 999\n", encoding="utf-8")

    prov = get_code_provenance(tmp_path)

    assert prov.git_dirty is True
