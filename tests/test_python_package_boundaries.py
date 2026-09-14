"""Regression checks for the Python dataset/model/experiment package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "observatory"

FORBIDDEN_IMPORTS = {
    "datasets": (
        "observatory.api",
        "observatory.experiments",
        "observatory.models",
        "observatory.runtime",
    ),
    "models": (
        "observatory.api",
        "observatory.datasets",
        "observatory.experiments",
        "observatory.runtime",
    ),
    "runtime": (
        "observatory.api",
        "observatory.datasets",
        "observatory.experiments",
        "observatory.models",
    ),
}


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules


def test_model_dataset_and_runtime_packages_do_not_depend_on_concrete_experiments() -> None:
    violations: list[str] = []
    for package, forbidden_prefixes in FORBIDDEN_IMPORTS.items():
        for path in (PACKAGE_ROOT / package).rglob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden_prefixes):
                    relative = path.relative_to(PACKAGE_ROOT)
                    violations.append(f"{relative} imports {module}")

    assert violations == []


def test_each_current_model_family_has_its_own_algorithm_package() -> None:
    assert (PACKAGE_ROOT / "models" / "linear_regression" / "gradient_descent.py").is_file()
    assert (PACKAGE_ROOT / "models" / "kmeans" / "lloyd.py").is_file()
