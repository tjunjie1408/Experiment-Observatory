"""Smoke test confirming the project package is importable under the locked environment."""

import observatory


def test_package_importable() -> None:
    assert observatory is not None
