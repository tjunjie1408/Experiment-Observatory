from __future__ import annotations

import numpy as np
import pytest

from observatory.datasets.synthetic.linear import (
    GENERATOR_ID,
    SyntheticLinearConfig,
    generate,
)


def make_config(**overrides: float | int) -> SyntheticLinearConfig:
    base: dict[str, float | int] = {
        "n_samples": 50,
        "true_bias": 1.0,
        "true_weight": 2.0,
        "noise_std": 0.0,
        "seed": 42,
    }
    base.update(overrides)
    return SyntheticLinearConfig(**base)  # type: ignore[arg-type]


def test_same_seed_produces_identical_dataset() -> None:
    config = make_config()
    a = generate(config)
    b = generate(config)
    assert a.sample_ids == b.sample_ids
    assert a.generator_id == GENERATOR_ID
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_zero_noise_lies_exactly_on_line() -> None:
    config = make_config(noise_std=0.0)
    dataset = generate(config)
    expected_y = config.true_bias + config.true_weight * dataset.x
    np.testing.assert_allclose(dataset.y, expected_y)


def test_different_seed_produces_different_samples() -> None:
    a = generate(make_config(seed=1))
    b = generate(make_config(seed=2))
    assert not np.array_equal(a.x, b.x)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_samples": 1},
        {"noise_std": -1.0},
        {"noise_std": float("nan")},
        {"true_bias": float("inf")},
        {"n_samples": True},
        {"seed": -1},
    ],
)
def test_rejects_invalid_config(overrides: dict[str, float | int]) -> None:
    config = make_config(**overrides)
    with pytest.raises(ValueError):
        generate(config)
