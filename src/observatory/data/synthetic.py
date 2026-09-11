"""Deterministic synthetic data generator for the V0 linear regression experiment.

Implements the data contract from BEHAVIOR_SPECIFICATION_V0.2.md section 3.1:
a fixed seed and generator id must reproduce identical sample ids, x, and y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENERATOR_ID = "synthetic_linear_v1"


@dataclass(frozen=True)
class SyntheticLinearConfig:
    n_samples: int
    true_bias: float
    true_weight: float
    noise_std: float
    seed: int


@dataclass(frozen=True)
class SyntheticLinearDataset:
    generator_id: str
    config: SyntheticLinearConfig
    sample_ids: list[str]
    x: np.ndarray
    y: np.ndarray


def _validate(config: SyntheticLinearConfig) -> None:
    if isinstance(config.n_samples, bool) or config.n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")
    if (
        isinstance(config.noise_std, bool)
        or not np.isfinite(config.noise_std)
        or config.noise_std < 0
    ):
        raise ValueError("noise_std must be a finite non-negative number")
    if not np.isfinite(config.true_bias) or not np.isfinite(config.true_weight):
        raise ValueError("true_bias and true_weight must be finite numbers")
    if isinstance(config.seed, bool) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")


def generate(config: SyntheticLinearConfig) -> SyntheticLinearDataset:
    """Generate a reproducible 1D linear regression dataset.

    Given the same config, this returns identical sample_ids, x, and y in the
    same environment.
    """
    _validate(config)

    rng = np.random.default_rng(config.seed)
    x = rng.uniform(-3.0, 3.0, size=config.n_samples)
    if config.noise_std > 0:
        noise = rng.normal(0.0, config.noise_std, size=config.n_samples)
    else:
        noise = np.zeros(config.n_samples)
    y = config.true_bias + config.true_weight * x + noise

    sample_ids = [f"s{i:04d}" for i in range(config.n_samples)]
    return SyntheticLinearDataset(
        generator_id=GENERATOR_ID,
        config=config,
        sample_ids=sample_ids,
        x=x,
        y=y,
    )
