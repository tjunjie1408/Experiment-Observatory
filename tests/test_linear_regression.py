"""Numerical validation against the independent fixture in
BEHAVIOR_SPECIFICATION_V0.2.md section 3.2 (x=[-1,0,1], y=[-1,1,3]).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from observatory.models.linear_regression import fit, gradient, least_squares_reference, mse

X = np.array([-1.0, 0.0, 1.0])
Y = np.array([-1.0, 1.0, 3.0])


def test_initial_state_matches_fixture() -> None:
    states = fit(X, Y, learning_rate=0.1, n_updates=0)
    assert len(states) == 1
    initial = states[0]
    assert initial.step == 0
    assert initial.b == pytest.approx(0.0)
    assert initial.w == pytest.approx(0.0)
    assert initial.mse == pytest.approx(11 / 3)
    assert initial.gradient_b == pytest.approx(-2.0)
    assert initial.gradient_w == pytest.approx(-8 / 3)


def test_one_update_matches_fixture() -> None:
    states = fit(X, Y, learning_rate=0.1, n_updates=1)
    assert len(states) == 2
    assert states[1].b == pytest.approx(0.2)
    assert states[1].w == pytest.approx(4 / 15)


@pytest.mark.parametrize(
    ("learning_rate", "check"),
    [
        (0.25, lambda m: m <= 1e-10),
        (0.001, lambda m: m > 1),
        (1.5, lambda m: m > 11 / 3),
    ],
    ids=["converges", "too_slow", "diverges_but_finite"],
)
def test_convergence_behavior_matches_fixture(
    learning_rate: float, check: Callable[[float], bool]
) -> None:
    states = fit(X, Y, learning_rate=learning_rate, n_updates=80)
    final_mse = states[-1].mse
    assert np.isfinite(final_mse)
    assert check(final_mse)


def test_gradient_matches_central_difference() -> None:
    b, w = 0.3, -0.7
    h = 1e-6
    grad_b, grad_w = gradient(b, w, X, Y)

    numerical_grad_b = (mse(b + h, w, X, Y) - mse(b - h, w, X, Y)) / (2 * h)
    numerical_grad_w = (mse(b, w + h, X, Y) - mse(b, w - h, X, Y)) / (2 * h)

    assert abs(grad_b - numerical_grad_b) <= 1e-7 + 1e-6 * abs(numerical_grad_b)
    assert abs(grad_w - numerical_grad_w) <= 1e-7 + 1e-6 * abs(numerical_grad_w)


def test_least_squares_reference_matches_optimum() -> None:
    b, w = least_squares_reference(X, Y)
    assert b == pytest.approx(1.0, abs=1e-9)
    assert w == pytest.approx(2.0, abs=1e-9)
    assert mse(b, w, X, Y) == pytest.approx(0.0, abs=1e-12)


def test_sklearn_reference_matches_lstsq() -> None:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-9)
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-9)


def test_snapshot_recompute_is_consistent() -> None:
    """A saved snapshot's prediction/residual/MSE must recompute from (b, w)."""
    states = fit(X, Y, learning_rate=0.1, n_updates=5)
    for state in states:
        recomputed_mse = mse(state.b, state.w, X, Y)
        assert recomputed_mse == pytest.approx(state.mse, abs=1e-10)


def test_diverging_update_raises_instead_of_recording_non_finite() -> None:
    with pytest.raises(FloatingPointError):
        fit(X, Y, learning_rate=1e250, n_updates=5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0, "n_updates": 10},
        {"learning_rate": -0.1, "n_updates": 10},
        {"learning_rate": float("nan"), "n_updates": 10},
        {"learning_rate": 0.1, "n_updates": -1},
        {"learning_rate": True, "n_updates": 10},
        {"learning_rate": 0.1, "n_updates": True},
    ],
)
def test_rejects_invalid_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        fit(X, Y, **kwargs)
