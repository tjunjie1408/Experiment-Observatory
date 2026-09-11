"""NumPy gradient-descent linear regression.

Implements the math contract from BEHAVIOR_SPECIFICATION_V0.2.md section 3.1:

    prediction_i(theta_k) = b_k + w_k * x_i
    residual_i(theta_k)   = prediction_i(theta_k) - y_i
    MSE(theta_k)          = sum(residual_i(theta_k)^2) / n
    gradient_b(theta_k)   = 2 * sum(residual_i(theta_k)) / n
    gradient_w(theta_k)   = 2 * sum(residual_i(theta_k) * x_i) / n
    theta_(k+1)           = theta_k - learning_rate * gradient(theta_k)

`gradient` in a recorded state is always the gradient at that state's own
(b, w), computed for use by the *next* update: step 0 is the initial state
before any update, step k is the state after k updates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LinearRegressionState:
    step: int
    b: float
    w: float
    predictions: FloatArray
    residuals: FloatArray
    mse: float
    gradient_b: float
    gradient_w: float


def predict(b: float, w: float, x: FloatArray) -> FloatArray:
    result: FloatArray = b + w * x
    return result


def residuals(b: float, w: float, x: FloatArray, y: FloatArray) -> FloatArray:
    result: FloatArray = predict(b, w, x) - y
    return result


def mse(b: float, w: float, x: FloatArray, y: FloatArray) -> float:
    r = residuals(b, w, x, y)
    return float(np.sum(r**2) / r.size)


def gradient(b: float, w: float, x: FloatArray, y: FloatArray) -> tuple[float, float]:
    r = residuals(b, w, x, y)
    n = r.size
    grad_b = float(2 * np.sum(r) / n)
    grad_w = float(2 * np.sum(r * x) / n)
    return grad_b, grad_w


def least_squares_reference(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Independent reference solution via np.linalg.lstsq (not matrix inversion)."""
    design = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def iter_fit(
    x: FloatArray,
    y: FloatArray,
    *,
    learning_rate: float,
    n_updates: int,
    b0: float = 0.0,
    w0: float = 0.0,
) -> Iterator[LinearRegressionState]:
    """Generator form of `fit`: yields one state at a time so a caller can
    persist each step and check for cancellation between steps, instead of
    computing the whole trajectory before anything is observable.

    Same validation and non-finite behavior as `fit`; `fit` is now
    implemented in terms of this generator to avoid duplicating the
    numerics.
    """
    if isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number")
    if isinstance(n_updates, bool) or n_updates < 0:
        raise ValueError("n_updates must be a non-negative integer")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of equal shape")
    if x.size < 2:
        raise ValueError("at least 2 samples are required")

    b, w = float(b0), float(w0)
    for step in range(n_updates + 1):
        preds = predict(b, w, x)
        r = preds - y
        current_mse = float(np.sum(r**2) / r.size)
        grad_b, grad_w = gradient(b, w, x, y)

        if not (np.isfinite(current_mse) and np.isfinite(grad_b) and np.isfinite(grad_w)):
            raise FloatingPointError(f"non-finite value encountered at step {step}")

        yield LinearRegressionState(
            step=step,
            b=b,
            w=w,
            predictions=preds,
            residuals=r,
            mse=current_mse,
            gradient_b=grad_b,
            gradient_w=grad_w,
        )

        if step < n_updates:
            b = b - learning_rate * grad_b
            w = w - learning_rate * grad_w


def fit(
    x: FloatArray,
    y: FloatArray,
    *,
    learning_rate: float,
    n_updates: int,
    b0: float = 0.0,
    w0: float = 0.0,
) -> list[LinearRegressionState]:
    """Run gradient descent, returning one state per step (0..n_updates).

    Raises ValueError for invalid configuration and FloatingPointError as
    soon as a non-finite value would be recorded: no illegal snapshot is
    ever produced.
    """
    return list(iter_fit(x, y, learning_rate=learning_rate, n_updates=n_updates, b0=b0, w0=w0))
