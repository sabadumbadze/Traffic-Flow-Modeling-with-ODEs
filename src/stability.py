"""Stability and stiffness helpers for the model.

The Jacobian of the model is lower bidiagonal, so its eigenvalues are the
diagonal entries  lambda_i = -(alpha_i + 2 * beta_i * x_i)  (all <= 0).

  explicit Euler : amplification |1 + h*lambda|, stable iff h*|lambda| < 2
  implicit Euler : amplification 1 / |1 - h*lambda| < 1 for every h > 0
"""

from __future__ import annotations

from typing import List, Sequence


def decay_rates(theta: Sequence[float], x_scale: float) -> List[float]:
    """|lambda_i| evaluated at the largest expected flow level ``x_scale``."""
    if len(theta) == 6:
        alphas, betas = theta[:3], theta[3:]
        return [a + 2.0 * b * x_scale for a, b in zip(alphas, betas)]
    elif len(theta) == 16:
        a1, a2, a3, b1, b2, b3, d1, d2, d3, g1, g2, k2, k3, c1, c2, c3 = theta
        return [
            a1 + 2.0 * b1 * x_scale + 3.0 * d1 * (x_scale**2) + g1,
            a2 + 2.0 * b2 * x_scale + 3.0 * d2 * (x_scale**2) + g2 + k2,
            a3 + 2.0 * b3 * x_scale + 3.0 * d3 * (x_scale**2) + k3,
        ]
    else:
        raise ValueError(f"theta must have length 6 or 16, got {len(theta)}")


def max_rate(theta: Sequence[float], x_scale: float) -> float:
    return max(decay_rates(theta, x_scale))


def stiffness_number(theta: Sequence[float], h: float, x_scale: float) -> float:
    """h * max|lambda|: dimensionless step size relative to the fastest decay."""
    return h * max_rate(theta, x_scale)


def explicit_euler_step_limit(theta: Sequence[float], x_scale: float) -> float:
    """Largest h (hours) for which explicit Euler is linearly stable."""
    rate = max_rate(theta, x_scale)
    return float("inf") if rate <= 0.0 else 2.0 / rate


def stiffness_ratio(theta: Sequence[float], x_scale: float) -> float:
    """max|lambda| / min|lambda| over the three states (inf if some rate is 0)."""
    rates = decay_rates(theta, x_scale)
    smallest = min(rates)
    return float("inf") if smallest <= 0.0 else max(rates) / smallest


def explicit_amplification(h_lambda: float) -> float:
    """|1 - z| for z = h*|lambda|."""
    return abs(1.0 - h_lambda)


def implicit_amplification(h_lambda: float) -> float:
    """1 / (1 + z) for z = h*|lambda|."""
    return 1.0 / (1.0 + h_lambda)
