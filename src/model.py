"""The three-state nonlinear traffic-flow ODE model.

    dx1/dt = alpha1 * (u(t) - x1) - beta1 * x1^2
    dx2/dt = alpha2 * (x1 - x2)   - beta2 * x2^2
    dx3/dt = alpha3 * (x2 - x3)   - beta3 * x3^2

State x = (x1, x2, x3) is the normalised flow at three consecutive sensors,
u(t) is the normalised flow at the upstream sensor (external input).
Parameter vector: theta = (alpha1, alpha2, alpha3, beta1, beta2, beta3), 1/hour.

This is a phenomenological model, not an established traffic law.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .config import N_STATES

Vector3 = Tuple[float, float, float]


def rhs(x: Sequence[float], u: float, theta: Sequence[float]) -> Vector3:
    """Right-hand side f(x, u; theta). Supports 6-param and 16-param high-capacity formulations."""
    x1, x2, x3 = x
    if len(theta) == 6:
        a1, a2, a3, b1, b2, b3 = theta
        return (
            a1 * (u - x1) - b1 * x1 * x1,
            a2 * (x1 - x2) - b2 * x2 * x2,
            a3 * (x2 - x3) - b3 * x3 * x3,
        )
    elif len(theta) == 16:
        a1, a2, a3, b1, b2, b3, d1, d2, d3, g1, g2, k2, k3, c1, c2, c3 = theta
        return (
            a1 * (u - x1) - b1 * x1 * x1 - d1 * (x1**3) + g1 * (x2 - x1) + c1,
            a2 * (x1 - x2) - b2 * x2 * x2 - d2 * (x2**3) + g2 * (x3 - x2) + k2 * (u - x2) + c2,
            a3 * (x2 - x3) - b3 * x3 * x3 - d3 * (x3**3) + k3 * (u - x3) + c3,
        )
    else:
        raise ValueError(f"theta must have length 6 or 16, got {len(theta)}")


def jacobian(x: Sequence[float], theta: Sequence[float]) -> List[List[float]]:
    """Jacobian df/dx. Supports 6-param (bidiagonal) and 16-param (tridiagonal) models."""
    x1, x2, x3 = x
    if len(theta) == 6:
        a1, a2, a3, b1, b2, b3 = theta
        return [
            [-a1 - 2.0 * b1 * x1, 0.0, 0.0],
            [a2, -a2 - 2.0 * b2 * x2, 0.0],
            [0.0, a3, -a3 - 2.0 * b3 * x3],
        ]
    elif len(theta) == 16:
        a1, a2, a3, b1, b2, b3, d1, d2, d3, g1, g2, k2, k3, c1, c2, c3 = theta
        return [
            [-a1 - 2.0 * b1 * x1 - 3.0 * d1 * x1 * x1 - g1, g1, 0.0],
            [a2, -a2 - 2.0 * b2 * x2 - 3.0 * d2 * x2 * x2 - g2 - k2, g2],
            [0.0, a3, -a3 - 2.0 * b3 * x3 - 3.0 * d3 * x3 * x3 - k3],
        ]
    else:
        raise ValueError(f"theta must have length 6 or 16, got {len(theta)}")


def check_parameters(theta: Sequence[float]) -> None:
    """Validate theta: 6 or 16 finite numbers with non-negative rate parameters."""
    if len(theta) not in (6, 16):
        raise ValueError(f"theta must have 6 or 16 entries, got {len(theta)}")
    if not all(math.isfinite(v) for v in theta):
        raise ValueError("theta contains non-finite values")
    n_rates = 6 if len(theta) == 6 else 13
    if any(v < 0.0 for v in theta[:n_rates]):
        raise ValueError("rate parameters must satisfy alpha_i, beta_i, delta_i, gamma_i, kappa_i >= 0")
