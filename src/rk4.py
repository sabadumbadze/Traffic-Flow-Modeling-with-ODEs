"""Classical fixed-step Runge-Kutta method of order 4.

Used as the integrator inside the parameter fit, so that the fitted
parameters describe the ODE itself and are not contaminated by the
first-order error of Euler's methods.
"""

from __future__ import annotations

from typing import Sequence

from .model import Vector3, rhs


def _shift(x: Sequence[float], a: float, k: Sequence[float]) -> Vector3:
    return (x[0] + a * k[0], x[1] + a * k[1], x[2] + a * k[2])


def rk4_step(x: Sequence[float], u: float, h: float, theta: Sequence[float]) -> Vector3:
    k1 = rhs(x, u, theta)
    k2 = rhs(_shift(x, 0.5 * h, k1), u, theta)
    k3 = rhs(_shift(x, 0.5 * h, k2), u, theta)
    k4 = rhs(_shift(x, h, k3), u, theta)
    c = h / 6.0
    return (
        x[0] + c * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        x[1] + c * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        x[2] + c * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    )
