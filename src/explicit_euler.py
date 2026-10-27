"""Explicit (forward) Euler:  x_{n+1} = x_n + h * f(x_n, u_n; theta).

Everything on the right-hand side is known at step n, so the update is direct.
Local truncation error O(h^2), global error O(h): first order.
Conditionally stable: for the decay rates lambda_i = -(alpha_i + 2 beta_i x_i)
of this model it needs h * |lambda_i| < 2.
"""

from __future__ import annotations

from typing import Sequence

from .model import Vector3, rhs


def explicit_euler_step(x: Sequence[float], u: float, h: float, theta: Sequence[float]) -> Vector3:
    f = rhs(x, u, theta)
    return (x[0] + h * f[0], x[1] + h * f[1], x[2] + h * f[2])
