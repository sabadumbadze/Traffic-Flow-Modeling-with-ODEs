"""Newton's method for small nonlinear systems G(y) = 0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

from .linalg import SingularMatrixError, is_finite_vector, norm_inf, solve_linear


@dataclass(frozen=True)
class NewtonResult:
    solution: List[float]
    iterations: int
    residual_norm: float
    converged: bool


def newton_solve(
    residual: Callable[[Sequence[float]], Sequence[float]],
    jacobian: Callable[[Sequence[float]], Sequence[Sequence[float]]],
    y0: Sequence[float],
    step_tol: float = 1e-13,
    residual_tol: float = 1e-15,
    max_iterations: int = 50,
) -> NewtonResult:
    """Solve G(y) = 0 starting from ``y0``.

    Converged when the residual is below ``residual_tol`` or the Newton step is
    below ``step_tol * (1 + ||y||_inf)``.
    """
    y = list(y0)
    r = list(residual(y))
    for iteration in range(1, max_iterations + 1):
        if not is_finite_vector(r):
            return NewtonResult(y, iteration - 1, math.inf, False)
        r_norm = norm_inf(r)
        if r_norm <= residual_tol:
            return NewtonResult(y, iteration - 1, r_norm, True)
        try:
            delta = solve_linear(jacobian(y), [-v for v in r])
        except SingularMatrixError:
            return NewtonResult(y, iteration - 1, r_norm, False)
        y = [yi + di for yi, di in zip(y, delta)]
        r = list(residual(y))
        if not is_finite_vector(y) or not is_finite_vector(r):
            return NewtonResult(y, iteration, math.inf, False)
        if norm_inf(delta) <= step_tol * (1.0 + norm_inf(y)):
            return NewtonResult(y, iteration, norm_inf(r), True)
    final = norm_inf(r) if is_finite_vector(r) else math.inf
    return NewtonResult(y, max_iterations, final, False)
