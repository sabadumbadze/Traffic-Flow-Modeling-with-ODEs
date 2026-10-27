"""Implicit (backward) Euler:  x_{n+1} = x_n + h * f(x_{n+1}, u; theta).

Each step solves the nonlinear system

    G(y) = y - x_n - h * f(y, u) = 0,      dG/dy = I - h * df/dx(y)

with Newton's method, started from the current state x_n. Local truncation
error O(h^2), global error O(h): first order. Unconditionally stable for the
decaying (negative) Jacobian diagonal of this model.

The input u is the constant value valid on the step (see input_signal.py).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .model import Vector3, jacobian, rhs
from .newton import newton_solve


class ImplicitEulerStepper:
    """Callable implicit-Euler step that also records Newton statistics."""

    def __init__(self, step_tol: float = 1e-13, residual_tol: float = 1e-15, max_iterations: int = 50) -> None:
        self.step_tol = step_tol
        self.residual_tol = residual_tol
        self.max_iterations = max_iterations
        self.calls = 0
        self.total_iterations = 0
        self.max_iterations_seen = 0
        self.failures = 0

    @property
    def mean_iterations(self) -> Optional[float]:
        return self.total_iterations / self.calls if self.calls else None

    def __call__(self, x: Sequence[float], u: float, h: float, theta: Sequence[float]) -> Vector3:
        x_n = list(x)

        def residual(y: Sequence[float]) -> List[float]:
            f = rhs(y, u, theta)
            return [y[i] - x_n[i] - h * f[i] for i in range(3)]

        def system_jacobian(y: Sequence[float]) -> List[List[float]]:
            jac = jacobian(y, theta)
            return [[(1.0 if i == j else 0.0) - h * jac[i][j] for j in range(3)] for i in range(3)]

        result = newton_solve(
            residual,
            system_jacobian,
            x_n,
            step_tol=self.step_tol,
            residual_tol=self.residual_tol,
            max_iterations=self.max_iterations,
        )
        self.calls += 1
        self.total_iterations += result.iterations
        self.max_iterations_seen = max(self.max_iterations_seen, result.iterations)
        if not result.converged:
            self.failures += 1
        return (result.solution[0], result.solution[1], result.solution[2])
