"""High-accuracy reference solution of the IVP.

Adaptive Dormand-Prince 5(4) integrator with tight tolerances, written in
pure Python (stand-in for scipy.integrate.solve_ivp). The piecewise-constant
input has jumps at the observation times, so integration is restarted at
every jump; inside each sub-interval the ODE is autonomous and smooth.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

from .ivp import InitialValueProblem
from .model import rhs

# Dormand-Prince tableau. Row i holds the coefficients a_{i,j} of stage i;
# the last row equals the 5th-order weights (FSAL property).
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
# Weights of (5th order - 4th order) solution: used for the error estimate.
_E = [71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0


def _dp45_step(
    f: Callable[[Sequence[float]], Sequence[float]],
    x: Sequence[float],
    h: float,
) -> Tuple[List[float], List[float]]:
    """One Dormand-Prince step. Returns (fifth-order solution, error estimate)."""
    n = len(x)
    stages: List[Sequence[float]] = []
    y: List[float] = list(x)
    for row in _A:
        y = [x[i] + h * sum(c * k[i] for c, k in zip(row, stages)) for i in range(n)]
        stages.append(f(y))
    error = [h * sum(e * k[i] for e, k in zip(_E, stages)) for i in range(n)]
    return y, error


def integrate_autonomous(
    f: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    t_a: float,
    t_b: float,
    rtol: float,
    atol: float,
    max_steps: int = 1_000_000,
) -> List[float]:
    """Integrate dx/dt = f(x) from t_a to t_b with adaptive step control."""
    x = list(x0)
    t = t_a
    h = t_b - t_a
    steps = 0
    while t_b - t > 1e-15 * max(1.0, abs(t_b)):
        steps += 1
        if steps > max_steps:
            raise RuntimeError("reference solver exceeded the maximum number of steps")
        last = h >= t_b - t
        if last:
            h = t_b - t
        x_new, error = _dp45_step(f, x, h)
        scale = [atol + rtol * max(abs(a), abs(b)) for a, b in zip(x, x_new)]
        err_norm = math.sqrt(sum((e / s) ** 2 for e, s in zip(error, scale)) / len(x))
        if err_norm <= 1.0:
            t = t_b if last else t + h
            x = x_new
        factor = _MAX_FACTOR if err_norm == 0.0 else _SAFETY * err_norm ** (-0.2)
        h *= min(_MAX_FACTOR, max(_MIN_FACTOR, factor))
    return x


def solve_reference(
    problem: InitialValueProblem,
    t_eval: Sequence[float],
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> List[Tuple[float, float, float]]:
    """Reference states at the times ``t_eval`` (``t_eval[0]`` must equal ``t0``)."""
    if abs(t_eval[0] - problem.t0) > 1e-12:
        raise ValueError("t_eval must start at the initial time of the problem")
    theta = problem.theta
    signal = problem.signal
    x = list(problem.x0)
    states = [tuple(x)]
    for t_a, t_b in zip(t_eval[:-1], t_eval[1:]):
        cuts = [t_a] + signal.breakpoints_between(t_a, t_b) + [t_b]
        for a, b in zip(cuts[:-1], cuts[1:]):
            u = signal(0.5 * (a + b))
            x = integrate_autonomous(lambda y, u=u: rhs(y, u, theta), x, a, b, rtol, atol)
        states.append((x[0], x[1], x[2]))
    return states
