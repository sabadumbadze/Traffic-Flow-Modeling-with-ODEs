"""Generic fixed-step time-stepping driver for any one-step method."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from .ivp import InitialValueProblem

Stepper = Callable[[Sequence[float], float, float, Sequence[float]], Tuple[float, float, float]]


def integrate(
    problem: InitialValueProblem,
    stepper: Stepper,
    substeps: int = 1,
) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    """March from t0 to t_end with h = interval / substeps.

    Returns (times, states) including the initial point. With
    ``substeps = 2**j`` the step never straddles an observation time, so the
    piecewise-constant input is a single constant on every step.
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    h = problem.signal.interval / substeps
    n_steps = problem.n_intervals * substeps
    x = problem.x0
    times = [problem.t0]
    states = [x]
    for n in range(n_steps):
        t_n = problem.t0 + n * h
        u = problem.signal.on_step(t_n, h)
        x = stepper(x, u, h, problem.theta)
        times.append(problem.t0 + (n + 1) * h)
        states.append(x)
    return times, states
