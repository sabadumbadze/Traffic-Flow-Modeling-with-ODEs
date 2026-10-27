"""Unified interface to simulate an IVP with explicit Euler, implicit Euler or RK4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .config import (
    OBS_INTERVAL_H,
    RK4_MAX_SUBSTEPS,
    RK4_MIN_SUBSTEPS,
    RK4_STIFFNESS_LIMIT,
    NewtonSettings,
)
from .explicit_euler import explicit_euler_step
from .implicit_euler import ImplicitEulerStepper
from .ivp import InitialValueProblem
from .rk4 import rk4_step
from .stability import max_rate
from .time_stepping import Stepper, integrate

METHODS = ("explicit", "implicit", "rk4")


@dataclass
class Trajectory:
    method: str
    substeps: int
    h_hours: float
    times: List[float]
    states: List[Tuple[float, float, float]]
    stepper: Any = None

    def at_observations(self) -> List[Tuple[float, float, float]]:
        """States at the 15-minute observation times (every ``substeps``-th state)."""
        return self.states[:: self.substeps]

    @property
    def mean_newton_iterations(self) -> Optional[float]:
        return getattr(self.stepper, "mean_iterations", None)


def make_stepper(method: str, newton: NewtonSettings = NewtonSettings()) -> Stepper:
    """Create a fresh stepper (a new object each time so statistics start at zero)."""
    if method == "explicit":
        return explicit_euler_step
    if method == "implicit":
        return ImplicitEulerStepper(newton.step_tol, newton.residual_tol, newton.max_iterations)
    if method == "rk4":
        return rk4_step
    raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")


def default_substeps(
    method: str,
    theta,
    x_scale: float,
    interval: float = OBS_INTERVAL_H,
) -> int:
    """Sub-steps per observation interval.

    Euler methods use one step per interval unless told otherwise. RK4 (the
    fitting integrator) picks enough sub-steps to keep h * max|lambda| below
    ``RK4_STIFFNESS_LIMIT`` so that it stays stable and accurate.
    """
    if method != "rk4":
        return 1
    needed = int(math.ceil(interval * max_rate(theta, x_scale) / RK4_STIFFNESS_LIMIT))
    return max(RK4_MIN_SUBSTEPS, min(RK4_MAX_SUBSTEPS, needed))


def simulate(
    problem: InitialValueProblem,
    method: str,
    substeps: int = 1,
    newton: NewtonSettings = NewtonSettings(),
) -> Trajectory:
    """Integrate ``problem`` with the chosen method and h = interval / substeps."""
    stepper = make_stepper(method, newton)
    times, states = integrate(problem, stepper, substeps)
    return Trajectory(method, substeps, problem.signal.interval / substeps, times, states, stepper)
