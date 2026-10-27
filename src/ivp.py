"""Formulation of the initial-value problem (IVP).

    dx/dt = f(x, u(t); theta),    t in [t0, t_end]
    x(t0) = x0 = (q1(t0), q2(t0), q3(t0)) / Q_ref        (observed values)

Boundary / input conditions
---------------------------
An ODE needs initial conditions, not spatial boundary conditions. The spatial
boundaries of the road section enter as follows:
  * upstream boundary  : the measured flow at the sensor just before sensor 1
                         is the external input u(t), piecewise constant on
                         every 15-minute observation interval;
  * downstream boundary: the model output at sensor 3 (free outflow, no
                         further condition needed).
Parameter constraints: alpha_i >= 0, beta_i >= 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple, TYPE_CHECKING

from .config import OBS_INTERVAL_H
from .input_signal import PiecewiseConstantInput
from .model import check_parameters

if TYPE_CHECKING:  # pragma: no cover
    from .preprocessing import ChainSeries


@dataclass(frozen=True)
class InitialValueProblem:
    theta: Tuple[float, ...]
    t0: float          # hours
    t_end: float       # hours
    x0: Tuple[float, float, float]
    signal: PiecewiseConstantInput

    @property
    def n_intervals(self) -> int:
        """Number of observation intervals between t0 and t_end."""
        return int(round((self.t_end - self.t0) / self.signal.interval))

    def validate(self) -> None:
        check_parameters(self.theta)
        if len(self.x0) != 3 or not all(math.isfinite(v) for v in self.x0):
            raise ValueError("x0 must contain three finite values")
        if not self.t_end > self.t0:
            raise ValueError("t_end must be larger than t0")


def build_ivp(
    series: "ChainSeries",
    theta: Sequence[float],
    start: int,
    stop: int,
    interval: float = OBS_INTERVAL_H,
) -> InitialValueProblem:
    """IVP over the observation indices start .. stop-1 (``stop - start >= 2``).

    Initial state: the observed normalised flows at index ``start``.
    Input: u_k = observed upstream flow at index k, held on [t_k, t_{k+1}).
    """
    if stop - start < 2:
        raise ValueError("An IVP needs at least two observation points")
    t0 = start * interval
    signal = PiecewiseConstantInput(series.u[start:stop], t0=t0, interval=interval)
    problem = InitialValueProblem(
        theta=tuple(float(v) for v in theta),
        t0=t0,
        t_end=(stop - 1) * interval,
        x0=series.state_at(start),
        signal=signal,
    )
    problem.validate()
    return problem


def describe_formulation(problem: InitialValueProblem) -> str:
    """Human-readable statement of the IVP (for logs and reports)."""
    a1, a2, a3, b1, b2, b3 = problem.theta
    return "\n".join(
        [
            "Initial-value problem",
            "  dx1/dt = alpha1*(u(t) - x1) - beta1*x1^2",
            "  dx2/dt = alpha2*(x1 - x2)   - beta2*x2^2",
            "  dx3/dt = alpha3*(x2 - x3)   - beta3*x3^2",
            f"  alpha = ({a1:.6g}, {a2:.6g}, {a3:.6g}) 1/h,  beta = ({b1:.6g}, {b2:.6g}, {b3:.6g}) 1/h",
            f"  t in [{problem.t0:.4g}, {problem.t_end:.4g}] h  ({problem.n_intervals} observation intervals)",
            "Initial condition (observed, normalised)",
            f"  x(t0) = ({problem.x0[0]:.6g}, {problem.x0[1]:.6g}, {problem.x0[2]:.6g})",
            "Boundary / input condition",
            "  upstream: u(t) = measured upstream flow / Q_ref, piecewise constant on each 15-min interval",
            "  downstream: model output at sensor 3 (no extra condition)",
            "Constraints: alpha_i >= 0, beta_i >= 0",
        ]
    )
