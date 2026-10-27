"""Piecewise-constant upstream input u(t).

The observation u_k = u(t_k) is held constant on the half-open interval
[t_k, t_{k+1}). Time is in hours; t_k = t0 + k * interval.

Time steps of both Euler methods (and of the reference solver) are chosen so
that they do not straddle an observation time (h = interval / 2^j). Within a
step the input is therefore a single constant, evaluated at the step midpoint
with :meth:`on_step`. This makes explicit and implicit Euler use exactly the
same input value, and makes the discretised problem consistent with the
continuous problem solved by the reference solver.
"""

from __future__ import annotations

import math
from typing import List, Sequence

_EPS = 1e-9


class PiecewiseConstantInput:
    def __init__(self, values: Sequence[float], t0: float, interval: float) -> None:
        if len(values) < 1:
            raise ValueError("The input needs at least one value")
        if interval <= 0.0:
            raise ValueError("interval must be positive")
        self.values = tuple(float(v) for v in values)
        self.t0 = float(t0)
        self.interval = float(interval)

    def index_at(self, t: float) -> int:
        k = int(math.floor((t - self.t0) / self.interval + _EPS))
        return min(max(k, 0), len(self.values) - 1)

    def __call__(self, t: float) -> float:
        return self.values[self.index_at(t)]

    def on_step(self, t_n: float, h: float) -> float:
        """Input value valid on the whole step [t_n, t_n + h]."""
        return self(t_n + 0.5 * h)

    def breakpoints_between(self, t_a: float, t_b: float) -> List[float]:
        """Observation times strictly inside (t_a, t_b), i.e. jump locations of u."""
        lo = int(math.floor((t_a - self.t0) / self.interval + _EPS)) + 1
        hi = int(math.ceil((t_b - self.t0) / self.interval - _EPS)) - 1
        return [
            self.t0 + k * self.interval
            for k in range(max(lo, 1), min(hi, len(self.values) - 1) + 1)
        ]
