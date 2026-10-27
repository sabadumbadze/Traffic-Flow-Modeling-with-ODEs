"""Extraction and normalisation of the chain data.

State variables:  x_i(t) = q_i(t) / Q_ref   (i = 1, 2, 3)
Input signal:     u(t)   = q_0(t) / Q_ref   (upstream sensor)

Q_ref is estimated from TRAINING data only and then reused unchanged for the
test data, so no information from the test period leaks into the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .dataset import TrafficDataset
from .sensor_selection import SensorChain


@dataclass(frozen=True)
class ChainSeries:
    """Normalised measurements of one period (train or test)."""

    q_ref: float
    u: Tuple[float, ...]                      # upstream input, length n
    x: Tuple[Tuple[float, ...], ...]          # (x1, x2, x3), each of length n

    @property
    def n(self) -> int:
        return len(self.u)

    def state_at(self, k: int) -> Tuple[float, float, float]:
        return (self.x[0][k], self.x[1][k], self.x[2][k])


def chain_flows(dataset: TrafficDataset, chain: SensorChain, period: str) -> List[List[float]]:
    """Raw flow series of (upstream, s1, s2, s3) for ``period`` in {"train", "test"}."""
    if period == "train":
        source = dataset.flow_train
    elif period == "test":
        source = dataset.flow_test
    else:
        raise ValueError("period must be 'train' or 'test'")
    return [list(source[sensor]) for sensor in chain.ids]


def estimate_q_ref(flow_series: Sequence[Sequence[float]], method: str = "mean") -> float:
    """Representative flow level computed from training data."""
    values = sorted(v for series in flow_series for v in series)
    if not values:
        raise ValueError("No data to estimate Q_ref from")
    if method == "mean":
        q_ref = sum(values) / len(values)
    elif method == "median":
        q_ref = values[len(values) // 2]
    elif method == "p95":
        q_ref = values[int(0.95 * (len(values) - 1))]
    elif method == "max":
        q_ref = values[-1]
    else:
        raise ValueError(f"Unknown Q_ref method: {method}")
    if not (q_ref > 0.0 and math.isfinite(q_ref)):
        raise ValueError(f"Q_ref must be positive and finite, got {q_ref}")
    return q_ref


def build_chain_series(flow_series: Sequence[Sequence[float]], q_ref: float) -> ChainSeries:
    """Normalise raw (q0, q1, q2, q3) series by Q_ref."""
    if len(flow_series) != 4:
        raise ValueError("Expected four flow series: upstream + three states")
    scaled = [tuple(v / q_ref for v in series) for series in flow_series]
    return ChainSeries(q_ref=q_ref, u=scaled[0], x=(scaled[1], scaled[2], scaled[3]))


def series_scale(series: ChainSeries) -> float:
    """Largest normalised value in the series (used for stiffness estimates)."""
    return max(max(series.u), max(max(s) for s in series.x))
