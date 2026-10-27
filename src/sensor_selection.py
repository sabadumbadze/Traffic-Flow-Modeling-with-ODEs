"""Choosing a chain of four neighbouring sensors: one upstream input + three states.

A chain is a simple path ``a - b - c - d`` in the sensor adjacency graph where
``a`` is the upstream sensor (input u) and ``b, c, d`` are the states x1, x2, x3.

IMPORTANT ASSUMPTION: the adjacency matrix is undirected, so the direction of
travel along the chain is NOT given by the data. It is inferred heuristically
from lagged correlations (an upstream sensor should lead the downstream one by
one sample). Training data only is used. The result can be overridden by
passing ``manual_ids`` (upstream first, then s1, s2, s3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .dataset import SensorMeta, TrafficDataset

CONSTRAINT_LEVELS = ("road+direction", "road", "any")


@dataclass(frozen=True)
class SensorChain:
    upstream: int
    states: Tuple[int, int, int]
    score: float               # mean simultaneous correlation of adjacent sensors
    orientation_margin: float  # forward minus backward lagged correlation (>0: direction supported)
    constraints: str           # which admissibility rule produced the chain

    @property
    def ids(self) -> Tuple[int, int, int, int]:
        return (self.upstream,) + tuple(self.states)  # type: ignore[return-value]


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    s_ab = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    s_aa = sum((x - mean_a) ** 2 for x in a)
    s_bb = sum((y - mean_b) ** 2 for y in b)
    if s_aa <= 0.0 or s_bb <= 0.0:
        return 0.0
    return s_ab / math.sqrt(s_aa * s_bb)


def lagged_correlation(a: Sequence[float], b: Sequence[float], lag: int = 1) -> float:
    """Correlation between a(t) and b(t + lag)."""
    return pearson(a[:-lag], b[lag:])


def _neighbours(adjacency: List[List[int]], node: int) -> List[int]:
    return [j for j, value in enumerate(adjacency[node]) if value and j != node]


def _admissible(path: Sequence[int], meta: List[SensorMeta], level: str) -> bool:
    roads = {meta[s].road for s in path}
    directions = {meta[s].direction for s in path}
    if level == "road+direction":
        return len(roads) == 1 and len(directions) == 1
    if level == "road":
        return len(roads) == 1
    return True


def _chain_scores(flows: List[List[float]], ids: Sequence[int]) -> Tuple[float, float]:
    pairs = list(zip(ids[:-1], ids[1:]))
    simultaneous = sum(pearson(flows[a], flows[b]) for a, b in pairs) / len(pairs)
    forward = sum(lagged_correlation(flows[a], flows[b]) for a, b in pairs)
    backward = sum(lagged_correlation(flows[b], flows[a]) for a, b in pairs)
    return simultaneous, forward - backward


def rank_chains(dataset: TrafficDataset, level: str = "road+direction") -> List[SensorChain]:
    """All admissible 4-sensor paths, oriented by lagged correlation, best first."""
    adjacency = dataset.adjacency
    flows = dataset.flow_train
    seen = set()
    chains: List[SensorChain] = []
    for a in range(dataset.n_sensors):
        for b in _neighbours(adjacency, a):
            for c in _neighbours(adjacency, b):
                if c in (a, b):
                    continue
                for d in _neighbours(adjacency, c):
                    if d in (a, b, c):
                        continue
                    path = (a, b, c, d)
                    key = min(path, path[::-1])
                    if key in seen:
                        continue
                    seen.add(key)
                    if not _admissible(path, dataset.meta, level):
                        continue
                    simultaneous, margin = _chain_scores(flows, path)
                    if margin < 0.0:
                        path = path[::-1]
                        margin = -margin
                    chains.append(
                        SensorChain(path[0], (path[1], path[2], path[3]), simultaneous, margin, level)
                    )
    chains.sort(key=lambda ch: (-ch.score, ch.ids))
    return chains


def evaluate_chain(dataset: TrafficDataset, ids: Sequence[int], constraints: str = "manual") -> SensorChain:
    """Score a user-supplied chain without re-orienting it."""
    ids = tuple(int(i) for i in ids)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("A chain needs four distinct sensor ids: (upstream, s1, s2, s3)")
    for a, b in zip(ids[:-1], ids[1:]):
        if not dataset.adjacency[a][b]:
            raise ValueError(f"Sensors {a} and {b} are not adjacent in the sensor graph")
    simultaneous, margin = _chain_scores(dataset.flow_train, ids)
    return SensorChain(ids[0], (ids[1], ids[2], ids[3]), simultaneous, margin, constraints)


def select_chain(dataset: TrafficDataset, manual_ids: Optional[Sequence[int]] = None) -> SensorChain:
    """Pick the chain used by the project (manual override or best-scoring candidate)."""
    if manual_ids is not None:
        return evaluate_chain(dataset, manual_ids)
    for level in CONSTRAINT_LEVELS:
        ranked = rank_chains(dataset, level)
        if ranked:
            return ranked[0]
    raise ValueError("No chain of four adjacent sensors exists in the adjacency matrix")
