"""Contiguous time segments of the data.

The dataset contains Monday..Friday only (weekends were removed), so the time
series has a jump at every Friday 23:45 -> Monday 00:00 boundary, i.e. every
STEPS_PER_WEEK = 480 samples counted from the first training sample (Monday
00:00). Treating the series as one continuous signal across these jumps would
violate the ODE's continuity, so simulations are restarted from an observed
state at the start of every contiguous segment.

The test series continues the training series without a gap, so its global
sample index is ``n_train + k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import STEPS_PER_WEEK


@dataclass(frozen=True)
class Segment:
    """Half-open range [start, stop) of local sample indices."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def contiguous_segments(
    n_samples: int,
    global_offset: int = 0,
    respect_gaps: bool = True,
    steps_per_week: int = STEPS_PER_WEEK,
    min_length: int = 2,
) -> List[Segment]:
    """Split ``n_samples`` samples into gap-free segments.

    ``global_offset`` is the global index of local sample 0 (0 for training,
    ``n_train`` for test).
    """
    if not respect_gaps:
        return [Segment(0, n_samples)]
    cuts = [0]
    for k in range(1, n_samples):
        if (global_offset + k) % steps_per_week == 0:
            cuts.append(k)
    cuts.append(n_samples)
    return [Segment(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_length]
