"""One-call data preparation: load, select sensors, normalise, split into segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import MAT_FILE, DataSettings
from .dataset import TrafficDataset, load_traffic_dataset
from .preprocessing import ChainSeries, build_chain_series, chain_flows, estimate_q_ref
from .segments import Segment, contiguous_segments
from .sensor_selection import SensorChain, select_chain


@dataclass(frozen=True)
class PreparedData:
    settings: DataSettings
    dataset: TrafficDataset
    chain: SensorChain
    q_ref: float
    train: ChainSeries
    test: ChainSeries
    train_segments: List[Segment]
    test_segments: List[Segment]


def prepare_data(settings: DataSettings = DataSettings(), mat_path: Path = MAT_FILE) -> PreparedData:
    """Prepare everything the experiments need.

    Q_ref and the sensor ranking use training data only.
    """
    dataset = load_traffic_dataset(mat_path)
    chain = select_chain(dataset, settings.sensor_ids)

    train_flows = chain_flows(dataset, chain, "train")
    test_flows = chain_flows(dataset, chain, "test")
    q_ref = estimate_q_ref(train_flows, settings.q_ref_method)

    train = build_chain_series(train_flows, q_ref)
    test = build_chain_series(test_flows, q_ref)

    train_segments = contiguous_segments(train.n, 0, settings.respect_weekend_gaps)
    # The test series continues the training series without a gap.
    test_segments = contiguous_segments(test.n, dataset.n_train, settings.respect_weekend_gaps)

    return PreparedData(settings, dataset, chain, q_ref, train, test, train_segments, test_segments)
