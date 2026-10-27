"""Loading of the UCI Traffic Flow Forecasting dataset (36 sensors, 15-min sampling).

Layout of the feature matrix X (36 sensors x 48 columns) of the first snapshot:
    cols 0-9    : 10 lagged flow values
    cols 10-14  : day of week one-hot (Mon..Fri)
    cols 15-38  : hour of day one-hot
    col  39     : number of lanes (static)
    cols 40-43  : road direction one-hot (static)
    cols 44-47  : road identifier one-hot (static)
Only the static columns are used here (to find sensors on the same road and
direction); the flow histories come from ``tra_Y_tr`` / ``tra_Y_te``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import MAT_FILE
from .mat_reader import load_mat

LANES_COLUMN = 39
DIRECTION_COLUMNS = range(40, 44)
ROAD_COLUMNS = range(44, 48)


@dataclass(frozen=True)
class SensorMeta:
    sensor_id: int
    lanes: int
    direction: int  # 0..3
    road: int       # 0..3


@dataclass(frozen=True)
class TrafficDataset:
    flow_train: List[List[float]]   # [sensor][time], 36 x 1261
    flow_test: List[List[float]]    # [sensor][time], 36 x 840
    adjacency: List[List[int]]      # 36 x 36, symmetric 0/1
    meta: List[SensorMeta]

    @property
    def n_sensors(self) -> int:
        return len(self.flow_train)

    @property
    def n_train(self) -> int:
        return len(self.flow_train[0])

    @property
    def n_test(self) -> int:
        return len(self.flow_test[0])


def _one_hot_index(row: List[float], columns: range) -> int:
    values = [row[c] for c in columns]
    return max(range(len(values)), key=values.__getitem__)


def load_traffic_dataset(path: Path = MAT_FILE) -> TrafficDataset:
    """Read ``traffic_dataset.mat`` using the pure-Python MAT reader."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    arrays = load_mat(path, names=("tra_Y_tr", "tra_Y_te", "tra_adj_mat"))
    missing = [n for n in ("tra_Y_tr", "tra_Y_te", "tra_adj_mat") if n not in arrays]
    if missing:
        raise ValueError(f"Variables missing from {path.name}: {missing}")

    first_snapshot = load_mat(path, names=("tra_X_tr",), max_cells=1)["tra_X_tr"][0]

    flow_train = arrays["tra_Y_tr"]
    flow_test = arrays["tra_Y_te"]
    adjacency = [[1 if value > 0 else 0 for value in row] for row in arrays["tra_adj_mat"]]

    n_sensors = len(flow_train)
    if len(flow_test) != n_sensors or len(adjacency) != n_sensors or len(first_snapshot) != n_sensors:
        raise ValueError("Inconsistent number of sensors across the dataset arrays")

    meta = [
        SensorMeta(
            sensor_id=i,
            lanes=int(round(first_snapshot[i][LANES_COLUMN])),
            direction=_one_hot_index(first_snapshot[i], DIRECTION_COLUMNS),
            road=_one_hot_index(first_snapshot[i], ROAD_COLUMNS),
        )
        for i in range(n_sensors)
    ]
    return TrafficDataset(flow_train, flow_test, adjacency, meta)
