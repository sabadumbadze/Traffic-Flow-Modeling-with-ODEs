"""Project-wide configuration: paths, time axis, parameter bounds, solver settings.

The time unit of the ODE is ONE HOUR. The dataset is sampled every 15 minutes,
so the observation interval is 0.25 h and all rate constants (alpha, beta)
are expressed in 1/hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MAT_FILE = DATA_DIR / "traffic_dataset.mat"
VISUALIZATION_DIR = ROOT_DIR / "visualization"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

# ---------------------------------------------------------------- time axis
OBS_INTERVAL_MIN = 15.0
OBS_INTERVAL_H = OBS_INTERVAL_MIN / 60.0
STEPS_PER_DAY = 96
WEEKDAYS_PER_WEEK = 5  # the dataset only contains Monday..Friday
STEPS_PER_WEEK = STEPS_PER_DAY * WEEKDAYS_PER_WEEK

# ---------------------------------------------------------------- model
N_STATES = 3
PARAM_NAMES = ("alpha1", "alpha2", "alpha3", "beta1", "beta2", "beta3")
PARAM_LOWER = (0.0,) * 6  # alpha_i >= 0, beta_i >= 0
PARAM_UPPER = (200.0,) * 6  # generous upper bound, 1/hour

# High-capacity extended 16-parameter formulation
HIGH_CAPACITY_PARAM_NAMES = (
    "alpha1", "alpha2", "alpha3",
    "beta1", "beta2", "beta3",
    "delta1", "delta2", "delta3",
    "gamma1", "gamma2",
    "kappa2", "kappa3",
    "c1", "c2", "c3",
)
HIGH_CAPACITY_PARAM_LOWER = (
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    -10.0, -10.0, -10.0,
)
HIGH_CAPACITY_PARAM_UPPER = (
    50.0, 50.0, 50.0,
    50.0, 50.0, 50.0,
    50.0, 50.0, 50.0,
    50.0, 50.0,
    50.0, 50.0,
    10.0, 10.0, 10.0,
)

# ---------------------------------------------------------------- RK4 used for fitting
# The fitting integrator is fixed-step RK4 with the number of sub-steps per
# observation interval chosen so that h * (largest decay rate) <= limit.
RK4_STIFFNESS_LIMIT = 0.5
RK4_MIN_SUBSTEPS = 4
RK4_MAX_SUBSTEPS = 512


@dataclass(frozen=True)
class DataSettings:
    """Which sensors to use and how to normalise / split the data."""

    sensor_ids: Optional[Tuple[int, int, int, int]] = None  # (upstream, s1, s2, s3), zero-based
    q_ref_method: str = "mean"  # "mean", "median", "p95" or "max" (training data only)
    respect_weekend_gaps: bool = True  # restart the simulation where weekends were removed


@dataclass(frozen=True)
class FitSettings:
    """Options of the bounded Levenberg-Marquardt parameter estimation."""

    max_iterations: int = 80
    ftol: float = 1e-12
    xtol: float = 1e-10
    gtol: float = 1e-9
    initial_damping: float = 1e-2
    fd_rel_step: float = 1e-6


@dataclass(frozen=True)
class NewtonSettings:
    """Options of the Newton solver used by implicit Euler."""

    step_tol: float = 1e-13
    residual_tol: float = 1e-15
    max_iterations: int = 50


@dataclass(frozen=True)
class ConvergenceSettings:
    """Options of the order-of-accuracy study (steps h = 15 min / 2**level)."""

    n_levels: int = 8
    window_start: int = 24  # first observation index of the window (06:00 on day 1)
    window_steps: int = 96  # one day
    ref_rtol: float = 1e-12
    ref_atol: float = 1e-14


@dataclass(frozen=True)
class ValidationSettings:
    """Forecast horizons (in observation steps) used on the held-out data."""

    horizons_steps: Tuple[int, ...] = (1, 2, 4, 8, 16, 48, 96)
