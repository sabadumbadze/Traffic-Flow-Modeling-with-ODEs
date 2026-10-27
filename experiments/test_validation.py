"""Validation experiment on the held-out test dataset split.

Evaluates explicit Euler, implicit Euler, and RK4 on test segments from tra_Y_te:
1. Continuous tracking over test segments (MAE, RMSE, Max Error, Relative Error).
2. Rolling forecast evaluation across multiple forecast horizons (ValidationSettings).
3. Newton solver convergence statistics for implicit Euler.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import (
    OBS_INTERVAL_H,
    OBS_INTERVAL_MIN,
    RESULTS_DIR,
    ValidationSettings,
)
from src.ivp import build_ivp
from src.pipeline import PreparedData, prepare_data
from src.simulation import simulate


def load_fitted_theta() -> Tuple[float, ...]:
    path = RESULTS_DIR / "fitted_parameters.json"
    if path.is_file():
        with open(path) as f:
            d = json.load(f)
            return (
                d["alpha1"],
                d["alpha2"],
                d["alpha3"],
                d["beta1"],
                d["beta2"],
                d["beta3"],
            )
    return (3.111091, 2.241434, 2.204499, 0.849505, 0.406662, 0.0)


def compute_metrics(
    sim_states: List[Tuple[float, float, float]],
    obs_states: List[Tuple[float, float, float]],
) -> Dict[str, Any]:
    n = len(sim_states)
    assert n == len(obs_states)

    metrics = {}
    total_abs_err = [0.0, 0.0, 0.0]
    total_sq_err = [0.0, 0.0, 0.0]
    max_err = [0.0, 0.0, 0.0]
    total_rel_err = [0.0, 0.0, 0.0]

    for sim, obs in zip(sim_states, obs_states):
        for i in range(3):
            err = abs(sim[i] - obs[i])
            total_abs_err[i] += err
            total_sq_err[i] += err * err
            max_err[i] = max(max_err[i], err)
            denom = max(1e-4, abs(obs[i]))
            total_rel_err[i] += err / denom

    mae = [e / n for e in total_abs_err]
    rmse = [math.sqrt(e / n) for e in total_sq_err]
    mre = [e / n for e in total_rel_err]

    metrics["mae_per_state"] = mae
    metrics["rmse_per_state"] = rmse
    metrics["max_err_per_state"] = max_err
    metrics["mre_per_state"] = mre
    metrics["overall_mae"] = sum(mae) / 3.0
    metrics["overall_rmse"] = math.sqrt(sum(rmse[i] ** 2 for i in range(3)) / 3.0)
    metrics["overall_max_err"] = max(max_err)
    return metrics


def evaluate_test_segments(
    data: PreparedData,
    theta: Tuple[float, ...],
) -> Dict[str, Any]:
    """Simulate all test segments and evaluate full trajectory tracking."""
    results = {}
    methods = ("explicit", "implicit", "rk4")

    # Aggregate over all test segments
    segment_evals = []
    saved_trajectories = {}

    for seg_idx, seg in enumerate(data.test_segments):
        if seg.length < 4:
            continue

        ivp = build_ivp(data.test, theta, seg.start, seg.stop)
        obs_states = [data.test.state_at(k) for k in range(seg.start, seg.stop)]

        seg_result = {"segment_index": seg_idx, "length": seg.length}

        for method in methods:
            substeps = 4 if method == "rk4" else 1
            traj = simulate(ivp, method, substeps=substeps)
            sim_at_obs = traj.at_observations()

            m = compute_metrics(sim_at_obs, obs_states)
            if method == "implicit":
                m["mean_newton_iterations"] = traj.mean_newton_iterations

            seg_result[method] = m

            # Store the largest segment for visualization
            if seg_idx == 1:
                saved_trajectories[method] = [list(s) for s in sim_at_obs]

        if seg_idx == 1:
            saved_trajectories["observed"] = [list(s) for s in obs_states]
            saved_trajectories["upstream"] = list(data.test.u[seg.start : seg.stop])
            saved_trajectories["times_h"] = [k * OBS_INTERVAL_H for k in range(seg.length)]

        segment_evals.append(seg_result)

    # Average metrics across segments
    summary_by_method = {}
    for method in methods:
        avg_mae = sum(s[method]["overall_mae"] for s in segment_evals) / len(segment_evals)
        avg_rmse = sum(s[method]["overall_rmse"] for s in segment_evals) / len(segment_evals)
        avg_max = max(s[method]["overall_max_err"] for s in segment_evals)
        summary_by_method[method] = {
            "mean_mae": avg_mae,
            "mean_rmse": avg_rmse,
            "worst_max_err": avg_max,
        }

    results["summary_by_method"] = summary_by_method
    results["segment_evaluations"] = segment_evals
    results["saved_trajectories"] = saved_trajectories
    return results


def evaluate_forecast_horizons(
    data: PreparedData,
    theta: Tuple[float, ...],
    settings: ValidationSettings = ValidationSettings(),
) -> Dict[str, Any]:
    """Rolling forecast test across different prediction horizons."""
    horizon_results = {}

    for H in settings.horizons_steps:
        horizon_min = H * OBS_INTERVAL_MIN
        errors = {"explicit": [], "implicit": [], "rk4": []}

        for seg in data.test_segments:
            if seg.length <= H + 2:
                continue
            # Sample every 4 steps (1 hour) as forecast origin
            for origin in range(seg.start, seg.stop - H, 4):
                ivp = build_ivp(data.test, theta, origin, origin + H + 1)
                obs_target = data.test.state_at(origin + H)

                for method in ("explicit", "implicit", "rk4"):
                    substeps = 4 if method == "rk4" else 1
                    traj = simulate(ivp, method, substeps=substeps)
                    sim_target = traj.states[-1]
                    err = sum(abs(sim_target[i] - obs_target[i]) for i in range(3)) / 3.0
                    errors[method].append(err)

        horizon_results[f"horizon_{H}_steps_{int(horizon_min)}min"] = {
            "horizon_steps": H,
            "horizon_minutes": horizon_min,
            "horizon_hours": horizon_min / 60.0,
            "mae_explicit": sum(errors["explicit"]) / len(errors["explicit"]),
            "mae_implicit": sum(errors["implicit"]) / len(errors["implicit"]),
            "mae_rk4": sum(errors["rk4"]) / len(errors["rk4"]),
        }

    return horizon_results


def run_test_validation_experiment() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING SIMULATION & FORECAST VALIDATION ON HELD-OUT TEST DATA")
    print("=" * 70)

    data = prepare_data()
    theta = load_fitted_theta()
    print(f"Selected Sensor Chain: upstream={data.chain.upstream}, states={data.chain.states}")
    print(f"Test Series Length: {data.test.n} steps ({data.test.n * OBS_INTERVAL_H:.1f} hours)")
    print(f"Test Contiguous Segments: {len(data.test_segments)} segments")
    print(f"Model parameters: theta = {[round(v, 4) for v in theta]}")

    print("\n1. Simulating contiguous test segments...")
    segment_results = evaluate_test_segments(data, theta)

    print("\nTest Tracking Performance Summary (Averaged over all test segments):")
    print(f"{'Method':12s} | {'Overall MAE':12s} | {'Overall RMSE':12s} | {'Worst Max Error':16s}")
    print("-" * 60)
    for method, summary in segment_results["summary_by_method"].items():
        print(
            f"{method:12s} | {summary['mean_mae']:12.5f} | {summary['mean_rmse']:12.5f} | {summary['worst_max_err']:16.5f}"
        )

    print("\n2. Evaluating rolling multi-step forecast horizons...")
    forecast_results = evaluate_forecast_horizons(data, theta)

    print(f"{'Horizon':20s} | {'Explicit MAE':14s} | {'Implicit MAE':14s} | {'RK4 MAE':12s}")
    print("-" * 65)
    for h_name, h_data in forecast_results.items():
        hrs = h_data["horizon_hours"]
        name = f"{hrs:4.2f}h ({int(h_data['horizon_minutes']):4d}m)"
        print(
            f"{name:20s} | {h_data['mae_explicit']:14.5f} | {h_data['mae_implicit']:14.5f} | {h_data['mae_rk4']:12.5f}"
        )

    full_output = {
        "parameters": list(theta),
        "tracking_evaluations": segment_results["summary_by_method"],
        "forecast_horizons": forecast_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "test_validation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)

    traj_path = RESULTS_DIR / "test_trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(segment_results["saved_trajectories"], f, indent=2)

    print(f"\nSaved test metrics to {out_path}")
    print(f"Saved test trajectories to {traj_path}")
    return full_output


if __name__ == "__main__":
    run_test_validation_experiment()
