"""Generates visual charts (SVG) from experiment results.

Produces:
1. visualization/test_trajectory_comparison.svg
2. visualization/convergence_order.svg
3. visualization/forecast_horizon_error.svg
4. visualization/stability_amplification.svg
"""

from __future__ import annotations

import json
from pathlib import Path
from src.config import RESULTS_DIR, VISUALIZATION_DIR
from visualization.svg_plotter import SVGPlot


def plot_trajectories() -> None:
    traj_file = RESULTS_DIR / "test_trajectories.json"
    if not traj_file.is_file():
        print(f"Skipping trajectory plot (file not found: {traj_file})")
        return

    with open(traj_file) as f:
        data = json.load(f)

    times = data["times_h"]
    # Plot state 1 (sensor 34)
    obs_s1 = [s[0] for s in data["observed"]]
    exp_s1 = [s[0] for s in data["explicit"]]
    imp_s1 = [s[0] for s in data["implicit"]]
    rk4_s1 = [s[0] for s in data["rk4"]]
    upstream = data["upstream"]

    plot = SVGPlot(
        title="Test Set Trajectory Tracking: Sensor State x1 (Normalized Flow)",
        xlabel="Time (hours from start of test segment)",
        ylabel="Normalized Traffic Volume",
        width=950,
        height=520,
    )
    plot.add_series("Observed Ground Truth (Sensor 34)", times, obs_s1, color="#0f172a", stroke_width=2.5)
    plot.add_series("Upstream Input u(t) (Sensor 31)", times, upstream, color="#94a3b8", stroke_width=1.5, dashed=True)
    plot.add_series("RK4 Integration", times, rk4_s1, color="#10b981", stroke_width=2.0)
    plot.add_series("Explicit Euler (h = 15m)", times, exp_s1, color="#ef4444", stroke_width=1.8, dashed=True)
    plot.add_series("Implicit Euler (h = 15m)", times, imp_s1, color="#3b82f6", stroke_width=1.8, dashed=True)

    out_path = VISUALIZATION_DIR / "test_trajectory_comparison.svg"
    plot.save(str(out_path))
    print(f"Generated trajectory plot: {out_path}")


def plot_convergence() -> None:
    conv_file = RESULTS_DIR / "convergence_results.json"
    if not conv_file.is_file():
        print(f"Skipping convergence plot (file not found: {conv_file})")
        return

    with open(conv_file) as f:
        data = json.load(f)

    h_vals = data["step_sizes_h"]
    exp_err = data["explicit_errors"]
    imp_err = data["implicit_errors"]

    # Reference O(h^1) slope line
    ref_line = [exp_err[0] * (h / h_vals[0]) for h in h_vals]

    plot = SVGPlot(
        title="Order-of-Accuracy Convergence: Global Error vs Step Size h (Log-Log)",
        xlabel="Step Size h (hours)",
        ylabel="Global L2 Error vs DP5(4) Reference",
        width=900,
        height=520,
        log_x=True,
        log_y=True,
    )
    plot.add_series("Explicit Euler (Slope p ≈ 1.00)", h_vals, exp_err, color="#ef4444", stroke_width=2.2, marker="circle")
    plot.add_series("Implicit Euler (Slope p ≈ 1.00)", h_vals, imp_err, color="#3b82f6", stroke_width=2.2, marker="square")
    plot.add_series("Theoretical O(h^1) Reference", h_vals, ref_line, color="#64748b", stroke_width=1.5, dashed=True)

    out_path = VISUALIZATION_DIR / "convergence_order.svg"
    plot.save(str(out_path))
    print(f"Generated convergence plot: {out_path}")


def plot_forecast_horizon() -> None:
    val_file = RESULTS_DIR / "test_validation_metrics.json"
    if not val_file.is_file():
        print(f"Skipping forecast horizon plot (file not found: {val_file})")
        return

    with open(val_file) as f:
        data = json.load(f)

    horizons_data = data["forecast_horizons"]
    hours = [h["horizon_hours"] for h in horizons_data.values()]
    exp_mae = [h["mae_explicit"] for h in horizons_data.values()]
    imp_mae = [h["mae_implicit"] for h in horizons_data.values()]
    rk4_mae = [h["mae_rk4"] for h in horizons_data.values()]

    plot = SVGPlot(
        title="Forecasting Error across Multi-Step Horizons on Test Data",
        xlabel="Forecast Horizon (hours)",
        ylabel="Mean Absolute Error (MAE)",
        width=900,
        height=500,
    )
    plot.add_series("Implicit Euler", hours, imp_mae, color="#3b82f6", stroke_width=2.2, marker="circle")
    plot.add_series("RK4 Integrator", hours, rk4_mae, color="#10b981", stroke_width=2.2, marker="square")
    plot.add_series("Explicit Euler", hours, exp_mae, color="#ef4444", stroke_width=2.0, dashed=True)

    out_path = VISUALIZATION_DIR / "forecast_horizon_error.svg"
    plot.save(str(out_path))
    print(f"Generated forecast horizon plot: {out_path}")


def plot_stability() -> None:
    stab_file = RESULTS_DIR / "stability_results.json"
    if not stab_file.is_file():
        print(f"Skipping stability plot (file not found: {stab_file})")
        return

    with open(stab_file) as f:
        data = json.load(f)

    z_vals = data["amplification_curves"]["z_values"]
    exp_amp = data["amplification_curves"]["explicit"]
    imp_amp = data["amplification_curves"]["implicit"]
    unit_line = [1.0] * len(z_vals)

    plot = SVGPlot(
        title="Stability Function: Amplification Factor vs Dimensionless Step z = h·|λ|",
        xlabel="z = h · |λ| (Dimensionless Decay Rate)",
        ylabel="Amplification Factor |R(z)|",
        width=900,
        height=520,
    )
    plot.add_series("Stability Threshold |R| = 1", z_vals, unit_line, color="#64748b", stroke_width=1.5, dashed=True)
    plot.add_series("Explicit Euler: |1 - z| (Unstable for z > 2)", z_vals, exp_amp, color="#ef4444", stroke_width=2.2)
    plot.add_series("Implicit Euler: 1 / (1 + z) (Unconditionally Stable)", z_vals, imp_amp, color="#3b82f6", stroke_width=2.2)

    out_path = VISUALIZATION_DIR / "stability_amplification.svg"
    plot.save(str(out_path))
    print(f"Generated stability amplification plot: {out_path}")


def generate_all_plots() -> None:
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("GENERATING VISUALIZATION CHARTS (SVG)")
    print("=" * 70)
    plot_trajectories()
    plot_convergence()
    plot_forecast_horizon()
    plot_stability()


if __name__ == "__main__":
    generate_all_plots()
