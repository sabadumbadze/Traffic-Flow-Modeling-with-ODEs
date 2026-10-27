"""Executes the Initial Value Problem (IVP) on real traffic data and renders high-res visual plots.

Produces:
- visualization/ivp_simulation_photo.png (1200x850 high-resolution raster photo)
- visualization/ivp_simulation_photo.svg (vector graphic)
- experiments/results/ivp_simulation_metrics.json (metrics summary)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.config import (
    OBS_INTERVAL_H,
    RESULTS_DIR,
    VISUALIZATION_DIR,
)
from src.ivp import build_ivp
from src.pipeline import prepare_data
from src.simulation import simulate
from experiments.test_validation import load_fitted_theta


def run_ivp_and_render_photo(
    segment_idx: int = 1,
    duration_steps: int = 96,  # 24 hours
) -> Dict[str, Any]:
    print("=" * 80)
    print("RUNNING INITIAL VALUE PROBLEM (IVP) ON REAL TRAFFIC DATA")
    print("=" * 80)

    data = prepare_data()
    theta = load_fitted_theta()

    # Choose a clean 24-hour window from the test data
    seg = data.test_segments[segment_idx]
    start = seg.start
    stop = min(seg.stop, start + duration_steps)
    n_steps = stop - start

    print(f"Test Segment: Index {segment_idx} | Steps [{start}, {stop}] ({n_steps} observation steps)")
    print(f"Time Horizon: {n_steps * OBS_INTERVAL_H:.1f} hours ({n_steps * 15} minutes)")
    print(f"Sensor Chain: Upstream={data.chain.upstream} -> State1={data.chain.states[0]}, "
          f"State2={data.chain.states[1]}, State3={data.chain.states[2]}")
    print(f"Initial State x0: {[round(v, 4) for v in data.test.state_at(start)]}")
    print(f"Model Parameters θ: {[round(v, 4) for v in theta]}")

    ivp = build_ivp(data.test, theta, start, stop)
    times_h = [k * OBS_INTERVAL_H for k in range(n_steps)]
    obs_states = [data.test.state_at(k) for k in range(start, stop)]
    upstream_u = list(data.test.u[start:stop])

    print("\nIntegrating ODE with Explicit Euler, Implicit Euler, and RK4...")
    traj_exp = simulate(ivp, "explicit", substeps=1)
    traj_imp = simulate(ivp, "implicit", substeps=1)
    traj_rk4 = simulate(ivp, "rk4", substeps=4)

    states_exp = traj_exp.at_observations()
    states_imp = traj_imp.at_observations()
    states_rk4 = traj_rk4.at_observations()

    # Metrics calculation
    def calc_stats(sim):
        mae = sum(sum(abs(s[i] - o[i]) for i in range(3)) for s, o in zip(sim, obs_states)) / (n_steps * 3)
        rmse = math.sqrt(sum(sum((s[i] - o[i])**2 for i in range(3)) for s, o in zip(sim, obs_states)) / (n_steps * 3))
        max_err = max(max(abs(s[i] - o[i]) for i in range(3)) for s, o in zip(sim, obs_states))
        return {"mae": mae, "rmse": rmse, "max_error": max_err}

    m_exp = calc_stats(states_exp)
    m_imp = calc_stats(states_imp)
    m_rk4 = calc_stats(states_rk4)

    print("\nIVP Tracking Error Results:")
    print(f"  Explicit Euler : MAE = {m_exp['mae']:.5f} | RMSE = {m_exp['rmse']:.5f} | Max Err = {m_exp['max_error']:.5f}")
    print(f"  Implicit Euler : MAE = {m_imp['mae']:.5f} | RMSE = {m_imp['rmse']:.5f} | Max Err = {m_imp['max_error']:.5f}")
    print(f"  RK4 Integrator : MAE = {m_rk4['mae']:.5f} | RMSE = {m_rk4['rmse']:.5f} | Max Err = {m_rk4['max_error']:.5f}")

    # =========================================================================
    # RENDER HIGH-RES VISUAL PHOTO (PNG via PIL)
    # =========================================================================
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    png_path = VISUALIZATION_DIR / "ivp_simulation_photo.png"
    svg_path = VISUALIZATION_DIR / "ivp_simulation_photo.svg"

    img_w, img_h = 1300, 920
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Palette
    c_bg = (248, 250, 252)
    c_border = (203, 213, 225)
    c_obs = (15, 23, 42)      # dark navy/black
    c_u = (148, 163, 184)     # slate gray
    c_exp = (239, 68, 68)     # red
    c_imp = (59, 130, 246)    # blue
    c_rk4 = (16, 185, 129)    # emerald green

    # Header title banner
    draw.rectangle([0, 0, img_w, 70], fill=(15, 23, 42))
    draw.text((30, 15), "TRAFFIC FLOW INITIAL VALUE PROBLEM (IVP) SIMULATION", fill=(255, 255, 255))
    draw.text((30, 42), f"24-Hour Real Highway Window (Jan 2017) | 15-min sampling | Upstream Sensor {data.chain.upstream} -> Sensors {data.chain.states}", fill=(148, 163, 184))

    # Metrics summary cards at top
    card_y = 85
    cards = [
        ("Explicit Euler (h=15m)", f"MAE: {m_exp['mae']:.4f}\nRMSE: {m_exp['rmse']:.4f}\nMax: {m_exp['max_error']:.4f}", (254, 242, 242), (239, 68, 68)),
        ("Implicit Euler (h=15m)", f"MAE: {m_imp['mae']:.4f}\nRMSE: {m_imp['rmse']:.4f}\nMax: {m_imp['max_error']:.4f}", (239, 246, 255), (59, 130, 246)),
        ("RK4 Method (4 substeps)", f"MAE: {m_rk4['mae']:.4f}\nRMSE: {m_rk4['rmse']:.4f}\nMax: {m_rk4['max_error']:.4f}", (236, 253, 245), (16, 185, 129)),
    ]
    card_w = 370
    for idx, (title, content, bg, border) in enumerate(cards):
        cx = 30 + idx * 420
        draw.rectangle([cx, card_y, cx + card_w, card_y + 65], fill=bg, outline=border, width=2)
        draw.text((cx + 12, card_y + 8), title, fill=(15, 23, 42))
        draw.text((cx + 12, card_y + 26), content, fill=(71, 85, 105))

    # 3 Main Trajectory Panels (State 1, State 2, State 3)
    panel_y_start = 170
    panel_h = 210
    panel_gap = 25
    panel_w = img_w - 120
    panel_x = 80

    state_labels = [
        f"State x1 (Sensor {data.chain.states[0]}) - Immediately Downstream of Input u(t)",
        f"State x2 (Sensor {data.chain.states[1]}) - Intermediate Road Segment",
        f"State x3 (Sensor {data.chain.states[2]}) - Free Outflow Downstream Sensor",
    ]

    for s_idx in range(3):
        py = panel_y_start + s_idx * (panel_h + panel_gap)
        draw.rectangle([panel_x, py, panel_x + panel_w, py + panel_h], fill=c_bg, outline=c_border, width=1)
        draw.text((panel_x + 15, py + 8), state_labels[s_idx], fill=(30, 41, 59))

        # Y scale for this state
        all_vals = (
            [o[s_idx] for o in obs_states]
            + [e[s_idx] for e in states_exp]
            + [i[s_idx] for i in states_imp]
            + [r[s_idx] for r in states_rk4]
            + upstream_u
        )
        min_v, max_v = 0.0, max(all_vals) * 1.15

        def to_coord(t_idx, val):
            cx = panel_x + 50 + (t_idx / (n_steps - 1)) * (panel_w - 70)
            cy = py + panel_h - 30 - (val - min_v) / (max_v - min_v) * (panel_h - 60)
            return cx, cy

        # Grid lines
        for g_i in range(5):
            g_v = min_v + g_i * (max_v - min_v) / 4.0
            _, gy = to_coord(0, g_v)
            draw.line([(panel_x + 45, gy), (panel_x + panel_w - 10, gy)], fill=(226, 232, 240), width=1)
            draw.text((panel_x + 10, gy - 6), f"{g_v:.2f}", fill=(100, 116, 139))

        # Time ticks
        for hour in range(0, int(times_h[-1]) + 1, 4):
            t_frac = hour / times_h[-1]
            tx = panel_x + 50 + t_frac * (panel_w - 70)
            draw.line([(tx, py + 25), (tx, py + panel_h - 30)], fill=(226, 232, 240), width=1)
            draw.text((tx - 10, py + panel_h - 22), f"{hour}h", fill=(100, 116, 139))

        # 1. Upstream u(t) line (thin dashed / dotted)
        pts_u = [to_coord(t, upstream_u[t]) for t in range(n_steps)]
        for i in range(len(pts_u) - 1):
            if i % 2 == 0:
                draw.line([pts_u[i], pts_u[i + 1]], fill=c_u, width=1)

        # 2. Explicit Euler (red)
        pts_exp = [to_coord(t, states_exp[t][s_idx]) for t in range(n_steps)]
        draw.line(pts_exp, fill=c_exp, width=2)

        # 3. Implicit Euler (blue)
        pts_imp = [to_coord(t, states_imp[t][s_idx]) for t in range(n_steps)]
        draw.line(pts_imp, fill=c_imp, width=2)

        # 4. RK4 (green)
        pts_rk4 = [to_coord(t, states_rk4[t][s_idx]) for t in range(n_steps)]
        draw.line(pts_rk4, fill=c_rk4, width=2)

        # 5. Observed Ground Truth (navy dots & line)
        pts_obs = [to_coord(t, obs_states[t][s_idx]) for t in range(n_steps)]
        draw.line(pts_obs, fill=c_obs, width=2)
        for pt in pts_obs[::4]:
            draw.ellipse([pt[0] - 2.5, pt[1] - 2.5, pt[0] + 2.5, pt[1] + 2.5], fill=c_obs)

    # Bottom shared legend
    leg_y = img_h - 40
    leg_items = [
        ("Observed Ground Truth (Real Data)", c_obs, 3),
        ("Upstream Input u(t)", c_u, 1),
        ("RK4 Method (Order 4)", c_rk4, 2),
        ("Explicit Euler (Order 1)", c_exp, 2),
        ("Implicit Euler (Order 1)", c_imp, 2),
    ]
    cur_x = 60
    for label, col, w in leg_items:
        draw.line([(cur_x, leg_y), (cur_x + 25, leg_y)], fill=col, width=w)
        draw.text((cur_x + 32, leg_y - 6), label, fill=(30, 41, 59))
        cur_x += 240

    img.save(str(png_path), "PNG")
    print(f"\nGenerated visual photo (PNG): {png_path}")

    # Also export vector SVG version
    from visualization.svg_plotter import SVGPlot
    plot = SVGPlot(
        title="Initial Value Problem Simulation: Observed vs Numerical Solvers",
        xlabel="Time (hours)",
        ylabel="Normalized Flow Volume",
        width=1000,
        height=550,
    )
    plot.add_series("Observed State x1 (Sensor 34)", times_h, [o[0] for o in obs_states], color="#0f172a", stroke_width=2.5)
    plot.add_series("Upstream Input u(t)", times_h, upstream_u, color="#94a3b8", stroke_width=1.5, dashed=True)
    plot.add_series("RK4 Integration", times_h, [r[0] for r in states_rk4], color="#10b981", stroke_width=2.0)
    plot.add_series("Explicit Euler", times_h, [e[0] for e in states_exp], color="#ef4444", stroke_width=1.8, dashed=True)
    plot.add_series("Implicit Euler", times_h, [i[0] for i in states_imp], color="#3b82f6", stroke_width=1.8, dashed=True)
    plot.save(str(svg_path))
    print(f"Generated vector graphic (SVG): {svg_path}")

    output_data = {
        "n_steps": n_steps,
        "duration_hours": n_steps * OBS_INTERVAL_H,
        "metrics": {"explicit": m_exp, "implicit": m_imp, "rk4": m_rk4},
        "artifacts": {"png": str(png_path), "svg": str(svg_path)},
    }
    with open(RESULTS_DIR / "ivp_simulation_metrics.json", "w") as f:
        json.dump(output_data, f, indent=2)

    return output_data


if __name__ == "__main__":
    run_ivp_and_render_photo()
