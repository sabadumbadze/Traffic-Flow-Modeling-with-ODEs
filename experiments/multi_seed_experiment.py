"""Multi-seed training and unseen test data evaluation suite.

Executes:
1. Trains traffic flow ODE model on 5 distinct seeds [42, 123, 456, 789, 2026] for 50 epochs each.
2. Uses CPU multiprocessing (num_workers=8) and data caching.
3. Tests each model on held-out UNSEEN data (tra_Y_te, 840 steps).
4. Generates visual photos (PNG & SVG) for each seed:
   - ivp_simulation_photo (24h unseen test window)
   - basin_of_attraction_photo (50x50 grid convergence & slope field)
   - forecast_horizon_error
5. Generates comprehensive multi-seed comparison photos and aggregate statistics.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OBS_INTERVAL_H,
    OBS_INTERVAL_MIN,
    PARAM_LOWER,
    PARAM_NAMES,
    PARAM_UPPER,
    RESULTS_DIR,
    VISUALIZATION_DIR,
)
from src.ivp import build_ivp
from src.pipeline import PreparedData
from src.simulation import simulate
from experiments.basin_of_attraction import (
    compute_basin_grid,
    compute_sample_trajectories,
    render_basin_photo,
)
from experiments.run_ivp_simulation import run_ivp_and_render_photo
from experiments.train import (
    AdamOptimizer,
    evaluate_window_chunk,
    get_cached_or_prepare_data,
    get_system_resource_snapshot,
    prepare_window_datasets,
    save_checkpoint,
)
from visualization.svg_plotter import SVGPlot

SEEDS = [42, 123, 456, 789, 2026]


# =====================================================================
# 1. SINGLE-SEED TRAINING FUNCTION
# =====================================================================

def train_single_seed(
    seed: int,
    data: PreparedData,
    epochs: int = 50,
    num_workers: int = 8,
    lr: float = 0.02,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STARTING 50-EPOCH TRAINING RUN FOR SEED {seed}")
    print("=" * 80)

    rng = random.Random(seed)

    # Perturb initial parameter vector by +/- 20%
    base_th = [2.90, 2.40, 2.50, 0.12, 0.09, 0.0]
    th_init = [
        max(PARAM_LOWER[i], min(PARAM_UPPER[i], base_th[i] * (1.0 + rng.uniform(-0.20, 0.20))))
        for i in range(6)
    ]
    print(f"[SEED {seed}] Initial perturbed theta: {[round(v, 4) for v in th_init]}")

    train_windows, val_windows = prepare_window_datasets(data, window_length=48, train_stride=8, val_stride=16)

    chunk_size = max(1, math.ceil(len(train_windows) / num_workers))
    val_chunks = [val_windows[i : i + chunk_size] for i in range(0, len(val_windows), chunk_size)]

    optimizer = AdamOptimizer(th_init, lr=lr)
    best_val_loss = float("inf")
    history: Dict[str, List[Any]] = {
        "epochs": [],
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "parameters": [],
    }

    pool = mp.Pool(processes=num_workers)
    t0_train = time.time()
    block_start = time.time()

    seed_ckpt_dir = RESULTS_DIR.parent / "checkpoints" / f"seed_{seed}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, epochs + 1):
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, epochs)))

            # Shuffle training windows with seed RNG for stochastic batching
            shuffled_windows = list(train_windows)
            rng.shuffle(shuffled_windows)
            train_chunks = [shuffled_windows[i : i + chunk_size] for i in range(0, len(shuffled_windows), chunk_size)]

            current_theta = tuple(optimizer.theta)
            worker_tasks = [(chunk, current_theta, OBS_INTERVAL_H) for chunk in train_chunks]
            chunk_results = pool.map(evaluate_window_chunk, worker_tasks)

            total_loss = sum(r[0] for r in chunk_results)
            total_abs = sum(r[1] for r in chunk_results)
            total_pts = sum(r[3] for r in chunk_results)

            mean_train_loss = total_loss / total_pts
            mean_train_mae = total_abs / total_pts

            agg_grad = [sum(r[4][j] for r in chunk_results) / total_pts for j in range(6)]
            grad_norm = math.sqrt(sum(g * g for g in agg_grad))

            updated_theta = optimizer.update(agg_grad, lr_scale=lr_scale)

            # Validation evaluation every 5 epochs
            val_loss = mean_train_loss
            val_mae = mean_train_mae
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                val_tasks = [(chunk, updated_theta, OBS_INTERVAL_H) for chunk in val_chunks]
                val_chunk_results = pool.map(evaluate_window_chunk, val_tasks)
                v_pts = sum(r[3] for r in val_chunk_results)
                val_loss = sum(r[0] for r in val_chunk_results) / v_pts
                val_mae = sum(r[1] for r in val_chunk_results) / v_pts
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            history["epochs"].append(epoch)
            history["train_loss"].append(mean_train_loss)
            history["train_mae"].append(mean_train_mae)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["parameters"].append(list(updated_theta))

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                block_dur = time.time() - block_start
                sys_snap = get_system_resource_snapshot()
                cores_str = " ".join(f"C{i}:{pct:.0f}%" for i, pct in enumerate(sys_snap["cpu_per_core"][:8]))
                gpu_str = f"GPU: {sys_snap['gpu'].get('gpu_util_percent', 0.0)}%" if sys_snap["gpu"]["available"] else "GPU: N/A"

                print(
                    f"[SEED {seed}] Epoch {epoch:2d}/50 | Time: {block_dur:5.2f}s | "
                    f"Train Loss: {mean_train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                    f"||∇θ||: {grad_norm:.4f} | CPU: {sys_snap['cpu_total_percent']:4.1f}% [{cores_str}] | {gpu_str}"
                )

                # Save checkpoint
                save_checkpoint(
                    epoch=epoch,
                    total_epochs=epochs,
                    optimizer=optimizer,
                    train_loss=mean_train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    history=history,
                    checkpoint_dir=seed_ckpt_dir,
                )
                block_start = time.time()

    finally:
        pool.close()
        pool.join()

    total_time = time.time() - t0_train
    final_theta = tuple(optimizer.theta)
    print(f"[SEED {seed}] Training completed in {total_time:.2f} seconds. Final theta: {[round(v, 4) for v in final_theta]}")

    return {
        "seed": seed,
        "final_theta": final_theta,
        "best_val_loss": best_val_loss,
        "training_time_sec": total_time,
        "history": history,
    }


# =====================================================================
# 2. EVALUATION ON UNSEEN DATA
# =====================================================================

def evaluate_on_unseen_data(
    data: PreparedData,
    theta: Tuple[float, ...],
    seed: int,
) -> Dict[str, Any]:
    """Tests the trained model on held-out test data across segments and forecast horizons."""
    print(f"\n[TESTING UNSEEN DATA - SEED {seed}]")
    n_test = data.test.n
    test_hours = n_test * OBS_INTERVAL_H

    # 1. Full tracking evaluation across all test segments
    segment_errors = {"explicit": [], "implicit": [], "rk4": []}
    segment_rmses = {"explicit": [], "implicit": [], "rk4": []}
    segment_maxes = {"explicit": [], "implicit": [], "rk4": []}

    for seg in data.test_segments:
        if seg.length < 4:
            continue
        ivp = build_ivp(data.test, theta, seg.start, seg.stop)
        obs_states = [data.test.state_at(k) for k in range(seg.start, seg.stop)]

        for method in ("explicit", "implicit", "rk4"):
            substeps = 4 if method == "rk4" else 1
            traj = simulate(ivp, method, substeps=substeps)
            sim = traj.at_observations()

            mae = sum(sum(abs(s[i] - o[i]) for i in range(3)) for s, o in zip(sim, obs_states)) / (seg.length * 3)
            rmse = math.sqrt(sum(sum((s[i] - o[i])**2 for i in range(3)) for s, o in zip(sim, obs_states)) / (seg.length * 3))
            m_err = max(max(abs(s[i] - o[i]) for i in range(3)) for s, o in zip(sim, obs_states))

            segment_errors[method].append(mae)
            segment_rmses[method].append(rmse)
            segment_maxes[method].append(m_err)

    avg_mae = {m: sum(segment_errors[m]) / len(segment_errors[m]) for m in segment_errors}
    avg_rmse = {m: sum(segment_rmses[m]) / len(segment_rmses[m]) for m in segment_rmses}
    worst_max = {m: max(segment_maxes[m]) for m in segment_maxes}

    print(f"  Unseen Test Tracking Performance (840 steps / {test_hours:.1f} hours):")
    print(f"    • Explicit Euler : MAE = {avg_mae['explicit']:.4f} | RMSE = {avg_rmse['explicit']:.4f} | Max = {worst_max['explicit']:.4f}")
    print(f"    • Implicit Euler : MAE = {avg_mae['implicit']:.4f} | RMSE = {avg_rmse['implicit']:.4f} | Max = {worst_max['implicit']:.4f}")
    print(f"    • RK4 Integrator : MAE = {avg_mae['rk4']:.4f} | RMSE = {avg_rmse['rk4']:.4f} | Max = {worst_max['rk4']:.4f}")

    # 2. Multi-step forecast horizon evaluation
    horizons = [1, 2, 4, 8, 16, 48, 96]  # 15m to 24h
    horizon_metrics = {}

    for H in horizons:
        h_min = H * OBS_INTERVAL_MIN
        errors = []
        for seg in data.test_segments:
            if seg.length <= H + 2:
                continue
            for origin in range(seg.start, seg.stop - H, 4):
                ivp = build_ivp(data.test, theta, origin, origin + H + 1)
                obs_target = data.test.state_at(origin + H)
                traj = simulate(ivp, "rk4", substeps=4)
                sim_target = traj.states[-1]
                err = sum(abs(sim_target[i] - obs_target[i]) for i in range(3)) / 3.0
                errors.append(err)

        mean_h_err = sum(errors) / len(errors) if errors else 0.0
        horizon_metrics[f"horizon_{H}_steps"] = {
            "steps": H,
            "minutes": h_min,
            "hours": h_min / 60.0,
            "mae": mean_h_err,
        }

    print(f"    • 15-min Forecast MAE : {horizon_metrics['horizon_1_steps']['mae']:.4f}")
    print(f"    • 1-hour Forecast MAE : {horizon_metrics['horizon_4_steps']['mae']:.4f}")
    print(f"    • 24-hr  Forecast MAE : {horizon_metrics['horizon_96_steps']['mae']:.4f}")

    # 3. 24-hour test window trajectory simulation for plotting
    seg = data.test_segments[1]
    window_steps = min(96, seg.length)
    ivp_window = build_ivp(data.test, theta, seg.start, seg.start + window_steps)
    traj_rk4 = simulate(ivp_window, "rk4", substeps=4).at_observations()
    obs_window = [data.test.state_at(k) for k in range(seg.start, seg.start + window_steps)]
    u_window = list(data.test.u[seg.start : seg.start + window_steps])
    times_h = [k * OBS_INTERVAL_H for k in range(window_steps)]

    return {
        "seed": seed,
        "theta": list(theta),
        "tracking_mae": avg_mae,
        "tracking_rmse": avg_rmse,
        "tracking_max": worst_max,
        "horizon_metrics": horizon_metrics,
        "window_24h": {
            "times_h": times_h,
            "observed": obs_window,
            "simulated": traj_rk4,
            "upstream_u": u_window,
        },
    }


# =====================================================================
# 3. SEED-SPECIFIC VISUALIZATION RENDERING
# =====================================================================

def render_seed_visualizations(
    seed: int,
    data: PreparedData,
    theta: Tuple[float, ...],
    unseen_eval: Dict[str, Any],
) -> Dict[str, Path]:
    seed_viz_dir = VISUALIZATION_DIR / f"seed_{seed}"
    seed_viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. IVP Simulation Photo on Unseen Data (PNG & SVG)
    seg = data.test_segments[1]
    duration_steps = min(96, seg.length)
    ivp = build_ivp(data.test, theta, seg.start, seg.start + duration_steps)
    times_h = [k * OBS_INTERVAL_H for k in range(duration_steps)]
    obs_states = [data.test.state_at(k) for k in range(seg.start, seg.start + duration_steps)]
    upstream_u = list(data.test.u[seg.start : seg.start + duration_steps])

    states_exp = simulate(ivp, "explicit", substeps=1).at_observations()
    states_imp = simulate(ivp, "implicit", substeps=1).at_observations()
    states_rk4 = simulate(ivp, "rk4", substeps=4).at_observations()

    # Draw PNG Photo
    img_w, img_h = 1300, 920
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, img_w, 70], fill=(15, 23, 42))
    draw.text((30, 15), f"IVP SIMULATION ON UNSEEN DATA (SEED {seed})", fill=(255, 255, 255))
    draw.text((30, 42), f"Held-out Test Window (24h) | Learned θ: {[round(v, 3) for v in theta]}", fill=(148, 163, 184))

    # Metrics cards
    card_y = 85
    card_w = 370
    cards = [
        ("Explicit Euler (h=15m)", f"MAE: {unseen_eval['tracking_mae']['explicit']:.4f}\nRMSE: {unseen_eval['tracking_rmse']['explicit']:.4f}", (254, 242, 242), (239, 68, 68)),
        ("Implicit Euler (h=15m)", f"MAE: {unseen_eval['tracking_mae']['implicit']:.4f}\nRMSE: {unseen_eval['tracking_rmse']['implicit']:.4f}", (239, 246, 255), (59, 130, 246)),
        ("RK4 Method (4 substeps)", f"MAE: {unseen_eval['tracking_mae']['rk4']:.4f}\nRMSE: {unseen_eval['tracking_rmse']['rk4']:.4f}", (236, 253, 245), (16, 185, 129)),
    ]
    for idx, (title, content, bg, border) in enumerate(cards):
        cx = 30 + idx * 420
        draw.rectangle([cx, card_y, cx + card_w, card_y + 65], fill=bg, outline=border, width=2)
        draw.text((cx + 12, card_y + 8), title, fill=(15, 23, 42))
        draw.text((cx + 12, card_y + 26), content, fill=(71, 85, 105))

    # Panels
    panel_y_start = 170
    panel_h = 210
    panel_gap = 25
    panel_w = img_w - 120
    panel_x = 80

    for s_idx in range(3):
        py = panel_y_start + s_idx * (panel_h + panel_gap)
        draw.rectangle([panel_x, py, panel_x + panel_w, py + panel_h], fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        draw.text((panel_x + 15, py + 8), f"Sensor State x{s_idx+1} (Sensor {data.chain.states[s_idx]}) - Unseen Traffic Flow", fill=(30, 41, 59))

        all_v = [o[s_idx] for o in obs_states] + [e[s_idx] for e in states_exp] + [r[s_idx] for r in states_rk4]
        max_v = max(all_v) * 1.15

        def to_coord(t_idx, val):
            cx = panel_x + 50 + (t_idx / (duration_steps - 1)) * (panel_w - 70)
            cy = py + panel_h - 30 - (val / max_v) * (panel_h - 60)
            return cx, cy

        pts_exp = [to_coord(t, states_exp[t][s_idx]) for t in range(duration_steps)]
        pts_imp = [to_coord(t, states_imp[t][s_idx]) for t in range(duration_steps)]
        pts_rk4 = [to_coord(t, states_rk4[t][s_idx]) for t in range(duration_steps)]
        pts_obs = [to_coord(t, obs_states[t][s_idx]) for t in range(duration_steps)]

        draw.line(pts_exp, fill=(239, 68, 68), width=2)
        draw.line(pts_imp, fill=(59, 130, 246), width=2)
        draw.line(pts_rk4, fill=(16, 185, 129), width=2)
        draw.line(pts_obs, fill=(15, 23, 42), width=2)
        for pt in pts_obs[::4]:
            draw.ellipse([pt[0] - 2.5, pt[1] - 2.5, pt[0] + 2.5, pt[1] + 2.5], fill=(15, 23, 42))

    ivp_png = seed_viz_dir / "ivp_simulation_photo.png"
    img.save(str(ivp_png), "PNG")

    # SVG IVP
    ivp_svg = seed_viz_dir / "ivp_simulation_photo.svg"
    plot = SVGPlot(
        title=f"Unseen Test Simulation (Seed {seed}): Observed vs Solvers",
        xlabel="Time (hours)",
        ylabel="Normalized Flow",
        width=950,
        height=520,
    )
    plot.add_series("Observed Ground Truth", times_h, [o[0] for o in obs_states], color="#0f172a", stroke_width=2.5)
    plot.add_series("RK4 Integrator", times_h, [r[0] for r in states_rk4], color="#10b981", stroke_width=2.0)
    plot.add_series("Explicit Euler", times_h, [e[0] for e in states_exp], color="#ef4444", stroke_width=1.8, dashed=True)
    plot.add_series("Implicit Euler", times_h, [i[0] for i in states_imp], color="#3b82f6", stroke_width=1.8, dashed=True)
    plot.save(str(ivp_svg))

    # 2. Basin of Attraction Study for this seed
    grid_data = compute_basin_grid(theta, u=1.0, grid_size=50, max_steps=400, h=0.02)
    sample_trajectories = compute_sample_trajectories(theta, u=1.0, n_steps=350, h=0.02)

    basin_png, basin_svg = render_basin_photo(grid_data, sample_trajectories, theta, u=1.0)
    # Copy or move into seed_viz_dir
    seed_basin_png = seed_viz_dir / "basin_of_attraction_photo.png"
    seed_basin_svg = seed_viz_dir / "basin_of_attraction_photo.svg"
    import shutil
    shutil.copy(basin_png, seed_basin_png)
    shutil.copy(basin_svg, seed_basin_svg)

    # 3. Forecast Horizon Error Plot
    horizon_svg = seed_viz_dir / "forecast_horizon_error.svg"
    h_data = unseen_eval["horizon_metrics"]
    hrs = [v["hours"] for v in h_data.values()]
    maes = [v["mae"] for v in h_data.values()]
    h_plot = SVGPlot(
        title=f"Multi-Step Forecast Error on Unseen Data (Seed {seed})",
        xlabel="Forecast Horizon (hours)",
        ylabel="Mean Absolute Error (MAE)",
        width=900,
        height=500,
    )
    h_plot.add_series(f"Seed {seed} RK4 Forecast", hrs, maes, color="#2563eb", stroke_width=2.5, marker="circle")
    h_plot.save(str(horizon_svg))

    return {
        "ivp_png": ivp_png,
        "ivp_svg": ivp_svg,
        "basin_png": seed_basin_png,
        "basin_svg": seed_basin_svg,
        "horizon_svg": horizon_svg,
    }


# =====================================================================
# 4. MULTI-SEED COMPARISON VISUAL PHOTO (PNG & SVG)
# =====================================================================

def render_multi_seed_comparison_photo(
    all_evals: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    png_path = VISUALIZATION_DIR / "multi_seed_comparison_photo.png"
    svg_path = VISUALIZATION_DIR / "multi_seed_comparison.svg"

    img_w, img_h = 1350, 980
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([0, 0, img_w, 75], fill=(15, 23, 42))
    draw.text((35, 16), "MULTI-SEED MODEL EVALUATION ON UNSEEN TEST DATA (5 SEEDS)", fill=(255, 255, 255))
    draw.text(
        (35, 45),
        "Trained across 5 distinct random seeds [42, 123, 456, 789, 2026] for 50 epochs each | Evaluated on held-out test data (840 steps / 210h)",
        fill=(148, 163, 184),
    )

    # Top panel: 24h Trajectory Comparison on Unseen Data (State 1)
    plot_x, plot_y = 70, 105
    plot_w, plot_h = img_w - 140, 360
    draw.rectangle([plot_x, plot_y, plot_x + plot_w, plot_y + plot_h], fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    draw.text((plot_x + 15, plot_y + 10), "Unseen Test Trajectory: Ground Truth vs 5 Trained Seeds (Sensor 34)", fill=(30, 41, 59))

    sample_eval = all_evals[0]["window_24h"]
    times = sample_eval["times_h"]
    n_pts = len(times)
    obs_s1 = [s[0] for s in sample_eval["observed"]]
    max_v = max(obs_s1) * 1.25

    def to_pixel(t_idx, val):
        px = plot_x + 50 + (t_idx / (n_pts - 1)) * (plot_w - 70)
        py = plot_y + plot_h - 35 - (val / max_v) * (plot_h - 70)
        return px, py

    # Time ticks
    for hr in range(0, int(times[-1]) + 1, 4):
        tx, _ = to_pixel(int(hr / OBS_INTERVAL_H), 0)
        draw.line([(tx, plot_y + 35), (tx, plot_y + plot_h - 35)], fill=(226, 232, 240), width=1)
        draw.text((tx - 10, plot_y + plot_h - 25), f"{hr}h", fill=(100, 116, 139))

    # Grid Y ticks
    for gy_i in range(5):
        gv = gy_i * max_v / 4.0
        _, gy = to_pixel(0, gv)
        draw.line([(plot_x + 45, gy), (plot_x + plot_w - 10, gy)], fill=(226, 232, 240), width=1)
        draw.text((plot_x + 10, gy - 6), f"{gv:.2f}", fill=(100, 116, 139))

    # Plot each seed's simulation line
    seed_colors = [
        (239, 68, 68),   # red
        (59, 130, 246),  # blue
        (16, 185, 129),  # green
        (217, 119, 6),   # amber
        (147, 51, 234),  # purple
    ]

    for idx, e in enumerate(all_evals):
        sim_s1 = [s[0] for s in e["window_24h"]["simulated"]]
        pts = [to_pixel(t, sim_s1[t]) for t in range(n_pts)]
        draw.line(pts, fill=seed_colors[idx], width=2)

    # Plot Ground Truth (solid bold black)
    pts_obs = [to_pixel(t, obs_s1[t]) for t in range(n_pts)]
    draw.line(pts_obs, fill=(15, 23, 42), width=3)
    for pt in pts_obs[::4]:
        draw.ellipse([pt[0] - 2.5, pt[1] - 2.5, pt[0] + 2.5, pt[1] + 2.5], fill=(15, 23, 42))

    # Trajectory legend
    leg_y = plot_y + plot_h - 25
    draw.line([(plot_x + 60, leg_y), (plot_x + 85, leg_y)], fill=(15, 23, 42), width=3)
    draw.text((plot_x + 92, leg_y - 6), "Ground Truth (Real Data)", fill=(15, 23, 42))
    for idx, e in enumerate(all_evals):
        lx = plot_x + 280 + idx * 165
        draw.line([(lx, leg_y), (lx + 20, leg_y)], fill=seed_colors[idx], width=2)
        draw.text((lx + 26, leg_y - 6), f"Seed {e['seed']}", fill=(51, 65, 85))

    # Bottom Panels:
    # Left Bottom: Multi-Step Forecast Error Curves across 5 Seeds
    b_plot_w = (plot_w - 30) // 2
    b_plot_h = 420
    b_plot_y = plot_y + plot_h + 30
    draw.rectangle([plot_x, b_plot_y, plot_x + b_plot_w, b_plot_y + b_plot_h], fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    draw.text((plot_x + 15, b_plot_y + 10), "Forecast MAE vs Horizon on Unseen Data (5 Seeds)", fill=(30, 41, 59))

    horizons_h = [v["hours"] for v in all_evals[0]["horizon_metrics"].values()]
    all_maes = [v["mae"] for e in all_evals for v in e["horizon_metrics"].values()]
    max_h_mae = max(all_maes) * 1.25

    def to_b_pixel(hr_val, err_val):
        bx = plot_x + 50 + (hr_val / 24.0) * (b_plot_w - 70)
        by = b_plot_y + b_plot_h - 40 - (err_val / max_h_mae) * (b_plot_h - 80)
        return bx, by

    for hr_tick in [0.25, 1, 4, 12, 24]:
        hx, _ = to_b_pixel(hr_tick, 0)
        draw.line([(hx, b_plot_y + 35), (hx, b_plot_y + b_plot_h - 40)], fill=(226, 232, 240), width=1)
        draw.text((hx - 8, b_plot_y + b_plot_h - 30), f"{hr_tick:g}h", fill=(100, 116, 139))

    for idx, e in enumerate(all_evals):
        h_vals = [v["mae"] for v in e["horizon_metrics"].values()]
        pts = [to_b_pixel(hr, mv) for hr, mv in zip(horizons_h, h_vals)]
        draw.line(pts, fill=seed_colors[idx], width=2)
        for pt in pts:
            draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill=seed_colors[idx])

    # Right Bottom: Aggregate Summary Statistics Table
    t_x = plot_x + b_plot_w + 30
    draw.rectangle([t_x, b_plot_y, t_x + b_plot_w, b_plot_y + b_plot_h], fill=(255, 255, 255), outline=(203, 213, 225), width=1)
    draw.text((t_x + 15, b_plot_y + 12), "MULTI-SEED PERFORMANCE SUMMARY ON UNSEEN DATA", fill=(15, 23, 42))

    # Table Header
    th_y = b_plot_y + 45
    draw.rectangle([t_x + 10, th_y, t_x + b_plot_w - 10, th_y + 26], fill=(241, 245, 249))
    headers = [("Seed", 40), ("Test MAE", 75), ("Test RMSE", 75), ("15m MAE", 75), ("Learned θ (α1, α2, α3)", 150)]
    cur_hx = t_x + 15
    for title, width in headers:
        draw.text((cur_hx, th_y + 5), title, fill=(30, 41, 59))
        cur_hx += width

    # Rows
    for idx, e in enumerate(all_evals):
        row_y = th_y + 32 + idx * 36
        bg_col = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
        draw.rectangle([t_x + 10, row_y, t_x + b_plot_w - 10, row_y + 32], fill=bg_col)
        th = e["theta"]
        th_str = f"({th[0]:.2f}, {th[1]:.2f}, {th[2]:.2f})"
        mae_15m = e["horizon_metrics"]["horizon_1_steps"]["mae"]
        row_data = [
            (str(e["seed"]), 40),
            (f"{e['tracking_mae']['rk4']:.4f}", 75),
            (f"{e['tracking_rmse']['rk4']:.4f}", 75),
            (f"{mae_15m:.4f}", 75),
            (th_str, 150),
        ]
        cur_rx = t_x + 15
        for text, width in row_data:
            draw.text((cur_rx, row_y + 8), text, fill=(51, 65, 85))
            cur_rx += width

    # Ensemble statistics row
    ens_y = th_y + 32 + len(all_evals) * 36 + 15
    draw.rectangle([t_x + 10, ens_y, t_x + b_plot_w - 10, ens_y + 70], fill=(236, 253, 245), outline=(16, 185, 129), width=1)
    maes = [e["tracking_mae"]["rk4"] for e in all_evals]
    rmses = [e["tracking_rmse"]["rk4"] for e in all_evals]
    m_15s = [e["horizon_metrics"]["horizon_1_steps"]["mae"] for e in all_evals]

    mean_mae = sum(maes) / len(maes)
    std_mae = math.sqrt(sum((m - mean_mae)**2 for m in maes) / len(maes))
    mean_rmse = sum(rmses) / len(rmses)
    std_rmse = math.sqrt(sum((r - mean_rmse)**2 for r in rmses) / len(rmses))
    mean_15m = sum(m_15s) / len(m_15s)
    std_15m = math.sqrt(sum((x - mean_15m)**2 for x in m_15s) / len(m_15s))

    draw.text((t_x + 20, ens_y + 10), "ENSEMBLE MEAN ± STD OVER 5 SEEDS:", fill=(6, 95, 70))
    draw.text((t_x + 20, ens_y + 30), f"• Test Tracking MAE  : {mean_mae:.4f} ± {std_mae:.4f}", fill=(4, 120, 87))
    draw.text((t_x + 20, ens_y + 48), f"• Test Tracking RMSE : {mean_rmse:.4f} ± {std_rmse:.4f} | 15-min Forecast MAE: {mean_15m:.4f} ± {std_15m:.4f}", fill=(4, 120, 87))

    img.save(str(png_path), "PNG")
    print(f"\nGenerated multi-seed comparison photo (PNG): {png_path}")

    # Vector SVG
    multi_svg = SVGPlot(
        title="Unseen Test Set Trajectories: Ground Truth vs 5 Trained Seeds",
        xlabel="Time (hours)",
        ylabel="Normalized Flow",
        width=1000,
        height=550,
    )
    multi_svg.add_series("Ground Truth (Observed)", times, obs_s1, color="#0f172a", stroke_width=2.8)
    for idx, e in enumerate(all_evals):
        sim_s1 = [s[0] for s in e["window_24h"]["simulated"]]
        hex_cols = ["#ef4444", "#3b82f6", "#10b981", "#d97706", "#9333ea"]
        multi_svg.add_series(f"Seed {e['seed']} (MAE: {e['tracking_mae']['rk4']:.3f})", times, sim_s1, color=hex_cols[idx], stroke_width=1.8, dashed=True)
    multi_svg.save(str(svg_path))
    print(f"Generated multi-seed vector graphic (SVG): {svg_path}")

    return png_path, svg_path


# =====================================================================
# 5. MAIN MULTI-SEED EXPERIMENT RUNNER
# =====================================================================

def run_multi_seed_experiment(
    seeds: List[int] = SEEDS,
    epochs: int = 50,
    num_workers: int = 8,
) -> Dict[str, Any]:
    print("=" * 80)
    print("TRAFFIC FLOW ODE - 5-SEED TRAINING & UNSEEN DATA TESTING SUITE")
    print("=" * 80)
    print(f"Seeds to Evaluate : {seeds}")
    print(f"Epochs per Seed   : {epochs}")
    print(f"Parallel Workers  : {num_workers} CPU cores")

    data = get_cached_or_prepare_data()

    all_seed_evals: List[Dict[str, Any]] = []

    for s_idx, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"STAGE {s_idx+1}/5: TRAINING & EVALUATING SEED {seed}")
        print(f"{'#'*80}")

        # Train model for this seed
        train_res = train_single_seed(
            seed=seed,
            data=data,
            epochs=epochs,
            num_workers=num_workers,
            lr=0.02,
        )

        # Test on UNSEEN test data
        unseen_eval = evaluate_on_unseen_data(data, train_res["final_theta"], seed)

        # Generate seed-specific visualizations
        viz_files = render_seed_visualizations(seed, data, train_res["final_theta"], unseen_eval)

        # Save seed results JSON
        seed_res_dir = RESULTS_DIR / f"seed_{seed}"
        seed_res_dir.mkdir(parents=True, exist_ok=True)
        seed_output = {
            "seed": seed,
            "learned_theta": list(train_res["final_theta"]),
            "best_val_loss": train_res["best_val_loss"],
            "training_time_sec": train_res["training_time_sec"],
            "unseen_test_evaluation": unseen_eval,
            "artifacts": {k: str(v) for k, v in viz_files.items()},
        }
        with open(seed_res_dir / "seed_evaluation_results.json", "w") as f:
            json.dump(seed_output, f, indent=2)

        all_seed_evals.append(unseen_eval)

    # Multi-seed aggregate comparison visualization
    print("\n" + "=" * 80)
    print("GENERATING MULTI-SEED SYNTHESIS VISUALIZATIONS & REPORT")
    print("=" * 80)
    multi_png, multi_svg = render_multi_seed_comparison_photo(all_seed_evals)

    # Save multi-seed summary JSON
    summary_path = RESULTS_DIR / "multi_seed_summary.json"
    summary_data = {
        "seeds": seeds,
        "epochs_per_seed": epochs,
        "seed_evaluations": [
            {
                "seed": e["seed"],
                "theta": e["theta"],
                "tracking_mae": e["tracking_mae"],
                "tracking_rmse": e["tracking_rmse"],
                "tracking_max": e["tracking_max"],
                "horizon_15m_mae": e["horizon_metrics"]["horizon_1_steps"]["mae"],
                "horizon_1h_mae": e["horizon_metrics"]["horizon_4_steps"]["mae"],
                "horizon_24h_mae": e["horizon_metrics"]["horizon_96_steps"]["mae"],
            }
            for e in all_seed_evals
        ],
        "ensemble_metrics": {
            "mean_test_mae": sum(e["tracking_mae"]["rk4"] for e in all_seed_evals) / len(seeds),
            "std_test_mae": math.sqrt(sum((e["tracking_mae"]["rk4"] - (sum(x["tracking_mae"]["rk4"] for x in all_seed_evals)/len(seeds)))**2 for e in all_seed_evals) / len(seeds)),
            "mean_test_rmse": sum(e["tracking_rmse"]["rk4"] for e in all_seed_evals) / len(seeds),
            "std_test_rmse": math.sqrt(sum((e["tracking_rmse"]["rk4"] - (sum(x["tracking_rmse"]["rk4"] for x in all_seed_evals)/len(seeds)))**2 for e in all_seed_evals) / len(seeds)),
            "mean_15m_mae": sum(e["horizon_metrics"]["horizon_1_steps"]["mae"] for e in all_seed_evals) / len(seeds),
            "std_15m_mae": math.sqrt(sum((e["horizon_metrics"]["horizon_1_steps"]["mae"] - (sum(x["horizon_metrics"]["horizon_1_steps"]["mae"] for x in all_seed_evals)/len(seeds)))**2 for e in all_seed_evals) / len(seeds)),
        },
        "artifacts": {
            "multi_seed_png": str(multi_png),
            "multi_seed_svg": str(multi_svg),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Saved complete multi-seed summary to {summary_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("FINAL 5-SEED PERFORMANCE SUMMARY TABLE (ON UNSEEN HELD-OUT TEST DATA)")
    print("=" * 90)
    print(f"{'Seed':8s} | {'Test MAE':10s} | {'Test RMSE':10s} | {'15m MAE':10s} | {'1h MAE':10s} | {'Learned Parameters θ (α1, α2, α3, β1, β2, β3)':35s}")
    print("-" * 90)
    for e in all_seed_evals:
        th = e["theta"]
        th_s = f"({th[0]:.2f}, {th[1]:.2f}, {th[2]:.2f}, {th[3]:.2f}, {th[4]:.2f}, {th[5]:.2f})"
        mae_15m = e["horizon_metrics"]["horizon_1_steps"]["mae"]
        mae_1h = e["horizon_metrics"]["horizon_4_steps"]["mae"]
        print(f"{e['seed']:8d} | {e['tracking_mae']['rk4']:10.4f} | {e['tracking_rmse']['rk4']:10.4f} | {mae_15m:10.4f} | {mae_1h:10.4f} | {th_s:35s}")
    print("-" * 90)
    ens = summary_data["ensemble_metrics"]
    print(f"{'ENSEMBLE':8s} | {ens['mean_test_mae']:10.4f} | {ens['mean_test_rmse']:10.4f} | {ens['mean_15m_mae']:10.4f} | {'—':10s} | (Mean across 5 seeds ± std: {ens['std_test_mae']:.4f})")
    print("=" * 90)

    return summary_data


if __name__ == "__main__":
    run_multi_seed_experiment()
