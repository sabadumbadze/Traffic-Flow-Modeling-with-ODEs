"""Experiments on ODE Stability, Stiffness, and Error Analysis.

Directly implements and tests every stability and stiffness concept from the course:
1. Test equation y' = -a*y and A-stability (Explicit Euler: h <= 2/a vs Implicit Euler: all h > 0)
2. Stiff ODE benchmark: y' = -50*(y - sin(t)) + cos(t), exact y(t) = sin(t) + exp(-50*t)
   with critical step h_crit = 2/|lambda| = 0.04 (testing h=0.03 vs h=0.05)
3. Traffic ODE Jacobian eigenvalues and stiffness threshold h_crit = 2/max|lambda_i|
4. Error analysis: Local Truncation Error (LTE = O(h^2)), Global Truncation Error (GTE = O(h)),
   and Residual Error (R_n = 0).
5. Discrete Gronwall bound on error growth for stiff vs non-stiff systems.

Generates visual figures for the presentation using Pillow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.config import RESULTS_DIR, VISUALIZATION_DIR
from src.pipeline import prepare_data
from src.stability import (
    decay_rates,
    explicit_amplification,
    explicit_euler_step_limit,
    implicit_amplification,
    max_rate,
    stiffness_ratio,
)
from experiments.test_validation import load_fitted_theta


# =============================================================================
# EXPERIMENT 1: Test equation y' = -a*y, a = 10, y(0) = 1
# =============================================================================
def run_test_equation_experiment() -> Dict[str, Any]:
    a = 10.0
    y0 = 1.0
    t_end = 1.0

    # Theoretical thresholds:
    # monotonic decay iff h < 1/a = 0.10
    # stable iff h <= 2/a = 0.20
    # unstable iff h > 2/a = 0.20
    h_values = [0.05, 0.15, 0.25]
    results = {}

    for h in h_values:
        n_steps = int(round(t_end / h))
        t_grid = [i * h for i in range(n_steps + 1)]
        y_exact = [y0 * math.exp(-a * t) for t in t_grid]

        # Explicit Euler: y_{n+1} = (1 - a*h) * y_n
        y_exp = [y0]
        for _ in range(n_steps):
            y_next = (1.0 - a * h) * y_exp[-1]
            # Cap huge values to avoid overflow
            if abs(y_next) > 1e6:
                y_next = 1e6 if y_next > 0 else -1e6
            y_exp.append(y_next)

        # Implicit Euler: y_{n+1} = y_n / (1 + a*h)
        y_imp = [y0]
        for _ in range(n_steps):
            y_imp.append(y_imp[-1] / (1.0 + a * h))

        results[f"h_{h}"] = {
            "h": h,
            "factor_explicit": 1.0 - a * h,
            "factor_implicit": 1.0 / (1.0 + a * h),
            "t": t_grid,
            "exact": y_exact,
            "explicit": y_exp,
            "implicit": y_imp,
            "stable_explicit": h <= 2.0 / a,
            "monotonic_explicit": h < 1.0 / a,
        }

    return results


# =============================================================================
# EXPERIMENT 2: Stiff ODE Benchmark (from Lecture 2.3, Slide 15)
# y' = -50*(y - sin(t)) + cos(t),  y(0) = 1
# Exact: y(t) = sin(t) + exp(-50*t)
# =============================================================================
def run_stiff_benchmark_experiment() -> Dict[str, Any]:
    lam = -50.0
    a = -lam  # 50.0
    y0 = 1.0
    t_end = 0.6
    h_crit = 2.0 / a  # 0.04

    # Test h = 0.03 (stable) and h = 0.05 (unstable)
    test_h = [0.03, 0.05]
    results = {"h_crit": h_crit, "runs": {}}

    for h in test_h:
        n_steps = int(round(t_end / h))
        t_grid = [i * h for i in range(n_steps + 1)]
        y_exact = [math.sin(t) + math.exp(-50.0 * t) for t in t_grid]

        # Explicit Euler: y_{n+1} = y_n + h*(-50*(y_n - sin(t_n)) + cos(t_n))
        y_exp = [y0]
        for n in range(n_steps):
            tn = t_grid[n]
            yn = y_exp[-1]
            fn = -50.0 * (yn - math.sin(tn)) + math.cos(tn)
            y_next = yn + h * fn
            if abs(y_next) > 1e4:
                y_next = 1e4 if y_next > 0 else -1e4
            y_exp.append(y_next)

        # Implicit Euler:
        # y_{n+1} = y_n + h*(-50*(y_{n+1} - sin(t_{n+1})) + cos(t_{n+1}))
        # (1 + 50*h)*y_{n+1} = y_n + 50*h*sin(t_{n+1}) + h*cos(t_{n+1})
        y_imp = [y0]
        for n in range(n_steps):
            t_next = t_grid[n + 1]
            rhs = y_imp[-1] + 50.0 * h * math.sin(t_next) + h * math.cos(t_next)
            y_next = rhs / (1.0 + 50.0 * h)
            y_imp.append(y_next)

        results["runs"][f"h_{h}"] = {
            "h": h,
            "t": t_grid,
            "exact": y_exact,
            "explicit": y_exp,
            "implicit": y_imp,
            "max_err_explicit": max(abs(e - ex) for e, ex in zip(y_exp, y_exact)),
            "max_err_implicit": max(abs(i - ex) for i, ex in zip(y_imp, y_exact)),
        }

    return results


# =============================================================================
# EXPERIMENT 3: Traffic Flow ODE System Stiffness & Stability Limit
# =============================================================================
def run_traffic_stability_experiment() -> Dict[str, Any]:
    theta = load_fitted_theta()
    # At different flow levels x:
    flow_levels = [0.0, 0.5, 1.0, 1.5]
    table = []

    for x in flow_levels:
        rates = decay_rates(theta, x)
        m_rate = max(rates)
        h_lim = 2.0 / m_rate if m_rate > 0 else float("inf")
        stiff_rat = stiffness_ratio(theta, x)
        table.append({
            "flow_level": x,
            "rates": rates,
            "max_decay_rate": m_rate,
            "h_crit_hours": h_lim,
            "h_crit_minutes": h_lim * 60.0,
            "stiffness_ratio": stiff_rat,
        })

    # Test Explicit vs Implicit on a fast highway transient
    # Using observation step h = 0.25 (15 min) vs large step h = 0.60 h
    return {
        "stability_table": table,
        "normal_step_hours": 0.25,
        "is_normal_step_stable": 0.25 < min(r["h_crit_hours"] for r in table if r["flow_level"] <= 1.0),
    }


# =============================================================================
# EXPERIMENT 4: Truncation Errors (LTE, GTE, Residual) and Gronwall Bound
# =============================================================================
def run_error_and_gronwall_experiment() -> Dict[str, Any]:
    # Test on y' = -y, y(0)=1 on [0, 2]
    # Exact: y(t) = exp(-t). Exact u''(t) = exp(-t) <= M=1.0. Lipschitz L=1.0.
    t_end = 2.0
    h_steps = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    error_data = []

    for h in h_steps:
        n = int(round(t_end / h))
        y = 1.0
        max_lte = 0.0
        residual_err = 0.0
        for step in range(n):
            t_curr = step * h
            y_exact_curr = math.exp(-t_curr)
            # One-step exact update from y_exact_curr
            y_one_step = y_exact_curr + h * (-y_exact_curr)
            y_exact_next = math.exp(-(t_curr + h))
            lte = abs(y_exact_next - y_one_step)
            if lte > max_lte:
                max_lte = lte

            # Euler step
            y_next = y + h * (-y)
            # Residual error R_n = |(y_{n+1} - y_n)/h - f(t_n, y_n)|
            res = abs((y_next - y) / h - (-y))
            if res > residual_err:
                residual_err = res
            y = y_next

        gte = abs(y - math.exp(-t_end))

        # Gronwall bound: M/(2L) * (exp(L*t_end) - 1) * h
        # With M=1, L=1:
        gronwall_bound = 0.5 * (math.exp(t_end) - 1.0) * h

        error_data.append({
            "h": h,
            "max_lte": max_lte,
            "gte": gte,
            "residual_error": residual_err,
            "gronwall_bound": gronwall_bound,
        })

    return {"steps": error_data}


# =============================================================================
# RENDER VISUAL PLOT OF EXPERIMENTS (PNG)
# =============================================================================
def render_stability_plots(test_eq_res: Dict[str, Any], stiff_res: Dict[str, Any]) -> Path:
    img_w, img_h = 1200, 680
    img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
    font_head = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
    font_text = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 13)

    # Header banner
    draw.rectangle([0, 0, img_w, 65], fill=(30, 41, 59))
    draw.text((30, 10), "STABILITY & STIFFNESS NUMERICAL EXPERIMENTS", fill=(255, 255, 255), font=font_title)
    draw.text((30, 38), "Direct Verification of A-Stability, Critical Step Sizes, and Stiff Divergence", fill=(148, 163, 184), font=font_sub)

    # Panel 1: Test Equation y' = -10y (Left panel)
    p1_x0, p1_y0, p1_x1, p1_y1 = 40, 80, 580, 640
    draw.rectangle([p1_x0, p1_y0, p1_x1, p1_y1], fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.text((p1_x0 + 15, p1_y0 + 12), "1. Test Equation: y' = -10y (h_crit = 2/10 = 0.20)", fill=(15, 23, 42), font=font_head)

    # Plot area inside panel 1
    plot1_x0, plot1_y0 = p1_x0 + 50, p1_y0 + 50
    plot1_x1, plot1_y1 = p1_x1 - 25, p1_y1 - 65

    # Axes
    draw.line([plot1_x0, plot1_y1, plot1_x1, plot1_y1], fill=(100, 116, 139), width=2)
    draw.line([plot1_x0, plot1_y0, plot1_x0, plot1_y1], fill=(100, 116, 139), width=2)

    # Zero line y = 0
    y_zero_1 = plot1_y1 - int((plot1_y1 - plot1_y0) * 0.45)
    draw.line([plot1_x0, y_zero_1, plot1_x1, y_zero_1], fill=(203, 213, 225), width=1)

    def map_p1(t: float, y: float) -> Tuple[int, int]:
        px = plot1_x0 + int((t / 1.0) * (plot1_x1 - plot1_x0))
        y_clamped = max(-1.5, min(1.5, y))
        py = y_zero_1 - int((y_clamped / 1.5) * (y_zero_1 - plot1_y0))
        return px, py

    # Draw curves in panel 1
    res_stable = test_eq_res["h_0.05"]
    pts_exact = [map_p1(t, y) for t, y in zip(res_stable["t"], res_stable["exact"])]
    draw.line(pts_exact, fill=(15, 23, 42), width=3)

    pts_exp_stable = [map_p1(t, y) for t, y in zip(res_stable["t"], res_stable["explicit"])]
    draw.line(pts_exp_stable, fill=(34, 197, 94), width=3)

    res_osc = test_eq_res["h_0.15"]
    pts_exp_osc = [map_p1(t, y) for t, y in zip(res_osc["t"], res_osc["explicit"])]
    draw.line(pts_exp_osc, fill=(234, 179, 8), width=3)

    res_unstable = test_eq_res["h_0.25"]
    pts_exp_unstable = [map_p1(t, y) for t, y in zip(res_unstable["t"], res_unstable["explicit"])]
    draw.line(pts_exp_unstable, fill=(239, 68, 68), width=3)

    pts_imp = [map_p1(t, y) for t, y in zip(res_unstable["t"], res_unstable["implicit"])]
    draw.line(pts_imp, fill=(59, 130, 246), width=3)

    # Legend for Panel 1
    leg_y = plot1_y1 + 10
    draw.text((plot1_x0, leg_y), "Exact: Black | h=0.05: Green | h=0.15: Yellow", fill=(30, 41, 59), font=font_text)
    draw.text((plot1_x0, leg_y + 22), "Explicit h=0.25: RED (EXPLODES!) | Implicit h=0.25: BLUE (STABLE)", fill=(30, 41, 59), font=font_text)

    # Panel 2: Stiff ODE Benchmark (Right panel)
    p2_x0, p2_y0, p2_x1, p2_y1 = 610, 80, 1160, 640
    draw.rectangle([p2_x0, p2_y0, p2_x1, p2_y1], fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.text((p2_x0 + 15, p2_y0 + 12), "2. Stiff ODE: y' = -50(y - sin t) + cos t (h_crit = 0.04)", fill=(15, 23, 42), font=font_head)

    plot2_x0, plot2_y0 = p2_x0 + 50, p2_y0 + 50
    plot2_x1, plot2_y1 = p2_x1 - 25, p2_y1 - 65

    draw.line([plot2_x0, plot2_y1, plot2_x1, plot2_y1], fill=(100, 116, 139), width=2)
    draw.line([plot2_x0, plot2_y0, plot2_x0, plot2_y1], fill=(100, 116, 139), width=2)

    y_zero_2 = plot2_y1 - int((plot2_y1 - plot2_y0) * (1.5 / 4.0))
    draw.line([plot2_x0, y_zero_2, plot2_x1, y_zero_2], fill=(203, 213, 225), width=1)

    def map_p2(t: float, y: float) -> Tuple[int, int]:
        px = plot2_x0 + int((t / 0.6) * (plot2_x1 - plot2_x0))
        y_clamped = max(-1.5, min(2.5, y))
        py = y_zero_2 - int((y_clamped / 2.5) * (y_zero_2 - plot2_y0))
        return px, py

    stiff_003 = stiff_res["runs"]["h_0.03"]
    stiff_005 = stiff_res["runs"]["h_0.05"]

    pts_stiff_exact = [map_p2(t, y) for t, y in zip(stiff_003["t"], stiff_003["exact"])]
    draw.line(pts_stiff_exact, fill=(15, 23, 42), width=3)

    pts_stiff_exp_003 = [map_p2(t, y) for t, y in zip(stiff_003["t"], stiff_003["explicit"])]
    draw.line(pts_stiff_exp_003, fill=(34, 197, 94), width=3)

    pts_stiff_exp_005 = [map_p2(t, y) for t, y in zip(stiff_005["t"], stiff_005["explicit"])]
    draw.line(pts_stiff_exp_005, fill=(239, 68, 68), width=3)

    pts_stiff_imp_005 = [map_p2(t, y) for t, y in zip(stiff_005["t"], stiff_005["implicit"])]
    draw.line(pts_stiff_imp_005, fill=(59, 130, 246), width=3)

    leg2_y = plot2_y1 + 10
    draw.text((plot2_x0, leg2_y), "Exact: Black | Explicit h=0.03 (< h_crit): Green", fill=(30, 41, 59), font=font_text)
    draw.text((plot2_x0, leg2_y + 22), "Explicit h=0.05 (> h_crit): RED (BLOWS UP) | Implicit h=0.05: BLUE", fill=(30, 41, 59), font=font_text)

    # Save to presentation/figures/
    out_dir = Path("presentation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stiffness_and_stability_experiments.png"
    img.save(out_path, format="PNG")
    print(f"Generated photo: {out_path}")
    return out_path


def main() -> None:
    print("=" * 70)
    print("RUNNING ALL STABILITY AND STIFFNESS EXPERIMENTS (COURSE SLIDES)")
    print("=" * 70)

    print("\n--- 1. Test Equation y' = -10y ---")
    test_eq = run_test_equation_experiment()
    print("h=0.05 factor:", test_eq["h_0.05"]["factor_explicit"], "-> monotonic decay")
    print("h=0.15 factor:", test_eq["h_0.15"]["factor_explicit"], "-> oscillatory decay")
    print("h=0.25 factor:", test_eq["h_0.25"]["factor_explicit"], "-> EXPLOSIVE blowup")

    print("\n--- 2. Stiff Benchmark y' = -50(y - sin t) + cos t ---")
    stiff = run_stiff_benchmark_experiment()
    print(f"h_crit = {stiff['h_crit']:.4f}")
    print("h=0.03 explicit max error:", stiff["runs"]["h_0.03"]["max_err_explicit"])
    print("h=0.05 explicit max error:", stiff["runs"]["h_0.05"]["max_err_explicit"], "(BLOWUP)")
    print("h=0.05 implicit max error:", stiff["runs"]["h_0.05"]["max_err_implicit"], "(STABLE)")

    print("\n--- 3. Traffic Model Stability & Stiffness ---")
    traffic = run_traffic_stability_experiment()
    for row in traffic["stability_table"]:
        print(f"Flow x={row['flow_level']}: max decay rate = {row['max_decay_rate']:.2f}, h_crit = {row['h_crit_hours']:.3f} h ({row['h_crit_minutes']:.1f} min)")

    print("\n--- 4. Truncation Error & Gronwall Analysis ---")
    errors = run_error_and_gronwall_experiment()
    print(f"{'h':8s} | {'LTE':12s} | {'GTE':12s} | {'Residual':12s} | {'Gronwall Bound':14s}")
    for row in errors["steps"]:
        print(f"{row['h']:8.5f} | {row['max_lte']:12.6f} | {row['gte']:12.6f} | {row['residual_error']:12.6f} | {row['gronwall_bound']:14.6f}")

    # Render figure
    fig_path = render_stability_plots(test_eq, stiff)

    # Save summary json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "lecture_stability_experiments.json"
    with open(summary_path, "w") as f:
        json.dump({
            "test_equation": {
                k: {kk: vv for kk, vv in v.items() if kk not in ["t", "exact", "explicit", "implicit"]}
                for k, v in test_eq.items()
            },
            "stiff_benchmark": {
                "h_crit": stiff["h_crit"],
                "runs": {
                    k: {kk: vv for kk, vv in v.items() if kk not in ["t", "exact", "explicit", "implicit"]}
                    for k, v in stiff["runs"].items()
                },
            },
            "traffic_stability": traffic,
            "errors_and_gronwall": errors,
        }, f, indent=2)

    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
