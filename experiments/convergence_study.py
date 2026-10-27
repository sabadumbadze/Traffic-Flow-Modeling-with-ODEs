"""Order-of-accuracy convergence study for explicit and implicit Euler.

Compares numerical solutions at step sizes h = 15 min / 2^level (level = 0..7)
against the high-accuracy Dormand-Prince DP5(4) reference solver.
Computes empirical orders of convergence p:
    p = log(E(h_{k-1}) / E(h_k)) / log(h_{k-1} / h_k) -> 1.0
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

from src.config import (
    ConvergenceSettings,
    OBS_INTERVAL_H,
    RESULTS_DIR,
)
from src.ivp import build_ivp
from src.pipeline import prepare_data
from src.reference_solver import solve_reference
from src.simulation import simulate
from experiments.test_validation import load_fitted_theta


def run_convergence_study(settings: ConvergenceSettings = ConvergenceSettings()) -> Dict[str, Any]:
    print("=" * 70)
    print("ORDER-OF-ACCURACY CONVERGENCE STUDY (EULER vs DP5(4) REFERENCE)")
    print("=" * 70)

    data = prepare_data()
    theta = load_fitted_theta()

    start = settings.window_start
    stop = start + settings.window_steps
    problem = build_ivp(data.train, theta, start, stop)

    print(f"Convergence Window: observation indices [{start}, {stop}] ({problem.n_intervals} intervals)")
    print(f"Time range: [{problem.t0:.2f} h, {problem.t_end:.2f} h] = {problem.t_end - problem.t0:.2f} hours")
    print(f"Initial state: {[round(v, 4) for v in problem.x0]}")

    # Compute high-accuracy reference solution at all observation points
    t_eval = [problem.t0 + k * OBS_INTERVAL_H for k in range(problem.n_intervals + 1)]
    print("\nComputing Dormand-Prince 5(4) reference trajectory (tight tolerances)...")
    ref_states = solve_reference(problem, t_eval, rtol=settings.ref_rtol, atol=settings.ref_atol)
    print(f"Reference trajectory computed: {len(ref_states)} points")

    levels = list(range(settings.n_levels))
    explicit_errors: List[float] = []
    implicit_errors: List[float] = []
    step_sizes_h: List[float] = []

    print("\n" + "-" * 75)
    print(
        f"{'Level':5s} | {'Substeps':8s} | {'h (hours)':10s} | {'Explicit Err':14s} | {'p (exp)':8s} | {'Implicit Err':14s} | {'p (imp)':8s}"
    )
    print("-" * 75)

    exp_orders: List[float] = []
    imp_orders: List[float] = []

    for lvl in levels:
        substeps = 2**lvl
        h = OBS_INTERVAL_H / substeps
        step_sizes_h.append(h)

        traj_exp = simulate(problem, "explicit", substeps=substeps)
        traj_imp = simulate(problem, "implicit", substeps=substeps)

        states_exp = traj_exp.at_observations()
        states_imp = traj_imp.at_observations()

        # Global L2 error across the trajectory
        err_exp = math.sqrt(
            sum(
                sum((se[i] - sr[i]) ** 2 for i in range(3))
                for se, sr in zip(states_exp, ref_states)
            )
            / (len(ref_states) * 3)
        )
        err_imp = math.sqrt(
            sum(
                sum((si[i] - sr[i]) ** 2 for i in range(3))
                for si, sr in zip(states_imp, ref_states)
            )
            / (len(ref_states) * 3)
        )

        explicit_errors.append(err_exp)
        implicit_errors.append(err_imp)

        p_exp_str = "-"
        p_imp_str = "-"
        if lvl > 0:
            h_ratio = step_sizes_h[-2] / step_sizes_h[-1]
            p_exp = math.log(explicit_errors[-2] / explicit_errors[-1]) / math.log(h_ratio)
            p_imp = math.log(implicit_errors[-2] / implicit_errors[-1]) / math.log(h_ratio)
            exp_orders.append(p_exp)
            imp_orders.append(p_imp)
            p_exp_str = f"{p_exp:6.3f}"
            p_imp_str = f"{p_imp:6.3f}"

        print(
            f"{lvl:5d} | {substeps:8d} | {h:10.6f} | {err_exp:14.6e} | {p_exp_str:8s} | {err_imp:14.6e} | {p_imp_str:8s}"
        )

    results = {
        "levels": levels,
        "step_sizes_h": step_sizes_h,
        "explicit_errors": explicit_errors,
        "implicit_errors": implicit_errors,
        "explicit_orders": exp_orders,
        "implicit_orders": imp_orders,
        "final_order_explicit": exp_orders[-1] if exp_orders else None,
        "final_order_implicit": imp_orders[-1] if imp_orders else None,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "convergence_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved convergence results to {out_path}")
    return results


if __name__ == "__main__":
    run_convergence_study()
