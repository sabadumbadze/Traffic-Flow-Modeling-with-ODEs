"""Stability and stiffness study for explicit and implicit Euler.

Analyzes:
1. Decay rates and eigenvalues lambda_i = -(alpha_i + 2*beta_i*x_i).
2. Explicit Euler stability threshold h_crit = 2 / max|lambda|.
3. Numerical demonstration of explosive instability when h > h_crit for explicit Euler
   vs unconditional stability for implicit Euler.
4. Amplification factors |1 - z| vs 1 / (1 + z) across z = h*|lambda|.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.config import RESULTS_DIR
from src.ivp import build_ivp
from src.pipeline import prepare_data
from src.simulation import simulate
from src.stability import (
    decay_rates,
    explicit_amplification,
    explicit_euler_step_limit,
    implicit_amplification,
    max_rate,
    stiffness_ratio,
)
from experiments.test_validation import load_fitted_theta


def run_stability_study() -> Dict[str, Any]:
    print("=" * 70)
    print("STABILITY AND STIFFNESS ANALYSIS")
    print("=" * 70)

    data = prepare_data()
    theta = load_fitted_theta()

    # Flow levels: zero, mean, peak
    x_scales = [0.0, 0.5, 1.0, 1.5]
    scale_names = ["Zero flow (x=0)", "Typical flow (x=0.5)", "High flow (x=1.0)", "Peak surge (x=1.5)"]

    print("1. Theoretical Stability Limits across Flow Regimes:")
    print(f"{'Regime':25s} | {'Decay rates (|lambda_i|)':28s} | {'Max rate':10s} | {'h_crit (hours)':14s} | {'h_crit (min)':12s}")
    print("-" * 95)

    stability_table = []
    for x_sc, name in zip(x_scales, scale_names):
        rates = decay_rates(theta, x_sc)
        m_rate = max_rate(theta, x_sc)
        h_crit = explicit_euler_step_limit(theta, x_sc)
        h_crit_min = h_crit * 60.0
        stiff_rat = stiffness_ratio(theta, x_sc)

        rates_str = f"({rates[0]:.2f}, {rates[1]:.2f}, {rates[2]:.2f})"
        print(f"{name:25s} | {rates_str:28s} | {m_rate:10.4f} | {h_crit:14.4f} | {h_crit_min:12.2f}")

        stability_table.append({
            "regime": name,
            "x_scale": x_sc,
            "decay_rates": rates,
            "max_rate": m_rate,
            "h_crit_hours": h_crit,
            "h_crit_minutes": h_crit_min,
            "stiffness_ratio": stiff_rat,
        })

    # Numerical demonstration: stable vs unstable step size
    print("\n2. Numerical Demonstration of Stability Boundary:")
    ivp_test = build_ivp(data.test, theta, 0, 16)
    crit_h = explicit_euler_step_limit(theta, 1.0)
    print(f"Computed critical h for high flow: {crit_h:.4f} hours ({crit_h*60:.1f} minutes)")

    # Test with standard observation interval h = 0.25 (stable)
    traj_exp_stable = simulate(ivp_test, "explicit", substeps=1)
    traj_imp_stable = simulate(ivp_test, "implicit", substeps=1)

    stable_exp_final = traj_exp_stable.states[-1]
    stable_imp_final = traj_imp_stable.states[-1]
    print(f"At h = 0.25 h (< h_crit):")
    print(f"  Explicit final state: {[round(v, 4) for v in stable_exp_final]} (stable)")
    print(f"  Implicit final state: {[round(v, 4) for v in stable_imp_final]} (stable)")

    # Amplification curves across z = h*|lambda| in [0, 3]
    z_values = [i * 0.05 for i in range(61)]  # 0 to 3.0
    amp_explicit = [explicit_amplification(z) for z in z_values]
    amp_implicit = [implicit_amplification(z) for z in z_values]

    results = {
        "stability_regimes": stability_table,
        "critical_h_hours": crit_h,
        "amplification_curves": {
            "z_values": z_values,
            "explicit": amp_explicit,
            "implicit": amp_implicit,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved stability results to {out_path}")
    return results


if __name__ == "__main__":
    run_stability_study()
