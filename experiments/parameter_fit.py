"""Parameter estimation for the three-state traffic flow ODE model.

Uses training segments to estimate theta = (alpha1, alpha2, alpha3, beta1, beta2, beta3)
subject to bounds PARAM_LOWER <= theta <= PARAM_UPPER using linear least squares
initialization followed by Gauss-Newton / Levenberg-Marquardt refinement with RK4 integration.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Tuple

from src.config import (
    FitSettings,
    OBS_INTERVAL_H,
    PARAM_LOWER,
    PARAM_NAMES,
    PARAM_UPPER,
    RESULTS_DIR,
)
from src.ivp import build_ivp
from src.pipeline import PreparedData, prepare_data
from src.rk4 import rk4_step
from src.time_stepping import integrate


def initial_linear_fit(series, interval: float = OBS_INTERVAL_H) -> Tuple[float, ...]:
    """Estimate alpha_i and beta_i by linear least squares on finite differences."""
    u = series.u
    x = series.x
    n = series.n
    inputs = [u, x[0], x[1]]
    thetas: List[Tuple[float, float]] = []

    for i in range(3):
        in_sig = inputs[i]
        state = x[i]
        A11, A12, A22 = 0.0, 0.0, 0.0
        B1, B2 = 0.0, 0.0
        for k in range(n - 1):
            dxdt = (state[k + 1] - state[k]) / interval
            c1 = in_sig[k] - state[k]
            c2 = -(state[k] ** 2)
            A11 += c1 * c1
            A12 += c1 * c2
            A22 += c2 * c2
            B1 += c1 * dxdt
            B2 += c2 * dxdt
        det = A11 * A22 - A12 * A12
        if abs(det) > 1e-12:
            a = (B1 * A22 - B2 * A12) / det
            b = (A11 * B2 - A12 * B1) / det
        else:
            a, b = 1.0, 0.1
        a = max(PARAM_LOWER[i], min(PARAM_UPPER[i], a if a > 0 else 0.5))
        b = max(PARAM_LOWER[i + 3], min(PARAM_UPPER[i + 3], b if b >= 0 else 0.0))
        thetas.append((a, b))

    alphas = [t[0] for t in thetas]
    betas = [t[1] for t in thetas]
    return tuple(alphas + betas)


def trajectory_residuals(data: PreparedData, theta: Sequence[float], max_steps_per_seg: int = 96) -> Tuple[float, List[float]]:
    """Compute sum of squared errors and residual vector across training segments."""
    total_sse = 0.0
    residuals: List[float] = []
    for seg in data.train_segments:
        stop = min(seg.stop, seg.start + max_steps_per_seg)
        if stop - seg.start < 2:
            continue
        ivp = build_ivp(data.train, theta, seg.start, stop)
        _, states = integrate(ivp, rk4_step, substeps=2)
        obs_states = [data.train.state_at(k) for k in range(seg.start, stop)]
        for sim, obs in zip(states, obs_states):
            for s_sim, s_obs in zip(sim, obs):
                diff = s_sim - s_obs
                residuals.append(diff)
                total_sse += diff * diff
    return total_sse, residuals


def fit_parameters(
    data: PreparedData,
    settings: FitSettings = FitSettings(),
) -> Tuple[float, ...]:
    """Fit theta using bounded optimization initialized by finite differences."""
    theta = list(initial_linear_fit(data.train))
    current_sse, _ = trajectory_residuals(data, theta)

    damping = settings.initial_damping
    for iteration in range(settings.max_iterations):
        grad = [0.0] * 6
        step_sizes = [max(1e-5, abs(th) * settings.fd_rel_step) for th in theta]

        # Numerical gradient of SSE
        for p in range(6):
            th_plus = list(theta)
            th_plus[p] += step_sizes[p]
            sse_plus, _ = trajectory_residuals(data, th_plus)
            grad[p] = (sse_plus - current_sse) / step_sizes[p]

        grad_norm = math.sqrt(sum(g * g for g in grad))
        if grad_norm < settings.gtol:
            break

        # Candidate step with damping
        step_taken = False
        for _ in range(5):
            candidate = list(theta)
            for p in range(6):
                cand_val = candidate[p] - (grad[p] / (grad_norm + damping)) * 0.05
                candidate[p] = max(PARAM_LOWER[p], min(PARAM_UPPER[p], cand_val))

            cand_sse, _ = trajectory_residuals(data, candidate)
            if cand_sse < current_sse:
                if (current_sse - cand_sse) / max(1.0, current_sse) < settings.ftol:
                    step_taken = True
                    theta = candidate
                    current_sse = cand_sse
                    break
                theta = candidate
                current_sse = cand_sse
                damping = max(1e-5, damping * 0.5)
                step_taken = True
                break
            damping *= 2.0

        if not step_taken:
            break

    return tuple(theta)


def run_parameter_fit_experiment() -> Dict[str, float]:
    data = prepare_data()
    print("Fitting model parameters on training segments...")
    theta = fit_parameters(data)
    result = {name: val for name, val in zip(PARAM_NAMES, theta)}
    print("Fitted parameters:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fitted_parameters.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved parameters to {out_path}")
    return result


if __name__ == "__main__":
    run_parameter_fit_experiment()
