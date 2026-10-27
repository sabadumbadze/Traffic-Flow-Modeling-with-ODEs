"""Master experiment and visualization runner.

Executes the complete experimental workflow:
1. Parameter estimation on training segments
2. Numerical stability and stiffness analysis
3. Order-of-accuracy convergence study (Euler vs DP5(4) reference)
4. Trajectory tracking and forecast horizon evaluation on held-out test data
5. Initial Value Problem (IVP) simulation and photo generation
6. Basin of attraction & phase-space geometric study (from docs/ Lecture 2.3)
7. SVG chart rendering in visualization/
"""

from __future__ import annotations

import sys
from experiments.parameter_fit import run_parameter_fit_experiment
from experiments.stability_study import run_stability_study
from experiments.convergence_study import run_convergence_study
from experiments.test_validation import run_test_validation_experiment
from experiments.run_ivp_simulation import run_ivp_and_render_photo
from experiments.basin_of_attraction import run_basin_of_attraction_study
from visualization.plot_results import generate_all_plots


def main() -> None:
    print("=" * 80)
    print("TRAFFIC FLOW ODE - COMPLETE SCIENTIFIC COMPUTATION EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Parameter estimation
    run_parameter_fit_experiment()
    print()

    # 2. Stability analysis
    run_stability_study()
    print()

    # 3. Order-of-accuracy convergence study
    run_convergence_study()
    print()

    # 4. Held-out test data validation
    run_test_validation_experiment()
    print()

    # 5. IVP Simulation Photo Generation
    run_ivp_and_render_photo()
    print()

    # 6. Basin of Attraction Study
    run_basin_of_attraction_study()
    print()

    # 7. Visualization rendering
    generate_all_plots()
    print()

    print("=" * 80)
    print("ALL EXPERIMENTS AND VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
