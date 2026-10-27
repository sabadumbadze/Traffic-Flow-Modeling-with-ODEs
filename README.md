# Traffic-Flow-Modeling-wirh-ODEs

**Traffic Flow Modeling and Prediction Using Nonlinear Ordinary Differential Equations**

**Author:** Saba Dumbadze  
**Course:** Scientific Computation  
**Institution:** Kutaisi International University, Kutaisi, Georgia  
**Date:** October 2026  

---

## 📌 Project Overview
This repository contains the complete theoretical derivations, numerical implementations, benchmark experiments, and empirical real-world model calibrations for macroscopic highway traffic flow modeling using systems of nonlinear Ordinary Differential Equations (ODEs).

The project investigates vehicular density dynamics across three interconnected highway segments governed by vehicle conservation and quadratic Greenshields outflow functions:
$$\frac{dx_i}{dt} = q_{\text{in}, i} - (\alpha_i x_i + \beta_i x_i^2), \quad i \in \{1, 2, 3\}$$

---

## 🔬 Core Components

### 1. Mathematical Foundations
- **Initial Value Problem (IVP)**: Formulated from continuous vehicle conservation and order reduction.
- **Initial vs. Spatial Boundary Conditions**: 
  - *Initial Conditions*: Temporal distribution across all segments at $t=0$.
  - *Boundary Conditions*: Inflow time series from upstream road sensors $u(t)$ and downstream corridor discharge rules.
- **Existence & Uniqueness**:
  - **Peano Existence Theorem**: Local existence under continuity (via Arzelà–Ascoli).
  - **Non-Uniqueness Counterexample**: Demonstrating that continuity alone allows branching solutions ($y' = 3y^{2/3}$ with $y_1=0$ and $y_2=t^3$).
  - **Picard–Lindelöf Theorem**: Guaranteed unique solutions through Lipschitz continuity and contraction mapping.
  - **Osgood Uniqueness Criterion** and Jacobian matrix boundedness.

### 2. Numerical Integration Schemes
Implements and evaluates five classical numerical schemes:
1. **Explicit Euler** (Forward Euler, Order 1)
2. **Implicit Euler** (Backward Euler, Order 1, A-stable)
3. **Adams–Bashforth 2** (AB2, Order 2 explicit)
4. **Adams–Moulton 2** (AM2, Order 3 implicit)
5. **Backward Differentiation Formula 2** (BDF2, Order 2 implicit, A-stable)

### 3. Scientific Computing Benchmarks (Lecture 3.7)
- **Experiment 1 (Convergence Order)**: Empirical slopes verified as $0.97$ (Explicit Euler), $1.03$ (Implicit Euler), $2.02$ (AB2), $2.02$ (BDF2), and $3.01$ (AM2).
- **Experiment 2 (Stability Limits)**: Amplification factors on $y' = -10y$ showing explicit stability bound $h < 0.20$ ($h=0.05$ decay, $h=0.15$ oscillation, $h=0.25$ explosion) vs. unconditional implicit decay.
- **Experiment 3 (Stiff Differential Equations)**: Tested on $y' = -1000(y - \cos t) - \sin t$. BDF2 ($h=0.05$, 20 steps) matches accuracy with 33x fewer steps than Explicit Euler ($h=0.0015$, 667 steps), while Explicit Euler explodes to $3.4 \cdot 10^{64}$ when $h=0.0025 > 0.002$.
- **Experiment 4 (Zero-Stability)**: Multi-step Method M with root $w=2 > 1$ explodes to $7 \cdot 10^{20}$ ($2^{80} \approx 1.2 \cdot 10^{24}$), verifying the Dahlquist Equivalence Theorem.

### 4. Empirical Model Comparison & Loss Optimization
Calibrated on real highway sensor density datasets using trajectory Mean Squared Error (MSE) loss:
$$\mathcal{L}(\theta) = \frac{1}{K \cdot M} \sum_{k=1}^K \sum_{n=1}^M \sum_{i=1}^3 \left(x_i(t_n; \theta) - x_{i,\text{obs}}(t_n)\right)^2 + \lambda_{\text{reg}} \|\theta\|_2^2$$

| Model Architecture | Params | Epochs | Best Loss | Test MAE | Test RMSE | 24h Forecast MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Greenshields (Single)** | 6 | 50 | 0.07988 | 0.3112 | 0.4390 | 0.3203 |
| **Baseline (5-Seed Ensemble)** | 6 | 50 | 0.07988 | 0.2019 | 0.2639 | 0.2086 |
| **High-Capacity Extended ODE** | 16 | 100 | 0.03119 | 0.1295 | 0.1679 | 0.1425 |
| **1,000-Epoch Full Data Model** | **16** | **1000** | **0.01330** | **0.0906** | **0.1239** | **0.0890** |

*Result:* 83.3% loss reduction from Baseline to 1,000-Epoch Full Data model, maintaining consistent sub-0.090 MAE across all forecast horizons from 15 minutes to 24 hours.

---

## 📁 Repository Structure
```
├── README.md               # Project documentation
├── .gitignore              # Ignored build and temporary files
├── report/
│   ├── report.tex          # 10-page Springer LNCS LaTeX source
│   ├── report.pdf          # Compiled 10-page research report
│   ├── llncs.cls           # Springer LNCS document class
│   ├── generate_enhanced_figures.py # Matplotlib 300 DPI figure generator
│   └── figures/            # High-resolution benchmark & loss plots
├── presentation/
│   ├── presentation.tex    # Madrid-theme LaTeX Beamer presentation
│   └── ...
├── src/                    # ODE solvers and traffic simulation routines
├── data/                   # Highway sensor density and boundary flow series
├── experiments/            # Benchmark and optimization experiment scripts
└── visualization/          # Trajectory plotting and animation utilities
```

---

## 📜 Citation & References
1. Hairer, E., Wanner, G.: *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer, Berlin (1996).
2. Butcher, J.C.: *Numerical Methods for Ordinary Differential Equations*, 3rd edn. John Wiley & Sons, Chichester (2016).
3. Botchorishvili, R.: *Scientific Computing: Ordinary Differential Equations and Numerical Experiments*. Lecture Notes, Kutaisi International University (2026).
4. Greenshields, B.D.: *A study of traffic capacity*. Highway Research Board Proceedings 14, 448–477 (1935).
