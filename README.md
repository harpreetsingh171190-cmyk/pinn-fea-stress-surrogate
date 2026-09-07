# Physics-Informed ML Surrogate for 2D Structural Stress Concentration

An end-to-end Physics-Informed Machine Learning (PIML) surrogate framework engineered to predict 2D Von Mises stress fields and critical stress concentration factors ($K_t$) for finite-width isotropic plates with a central circular hole subjected to uniaxial tension.

---

## 📌 Project Overview
Standard Finite Element Analysis (FEA) provides high-fidelity structural simulations but is computationally expensive for rapid design optimization and real-time stress monitoring. This project bridges classical solid mechanics and deep learning by:
1. **Accelerating Inference:** Evaluating spatial stress distributions in milliseconds (<15 ms) versus minutes in high-density mesh FEA solvers.
2. **Embedding Analytical Mechanics:** Enforcing Kirsch equations and Peterson finite-width corrections as inductive biases within feature engineering and network loss objectives.
3. **High Critical-Zone Fidelity:** Ensuring tight bounded error (<2%) in the critical notch stress concentration zone ($x/r \le 1.15$).

---

## ⚙️ Theoretical Formulation & Governing Equations

Consider a flat plate of width $W$, thickness $t$, and hole diameter $d = 2r$ subjected to remote tensile force $F$.

The nominal remote tensile stress applied across the gross cross-section is:
$$\sigma_\infty = \frac{F}{W \cdot t}$$

### 1. Peterson Finite-Width Correction ($K_t$)
For an infinite plate, classical elasticity dictates $K_t = 3.0$. For finite-width strips with ratio $d/W \in [0.10, 0.45]$, Peterson's net stress concentration empirical polynomial governs the peak boundary stress:
$$K_t = 3.0 - 3.14 \left(\frac{d}{W}\right) + 3.667 \left(\frac{d}{W}\right)^2 - 1.527 \left(\frac{d}{W}\right)^3$$

### 2. Kirsch Stress Decay Field
Along the transverse symmetry line ($y = 0$, orthogonal to loading direction), stress components decay radially as a function of the distance ratio $\xi = x/r$:
$$\sigma_y(x) = \frac{\sigma_\infty}{2} \left[2 + \left(\frac{r}{x}\right)^2 + 3 \left(\frac{r}{x}\right)^4\right]$$
$$\sigma_x(x) = \frac{\sigma_\infty}{2} \left[3 \left(\frac{r}{x}\right)^4 - \left(\frac{r}{x}\right)^2 - 2\right]$$

The equivalent planar Von Mises stress target is calculated as:
$$\sigma_{\text{vm}} = \sqrt{\sigma_y^2 - \sigma_y \sigma_x + \sigma_x^2}$$

---

## 🏗️ Architecture & Physics Loss Formulation

The surrogate integrates a custom multi-objective loss function operating in normalized space:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{asymptote}} \mathcal{L}_{\text{asymptote}}$$

* **Data Discrepancy Loss ($\mathcal{L}_{\text{data}}$):** Standard supervised Mean Squared Error against synthetic FEA ground truth.
* **Boundary Condition Loss ($\mathcal{L}_{\text{BC}}$):** Soft constraint penalizing deviations from Peterson peak stress at the notch edge ($x/r \to 1.0$) with exponential spatial weighting:
  $$\mathcal{L}_{\text{BC}} = \frac{1}{N} \sum \exp\left(-3.5 \left(\frac{x}{r} - 1.0\right)\right) \left(\hat{\sigma} - K_t \sigma_\infty\right)^2$$
* **Far-Field Decay Loss ($\mathcal{L}_{\text{asymptote}}$):** Asymptotic penalty pulling predicted stress toward nominal stress $\sigma_\infty$ in undisturbed regions ($x/r \ge 3.0$).

---

## 📊 Benchmark & Performance Comparison

Evaluated on a strictly isolated 15% test partition ($N = 225$, stratified across geometric ratios $d/W$):

| Model Architecture | Global $R^2$ | MAE (MPa) | RMSE (MPa) | Notch Zone ($x/r < 1.3$) Mean Error |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.8124 | 48.20 | 66.12 | 18.40% |
| **Random Forest (150 trees)** | 0.9432 | 17.31 | 36.39 | 7.85% |
| **PINN (Physics-Informed NN)** | **0.9986** | **2.85** | **4.92** | **< 1.85%** |

---

## 📂 Repository Structure

```text
├── dataset/
│   └── fea_stress_surrogate_dataset.csv       # Parametric Kirsch-Peterson FEA data
├── outputs/
│   ├── stress_surrogate_diagnostics.png      # Parity, residual, and feature importance plots
│   └── stress_surrogate_test_predictions.csv  # Model inferences against test ground truth
├── stress_surrogate_baseline.py               # Baseline data pipeline, RF, and linear models
├── pinn_surrogate_training.py                 # PyTorch PINN model with multi-objective physics loss
├── stress_field_visualization.py              # 2D contour mesh generation script
└── README.md                                  # Project documentation
