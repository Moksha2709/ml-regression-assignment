# Polynomial Regression — Machine Learning Assignment 1

**Roll Number:** `BT2024234`  

This repository contains the complete code used to train polynomial regression models, evaluate cross-validation stability, generate diagnostic figures, and perform inference for two personalized engineering datasets:
- **Phase 1 (VAR1):** Power Plant Steam Turbine Optimization (6 operational parameters, Net Power Score prediction).
- **Phase 2 (VAR2):** Subterranean Thermal Reservoir Mapping (3 spatial coordinates, Thermal Anomaly Score prediction).

---

## Final Model Summary

| Problem | Best Degree | Features ($p$) | Model / Regularizer | Optimal $\alpha$ | 5-Fold CV MSE | 5-Fold CV $R^2$ | Train $R^2$ |
|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **VAR1 (Turbine)** | **5** | 461 | Ridge Regression ($L_2$) | 2.8480 | **0.4730** | **0.9484** | **0.9788** |
| **VAR2 (Thermal)** | **12** | 454 | Ridge Regression ($L_2$) | 0.1520 | **0.2590** | **0.9949** | **0.9966** |

*(Note: In VAR2, Degree 10 with 285 features is tightly competitive with CV MSE = 0.2596 and CV $R^2$ = 0.9949).*

---

## Directory Layout

```text
.
├── BT2024234_train_var1.csv          # Provided training data for Phase 1
├── BT2024234_test_var1.csv           # Provided test data for Phase 1
├── BT2024234_train_var2.csv          # Provided training data for Phase 2
├── BT2024234_test_var2.csv           # Provided test data for Phase 2
├── BT2024234_pred_var1.csv           # Final predictions for Phase 1 (1,000 samples)
├── BT2024234_pred_var2.csv           # Final predictions for Phase 2 (1,000 samples)
├── BT2024234_report.pdf              # Concise 5-page Technical Report (PDF)
├── run_all.py                        # Master script: sweeps, diagnostics & predictions
├── inference.py                      # Standalone inference on test CSVs
├── requirements.txt                  # Python dependencies
├── outputs/
│   ├── pipeline_results.json         # Raw numerical metrics and sweep results
│   ├── BT2024234_pred_var1.csv       # Mirrored prediction output
│   ├── BT2024234_pred_var2.csv       # Mirrored prediction output
│   └── figures/                      # High-resolution diagnostic plots (300 DPI)
│       ├── var1_degree_vs_metric.png # Figure 1: CV MSE and R² vs Degree (1 to 10)
│       ├── var1_residuals.png        # Figure 2: Residual diagnostics (VAR1)
│       ├── var1_actual_vs_predicted.png # Figure 3: Actual vs Fitted (VAR1)
│       ├── var2_degree_vs_metric.png # Figure 4: CV MSE and R² vs Degree (1 to 20)
│       ├── var2_residuals.png        # Figure 5: Residual diagnostics (VAR2)
│       └── var2_actual_vs_predicted.png # Figure 6: Actual vs Fitted (VAR2)
└── src/
    ├── __init__.py
    ├── config.py                     # Centralized paths, seeds, and hyperparameters
    ├── utils.py                      # Polynomial generation, Ridge LOO-CV & plotting
    ├── train_var1.py                 # Dedicated training pipeline for VAR1
    └── train_var2.py                 # Dedicated training pipeline for VAR2
```

---

## How to Reproduce Results

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Training, Sweeps, Plots & Predictions)
```bash
python run_all.py
```
This executes:
1. Feature combinatorial checks and OLS breakdown verification.
2. Degree sweeps (1 to 10 for VAR1, 1 to 20 for VAR2) using Ridge regression with Leave-One-Out CV for $\alpha$.
3. 5-fold cross-validation and 5-seed stability verification across seeds `[42, 123, 456, 789, 2024]`.
4. Generating all 6 high-resolution diagnostic figures in `outputs/figures/`.
5. Model refit on all 1,000 training observations and export of predictions (`BT2024234_pred_var1.csv` and `BT2024234_pred_var2.csv`).

### 3. Generate Predictions Only (Inference)
```bash
python inference.py
```
This loads test inputs directly and outputs the required 1,000 predictions with column header `y` for both variants.
