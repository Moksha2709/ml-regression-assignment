import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.utils import (
    load_data, generate_polynomial_features, evaluate_ridge,
    save_predictions, plot_degree_vs_metric, plot_residuals, plot_actual_vs_predicted
)


def main():
    print("=========================================================")
    print("Running VAR2 Pipeline (Subterranean Thermal Reservoir)")
    print("=========================================================")
    
    # 1. Load data
    X_train, y_train, X_test, feat_names = load_data(TRAIN_VAR2, TEST_VAR2)
    print(f"Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples:  {X_test.shape[0]}")
    
    # 2. OLS breakdown inspection
    ols_breakdowns = {}
    print("\nOLS Baseline vs High Degree Dimensionality:")
    for d in [1, 3, 5, 7, 9, 10, 12, 14]:
        X_p, _, _ = generate_polynomial_features(X_train, None, d)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        ols_score = -cross_val_score(LinearRegression(), X_p, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
        ols_breakdowns[d] = float(ols_score)
        print(f"  OLS Degree {d:2d} (p={X_p.shape[1]:4d}): CV MSE = {ols_score:.4f}")
    
    # 3. Polynomial degree sweep with Ridge
    print("\nScanning Degrees 1 to 20 with Ridge:")
    sweep_results = []
    best_deg = None
    best_mse = float('inf')
    best_alpha = None
    best_r2 = None
    
    degs, mses, r2s = [], [], []
    
    for deg in VAR2_DEGREES:
        X_poly, _, _ = generate_polynomial_features(X_train, None, deg)
        n_features = X_poly.shape[1]
        alpha, mse, r2 = evaluate_ridge(X_poly, y_train, RIDGE_ALPHAS, N_FOLDS, RANDOM_SEED)
        
        sweep_results.append({
            "degree": deg,
            "features": n_features,
            "alpha": alpha,
            "cv_mse": mse,
            "cv_r2": r2
        })
        degs.append(deg)
        mses.append(mse)
        r2s.append(r2)
        
        print(f"  Degree {deg:2d} | Features: {n_features:4d} | Best Alpha: {alpha:8.4f} | CV MSE: {mse:7.4f} | CV R²: {r2:7.4f}")
        
        if mse < best_mse:
            best_mse = mse
            best_deg = deg
            best_alpha = alpha
            best_r2 = r2
            
    print(f"\n>>> Selected VAR2 Optimal Degree: {best_deg} (Alpha = {best_alpha:.4f}, CV MSE = {best_mse:.4f}, CV R² = {best_r2:.4f})")
    
    # 4. Multi-seed stability analysis
    print("\nMulti-Seed Stability Verification (5 seeds x 5-fold CV):")
    X_poly_best, X_test_poly_best, _ = generate_polynomial_features(X_train, X_test, best_deg)
    seed_scores = {}
    for s in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        s_mse = -cross_val_score(Ridge(alpha=best_alpha), X_poly_best, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
        seed_scores[s] = float(s_mse)
        print(f"  Seed {s:5d}: CV MSE = {s_mse:.4f}")
        
    seed_vals = list(seed_scores.values())
    seed_mean, seed_std = float(np.mean(seed_vals)), float(np.std(seed_vals))
    print(f"  Mean ± Std: {seed_mean:.4f} ± {seed_std:.4f} (CV Std/Mean = {100*seed_std/seed_mean:.2f}%)")
    
    # 5. Final model fit on full training set
    final_model = Ridge(alpha=best_alpha).fit(X_poly_best, y_train)
    train_preds = final_model.predict(X_poly_best)
    train_mse = float(mean_squared_error(y_train, train_preds))
    train_r2 = float(r2_score(y_train, train_preds))
    print(f"\nFinal Model Fit on Full Train: Train MSE = {train_mse:.4f}, Train R² = {train_r2:.4f}")
    
    # 6. Diagnostic plots
    fig4_path = os.path.join(FIGURE_DIR, "var2_degree_vs_metric.png")
    fig5_path = os.path.join(FIGURE_DIR, "var2_residuals.png")
    fig6_path = os.path.join(FIGURE_DIR, "var2_actual_vs_predicted.png")
    
    plot_degree_vs_metric(degs, mses, r2s, best_deg, fig4_path, "VAR2 (Thermal Reservoir)")
    plot_residuals(y_train, train_preds, fig5_path, "VAR2")
    plot_actual_vs_predicted(y_train, train_preds, fig6_path, "VAR2", train_r2)
    print("Generated Figures 4, 5, and 6.")
    
    # 7. Test inference & distribution checks
    test_preds = final_model.predict(X_test_poly_best)
    save_predictions(test_preds, PRED_VAR2)
    save_predictions(test_preds, ROOT_PRED_VAR2)
    
    y_tr_range = float(np.max(y_train) - np.min(y_train))
    y_te_range = float(np.max(test_preds) - np.min(test_preds))
    range_ratio = y_te_range / y_tr_range if y_tr_range > 0 else 1.0
    
    dist_stats = {
        "train_mean": float(np.mean(y_train)),
        "test_mean": float(np.mean(test_preds)),
        "train_std": float(np.std(y_train)),
        "test_std": float(np.std(test_preds)),
        "train_min": float(np.min(y_train)),
        "train_max": float(np.max(y_train)),
        "test_min": float(np.min(test_preds)),
        "test_max": float(np.max(test_preds)),
        "range_ratio": range_ratio
    }
    print(f"Distribution: Train Mean={dist_stats['train_mean']:.4f}, Test Mean={dist_stats['test_mean']:.4f}, Range Ratio={range_ratio:.2f}x")
    
    return {
        "variant": "VAR2",
        "sweep_table": sweep_results,
        "best_degree": best_deg,
        "best_alpha": best_alpha,
        "best_cv_mse": best_mse,
        "best_cv_r2": best_r2,
        "ols_breakdowns": ols_breakdowns,
        "seed_scores": seed_scores,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "train_mse": train_mse,
        "train_r2": train_r2,
        "dist_stats": dist_stats
    }


if __name__ == "__main__":
    main()
