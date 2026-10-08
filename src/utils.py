import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """Load train and test CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('y', axis=1).values
    y_train = train_df['y'].values
    X_test = test_df.values
    feature_names = [col for col in train_df.columns if col != 'y']
    
    return X_train, y_train, X_test, feature_names


def generate_polynomial_features(X_train, X_test, degree):
    """Expand features up to the specified degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test) if X_test is not None else None
    return X_train_poly, X_test_poly, poly


def evaluate_ridge(X_poly, y, alphas, n_folds=5, seed=42):
    """Ridge regression with LOO-CV to select best alpha, then 5-fold CV score."""
    rcv = RidgeCV(alphas=alphas, cv=None).fit(X_poly, y)
    best_alpha = float(rcv.alpha_)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    model = Ridge(alpha=best_alpha)
    mse = float(-cross_val_score(model, X_poly, y, cv=kf, scoring='neg_mean_squared_error').mean())
    r2 = float(cross_val_score(model, X_poly, y, cv=kf, scoring='r2').mean())
    return best_alpha, mse, r2


def save_predictions(y_pred, output_path):
    """Save 1D prediction array to CSV matching requirements."""
    df = pd.DataFrame({"y": y_pred})
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path} ({len(df)} rows)")


def plot_degree_vs_metric(degrees, mses, r2s, best_deg, out_path, var_name):
    """Plot CV MSE and R2 across candidate degrees with professional publication aesthetic."""
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6), dpi=300)
    
    col_err = '#3b82f6'   # Sapphire blue
    col_fit = '#10b981'   # Emerald green
    
    ax1.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#ffffff')
    
    ax1.set_xlabel('Polynomial Order ($d$)', fontsize=10.5, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Empirical 5-Fold CV MSE (log scale)', color=col_err, fontsize=10.5, fontweight='600')
    line1 = ax1.plot(degrees, mses, marker='o', color=col_err, linewidth=2.2, markersize=5.5, label='CV Loss (MSE)')
    ax1.tick_params(axis='y', labelcolor=col_err, labelsize=9)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cross-Validated $R^2$ Score', color=col_fit, fontsize=10.5, fontweight='600')
    line2 = ax2.plot(degrees, r2s, marker='^', color=col_fit, linewidth=2.0, markersize=5.5, linestyle='-.', label='Validation $R^2$')
    ax2.tick_params(axis='y', labelcolor=col_fit, labelsize=9)
    
    # Selected degree threshold
    vline = ax1.axvline(best_deg, color='#e11d48', linestyle='--', linewidth=1.8, label=f'Chosen $d^* = {best_deg}$')
    
    lines = line1 + line2 + [vline]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5)
    
    ax1.set_title(f'{var_name} — Model Capacity Optimization Curve', fontsize=11.5, fontweight='700', color='#0f172a', pad=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_path, var_name):
    """Plot residuals vs fitted values and distribution with modern styling."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # Residuals vs Fitted
    ax1.set_facecolor('#fafafa')
    ax1.scatter(y_pred, residuals, alpha=0.55, edgecolors='none', color='#475569', s=16)
    ax1.axhline(0, color='#e11d48', linestyle='--', linewidth=1.4)
    ax1.set_xlabel(r'Model Inferred Values ($\hat{y}$)', fontsize=9.5, fontweight='600', color='#1e293b')
    ax1.set_ylabel(r'Residual Deviations ($y - \hat{y}$)', fontsize=9.5, fontweight='600', color='#1e293b')
    ax1.set_title('Residual Scatter (Homoscedasticity)', fontsize=10.5, fontweight='700', color='#0f172a')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    ax1.tick_params(labelsize=8.5)
    
    # Histogram of Residuals
    ax2.set_facecolor('#fafafa')
    ax2.hist(residuals, bins=28, color='#64748b', edgecolor='#334155', alpha=0.75, density=True)
    mu, std = float(np.mean(residuals)), float(np.std(residuals))
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 120)
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax2.plot(x, p, color='#0284c7', linewidth=1.8, label=rf'$\mu={mu:.2f}, \sigma={std:.2f}$')
    ax2.set_xlabel('Error Magnitude', fontsize=9.5, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Probability Density', fontsize=9.5, fontweight='600', color='#1e293b')
    ax2.set_title('Error Density vs Gaussian Profile', fontsize=10.5, fontweight='700', color='#0f172a')
    ax2.legend(fontsize=8, loc='upper right', frameon=True, facecolor='#ffffff')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    ax2.tick_params(labelsize=8.5)
    
    fig.suptitle(f'{var_name} — Error Structure & Diagnostic Verification', fontsize=11.5, fontweight='700', color='#0f172a', y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, out_path, var_name, r2_val):
    """Plot ground truth against model predictions."""
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')
    
    ax.scatter(y_true, y_pred, alpha=0.55, edgecolors='none', color='#0d9488', s=16)
    
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color='#e11d48', linestyle='--', linewidth=1.4, label=r'Identity Baseline ($y = \hat{y}$)')
    
    ax.set_xlabel('Observed Ground Truth ($y$)', fontsize=9.5, fontweight='600', color='#1e293b')
    ax.set_ylabel(r'Reconstructed Prediction ($\hat{y}$)', fontsize=9.5, fontweight='600', color='#1e293b')
    ax.set_title(f'{var_name}: Fitted Consistency ($R^2 = {r2_val:.4f}$)', fontsize=10.5, fontweight='700', color='#0f172a')
    ax.legend(fontsize=8.5, loc='upper left', frameon=True, facecolor='#ffffff')
    ax.grid(True, linestyle=':', alpha=0.6, color='#94a3b8')
    ax.tick_params(labelsize=8.5)
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
