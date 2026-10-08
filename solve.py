"""Polynomial regression assignment - roll no BT2024234.

Run:  python solve.py
CSVs must be in the same directory as this script (flat layout, no subfolder).
Outputs:
  BT2024234_pred_var1.csv, BT2024234_pred_var2.csv   (submission files)
  results/results.json, results/*.png                (used by make_report.py)
"""
import json
import warnings
from math import comb
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROLL_NO = "BT2024234"
ROOT = Path(__file__).resolve().parent
# CSVs are in the same folder as this script (flat layout, no subfolder needed)
DATA_DIR = ROOT
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
# Wider, finer alpha grid: Ridge needs larger alphas at high degrees
ALPHAS = np.logspace(-4, 6, 200)
CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)


def build_model(degree, kind):
    """Full polynomial (all cross terms, total degree <= degree) -> scale -> linear fit."""
    if kind == "ols":
        reg = LinearRegression()
    elif kind == "ridge":
        reg = RidgeCV(alphas=ALPHAS, cv=None)  # uses LOO-CV internally, very fast
    elif kind == "lasso":
        reg = LassoCV(alphas=ALPHAS, cv=5, max_iter=10000, random_state=SEED)
    else:
        raise ValueError(kind)
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        reg,
    )


def sweep(X, y, degree_limit, kind):
    """Repeated 5-fold CV for each degree. Returns list of dicts."""
    rows = []
    n_in = X.shape[1]
    for d in range(1, degree_limit + 1):
        cv = cross_validate(
            build_model(d, kind), X, y, cv=CV,
            scoring=("r2", "neg_mean_squared_error"), return_train_score=True,
        )
        row = {
            "degree": d,
            "n_features": comb(n_in + d, d) - 1,
            "cv_mse": float(-cv["test_neg_mean_squared_error"].mean()),
            "cv_r2": float(cv["test_r2"].mean()),
            "train_mse": float(-cv["train_neg_mean_squared_error"].mean()),
        }
        rows.append(row)
        print(f"  [{kind:5s}] deg {d:2d} feats {row['n_features']:5d} "
              f"CV MSE {row['cv_mse']:.4f} R2 {row['cv_r2']:.4f}")
    return rows


def shift_check(X, y, degrees, kind):
    """Extrapolation check: train on inner 70% (by ||x||), validate on outer 30%."""
    r = np.linalg.norm(X, axis=1)
    idx = np.argsort(r)
    cut = int(0.7 * len(r))
    tr, va = idx[:cut], idx[cut:]
    out = []
    for d in degrees:
        m = build_model(d, kind).fit(X[tr], y[tr])
        p = m.predict(X[va])
        out.append({"degree": d, "mse": float(mean_squared_error(y[va], p)),
                    "r2": float(r2_score(y[va], p))})
    return out


def solve_variant(variant, degree_limit):
    print(f"\n=== {variant} ===")
    train = pd.read_csv(DATA_DIR / f"{ROLL_NO}_train_{variant}.csv")
    test  = pd.read_csv(DATA_DIR / f"{ROLL_NO}_test_{variant}.csv")
    X = train.drop(columns="y").values
    y = train["y"].values
    Xt = test[train.drop(columns="y").columns].values

    # 1) Hold-out (20%) kept untouched, used only once for an honest accuracy estimate.
    X_dev, X_ho, y_dev, y_ho = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 2) Degree sweep with CV on the 80% dev part.
    #    Three families: plain OLS, Ridge (L2), Lasso (L1).
    ols   = sweep(X_dev, y_dev, degree_limit, "ols")
    ridge = sweep(X_dev, y_dev, degree_limit, "ridge")
    lasso = sweep(X_dev, y_dev, degree_limit, "lasso")

    # 3) Select by lowest CV MSE across all families and all degrees.
    best_ols   = min(ols,   key=lambda r: r["cv_mse"])
    best_ridge = min(ridge, key=lambda r: r["cv_mse"])
    best_lasso = min(lasso, key=lambda r: r["cv_mse"])
    candidates = [("ols", best_ols), ("ridge", best_ridge), ("lasso", best_lasso)]
    kind, best = min(candidates, key=lambda t: t[1]["cv_mse"])
    print(f"  best OLS   : deg {best_ols['degree']} CV MSE {best_ols['cv_mse']:.4f}")
    print(f"  best Ridge : deg {best_ridge['degree']} CV MSE {best_ridge['cv_mse']:.4f}")
    print(f"  best Lasso : deg {best_lasso['degree']} CV MSE {best_lasso['cv_mse']:.4f}")
    print(f"  SELECTED   : {kind} degree {best['degree']}")

    # 4) Honest hold-out score of the selected model (trained on dev part only).
    m = build_model(best["degree"], kind).fit(X_dev, y_dev)
    p = m.predict(X_ho)
    holdout = {"mse": float(mean_squared_error(y_ho, p)), "r2": float(r2_score(y_ho, p))}
    print(f"  hold-out   : MSE {holdout['mse']:.4f} R2 {holdout['r2']:.4f}")
    # Also hold-out for the best plain-OLS model (for the report comparison).
    m_o = build_model(best_ols["degree"], "ols").fit(X_dev, y_dev)
    p_o = m_o.predict(X_ho)
    holdout_ols = {"degree": best_ols["degree"], "mse": float(mean_squared_error(y_ho, p_o)),
                   "r2": float(r2_score(y_ho, p_o))}

    # 5) Full-data CV sanity check — confirms degree generalises on all 1000 rows.
    full_cv_result = cross_validate(
        build_model(best["degree"], kind), X, y, cv=CV,
        scoring=("r2", "neg_mean_squared_error"),
    )
    full_cv = {
        "cv_mse": float(-full_cv_result["test_neg_mean_squared_error"].mean()),
        "cv_r2":  float(full_cv_result["test_r2"].mean()),
    }
    print(f"  full-data CV: MSE {full_cv['cv_mse']:.4f} R2 {full_cv['cv_r2']:.4f}")

    # 6) Extrapolation check (matters for var1: test has many more |x|=1 points).
    shift_degs = sorted({max(1, best["degree"] - 1), best["degree"], best["degree"] + 1,
                         best_ols["degree"]})
    shift = {"ols": shift_check(X, y, shift_degs, "ols"),
             "ridge": shift_check(X, y, shift_degs, "ridge")}

    # 7) Refit chosen model on ALL training data and predict test.
    final = build_model(best["degree"], kind).fit(X, y)
    pred  = final.predict(Xt)
    last  = final[-1]
    if kind == "ridge":
        alpha = float(last.alpha_)
    elif kind == "lasso":
        alpha = float(last.alpha_)
    else:
        alpha = None
    out = ROOT / f"{ROLL_NO}_pred_{variant}.csv"
    pd.DataFrame({"y": pred}).to_csv(out, index=False)
    print(f"  saved {out.name}  ({len(pred)} rows, alpha={alpha})")

    # 8) U-curve plot.
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r["degree"] for r in ols],   [r["cv_mse"] for r in ols],   "o-", label="OLS (CV)")
    # train MSE is clipped at 1e-3 so near-zero values do not squash the CV curves
    ax.plot([r["degree"] for r in ols],   [max(r["train_mse"], 1e-3) for r in ols], "s--", alpha=.6,
            label="OLS (train, floor 1e-3)")
    ax.plot([r["degree"] for r in ridge], [r["cv_mse"] for r in ridge], "^-", label="Ridge (CV)")
    ax.plot([r["degree"] for r in lasso], [r["cv_mse"] for r in lasso], "D-", label="Lasso (CV)", alpha=0.8)
    ax.axvline(best["degree"], color="gray", ls=":", label=f"chosen: {kind} deg {best['degree']}")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 100 if variant == "var1" else 1e11)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(f"{variant}: error vs degree")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ucurve_{variant}.png", dpi=170)
    plt.close(fig)

    # 9) Predicted vs actual on hold-out.
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.scatter(y_ho, p, s=10, alpha=.7)
    lo, hi = min(y_ho.min(), p.min()), max(y_ho.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_title(f"{variant}: hold-out")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"holdout_{variant}.png", dpi=170)
    plt.close(fig)

    return {
        "variant": variant, "n_train": len(y), "n_test": len(Xt), "n_inputs": X.shape[1],
        "ols": ols, "ridge": ridge, "lasso": lasso,
        "selected_kind": kind, "selected_degree": best["degree"],
        "selected_cv": best, "alpha": alpha,
        "holdout": holdout, "holdout_best_ols": holdout_ols,
        "full_data_cv": full_cv,
        "shift": shift,
        "pred_stats": {"mean": float(pred.mean()), "std": float(pred.std()),
                       "min": float(pred.min()), "max": float(pred.max())},
        "train_y_stats": {"mean": float(y.mean()), "std": float(y.std()),
                          "min": float(y.min()), "max": float(y.max())},
    }


if __name__ == "__main__":
    res = {"var1": solve_variant("var1", 10), "var2": solve_variant("var2", 20)}
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2))
    print("\nDone. Results in results/results.json")
