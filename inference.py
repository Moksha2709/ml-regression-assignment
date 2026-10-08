import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from src.config import TRAIN_VAR1, TEST_VAR1, TRAIN_VAR2, TEST_VAR2, PRED_VAR1, PRED_VAR2, ROOT_PRED_VAR1, ROOT_PRED_VAR2
from src.utils import load_data, save_predictions


def infer_var1(test_path=TEST_VAR1, output_path=PRED_VAR1):
    """Train optimal model (Degree 5, Ridge) and predict on test dataset."""
    print("Running inference for VAR1 (Power Plant Steam Turbine)...")
    X_train, y_train, X_test, _ = load_data(TRAIN_VAR1, test_path)
    
    # Optimal parameters for VAR1: Degree 5, Ridge alpha = 2.8480
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = Ridge(alpha=2.8480)
    model.fit(X_train_poly, y_train)
    preds = model.predict(X_test_poly)
    
    save_predictions(preds, output_path)
    save_predictions(preds, ROOT_PRED_VAR1)
    return preds


def infer_var2(test_path=TEST_VAR2, output_path=PRED_VAR2):
    """Train optimal model (Degree 10, Ridge) and predict on test dataset."""
    print("Running inference for VAR2 (Subterranean Thermal Reservoir)...")
    X_train, y_train, X_test, _ = load_data(TRAIN_VAR2, test_path)
    
    # Optimal parameters for VAR2: Degree 10, Ridge alpha = 0.0534
    poly = PolynomialFeatures(degree=10, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = Ridge(alpha=0.0534)
    model.fit(X_train_poly, y_train)
    preds = model.predict(X_test_poly)
    
    save_predictions(preds, output_path)
    save_predictions(preds, ROOT_PRED_VAR2)
    return preds


def main():
    infer_var1()
    infer_var2()
    print("All test inferences completed successfully.")


if __name__ == "__main__":
    main()
