import os
import numpy as np

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLL_NO = "BT2024234"

DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

TRAIN_VAR1 = os.path.join(DATA_DIR, f"{ROLL_NO}_train_var1.csv")
TEST_VAR1 = os.path.join(DATA_DIR, f"{ROLL_NO}_test_var1.csv")
TRAIN_VAR2 = os.path.join(DATA_DIR, f"{ROLL_NO}_train_var2.csv")
TEST_VAR2 = os.path.join(DATA_DIR, f"{ROLL_NO}_test_var2.csv")

PRED_VAR1 = os.path.join(OUTPUT_DIR, f"{ROLL_NO}_pred_var1.csv")
PRED_VAR2 = os.path.join(OUTPUT_DIR, f"{ROLL_NO}_pred_var2.csv")
ROOT_PRED_VAR1 = os.path.join(BASE_DIR, f"{ROLL_NO}_pred_var1.csv")
ROOT_PRED_VAR2 = os.path.join(BASE_DIR, f"{ROLL_NO}_pred_var2.csv")

# Random seed for reproducibility
RANDOM_SEED = 42
CV_SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5

# Candidate degrees for each problem
VAR1_DEGREES = list(range(1, 11))
VAR2_DEGREES = list(range(1, 21))

# Ridge regularization alphas
RIDGE_ALPHAS = np.logspace(-4, 5, 100)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
