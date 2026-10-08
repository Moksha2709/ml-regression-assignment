import os
import sys
import time
import json

from src.train_var1 import main as train_var1
from src.train_var2 import main as train_var2
from src.config import OUTPUT_DIR, ROLL_NO


def main():
    t0 = time.time()
    print(f"Starting complete pipeline for {ROLL_NO}...\n")
    
    res_var1 = train_var1()
    print("\n" + "=" * 50 + "\n")
    
    res_var2 = train_var2()
    print("\n" + "=" * 50 + "\n")
    
    # Save combined results
    combined_results = {
        "roll_no": ROLL_NO,
        "var1": res_var1,
        "var2": res_var2,
        "elapsed_seconds": round(time.time() - t0, 2)
    }
    
    json_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    with open(json_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    print("\n" + "=" * 50)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {combined_results['elapsed_seconds']}s")
    print("=" * 50)
    print(f"VAR1 Optimal: Degree {res_var1['best_degree']}, Alpha {res_var1['best_alpha']:.4f}, CV MSE {res_var1['best_cv_mse']:.4f}, CV R² {res_var1['best_cv_r2']:.4f}")
    print(f"VAR2 Optimal: Degree {res_var2['best_degree']}, Alpha {res_var2['best_alpha']:.4f}, CV MSE {res_var2['best_cv_mse']:.4f}, CV R² {res_var2['best_cv_r2']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
