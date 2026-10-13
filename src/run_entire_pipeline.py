"""
CLV Predictive Analytics — Master Pipeline Runner
=================================================
Runs the entire end-to-end workflow sequentially:
  Stage 0: Data Cleaning & Injected Anomaly Handling (clv_data_cleaning.py)
  Stage 1: Model Training, 5-Fold Group CV & Model Selection (clv_model_training.py)
  Stage 2: Probability Calibration & SHAP Explainability (clv_calibration_shap.py)
  Stage 3: 66 Unlabeled Array Scoring & SHAP Extraction (clv_score_unlabeled.py)
  Stage 4: Forecast Bias Back-Testing & Validation Report (clv_backtest.py)
  Stage 5: Snowflake SQL Ingestion Script Generation (generate_sql_data.py)
"""

import subprocess
import sys
import os

SCRIPTS = [
    ("Stage 0: Data Cleaning", "clv_data_cleaning.py"),
    ("Stage 1: Model Training & Validation", "clv_model_training.py"),
    ("Stage 2: Calibration & SHAP Explainability", "clv_calibration_shap.py"),
    ("Stage 3: Scoring 66 Unlabeled Arrays", "clv_score_unlabeled.py"),
    ("Stage 4: Back-Testing & Validation Report", "clv_backtest.py"),
    ("Stage 5: Snowflake Ingestion Generation", "generate_sql_data.py"),
]

def main():
    print("=" * 80)
    print("STORAGE HARDWARE CLV PREDICTIVE ANALYTICS — END-TO-END REPRODUCIBLE PIPELINE")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for name, script in SCRIPTS:
        print(f"\n>>> EXECUTING: {name} ({script})")
        print("-" * 80)
        src_path = os.path.join(base_dir, "src", script)
        script_path = src_path if os.path.exists(src_path) else os.path.join(base_dir, script)
        res = subprocess.run([sys.executable, script_path], cwd=base_dir)
        if res.returncode != 0:
            print(f"FAILED at {name} (exit code {res.returncode})")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("ALL STAGES COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
