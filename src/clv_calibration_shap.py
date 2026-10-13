"""
CLV Predictive Analytics — Step 1: Probability Calibration & SHAP Explainability
================================================================================
Role: AI Validation Specialist / Senior ML Engineer

Objectives:
  1. Calibrate the winning XGBoost model using group-aware cross-validation
     (StratifiedGroupKFold on account_id) to eliminate probability distortion.
  2. Measure Brier score reduction and plot reliability curves (pre vs post calibration).
  3. Generate SHAP feature attributions via TreeExplainer to quantify global
     and array-level risk drivers.
  4. Export the calibrated model artifact and SHAP summary charts.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5

_this_dir   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "src" else _this_dir
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

LABELED_CSV     = os.path.join(DATA_DIR, "clv_cleaned_labeled.csv")
RAW_MODEL_PATH  = os.path.join(MODELS_DIR, "clv_final_model.pkl")
CAL_MODEL_PATH  = os.path.join(MODELS_DIR, "clv_calibrated_model.pkl")
CAL_META_PATH   = os.path.join(MODELS_DIR, "clv_calibrated_metadata.json")

FEATURE_COLS = [
    "utilization_rate_pct",
    "current_capacity_tb",
    "profit_symlog",
    "is_unprofitable",
    "profit_per_tb_symlog",
    "days_until_contract_end",
    "expansion_High",
    "expansion_Low",
    "expansion_Medium",
    "expansion_UNKNOWN",
]

TARGET_COL = "churn_event_next_12m"
GROUP_COL  = "account_id"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA & RAW MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_data_and_model():
    print("=" * 75)
    print("STEP 1.1: LOADING LABELED DATA & TRAINED WINNING MODEL")
    print("=" * 75)

    df = pd.read_csv(LABELED_CSV)
    print(f"  Loaded labeled data: {df.shape[0]} rows x {df.shape[1]} cols")

    saved_obj = joblib.load(RAW_MODEL_PATH)
    raw_model = saved_obj["model"]
    raw_meta  = saved_obj["metadata"]
    print(f"  Loaded model: {raw_meta['model_name']} ({raw_meta['model_version']})")
    print(f"  Underlying algorithm: {type(raw_model).__name__}")
    print()
    return df, raw_model, raw_meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def run_calibration(df, raw_model):
    print("=" * 75)
    print("STEP 1.2: PROBABILITY CALIBRATION (Group-Aware Cross-Validation)")
    print("=" * 75)

    X      = df[FEATURE_COLS].values.astype(float)
    y      = df[TARGET_COL].values.astype(int)
    groups = df[GROUP_COL].values

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1. Raw Out-of-Fold Predictions
    raw_oof_proba = np.zeros(len(y))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        clone_model = joblib.load(RAW_MODEL_PATH)["model"]
        clone_model.fit(X[train_idx], y[train_idx])
        raw_oof_proba[val_idx] = clone_model.predict_proba(X[val_idx])[:, 1]

    raw_brier = brier_score_loss(y, raw_oof_proba)
    raw_prauc = average_precision_score(y, raw_oof_proba)
    raw_roc   = roc_auc_score(y, raw_oof_proba)
    static5_brier = brier_score_loss(y, np.full(len(y), 0.05))

    print(f"  Raw Model (Out-of-Fold):")
    print(f"    Brier Score: {raw_brier:.4f} (Static 5% Baseline: {static5_brier:.4f})")
    print(f"    PR-AUC:      {raw_prauc:.4f}")
    print(f"    ROC-AUC:     {raw_roc:.4f}")
    print()

    # 2. Fit CalibratedClassifierCV using Sigmoid (Platt Scaling) with Group CV
    # Note: precomputed cv_splits preserves group independence without passing groups to estimator.fit
    cv_splits = list(cv.split(X, y, groups=groups))
    cal_model = CalibratedClassifierCV(
        estimator=raw_model,
        method="sigmoid",
        cv=cv_splits
    )
    cal_model.fit(X, y)

    # Out-of-fold calibrated probabilities
    cal_oof_proba = np.zeros(len(y))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        fold_cal = CalibratedClassifierCV(
            estimator=joblib.load(RAW_MODEL_PATH)["model"],
            method="sigmoid",
            cv=3
        )
        fold_cal.fit(X[train_idx], y[train_idx])
        cal_oof_proba[val_idx] = fold_cal.predict_proba(X[val_idx])[:, 1]

    cal_brier = brier_score_loss(y, cal_oof_proba)
    cal_prauc = average_precision_score(y, cal_oof_proba)
    cal_roc   = roc_auc_score(y, cal_oof_proba)

    print(f"  Calibrated Model (Out-of-Fold):")
    print(f"    Brier Score: {cal_brier:.4f} (Error reduced by {(raw_brier - cal_brier)/raw_brier*100:.2f}%)")
    print(f"    PR-AUC:      {cal_prauc:.4f}")
    print(f"    ROC-AUC:     {cal_roc:.4f}")
    print()

    # 3. Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    prob_true_raw, prob_pred_raw = calibration_curve(y, raw_oof_proba, n_bins=8, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y, cal_oof_proba, n_bins=8, strategy="uniform")

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#e74c3c", label=f"Raw XGBoost (Brier = {raw_brier:.4f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#27ae60", label=f"Calibrated XGBoost (Brier = {cal_brier:.4f})")

    ax.set_xlabel("Mean Predicted Churn Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Fraction of Churn", fontsize=11, fontweight="bold")
    ax.set_title("Reliability Diagram: Pre- vs Post-Calibration (5-Fold CV)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    curve_path = os.path.join(REPORTS_DIR, "calibration_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"  Saved reliability curve: {curve_path}")

    # Save out-of-fold predictions CSV for downstream reporting & notebook
    oof_df = pd.DataFrame({
        "array_id": df["array_id"],
        "account_id": df["account_id"],
        "actual_churn": y,
        "raw_xgb_proba": raw_oof_proba,
        "calibrated_xgb_proba": cal_oof_proba
    })
    oof_csv_path = os.path.join(REPORTS_DIR, "clv_calibration_oof.csv")
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"  Saved calibration OOF CSV: {oof_csv_path}")

    # 4. Save Final Calibrated Model Artifact
    cal_meta = {
        "model_name":          "xgb_calibrated_sigmoid",
        "base_model":          "xgb_tuned",
        "calibration_method":  "sigmoid (Platt Scaling)",
        "brier_score_raw":     round(float(raw_brier), 6),
        "brier_score_calibrated": round(float(cal_brier), 6),
        "brier_reduction_pct": round(float((raw_brier - cal_brier)/raw_brier*100), 2),
        "static_5pct_brier":   round(float(static5_brier), 6),
        "pr_auc":              round(float(cal_prauc), 6),
        "roc_auc":             round(float(cal_roc), 6),
        "feature_cols":        FEATURE_COLS,
        "n_training_rows":     len(y),
    }

    joblib.dump({"model": cal_model, "metadata": cal_meta}, CAL_MODEL_PATH)
    with open(CAL_META_PATH, "w") as f:
        json.dump(cal_meta, f, indent=2)

    print(f"  Saved calibrated model: {CAL_MODEL_PATH}")
    print(f"  Saved calibrated metadata: {CAL_META_PATH}")
    print()
    return cal_model, cal_meta


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP EXPLAINABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(df, raw_model):
    print("=" * 75)
    print("STEP 1.3: SHAP EXPLAINABILITY ANALYSIS (TreeExplainer)")
    print("=" * 75)

    X = df[FEATURE_COLS].values.astype(float)

    # Use TreeExplainer on underlying XGBoost model
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X)

    # Global feature importance: mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_csv_path = os.path.join(REPORTS_DIR, "shap_feature_importance.csv")
    shap_df.to_csv(shap_csv_path, index=False)
    print("  Global SHAP Feature Importance Ranking:")
    for idx, row in shap_df.iterrows():
        print(f"    {idx+1:2d}. {row['feature']:<25} | Mean |SHAP| = {row['mean_abs_shap']:.5f}")
    print()
    print(f"  Saved feature importance CSV: {shap_csv_path}")

    # Plot Bar Chart of Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    y_pos = np.arange(len(shap_df))
    ax.barh(y_pos, shap_df["mean_abs_shap"][::-1], color="#2980b9", edgecolor="#1a5276", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_df["feature"][::-1], fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Churn Output)", fontsize=11, fontweight="bold")
    ax.set_title("Global SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()

    shap_bar_path = os.path.join(REPORTS_DIR, "shap_importance.png")
    plt.savefig(shap_bar_path)
    plt.close()
    print(f"  Saved SHAP importance chart: {shap_bar_path}")

    # Plot Beeswarm Summary Plot
    plt.figure(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact on Churn Risk", fontsize=13, fontweight="bold")
    plt.tight_layout()
    shap_summary_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.savefig(shap_summary_path)
    plt.close()
    print(f"  Saved SHAP beeswarm summary plot: {shap_summary_path}")
    print()

    return explainer, shap_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 75)
    print("CLV PREDICTIVE ANALYTICS — STEP 1: CALIBRATION & SHAP ANALYSIS")
    print("=" * 75 + "\n")

    df, raw_model, raw_meta = load_data_and_model()
    cal_model, cal_meta     = run_calibration(df, raw_model)
    explainer, shap_df      = run_shap_analysis(df, raw_model)

    print("=" * 75)
    print("STEP 1 COMPLETE: CALIBRATION & SHAP EXPLAINABILITY ARTIFACTS READY")
    print(f"  Calibrated Model: {CAL_MODEL_PATH}")
    print(f"  Calibration Curve: {os.path.join(REPORTS_DIR, 'calibration_curve.png')}")
    print(f"  SHAP Importance:  {os.path.join(REPORTS_DIR, 'shap_importance.png')}")
    print(f"  SHAP Summary:     {os.path.join(REPORTS_DIR, 'shap_summary.png')}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
