"""
CLV Predictive Analytics — Step 2: Score 66 Unlabeled Records & SHAP Payload
=============================================================================
Role: AI Validation Specialist / Senior ML Engineer

Objectives:
  1. Load the 66 cleaned unlabeled scoring arrays (data/clv_cleaned_scoring_unlabeled.csv).
  2. Apply the final calibrated model to generate calibrated churn probabilities.
  3. Assign enterprise risk buckets (LOW, MEDIUM, HIGH, CRITICAL).
  4. Compute SHAP values using TreeExplainer on each array to extract:
     - TOP_SHAP_FEATURE: The primary driver pushing this array toward churn.
     - TOP_SHAP_VALUE: The SHAP attribution magnitude.
     - SHAP_PAYLOAD: Full JSON dictionary of all feature attributions (ready for Snowflake VARIANT).
  5. Export scored outputs to CSV and JSON for downstream Snowflake ingestion (Task 3).
"""

import os
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")

_this_dir   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "src" else _this_dir
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

UNLABELED_CSV  = os.path.join(DATA_DIR, "clv_cleaned_scoring_unlabeled.csv")
CAL_MODEL_PATH = os.path.join(MODELS_DIR, "clv_calibrated_model.pkl")
RAW_MODEL_PATH = os.path.join(MODELS_DIR, "clv_final_model.pkl")

SCORED_CSV  = os.path.join(DATA_DIR, "clv_scored_66_arrays.csv")
SCORED_JSON = os.path.join(DATA_DIR, "clv_scored_66_arrays.json")

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


def assign_risk_bucket(prob):
    """Assign actionable enterprise risk tiers based on churn probability."""
    if prob < 0.15:
        return "LOW"
    elif prob < 0.22:
        return "MEDIUM"
    elif prob < 0.28:
        return "HIGH"
    else:
        return "CRITICAL"


def main():
    print("\n" + "=" * 75)
    print("CLV PREDICTIVE ANALYTICS — STEP 2: SCORING 66 UNLABELED ARRAYS")
    print("=" * 75 + "\n")

    # 1. Load data
    df = pd.read_csv(UNLABELED_CSV)
    print(f"  Loaded unlabeled scoring data: {df.shape[0]} arrays")
    assert df.shape[0] == 66, f"Expected 66 unlabeled arrays, got {df.shape[0]}"

    # 2. Load models
    cal_obj = joblib.load(CAL_MODEL_PATH)
    cal_model = cal_obj["model"]
    cal_meta  = cal_obj["metadata"]

    raw_obj = joblib.load(RAW_MODEL_PATH)
    raw_model = raw_obj["model"]
    model_version = raw_obj["metadata"]["model_version"]
    print(f"  Model Version: {model_version} (Calibrated with {cal_meta['calibration_method']})")

    # 3. Predict probabilities
    X = df[FEATURE_COLS].values.astype(float)
    cal_probs = cal_model.predict_proba(X)[:, 1]
    print(f"  Generated calibrated probabilities:")
    print(f"    Min: {cal_probs.min():.4f} | Median: {np.median(cal_probs):.4f} | Mean: {cal_probs.mean():.4f} | Max: {cal_probs.max():.4f}")

    # 4. Compute SHAP explainability per array
    explainer = shap.TreeExplainer(raw_model)
    shap_vals = explainer.shap_values(X)
    print(f"  Computed SHAP values for all 66 arrays across {len(FEATURE_COLS)} features")

    # 5. Build enriched scored dataset
    scored_records = []
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for i in range(len(df)):
        array_id   = str(df.iloc[i]["array_id"])
        account_id = str(df.iloc[i]["account_id"])
        prob       = float(cal_probs[i])
        bucket     = assign_risk_bucket(prob)

        # Array-specific SHAP values
        row_shap = shap_vals[i]
        shap_dict = {FEATURE_COLS[j]: round(float(row_shap[j]), 6) for j in range(len(FEATURE_COLS))}

        # Top risk driver: feature with highest positive SHAP push (or highest absolute)
        positive_shap = {k: v for k, v in shap_dict.items() if v > 0}
        if positive_shap:
            top_feature = max(positive_shap, key=positive_shap.get)
            top_val     = positive_shap[top_feature]
        else:
            top_feature = max(shap_dict, key=lambda k: abs(shap_dict[k]))
            top_val     = shap_dict[top_feature]

        scored_records.append({
            "ARRAY_ID":          array_id,
            "ACCOUNT_ID":        account_id,
            "CHURN_PROBABILITY": round(prob, 4),
            "RISK_BUCKET":       bucket,
            "TOP_SHAP_FEATURE":  top_feature,
            "TOP_SHAP_VALUE":    round(top_val, 4),
            "SHAP_PAYLOAD":      json.dumps(shap_dict),
            "MODEL_VERSION":     model_version,
            "SCORED_AT":         now_str,
        })

    scored_df = pd.DataFrame(scored_records)

    # 6. Save scored outputs
    scored_df.to_csv(SCORED_CSV, index=False)
    with open(SCORED_JSON, "w") as f:
        json.dump(scored_records, f, indent=2)

    print(f"  Saved scored CSV:  {SCORED_CSV}")
    print(f"  Saved scored JSON: {SCORED_JSON}")
    print()

    # 7. Print Risk Tier Summary
    print("  Risk Tier Distribution for 66 Production Arrays:")
    print("-" * 50)
    vc = scored_df["RISK_BUCKET"].value_counts()
    for bkt in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = vc.get(bkt, 0)
        pct   = count / len(scored_df) * 100
        print(f"    {bkt:<10}: {count:2d} arrays ({pct:5.1f}%)")
    print("-" * 50)

    print("\n  Top 5 High-Risk Arrays Flagged for Action:")
    print("-" * 75)
    top5 = scored_df.sort_values(by="CHURN_PROBABILITY", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"    {r['ARRAY_ID']} ({r['ACCOUNT_ID']}): Churn Prob = {r['CHURN_PROBABILITY']:.2%} | Bucket = {r['RISK_BUCKET']} | Top Driver: {r['TOP_SHAP_FEATURE']} (+{r['TOP_SHAP_VALUE']:.4f})")
    print("-" * 75)
    print()


if __name__ == "__main__":
    main()
