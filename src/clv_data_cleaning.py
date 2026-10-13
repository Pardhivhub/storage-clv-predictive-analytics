"""
CLV Predictive Analytics - Rigorous Data Auditing, Cleaning & Segregation Pipeline
===================================================================================
Role Context: AI Validation & Model Risk Governance

Explicit Mandate & Remediation Rules:
1. Utilization Out-of-Bounds (> 100% / > 1.0):
   - Programmatically audited and excluded (862 records removed as corrupt/out-of-bounds).
   - Cleaned subset strictly satisfies: `utilization_rate_pct <= 1.0` (N = 1,138).
2. Negative Net Profit (down to -$100k):
   - Continuous transformation using Signed Log (Symlog): sign(x) * ln(1 + |x|).
   - Risk indicator flag: `is_unprofitable = (profit < 0).astype(int)`.
   - Verified that transformed values are finite and well-conditioned.
3. Missing Churn Labels in 'churn_event_next_12m':
   - Dynamic quarantine executed AFTER defect 1 filtering.
   - Cleaned Labeled Training/Back-Testing Partition: N = 1,072 (890 non-churn, 182 churn).
   - Cleaned Unlabeled Production Scoring Partition: N = 66 (held for Snowflake Task 3).
4. 'UNKNOWN' Expansion Signal:
   - Treated as a distinct categorical state (informative missingness).
   - Strictly One-Hot Encoded (avoiding artificial ordinal ranking assumptions).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset and log schema."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    return df


def signed_log_transform(series: pd.Series) -> pd.Series:
    """
    Symmetric Log (Symlog) Transform: sign(x) * log1p(|x|)
    Preserves negative loss magnitude while compressing multi-million positive scale.
    """
    return np.sign(series) * np.log1p(np.abs(series))


def execute_validation_cleaning_pipeline(
    raw_df: pd.DataFrame, 
    reference_date_str: str = "2026-10-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes sequential data cleaning with strict audit accounting at every gate.
    
    Returns:
        labeled_df: Cleaned dataframe for model training & CV (N = 1,072)
        unlabeled_df: Cleaned dataframe for warehouse scoring (N = 66)
        audit_log: Comprehensive metrics dictionary for the validation report
    """
    audit_log = {}
    
    # ---------------------------------------------------------
    # STAGE 0: Initial Audit of Raw Ingested Data
    # ---------------------------------------------------------
    total_raw_rows = len(raw_df)
    audit_log["stage0_raw_row_count"] = total_raw_rows
    
    raw_over_util = raw_df["utilization_rate_pct"] > 1.0
    raw_neg_profit = raw_df["historical_15yr_net_profit_usd"] < 0
    raw_missing_target = raw_df["churn_event_next_12m"].isna()
    raw_unknown_exp = raw_df["expansion_potential_signal"] == "UNKNOWN"
    
    audit_log["stage0_raw_util_out_of_bounds"] = int(raw_over_util.sum())
    audit_log["stage0_raw_neg_profit"] = int(raw_neg_profit.sum())
    audit_log["stage0_raw_missing_target"] = int(raw_missing_target.sum())
    audit_log["stage0_raw_unknown_expansion"] = int(raw_unknown_exp.sum())
    
    # Intersectional defect analysis
    audit_log["stage0_overlap_util_and_missing_target"] = int((raw_over_util & raw_missing_target).sum())
    audit_log["stage0_overlap_util_and_neg_profit"] = int((raw_over_util & raw_neg_profit).sum())
    
    # ---------------------------------------------------------
    # STAGE 1: Defect 1 - Enforce Hard Bounds on Utilization Rate
    # ---------------------------------------------------------
    # The assignment explicitly marks utilization > 100% (e.g. 1.45) as out-of-bounds anomaly.
    # We remove these records and document the 43.1% reduction.
    df_step1 = raw_df[raw_df["utilization_rate_pct"] <= 1.0].copy()
    
    audit_log["stage1_excluded_util_count"] = int(total_raw_rows - len(df_step1))
    audit_log["stage1_remaining_rows"] = len(df_step1)
    
    # Validation assertion
    assert (df_step1["utilization_rate_pct"] <= 1.0).all(), "Gate 1 Failed: Values > 1.0 still present!"
    assert len(df_step1) == 1138, f"Expected 1,138 records after filtering, got {len(df_step1)}"
    
    # ---------------------------------------------------------
    # STAGE 2: Defect 2 - Negative Net Profit Transformation
    # ---------------------------------------------------------
    # Symmetrical log transformation and unprofitability indicator
    df_step2 = df_step1.copy()
    df_step2["is_unprofitable"] = (df_step2["historical_15yr_net_profit_usd"] < 0).astype(int)
    df_step2["profit_symlog"] = signed_log_transform(df_step2["historical_15yr_net_profit_usd"])
    
    # Capacity unit economics
    df_step2["profit_per_tb"] = df_step2["historical_15yr_net_profit_usd"] / df_step2["current_capacity_tb"]
    df_step2["profit_per_tb_symlog"] = signed_log_transform(df_step2["profit_per_tb"])
    
    audit_log["stage2_negative_profit_in_clean"] = int(df_step2["is_unprofitable"].sum())
    
    # Validation assertion
    assert np.isfinite(df_step2["profit_symlog"]).all(), "Gate 2 Failed: Non-finite profit values!"
    assert np.isfinite(df_step2["profit_per_tb_symlog"]).all(), "Gate 2 Failed: Non-finite profit_per_tb values!"
    
    # ---------------------------------------------------------
    # STAGE 3: Defect 4 - 'UNKNOWN' Categorical Encoding
    # ---------------------------------------------------------
    # Preserve UNKNOWN as an informative state and One-Hot Encode (avoiding arbitrary ordinal ranking)
    df_step3 = df_step2.copy()
    audit_log["stage3_expansion_counts"] = df_step3["expansion_potential_signal"].value_counts().to_dict()
    
    one_hot_exp = pd.get_dummies(
        df_step3["expansion_potential_signal"], 
        prefix="expansion", 
        dtype=int
    )
    df_step3 = pd.concat([df_step3, one_hot_exp], axis=1)
    
    # Ensure expected dummy columns exist
    for col in ["expansion_Low", "expansion_Medium", "expansion_High", "expansion_UNKNOWN"]:
        assert col in df_step3.columns, f"Gate 3 Failed: Missing dummy column {col}"
    
    # ---------------------------------------------------------
    # STAGE 4: Feature Engineering - Date Proximity
    # ---------------------------------------------------------
    df_step4 = df_step3.copy()
    df_step4["contract_end_date"] = pd.to_datetime(df_step4["contract_end_date"])
    ref_date = pd.to_datetime(reference_date_str)
    df_step4["days_until_contract_end"] = (df_step4["contract_end_date"] - ref_date).dt.days
    
    # ---------------------------------------------------------
    # STAGE 5: Defect 3 - Dynamic Target Isolation & Segregation
    # ---------------------------------------------------------
    # Segregate remaining 1,138 records into Labeled (training/test) and Unlabeled (Snowflake scoring)
    labeled_mask = df_step4["churn_event_next_12m"].notna()
    
    labeled_df = df_step4[labeled_mask].copy()
    unlabeled_df = df_step4[~labeled_mask].copy()
    
    labeled_df["churn_event_next_12m"] = labeled_df["churn_event_next_12m"].astype(int)
    
    # Final counts verification
    audit_log["stage5_labeled_count"] = len(labeled_df)
    audit_log["stage5_unlabeled_count"] = len(unlabeled_df)
    audit_log["stage5_labeled_churn_0"] = int((labeled_df["churn_event_next_12m"] == 0).sum())
    audit_log["stage5_labeled_churn_1"] = int((labeled_df["churn_event_next_12m"] == 1).sum())
    audit_log["stage5_labeled_churn_rate"] = float(labeled_df["churn_event_next_12m"].mean())
    audit_log["stage5_distinct_accounts"] = int(labeled_df["account_id"].nunique())
    
    # Validation assertions
    assert len(labeled_df) == 1072, f"Gate 5 Failed: Expected 1,072 labeled rows, got {len(labeled_df)}"
    assert len(unlabeled_df) == 66, f"Gate 5 Failed: Expected 66 unlabeled rows, got {len(unlabeled_df)}"
    assert labeled_df["churn_event_next_12m"].isna().sum() == 0, "Gate 5 Failed: NaN remaining in labeled target!"
    assert unlabeled_df["churn_event_next_12m"].isna().all(), "Gate 5 Failed: Target present in unlabeled partition!"
    
    return labeled_df, unlabeled_df, audit_log


def print_audit_report(audit_log: Dict[str, Any]):
    """Print formal AI Validation governance audit report."""
    print("="*75)
    print("AI MODEL VALIDATION AUDIT REPORT: INJECTED DEFECT REMEDIATION & ACCOUNTING")
    print("="*75)
    print(f"Total Ingested Raw Population:          {audit_log['stage0_raw_row_count']} arrays")
    print("-" * 75)
    print("RAW ANOMALY IDENTIFICATION (Pre-Cleaning):")
    print(f"  * Out-of-bounds Utilization (> 100%):  {audit_log['stage0_raw_util_out_of_bounds']} rows (43.1% of dataset)")
    print(f"  * Negative Net Profit (< $0):          {audit_log['stage0_raw_neg_profit']} rows")
    print(f"  * Missing Churn Target (NaN):          {audit_log['stage0_raw_missing_target']} rows")
    print(f"  * Telemetry 'UNKNOWN' Expansion:       {audit_log['stage0_raw_unknown_expansion']} rows")
    print(f"  * Intersectional Defects:              {audit_log['stage0_overlap_util_and_missing_target']} records had BOTH Util > 100% AND Missing Target")
    print(f"                                         {audit_log['stage0_overlap_util_and_neg_profit']} records had BOTH Util > 100% AND Negative Profit")
    print("-" * 75)
    print("GATE 1: DEFECT 1 REMOVAL (utilization_rate_pct <= 1.0):")
    print(f"  * Records Excluded:                    {audit_log['stage1_excluded_util_count']} rows")
    print(f"  * Clean Population Retained:           {audit_log['stage1_remaining_rows']} rows (100% validated <= 1.0)")
    print("-" * 75)
    print("GATE 2: DEFECT 2 SCALING (Negative Profit Handling):")
    print(f"  * Negative Profit Rows in Clean Data:  {audit_log['stage2_negative_profit_in_clean']} rows")
    print(f"  * Transformation Applied:              Signed Log symlog(x) = sign(x)*ln(1+|x|) + binary flag")
    print(f"  * Finite Number Check:                 PASSED (zero NaN / Inf values)")
    print("-" * 75)
    print("GATE 3: DEFECT 4 CATEGORICAL ENCODING ('UNKNOWN' Retention):")
    print(f"  * Category Breakdown:                  {audit_log['stage3_expansion_counts']}")
    print(f"  * Encoding Method:                     One-Hot (no unverified ordinal assumptions)")
    print("-" * 75)
    print("GATE 4: DEFECT 3 SEGREGATION (Target Quarantine):")
    print(f"  * Labeled Training Cohort:             {audit_log['stage5_labeled_count']} rows")
    print(f"      - Class 0 (Renewed / Non-Churn):   {audit_log['stage5_labeled_churn_0']} ({1 - audit_log['stage5_labeled_churn_rate']:.2%})")
    print(f"      - Class 1 (Churned):               {audit_log['stage5_labeled_churn_1']} ({audit_log['stage5_labeled_churn_rate']:.2%})")
    print(f"      - Class Imbalance Ratio:           ~4.89 : 1")
    print(f"      - Distinct Accounts:               {audit_log['stage5_distinct_accounts']}")
    print(f"  * Unlabeled Production Scoring Cohort: {audit_log['stage5_unlabeled_count']} rows (reserved for Task 3 Snowflake)")
    print("="*75)


if __name__ == "__main__":
    _this_dir   = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "src" else _this_dir
    DATA_DIR    = os.path.join(BASE_DIR, "data")
    
    raw_path = os.path.join(DATA_DIR, "Case_Study_2_CLV_Predictive_Analytics.csv")
    raw_df = load_raw_data(raw_path)
    labeled_df, unlabeled_df, audit_log = execute_validation_cleaning_pipeline(raw_df)
    print_audit_report(audit_log)
    
    # Save partitioned clean datasets
    labeled_df.to_csv(os.path.join(DATA_DIR, "clv_cleaned_labeled.csv"), index=False)
    unlabeled_df.to_csv(os.path.join(DATA_DIR, "clv_cleaned_scoring_unlabeled.csv"), index=False)
    print("[SUCCESS] Verified cleaned datasets exported successfully.")
