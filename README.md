# Storage Fleet Analytics: Churn Calibration & 15-Year Valuation

## 12. Technical Handover Documentation

### Purpose
Replace the flat 5% churn assumption with a calibrated, data-driven forecast. Score arrays for churn,
quantify the effect on 15-year CLV, and load the results into Snowflake.

### Project structure
| Path | Contents |
|---|---|
| `data/Case_Study_2_CLV_Predictive_Analytics.csv` | Raw input (2,000 arrays) |
| `data/clv_cleaned_labeled.csv` | Cleaned training set (1,072 arrays, labelled) |
| `data/clv_cleaned_scoring_unlabeled.csv` | Cleaned arrays with no churn label (66) |
| `data/clv_scored_66_arrays.csv` | Scored output (probability, risk bucket, top SHAP feature) |
| `reports/` | Fold metrics, calibration predictions, cohort back-test, SHAP importance (CSV) |
| `src/clv_data_cleaning.py` | Cleaning, feature engineering, train/scoring split |
| `src/clv_model_training.py` | Nested grouped CV for Random Forest and XGBoost |
| `src/clv_calibration_shap.py` | Platt calibration and TreeSHAP |
| `src/clv_score_unlabeled.py` | Scores the 66 arrays and assigns risk buckets |
| `src/clv_backtest.py` | Cohort bias back-test against the 5% rule |
| `sql/clv_scored_output_ddl.sql` | Table, staging table and summary view |
| `sql/clv_scored_output_insert.sql` | MERGE and `SP_INGEST_CLV_SCORES` procedure |
| `sql/clv_scored_output_data_insert.sql` | Insert seed for the 66 scored arrays |
| `CLV_Predictive_Analytics_Case_Study.ipynb` | Analysis and report (reads the files above) |

### How to re-run
1. Create the environment: `pip install -r requirements.txt` (pinned dependencies: `pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.5.0`, `xgboost==2.0.3`, `shap==0.53.0`, `matplotlib==3.9.0`, `joblib==1.4.2`).
2. Run the scripts in order: cleaning → model training → calibration/SHAP → scoring → back-test (or execute `python run_entire_pipeline.py`).
3. Open the notebook and run all cells. It reads the CSVs the scripts generate.
4. Run the SQL files in order: DDL → data insert → procedure. Set the database and schema first. The default is `ANALYTICS_PROD.STORAGE`.

### Model details
- **Model version:** `xgb_tuned_v2_nested_clv`
- **Primary model:** XGBoost with Platt (sigmoid) calibration, tuned with nested 3-fold CV inside 5 outer folds.
- **Validation:** `StratifiedGroupKFold` grouped by `account_id`, so no account appears in both train and validation.
- **Reference date:** 2026-10-01, used for `days_until_contract_end`.
- **Random seed:** `42`
- **Risk buckets:** LOW below 16.5%, MEDIUM 16.5% to 17.2%, HIGH 17.2% to 18.0%, CRITICAL 18.0% and above.

### Data handling decisions
- Utilization above 100% (862 rows): dropped for the main model; capping tested as a sensitivity check.
- Negative profit: signed log transform plus an `is_unprofitable` flag.
- 'UNKNOWN' expansion signal: kept as its own category.
- Missing churn labels: those rows were held out as the scoring set.

### Known limitations
1. **Little individual-level signal.** Churn ROC-AUC is about 0.51 and expansion about 0.47. The model fixes the fleet-level forecast bias but does not reliably rank individual arrays. Don't use the risk buckets for account-level actions.
2. **Calibration is optimistic.** Platt scaling was fit on the same out-of-fold predictions it is scored on, so the near-zero fleet bias (−0.12 pp) is partly by construction. The honest benchmark is the CV Brier of 0.147.
3. **The 15-year view is an assumption.** Survival uses a constant annual hazard, S(t) = (1−p)^t. The data has 12-month snapshots only, so there are no multi-year time-to-event records.
4. **CLV assumptions.** Historical 15-year profit ÷ 15 is treated as forward annual cash flow, with a 10% discount rate. The 10,000-array figure scales up linearly from a filtered 1,072-array sample.
5. **Filtered sample.** Dropping 43% of rows may bias the training set. The capped sensitivity gives 15.8% churn vs 17.0%.
6. **No expansion outcome label.** The expansion model predicts the 'High' signal, not actual conversions.
7. **The back-test is cross-sectional.** It uses out-of-fold predictions across cohorts and has no time axis. The Expansion: UNKNOWN cohort (65 arrays) shows +4.8 pp from small-sample noise.
8. **Snowflake scripts are syntax-reviewed only.** They have not been executed on a live account. Test them in a development schema first. The `CLUSTER BY` clause is only useful at much larger scale.

### Recommended next steps
Add operational telemetry (hardware error logs, firmware recency, support tickets, IOPS trends), refit with time-based validation, and re-evaluate before any per-array use.
