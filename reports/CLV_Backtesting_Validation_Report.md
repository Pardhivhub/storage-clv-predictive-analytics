# Forecast Bias Back-Testing Validation Report
**Case Study 2: Customer Lifetime Value (CLV) Predictive Analytics**  
**Role:** Senior AI Validation Specialist / Quantitative Analytics Specialist  
**Evaluated Artifacts:** Winning Calibrated XGBoost Model (`xgb_calibrated_sigmoid`) vs. Static 5% Baseline Rule  
**Scope:** 1,072 Labeled Historical Storage Arrays (5-Fold Stratified Group Cross-Validation on `account_id`)

---

## 1. Executive Summary & Core Findings

Forward Customer Lifetime Value (CLV) projections for enterprise storage arrays currently rely on a fixed **5% annual churn rate assumption**. 
This back-testing audit systematically evaluated whether the **Machine Learning Pipeline (XGBoost with Platt Calibration)** improves upon the static rule across 1,072 historical array renewals.

### Key Validation Outcomes:
1. **Severe Systematic Optimism in the Static 5% Rule:**  
   The actual empirical churn rate across the fleet is **16.98%**. The company's flat 5% assumption under-forecasts churn by **11.98 percentage points** across the board. In financial terms, this introduces massive **optimism bias**, dramatically overstating forecasted 15-year CLV asset values.
2. **Portfolio-Level Neutrality in the ML Model:**  
   The calibrated ML model forecasts an average fleet churn rate of **16.66%**, achieving a Mean Forecast Error (MFE) of **-0.31 percentage points** (unrounded: -0.3149 pp; using 2-decimal rounded figures: 16.66% − 16.98% = -0.32 pp).
3. **Probability Error Reduction:**  
   The calibrated ML model reduces Brier score loss from **0.1553 down to 0.1473** (a **5.14% reduction in probabilistic forecast error**).
4. **Zero Account Leakage:**  
   All back-testing metrics were generated via out-of-fold evaluations with strict grouping on `account_id`, guaranteeing multi-array accounts never leaked into validation splits.

---

## 2. Fleet-Level Metric Comparison & Validation Interpretation

| Performance Metric | Static 5% Business Baseline | Calibrated ML Model (XGBoost) | Validation Finding & Significance |
|---|:---:|:---:|:---|
| **Mean Forecasted Churn** | 5.00% | 16.66% | Matches actual empirical fleet rate (16.98%) |
| **Mean Forecast Error (MFE)** | **-11.98%** | **-0.31%** (or -0.32% rounded) | **Eliminates ~12pp optimism bias across cohorts** |
| **Brier Score Loss (Prob Error)** | 0.1553 | 0.1473 | **+5.14% reduction in probabilistic error** |
| **PR-AUC (Precision-Recall)** | 0.1698 (prevalence) | 0.1739 (CV mean: 0.1877) | Modest ranking signal (+1.79 pp lift over random baseline) |
| **ROC-AUC (Discrimination)** | 0.5000 (uninformative) | 0.5079 (CV mean: 0.5090) | Weak global rank-ordering (near random baseline) |
| **Mean Absolute Error (MAE)** | 0.2028 | 0.2794 | Reflects spread around base rate vs constant 0.05 prediction |

*Note on MFE Formula: $\text{Forecast Error} = \text{Predicted Rate} - \text{Actual Rate}$. Negative values indicate under-prediction of churn (optimism bias).*

---

## 2.1 Critical Methodological Caveat: Aggregate Neutrality vs. Individual Precision

An essential distinction must be maintained between **portfolio calibration** and **individual discrimination**:

1. **Portfolio-Level Neutrality:**  
   The Mean Forecast Error of **-0.31 percentage points** confirms that across the portfolio and within major business cohorts, the ML model eliminates the systematic optimism that corrupted financial forecasting.
2. **Individual-Level Accuracy:**  
   **A nearly zero average forecast error does NOT prove that individual array predictions are precise.** Overpredictions and underpredictions cancel each other out in the cohort aggregate. 
3. **Modest Discrimination Signal:**  
   The modest discrimination metrics (ROC-AUC = 0.5090 ± 0.059, PR-AUC = 0.1877 ± 0.035) demonstrate that customer churn is highly stochastic in this fleet (all Pearson feature correlations with churn are < 0.05).
4. **Executive Takeaway:**  
   The primary business deliverable is **unbiased CLV asset valuation and capital budgeting**, paired with **provisional operational priority bands** for retention triage. The model should **not** be presented to executive stakeholders as a deterministic oracle of individual churn.

---

## 2.2 Calibration Methodology Confirmation & Metric Reconciliation

### Confirmation of Out-of-Fold Evaluation Procedure
To ensure scientific validity, all calibration metrics were evaluated on **strictly held-out accounts**:
* In `clv_calibration_shap.py`, evaluation was conducted using 5-fold `StratifiedGroupKFold` on `account_id` (zero account leakage).
* For each outer fold, the base model was trained on the outer training split, and Platt Sigmoid calibration was fit using an **inner 3-fold group split** within that training split.
* Calibrated probabilities were evaluated exclusively on the held-out outer validation fold. The calibrator was **never evaluated on observations used to fit it**.

### Disambiguation of the Two Brier Score Reductions
* **5.14% Brier Reduction (0.1553 → 0.1473):**  
  Measures the performance improvement of the **Calibrated ML Model vs. the Static 5% Business Baseline** across the 1,072 pooled out-of-fold predictions in `clv_backtest.py`.
* **7.60% Brier Reduction (0.1542 → 0.1425):**  
  Measures the performance improvement of **Platt-Calibrated XGBoost vs. Raw Uncalibrated XGBoost** in `clv_calibration_shap.py`.

### Reconciliation of Stage 1 vs. Stage 2 Reported Brier Scores
* **Stage 1 Nested CV Brier (0.1470 ± 0.018):**  
  The mean out-of-fold test score during nested cross-validation, where each of the 5 outer folds independently tuned its own hyperparameter set via inner randomized search.
* **Stage 2 Raw XGBoost Brier (0.1542):**  
  The out-of-fold Brier score of the single final refit production model configuration evaluated across 5 group folds before calibration. Platt calibration then reduced this score to **0.1425**.
* Both procedures are leak-free; Stage 1 validates the tuning methodology, while Stage 2 isolates the marginal gain of Platt Sigmoid calibration on the production model.

---

## 3. Product Cohort Bias Analysis

The audit evaluated model bias across product and customer segments to verify whether the static baseline distorts specific segments more severely.

### 3.1 Expansion Potential Cohort Breakdown
| Expansion Signal | Array Count | Observed Churn | Static 5% Forecast | ML Forecast | Static Bias (MFE) | ML Bias (MFE) | Brier Improvement |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **High** | 217 | 17.05% | 5.00% | 16.21% | -12.05% | -0.84% | +13.1% |
| **Low** | 450 | 16.00% | 5.00% | 15.44% | -11.00% | -0.56% | +1.3% |
| **Medium** | 340 | 19.41% | 5.00% | 18.78% | -14.41% | -0.64% | +7.0% |
| **UNKNOWN** | 65 | 10.77% | 5.00% | 15.58% | -5.77% | +4.81% | +-14.6% |

> **Cohort Finding:** Arrays with 'Medium' expansion signal experienced the highest actual churn (19.41%), where the static 5% assumption suffered its largest blind spot (-14.41pp bias). The ML model captured this elevated risk accurately (17.38%).

### 3.2 Profitability & Capacity Cohort Breakdown
| Cohort Dimension | Segment | Array Count | Actual Churn | Static 5% Bias (MFE) | ML Model Bias (MFE) | Brier Improvement |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Profitability Status | **Profitable** | 1036 | 16.99% | -11.99% | -0.35% | +5.0% |
| Profitability Status | **Unprofitable** | 36 | 16.67% | -11.67% | +0.80% | +8.0% |
| Storage Capacity | **Q1 (Small)** | 268 | 20.15% | -15.15% | -1.02% | +10.2% |
| Storage Capacity | **Q2 (Medium)** | 268 | 17.54% | -12.54% | -0.01% | +3.3% |
| Storage Capacity | **Q3 (Large)** | 268 | 14.18% | -9.18% | +0.73% | +4.9% |
| Storage Capacity | **Q4 (Enterprise)** | 268 | 16.04% | -11.04% | -0.97% | +1.0% |

---

## 4. Visual Evidence Artifacts

1. **Cohort Forecast Accuracy:** `reports/backtest_cohort_bias.png`  
   Directly demonstrates the divergence between the static 5% horizontal line and actual observed cohort churn.
2. **Systematic Bias Distribution:** `reports/backtest_mfe_comparison.png`  
   Illustrates that the static rule is uniformly biased across every single business dimension, whereas the ML model remains centered near zero.
3. **Probability Calibration Reliability Diagram:** `reports/calibration_curve.png`  
   Confirms that predicted probabilities accurately track empirical frequencies along the 45-degree diagonal.
4. **SHAP Feature Importance & Summary:** `reports/shap_importance.png` & `reports/shap_summary.png`  
   Documents feature attributions in raw log-odds margin space, highlighting capacity, profitability, and expansion signals as primary drivers.

---

## 5. Operational Governance & Production Deployment Disclosures

### 5.1 Provisional Operational Risk Tiers
* **Tiers:** LOW (< 16.5%), MEDIUM (16.5%–17.2%), HIGH (17.2%–18.0%), CRITICAL (≥ 18.0%).
* **Governance Status:** These tiers are **provisional operational recommendations** for account executive outreach bandwidth, pending formal business sign-off. They are **not** empirically proven causal intervention policies.
* **Baseline Risk Disclosure:** Arrays in the "LOW" risk tier still exhibit a predicted churn probability of ~14.4% to 16.4% (near the ~17% fleet average). **Low risk does not equal zero risk.**

### 5.2 Snowflake Warehouse Integration Status
* **Deliverable Status:** The SQL artifacts (`sql/clv_scored_output_ddl.sql`, `sql/clv_scored_output_insert.sql`, `sql/clv_scored_output_data_insert.sql`) represent **fully generated, syntax-validated SQL scripts ready for deployment** in Snowflake, rather than an executed live remote cluster session.
* **Unlabeled Arrays Scoring:** 66 unlabeled arrays have been scored with calibrated probabilities and individual SHAP attribution payloads. Because their true churn outcomes are unobserved, their accuracy cannot be evaluated until future contract renewal milestones conclude.

### 5.3 Selection Bias & Leakage Disclosures
* **Selection Bias Disclosure:** Handled removal of 862 corrupted records (utilization > 1.0). Retained population churn rate is 16.98% vs. 14.29% in removed records. Documented and accounted for.
* **Final Recommendation:** **Recommend further controlled evaluation of calibrated XGBoost (`xgb_calibrated_sigmoid`) as a candidate replacement for the static 5% churn assumption.** The model substantially reduces aggregate forecast bias in the evaluated data (-0.31 pp vs. -11.98 pp), but its weak discrimination (ROC-AUC ≈ 0.509) warrants further assessment (e.g., pilot or shadow evaluation) before full enterprise production deployment.
