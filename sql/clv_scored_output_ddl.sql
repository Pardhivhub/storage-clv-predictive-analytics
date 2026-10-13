-- =============================================================================
-- Snowflake DDL Creation Script: CLV_SCORED_OUTPUT
-- Case Study 2: Customer Lifetime Value (CLV) Predictive Analytics
-- Task 3: Production Warehouse Integration
-- =============================================================================
-- Purpose:
--   Enterprise table schema designed to store calibrated array-level renewal/churn
--   probabilities, risk tiers, and SHAP explainability metrics for forward 15-year
--   CLV calculations across tens of thousands of storage arrays.
--
-- Optimization & Architecture:
--   - Clustered by ACCOUNT_ID and RISK_BUCKET for micro-partition pruning.
--   - VARIANT column for semi-structured SHAP attribution payload.
--   - NOT NULL constraints on critical business attributes.
--   - Idempotent table definition with comprehensive column comments.
-- =============================================================================

USE ROLE ACCOUNTADMIN;
CREATE DATABASE IF NOT EXISTS ANALYTICS_PROD;
CREATE SCHEMA IF NOT EXISTS ANALYTICS_PROD.STORAGE;
USE SCHEMA ANALYTICS_PROD.STORAGE;

-- 1. Create Target Warehouse Table (Idempotent schema creation)
CREATE TABLE IF NOT EXISTS CLV_SCORED_OUTPUT (
    ARRAY_ID            VARCHAR(32)     NOT NULL,
    ACCOUNT_ID          VARCHAR(32)     NOT NULL,
    CHURN_PROBABILITY   NUMBER(5, 4)    NOT NULL,
    RISK_BUCKET         VARCHAR(16)     NOT NULL,
    TOP_SHAP_FEATURE    VARCHAR(64)     NOT NULL,
    TOP_SHAP_VALUE      NUMBER(8, 4)    NOT NULL,
    SHAP_PAYLOAD        VARIANT         NOT NULL,
    MODEL_VERSION       VARCHAR(32)     NOT NULL,
    SCORED_AT           TIMESTAMP_NTZ   NOT NULL,
    INSERTED_AT         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),

    -- Constraints
    CONSTRAINT PK_CLV_SCORED_OUTPUT PRIMARY KEY (ARRAY_ID)
)
CLUSTER BY (ACCOUNT_ID, RISK_BUCKET) -- Designed for multi-tenant partition pruning as fleet scales to millions of telemetry records
COMMENT = 'Production scored CLV array churn risks, risk buckets, and SHAP explainability metrics.';

-- 2. Column Documentation & Governance Metadata
COMMENT ON COLUMN CLV_SCORED_OUTPUT.ARRAY_ID          IS 'Unique identifier for the physical enterprise storage array (Primary Key).';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.ACCOUNT_ID        IS 'Enterprise parent account owning the storage array (Group identifier).';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.CHURN_PROBABILITY IS 'Calibrated 12-month annual churn probability (0.0000 to 1.0000) using Platt-calibrated XGBoost.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.RISK_BUCKET       IS 'Actionable risk tier for retention workflows: LOW, MEDIUM, HIGH, CRITICAL.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.TOP_SHAP_FEATURE  IS 'Primary behavioral or financial feature driving churn risk for this specific array.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.TOP_SHAP_VALUE    IS 'SHAP attribution value of the primary risk driver (positive values push toward churn).';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.SHAP_PAYLOAD      IS 'Semi-structured JSON payload of all 10 feature SHAP values for portal explainability.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.MODEL_VERSION     IS 'Unique model registry identifier for traceability and ML governance audit.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.SCORED_AT        IS 'UTC timestamp when the model scoring pipeline generated this prediction.';
COMMENT ON COLUMN CLV_SCORED_OUTPUT.INSERTED_AT      IS 'Warehouse ingestion audit timestamp.';

-- 3. Staging Table for Batch Loading
CREATE OR REPLACE TABLE STG_CLV_SCORED_INPUT (
    ARRAY_ID            VARCHAR(32),
    ACCOUNT_ID          VARCHAR(32),
    CHURN_PROBABILITY   NUMBER(5, 4),
    RISK_BUCKET         VARCHAR(16),
    TOP_SHAP_FEATURE    VARCHAR(64),
    TOP_SHAP_VALUE      NUMBER(8, 4),
    SHAP_PAYLOAD_RAW    VARCHAR,
    MODEL_VERSION       VARCHAR(32),
    SCORED_AT           TIMESTAMP_NTZ
)
COMMENT = 'Transient staging table for bulk file ingestion before MERGE into CLV_SCORED_OUTPUT.';

-- 4. Analytical Executive View
CREATE OR REPLACE VIEW VW_CLV_FLEET_RISK_SUMMARY AS
SELECT
    RISK_BUCKET,
    COUNT(ARRAY_ID)                     AS TOTAL_ARRAYS,
    COUNT(DISTINCT ACCOUNT_ID)           AS DISTINCT_ACCOUNTS,
    ROUND(AVG(CHURN_PROBABILITY), 4)    AS AVG_CHURN_PROBABILITY,
    ROUND(MIN(CHURN_PROBABILITY), 4)    AS MIN_CHURN_PROBABILITY,
    ROUND(MAX(CHURN_PROBABILITY), 4)    AS MAX_CHURN_PROBABILITY,
    MODE() WITHIN GROUP (ORDER BY TOP_SHAP_FEATURE) AS PRIMARY_FLEET_DRIVER
FROM CLV_SCORED_OUTPUT
GROUP BY RISK_BUCKET
ORDER BY AVG_CHURN_PROBABILITY DESC;
