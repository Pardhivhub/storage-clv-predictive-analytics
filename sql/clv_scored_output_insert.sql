-- =============================================================================
-- Snowflake Optimized Table Insert & Merge Procedure Script
-- Case Study 2: Customer Lifetime Value (CLV) Predictive Analytics
-- Task 3: Production Warehouse Integration
-- =============================================================================
-- Purpose:
--   Production-grade, idempotent MERGE procedure to upsert newly scored storage
--   arrays into CLV_SCORED_OUTPUT. Parses the JSON SHAP attributions into
--   Snowflake's native VARIANT type, tracks scoring timestamps, and avoids
--   duplicate keys.
-- =============================================================================

USE SCHEMA ANALYTICS_PROD.STORAGE;

-- 1. Optimized Idempotent MERGE Statement from Staging
MERGE INTO CLV_SCORED_OUTPUT AS target
USING (
    SELECT
        ARRAY_ID,
        ACCOUNT_ID,
        CHURN_PROBABILITY,
        RISK_BUCKET,
        TOP_SHAP_FEATURE,
        TOP_SHAP_VALUE,
        PARSE_JSON(SHAP_PAYLOAD_RAW) AS SHAP_PAYLOAD,
        MODEL_VERSION,
        SCORED_AT
    FROM STG_CLV_SCORED_INPUT
) AS source
ON target.ARRAY_ID = source.ARRAY_ID

WHEN MATCHED AND source.SCORED_AT >= target.SCORED_AT THEN
    UPDATE SET
        target.ACCOUNT_ID        = source.ACCOUNT_ID,
        target.CHURN_PROBABILITY = source.CHURN_PROBABILITY,
        target.RISK_BUCKET       = source.RISK_BUCKET,
        target.TOP_SHAP_FEATURE  = source.TOP_SHAP_FEATURE,
        target.TOP_SHAP_VALUE    = source.TOP_SHAP_VALUE,
        target.SHAP_PAYLOAD      = source.SHAP_PAYLOAD,
        target.MODEL_VERSION     = source.MODEL_VERSION,
        target.SCORED_AT         = source.SCORED_AT,
        target.INSERTED_AT       = CURRENT_TIMESTAMP()

WHEN NOT MATCHED THEN
    INSERT (
        ARRAY_ID,
        ACCOUNT_ID,
        CHURN_PROBABILITY,
        RISK_BUCKET,
        TOP_SHAP_FEATURE,
        TOP_SHAP_VALUE,
        SHAP_PAYLOAD,
        MODEL_VERSION,
        SCORED_AT,
        INSERTED_AT
    )
    VALUES (
        source.ARRAY_ID,
        source.ACCOUNT_ID,
        source.CHURN_PROBABILITY,
        source.RISK_BUCKET,
        source.TOP_SHAP_FEATURE,
        source.TOP_SHAP_VALUE,
        source.SHAP_PAYLOAD,
        source.MODEL_VERSION,
        source.SCORED_AT,
        CURRENT_TIMESTAMP()
    );

-- 2. Stored Procedure Wrapper for Scheduled Production Pipeline
CREATE OR REPLACE PROCEDURE SP_INGEST_CLV_SCORES()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
    ROWS_MERGED INTEGER;
BEGIN
    -- Execute the Merge Operation
    MERGE INTO CLV_SCORED_OUTPUT AS target
    USING (
        SELECT
            ARRAY_ID,
            ACCOUNT_ID,
            CHURN_PROBABILITY,
            RISK_BUCKET,
            TOP_SHAP_FEATURE,
            TOP_SHAP_VALUE,
            PARSE_JSON(SHAP_PAYLOAD_RAW) AS SHAP_PAYLOAD,
            MODEL_VERSION,
            SCORED_AT
        FROM STG_CLV_SCORED_INPUT
    ) AS source
    ON target.ARRAY_ID = source.ARRAY_ID
    WHEN MATCHED AND source.SCORED_AT >= target.SCORED_AT THEN
        UPDATE SET
            target.ACCOUNT_ID        = source.ACCOUNT_ID,
            target.CHURN_PROBABILITY = source.CHURN_PROBABILITY,
            target.RISK_BUCKET       = source.RISK_BUCKET,
            target.TOP_SHAP_FEATURE  = source.TOP_SHAP_FEATURE,
            target.TOP_SHAP_VALUE    = source.TOP_SHAP_VALUE,
            target.SHAP_PAYLOAD      = source.SHAP_PAYLOAD,
            target.MODEL_VERSION     = source.MODEL_VERSION,
            target.SCORED_AT         = source.SCORED_AT,
            target.INSERTED_AT       = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
        INSERT (
            ARRAY_ID,
            ACCOUNT_ID,
            CHURN_PROBABILITY,
            RISK_BUCKET,
            TOP_SHAP_FEATURE,
            TOP_SHAP_VALUE,
            SHAP_PAYLOAD,
            MODEL_VERSION,
            SCORED_AT,
            INSERTED_AT
        )
        VALUES (
            source.ARRAY_ID,
            source.ACCOUNT_ID,
            source.CHURN_PROBABILITY,
            source.RISK_BUCKET,
            source.TOP_SHAP_FEATURE,
            source.TOP_SHAP_VALUE,
            source.SHAP_PAYLOAD,
            source.MODEL_VERSION,
            source.SCORED_AT,
            CURRENT_TIMESTAMP()
        );

    -- Clear staging table upon successful load
    TRUNCATE TABLE STG_CLV_SCORED_INPUT;

    RETURN 'SUCCESS: CLV scored batch merged into CLV_SCORED_OUTPUT and staging truncated.';
END;
$$;
