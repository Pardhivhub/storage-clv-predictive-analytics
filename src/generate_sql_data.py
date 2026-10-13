"""
Generate Snowflake Data Insert Script for the 66 Scored Arrays
==============================================================
Reads data/clv_scored_66_arrays.csv and formats direct INSERT statements
into sql/clv_scored_output_data_insert.sql.
"""

import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
SQL_DIR    = os.path.join(BASE_DIR, "sql")
SCORED_CSV = os.path.join(DATA_DIR, "clv_scored_66_arrays.csv")
OUT_SQL    = os.path.join(SQL_DIR, "clv_scored_output_data_insert.sql")


def main():
    df = pd.read_csv(SCORED_CSV)
    print(f"Loaded {len(df)} scored arrays.")

    lines = [
        "-- =============================================================================",
        "-- Snowflake Direct Data Ingestion Script (66 Scored Storage Arrays)",
        "-- Generated for Task 3: Production Warehouse Integration",
        "-- =============================================================================",
        "",
        "USE SCHEMA ANALYTICS_PROD.STORAGE;",
        "",
        "INSERT INTO CLV_SCORED_OUTPUT (",
        "    ARRAY_ID,",
        "    ACCOUNT_ID,",
        "    CHURN_PROBABILITY,",
        "    RISK_BUCKET,",
        "    TOP_SHAP_FEATURE,",
        "    TOP_SHAP_VALUE,",
        "    SHAP_PAYLOAD,",
        "    MODEL_VERSION,",
        "    SCORED_AT",
        ")",
        "VALUES",
    ]

    value_rows = []
    for _, r in df.iterrows():
        # Escape single quotes in JSON payload if any
        payload_json = r["SHAP_PAYLOAD"].replace("'", "''")
        row_str = (
            f"    ('{r['ARRAY_ID']}', '{r['ACCOUNT_ID']}', {r['CHURN_PROBABILITY']:.4f}, "
            f"'{r['RISK_BUCKET']}', '{r['TOP_SHAP_FEATURE']}', {r['TOP_SHAP_VALUE']:.4f}, "
            f"PARSE_JSON('{payload_json}'), '{r['MODEL_VERSION']}', '{r['SCORED_AT']}')"
        )
        value_rows.append(row_str)

    lines.append(",\n".join(value_rows) + ";")
    lines.append("")
    lines.append("-- Verify ingestion:")
    lines.append("SELECT RISK_BUCKET, COUNT(*) AS ARRAY_COUNT, ROUND(AVG(CHURN_PROBABILITY), 4) AS AVG_PROB")
    lines.append("FROM CLV_SCORED_OUTPUT")
    lines.append("GROUP BY RISK_BUCKET;")

    with open(OUT_SQL, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {len(df)} SQL insert statements to {OUT_SQL}")


if __name__ == "__main__":
    main()
