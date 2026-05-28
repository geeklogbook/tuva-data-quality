"""
Run once locally to export DuckDB tables to Parquet files for Streamlit Cloud deployment.

    python visualizations/export_to_parquet.py

Requires duckdb: pip install duckdb
"""
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "tuva.duckdb"
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=True)


def export(name, sql):
    print(f"Exporting {name}...")
    df = con.execute(sql).df()
    path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  → {len(df):,} rows  {path.stat().st_size / 1024:.0f} KB")


# ── Home quick stats ───────────────────────────────────────────────────────────
export("quick_stats", """
    SELECT
        (SELECT COUNT(*)                    FROM main_input_layer.medical_claim) AS claims,
        (SELECT COUNT(DISTINCT person_id)   FROM main_input_layer.eligibility)   AS members,
        (SELECT SUM(paid_amount)            FROM main_input_layer.medical_claim) AS total_paid,
        (SELECT COUNT(DISTINCT payer)       FROM main_input_layer.medical_claim) AS payers
""")

export("dq_quick_stats", """
    SELECT
        SUM(CASE WHEN status = 'green'  THEN 1 ELSE 0 END) AS n_green,
        SUM(CASE WHEN status = 'yellow' THEN 1 ELSE 0 END) AS n_yellow,
        SUM(CASE WHEN status = 'red'    THEN 1 ELSE 0 END) AS n_red,
        COUNT(*) AS n_total
    FROM (
        SELECT CASE
            WHEN green IS NULL THEN 'no threshold'
            WHEN ROUND(100.0 * fill_num / NULLIF(denom,0),1) >= green THEN 'green'
            WHEN ROUND(100.0 * fill_num / NULLIF(denom,0),1) >= red   THEN 'yellow'
            ELSE 'red'
        END AS status
        FROM main_data_quality.summary
        WHERE red IS NOT NULL
    )
""")

# ── Tuva Explorer ──────────────────────────────────────────────────────────────
export("encounter_spend", """
    SELECT
        encounter_type,
        COUNT(DISTINCT claim_id)      AS claims,
        COUNT(DISTINCT person_id)     AS members,
        ROUND(SUM(paid_amount), 0)    AS total_paid,
        ROUND(AVG(paid_amount), 2)    AS avg_paid_per_line,
        ROUND(SUM(paid_amount) / NULLIF(COUNT(DISTINCT claim_id), 0), 0) AS avg_paid_per_claim
    FROM main_core.medical_claim
    WHERE encounter_type IS NOT NULL
    GROUP BY encounter_type
    ORDER BY total_paid DESC
""")

export("pmpm_payer", """
    SELECT year_month, medical_paid, acute_inpatient_paid, emergency_department_paid,
           office_based_visit_paid, outpatient_hospital_or_clinic_paid, lab_paid
    FROM main_financial_pmpm.pmpm_payer
    ORDER BY year_month
""")

export("member_gender", """
    SELECT gender, COUNT(DISTINCT person_id) AS members
    FROM main_input_layer.eligibility
    GROUP BY gender ORDER BY members DESC
""")

export("member_payer", """
    SELECT payer, COUNT(DISTINCT person_id) AS members
    FROM main_input_layer.eligibility
    GROUP BY payer ORDER BY members DESC
""")

export("member_dual", """
    SELECT COALESCE(dual_status_code, 'Not dual') AS dual_status,
           COUNT(DISTINCT person_id) AS members
    FROM main_input_layer.eligibility
    GROUP BY dual_status ORDER BY members DESC
""")

export("claim_type_summary", """
    SELECT
        claim_type,
        COUNT(DISTINCT claim_id)   AS claims,
        COUNT(DISTINCT person_id)  AS members,
        ROUND(SUM(paid_amount), 2) AS total_paid
    FROM main_input_layer.medical_claim
    GROUP BY claim_type ORDER BY claims DESC
""")

export("claim_monthly", """
    SELECT
        DATE_TRUNC('month', claim_end_date) AS month,
        claim_type,
        COUNT(DISTINCT claim_id) AS claims
    FROM main_input_layer.medical_claim
    WHERE claim_end_date IS NOT NULL
    GROUP BY 1, 2 ORDER BY 1
""")

export("top_providers", """
    SELECT
        rendering_npi,
        COUNT(DISTINCT claim_id)   AS claims,
        COUNT(DISTINCT person_id)  AS members_seen,
        ROUND(SUM(paid_amount), 2) AS total_paid,
        COUNT(DISTINCT payer)      AS payers
    FROM main_input_layer.medical_claim
    WHERE rendering_npi IS NOT NULL
    GROUP BY rendering_npi
    ORDER BY claims DESC
    LIMIT 20
""")

# ── Data Quality Explorer ──────────────────────────────────────────────────────
export("dq_summary", """
    SELECT
        summary_sk,
        table_name, claim_type, field_name, data_source,
        CAST(red   AS DOUBLE) AS red,
        CAST(green AS DOUBLE) AS green,
        valid_num, fill_num, denom,
        ROUND(100.0 * fill_num  / NULLIF(denom, 0), 1) AS fill_pct,
        ROUND(100.0 * valid_num / NULLIF(denom, 0), 1) AS valid_pct,
        CASE
            WHEN green IS NULL THEN 'no threshold'
            WHEN ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) >= green THEN 'green'
            WHEN ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) >= red   THEN 'yellow'
            ELSE 'red'
        END AS status
    FROM main_data_quality.summary
    ORDER BY table_name, claim_type, field_name
""")

export("dq_for_pbi", """
    SELECT table_name, field_name, bucket_name, invalid_reason,
           field_value, drill_down_key, drill_down_value, frequency
    FROM main_data_quality.data_quality_for_pbi
""")

export("dq_quality_trend", """
    SELECT
        t.first_day_of_month,
        s.table_name, s.field_name, s.claim_type,
        ROUND(100.0 * t.fill_num / NULLIF(t.denom, 0), 1) AS fill_pct
    FROM main_data_quality.quality_trend t
    JOIN main_data_quality.summary s ON t.summary_sk = s.summary_sk
    ORDER BY t.first_day_of_month, s.field_name
""")

# ── HCC Risk Predictor ─────────────────────────────────────────────────────────
export("hcc_summary", """
    SELECT person_id, payer, patient_sex, patient_birth_date, patient_age, suspecting_gaps
    FROM main_hcc_suspecting.summary
    WHERE patient_age IS NOT NULL
    ORDER BY suspecting_gaps DESC
""")

export("hcc_ml_dataset", """
    SELECT
        h.person_id, h.patient_age, h.patient_sex, h.suspecting_gaps,
        COALESCE(c.claim_lines, 0)   AS claim_lines,
        COALESCE(c.total_paid, 0.0)  AS total_paid,
        COALESCE(cond.conditions, 0) AS conditions
    FROM main_hcc_suspecting.summary h
    LEFT JOIN (
        SELECT person_id, COUNT(*) AS claim_lines, SUM(paid_amount) AS total_paid
        FROM main_input_layer.medical_claim
        GROUP BY person_id
    ) c ON h.person_id = c.person_id
    LEFT JOIN (
        SELECT person_id, COUNT(DISTINCT normalized_code) AS conditions
        FROM main_core.condition
        GROUP BY person_id
    ) cond ON h.person_id = cond.person_id
    WHERE h.patient_age IS NOT NULL
""")

export("hcc_list", """
    SELECT person_id, hcc_code, hcc_description, reason, contributing_factor, suspect_date
    FROM main_hcc_suspecting.list
""")

export("patient_conditions", """
    SELECT person_id, normalized_code, normalized_description, condition_type, recorded_date
    FROM main_core.condition
    ORDER BY person_id, recorded_date DESC
""")

con.close()
print("\nDone! All parquet files written to visualizations/data/")
