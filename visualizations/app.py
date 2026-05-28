import streamlit as st
import duckdb
import pandas as pd

DB_PATH = "/home/j/Documents/geeklogbook/tuva-data-quality/tuva.duckdb"

st.set_page_config(
    page_title="Tuva Analytics Hub",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data
def load_quick_stats():
    con = duckdb.connect(DB_PATH, read_only=True)
    stats = {}
    try:
        stats["claims"]  = con.execute("SELECT COUNT(*) FROM main_input_layer.medical_claim").fetchone()[0]
        stats["members"] = con.execute("SELECT COUNT(DISTINCT person_id) FROM main_input_layer.eligibility").fetchone()[0]
        stats["paid"]    = con.execute("SELECT SUM(paid_amount) FROM main_input_layer.medical_claim").fetchone()[0]
    except Exception:
        pass
    try:
        dq = con.execute("""
            SELECT
                SUM(CASE WHEN status = 'green'  THEN 1 ELSE 0 END) AS n_green,
                SUM(CASE WHEN status = 'yellow' THEN 1 ELSE 0 END) AS n_yellow,
                SUM(CASE WHEN status = 'red'    THEN 1 ELSE 0 END) AS n_red,
                COUNT(*) AS n_total
            FROM (
                SELECT
                    CASE
                        WHEN green IS NULL THEN 'no threshold'
                        WHEN ROUND(100.0 * fill_num / NULLIF(denom,0),1) >= green THEN 'green'
                        WHEN ROUND(100.0 * fill_num / NULLIF(denom,0),1) >= red   THEN 'yellow'
                        ELSE 'red'
                    END AS status
                FROM main_data_quality.summary
                WHERE red IS NOT NULL
            )
        """).fetchone()
        stats["dq_green"]  = dq[0]
        stats["dq_yellow"] = dq[1]
        stats["dq_red"]    = dq[2]
        stats["dq_total"]  = dq[3]
    except Exception:
        pass
    return stats


def home():
    st.title("🏥 Tuva Analytics Hub")
    st.caption("Claims · Members · Providers · Data Quality — Synthetic dataset v0.15/0.16")
    st.divider()

    # ── Quick stats ───────────────────────────────────────────────────────────
    stats = load_quick_stats()

    st.markdown("#### Dataset summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Medical claims",  f"{stats.get('claims', 'N/A'):,}" if isinstance(stats.get("claims"), int) else "N/A")
    c2.metric("Unique members",  f"{stats.get('members', 'N/A'):,}" if isinstance(stats.get("members"), int) else "N/A")
    c3.metric("Total paid",      f"${stats.get('paid', 0):,.0f}" if stats.get("paid") else "N/A")
    c4.metric("DQ fields 🟢",    f"{stats.get('dq_green', '—')}")
    c5.metric("DQ fields 🔴",    f"{stats.get('dq_red', '—')}")

    st.divider()

    # ── Report cards ──────────────────────────────────────────────────────────
    st.markdown("#### Available reports")

    reports = [
        {
            "icon": "🏥",
            "title": "Data Explorer",
            "desc": "Explore medical claims, member demographics, claim types, and provider activity.",
            "tags": ["Claims", "Members", "Providers", "Payers"],
            "page": "Tuva Data Explorer",
        },
        {
            "icon": "🔍",
            "title": "Data Quality Explorer",
            "desc": "Analyze fill rates, field validity, trends over time, and drill down into invalid records.",
            "tags": ["Fill rate", "Validity", "Trends", "Invalid records"],
            "page": "Data Quality Explorer",
        },
        {
            "icon": "🧬",
            "title": "HCC Risk & Condition Predictor",
            "desc": "Random Forest classifier predicting high-risk Medicare patients from HCC suspecting gaps, spend, and condition history.",
            "tags": ["HCC", "Risk score", "Random Forest", "Conditions"],
            "page": "HCC Risk Predictor",
        },
    ]

    cols = st.columns(len(reports), gap="large")
    for col, r in zip(cols, reports):
        with col:
            with st.container(border=True):
                st.markdown(f"### {r['icon']} {r['title']}")
                st.markdown(r["desc"])
                st.markdown(" ".join(f"`{t}`" for t in r["tags"]))
                st.markdown(f"→ Navigate using the sidebar: **{r['page']}**")

    st.divider()
    st.caption("Stack: DuckDB 1.5 · dbt 1.11 · Tuva Project 0.17.2 · Streamlit 1.57")


# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page(home,                        title="Home",                   icon="🏠", default=True),
        st.Page("tuva_explorer.py",          title="Tuva Data Explorer",     icon="🏥"),
        st.Page("data_quality_explorer.py",  title="Data Quality Explorer",  icon="🔍"),
        st.Page("predictions.py",            title="HCC Risk Predictor",     icon="🧬"),
    ]
)
pg.run()
