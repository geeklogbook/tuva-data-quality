import streamlit as st
import duckdb
import pandas as pd
import altair as alt

DB_PATH = "/home/j/Documents/geeklogbook/tuva-data-quality/tuva.duckdb"

st.set_page_config(
    page_title="Tuva Data Explorer",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Tuva Data Explorer")
st.caption("Claims · Members · Providers — Synthetic dataset v0.15/0.16")

@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_data
def query(sql):
    con = get_conn()
    return con.execute(sql).df()

# ── Metrics ──────────────────────────────────────────────────────────────────
try:
    claims_count = query("SELECT COUNT(*) as n FROM main_input_layer.medical_claim").iloc[0]["n"]
    members_count = query("SELECT COUNT(DISTINCT person_id) as n FROM main_input_layer.eligibility").iloc[0]["n"]
    payers_count = query("SELECT COUNT(DISTINCT payer) as n FROM main_input_layer.medical_claim").iloc[0]["n"]
    total_paid = query("SELECT SUM(paid_amount) as n FROM main_input_layer.medical_claim").iloc[0]["n"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Medical claims", f"{claims_count:,.0f}")
    c2.metric("Unique members", f"{members_count:,.0f}")
    c3.metric("Payers", f"{payers_count:,.0f}")
    c4.metric("Total paid amount", f"${total_paid:,.0f}" if total_paid else "N/A")
except Exception as e:
    st.warning(f"Could not load metrics: {e}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Spend & utilization", "Member demographics", "Claim types", "Provider activity"])

# ── Tab 1: Spend & utilization ────────────────────────────────────────────────
with tab1:
    st.subheader("Spend and utilization by encounter type")
    try:
        df_enc = query("""
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

        df_pmpm = query("""
            SELECT
                year_month,
                medical_paid,
                acute_inpatient_paid,
                emergency_department_paid,
                office_based_visit_paid,
                outpatient_hospital_or_clinic_paid,
                lab_paid
            FROM main_financial_pmpm.pmpm_payer
            ORDER BY year_month
        """)

        total_spend  = df_enc["total_paid"].sum()
        top_enc      = df_enc.iloc[0]
        top_pct      = 100 * top_enc["total_paid"] / max(total_spend, 1)
        avg_cost_inp = df_enc[df_enc["encounter_type"] == "acute inpatient"]["avg_paid_per_claim"].values
        avg_cost_ed  = df_enc[df_enc["encounter_type"] == "emergency department"]["avg_paid_per_claim"].values

        st.caption(
            f"**{top_enc['encounter_type'].title()}** accounts for "
            f"**{top_pct:.0f}% of total spend** (${top_enc['total_paid']:,.0f}) "
            f"from only {top_enc['claims']:,} claims — "
            f"avg ${(avg_cost_inp[0] if len(avg_cost_inp) else 0):,.0f} per inpatient episode "
            f"vs ${(avg_cost_ed[0] if len(avg_cost_ed) else 0):,.0f} per ED visit."
        )

        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.markdown("##### Total spend by encounter type")
            spend_chart = (
                alt.Chart(df_enc)
                .mark_bar()
                .encode(
                    x=alt.X("total_paid:Q", title="Total paid ($)"),
                    y=alt.Y("encounter_type:N", sort="-x", title=None),
                    color=alt.Color(
                        "total_paid:Q",
                        scale=alt.Scale(scheme="blues"),
                        legend=None,
                    ),
                    tooltip=[
                        "encounter_type",
                        alt.Tooltip("total_paid:Q", title="Total paid ($)", format="$,.0f"),
                        alt.Tooltip("claims:Q", title="Claims", format=","),
                        alt.Tooltip("members:Q", title="Members", format=","),
                        alt.Tooltip("avg_paid_per_claim:Q", title="Avg per claim ($)", format="$,.0f"),
                    ],
                )
                .properties(height=480)
            )
            st.altair_chart(spend_chart, use_container_width=True)

        with col_b:
            st.markdown("##### Avg cost per claim by type")
            st.caption("High cost per claim = episodes that drive the most spend per event.")
            cost_chart = (
                alt.Chart(df_enc[df_enc["avg_paid_per_claim"] > 0])
                .mark_bar(color="#f58518")
                .encode(
                    x=alt.X("avg_paid_per_claim:Q", title="Avg paid per claim ($)"),
                    y=alt.Y("encounter_type:N", sort="-x", title=None),
                    tooltip=[
                        "encounter_type",
                        alt.Tooltip("avg_paid_per_claim:Q", title="Avg per claim ($)", format="$,.0f"),
                        alt.Tooltip("claims:Q", title="Claims", format=","),
                    ],
                )
                .properties(height=480)
            )
            st.altair_chart(cost_chart, use_container_width=True)

        st.divider()

        # PMPM trend
        st.markdown("##### Monthly PMPM trend by service category")
        st.caption("Per-member-per-month paid amount — reveals seasonal patterns and cost shifts over time.")
        if not df_pmpm.empty:
            df_pmpm_melt = df_pmpm.melt(
                id_vars="year_month",
                value_vars=[
                    "acute_inpatient_paid", "emergency_department_paid",
                    "office_based_visit_paid", "outpatient_hospital_or_clinic_paid", "lab_paid",
                ],
                var_name="category", value_name="pmpm",
            )
            label_map = {
                "acute_inpatient_paid":              "Acute inpatient",
                "emergency_department_paid":         "Emergency dept",
                "office_based_visit_paid":           "Office visit",
                "outpatient_hospital_or_clinic_paid": "Outpatient hospital",
                "lab_paid":                          "Lab",
            }
            df_pmpm_melt["category"] = df_pmpm_melt["category"].map(label_map)
            df_pmpm_melt["year_month"] = pd.to_datetime(df_pmpm_melt["year_month"].astype(str), format="%Y%m")

            pmpm_chart = (
                alt.Chart(df_pmpm_melt)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year_month:T", title="Month"),
                    y=alt.Y("pmpm:Q", title="PMPM paid ($)"),
                    color=alt.Color("category:N", legend=alt.Legend(title="Service category")),
                    tooltip=[
                        alt.Tooltip("year_month:T", title="Month"),
                        "category",
                        alt.Tooltip("pmpm:Q", title="PMPM ($)", format="$,.2f"),
                    ],
                )
                .properties(title="PMPM paid by service category", height=300)
            )
            st.altair_chart(pmpm_chart, use_container_width=True)

        st.dataframe(
            df_enc,
            use_container_width=True,
            hide_index=True,
            column_config={
                "total_paid":         st.column_config.NumberColumn("Total paid ($)",       format="$%,.0f"),
                "avg_paid_per_claim": st.column_config.NumberColumn("Avg per claim ($)",    format="$%,.0f"),
                "avg_paid_per_line":  st.column_config.NumberColumn("Avg per line ($)",     format="$%,.2f"),
            },
        )
    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 2: Member demographics ────────────────────────────────────────────────
with tab2:
    st.subheader("Member demographics")
    try:
        df_gender = query("""
            SELECT gender, COUNT(DISTINCT person_id) AS members
            FROM main_input_layer.eligibility
            GROUP BY gender ORDER BY members DESC
        """)

        df_payer_enroll = query("""
            SELECT payer, COUNT(DISTINCT person_id) AS members
            FROM main_input_layer.eligibility
            GROUP BY payer ORDER BY members DESC
        """)

        df_dual = query("""
            SELECT
                COALESCE(dual_status_code, 'Not dual') AS dual_status,
                COUNT(DISTINCT person_id) AS members
            FROM main_input_layer.eligibility
            GROUP BY dual_status ORDER BY members DESC
        """)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.altair_chart(
                alt.Chart(df_gender).mark_arc(innerRadius=40).encode(
                    theta="members:Q",
                    color=alt.Color("gender:N", legend=alt.Legend(title="Gender")),
                    tooltip=["gender", "members"],
                ).properties(title="Gender distribution", height=240),
                use_container_width=True,
            )

        with col_b:
            st.altair_chart(
                alt.Chart(df_payer_enroll).mark_bar().encode(
                    x=alt.X("members:Q", title="Members"),
                    y=alt.Y("payer:N", sort="-x", title="Payer"),
                    color=alt.Color("payer:N", legend=None),
                    tooltip=["payer", "members"],
                ).properties(title="Members by payer", height=240),
                use_container_width=True,
            )

        with col_c:
            st.altair_chart(
                alt.Chart(df_dual).mark_bar().encode(
                    x=alt.X("members:Q", title="Members"),
                    y=alt.Y("dual_status:N", sort="-x", title="Dual status"),
                    color=alt.Color("dual_status:N", legend=None),
                    tooltip=["dual_status", "members"],
                ).properties(title="Dual status (Medicare/Medicaid)", height=240),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 3: Claim types ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Claim type breakdown")
    try:
        df_type = query("""
            SELECT
                claim_type,
                COUNT(DISTINCT claim_id) AS claims,
                COUNT(DISTINCT person_id) AS members,
                ROUND(SUM(paid_amount), 2) AS total_paid
            FROM main_input_layer.medical_claim
            GROUP BY claim_type ORDER BY claims DESC
        """)

        df_monthly = query("""
            SELECT
                DATE_TRUNC('month', claim_end_date) AS month,
                claim_type,
                COUNT(DISTINCT claim_id) AS claims
            FROM main_input_layer.medical_claim
            WHERE claim_end_date IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1
        """)

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.altair_chart(
                alt.Chart(df_type).mark_arc(innerRadius=50).encode(
                    theta="claims:Q",
                    color=alt.Color("claim_type:N", legend=alt.Legend(title="Claim type")),
                    tooltip=["claim_type", "claims", "members", "total_paid"],
                ).properties(title="Claims by type", height=300),
                use_container_width=True,
            )

        with col_b:
            if not df_monthly.empty:
                st.altair_chart(
                    alt.Chart(df_monthly).mark_line(point=True).encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("claims:Q", title="Claims"),
                        color=alt.Color("claim_type:N", legend=alt.Legend(title="Type")),
                        tooltip=["month", "claim_type", "claims"],
                    ).properties(title="Monthly claim volume by type", height=300),
                    use_container_width=True,
                )
            else:
                st.info("No date data available for monthly trend.")

        st.dataframe(df_type, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 4: Provider activity ──────────────────────────────────────────────────
with tab4:
    st.subheader("Provider activity (by rendering NPI)")
    try:
        df_providers = query("""
            SELECT
                rendering_npi,
                COUNT(DISTINCT claim_id) AS claims,
                COUNT(DISTINCT person_id) AS members_seen,
                ROUND(SUM(paid_amount), 2) AS total_paid,
                COUNT(DISTINCT payer) AS payers
            FROM main_input_layer.medical_claim
            WHERE rendering_npi IS NOT NULL
            GROUP BY rendering_npi
            ORDER BY claims DESC
            LIMIT 20
        """)

        st.altair_chart(
            alt.Chart(df_providers).mark_bar().encode(
                x=alt.X("claims:Q", title="Claims"),
                y=alt.Y("rendering_npi:N", sort="-x", title="Rendering NPI"),
                color=alt.Color("total_paid:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Paid ($)")),
                tooltip=["rendering_npi", "claims", "members_seen", "total_paid", "payers"],
            ).properties(title="Top 20 providers by claim volume", height=420),
            use_container_width=True,
        )

        st.dataframe(df_providers, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Data: Tuva synthetic dataset · DuckDB 1.5 · dbt 1.11 · Tuva Project 0.17.2")
