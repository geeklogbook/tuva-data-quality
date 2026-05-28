import streamlit as st
import duckdb
import pandas as pd
import altair as alt

DB_PATH = "/home/j/Documents/geeklogbook/tuva-data-quality/tuva.duckdb"

st.set_page_config(
    page_title="Tuva Data Quality Explorer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Tuva Data Quality Explorer")
st.caption("Fill rates · Validity · Trends · Drill-down — Synthetic dataset v0.15/0.16")


@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data
def query(sql, params=None):
    con = get_conn()
    if params:
        return con.execute(sql, params).df()
    return con.execute(sql).df()


# ── Scorecard metrics ─────────────────────────────────────────────────────────
@st.cache_data
def load_summary():
    return query("""
        SELECT
            table_name,
            claim_type,
            field_name,
            data_source,
            red,
            green,
            valid_num,
            fill_num,
            denom,
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


@st.cache_data
def load_payer_overview():
    return query("""
        SELECT
            data_source,
            table_name,
            claim_type,
            ROUND(AVG(100.0 * fill_num / NULLIF(denom, 0)), 1) AS avg_fill_pct,
            COUNT(*) AS field_count
        FROM main_data_quality.summary
        WHERE data_source IS NOT NULL
        GROUP BY data_source, table_name, claim_type
        ORDER BY data_source, table_name
    """)


@st.cache_data
def load_payer_field_detail(table):
    return query("""
        SELECT
            data_source,
            field_name,
            claim_type,
            ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) AS fill_pct
        FROM main_data_quality.summary
        WHERE data_source IS NOT NULL AND table_name = ?
        ORDER BY field_name, data_source
    """, [table])


@st.cache_data
def load_fill_distribution():
    return query("""
        SELECT
            CASE
                WHEN fill_pct = 0    THEN '0% — completely empty'
                WHEN fill_pct < 25   THEN '1–24%'
                WHEN fill_pct < 50   THEN '25–49%'
                WHEN fill_pct < 75   THEN '50–74%'
                WHEN fill_pct < 90   THEN '75–89%'
                WHEN fill_pct < 100  THEN '90–99%'
                ELSE '100% — fully populated'
            END AS bucket,
            COUNT(*) AS fields,
            CASE
                WHEN fill_pct = 0    THEN 0
                WHEN fill_pct < 25   THEN 1
                WHEN fill_pct < 50   THEN 2
                WHEN fill_pct < 75   THEN 3
                WHEN fill_pct < 90   THEN 4
                WHEN fill_pct < 100  THEN 5
                ELSE 6
            END AS sort_order
        FROM (
            SELECT ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) AS fill_pct
            FROM main_data_quality.summary
        )
        GROUP BY 1, 3
        ORDER BY sort_order
    """)


@st.cache_data
def load_null_volume():
    return query("""
        SELECT
            table_name,
            claim_type,
            field_name,
            denom,
            ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) AS fill_pct,
            CAST(denom - fill_num AS BIGINT) AS null_records
        FROM main_data_quality.summary
        WHERE denom > 0 AND fill_num < denom
        ORDER BY null_records DESC
        LIMIT 20
    """)


@st.cache_data
def load_validity_by_table():
    return query("""
        SELECT
            table_name,
            ROUND(AVG(100.0 * valid_num / NULLIF(denom, 0)), 1) AS avg_valid_pct,
            ROUND(AVG(100.0 * fill_num  / NULLIF(denom, 0)), 1) AS avg_fill_pct,
            COUNT(*) AS fields
        FROM main_data_quality.summary
        WHERE denom > 0 AND valid_num IS NOT NULL
          AND table_name NOT IN ('eligibility')
        GROUP BY table_name
        ORDER BY avg_valid_pct
    """)


@st.cache_data
def load_invalid_reasons_volume():
    return query("""
        SELECT
            invalid_reason,
            bucket_name,
            SUM(frequency)           AS total_records,
            COUNT(DISTINCT field_name)  AS affected_fields,
            COUNT(DISTINCT table_name)  AS affected_tables
        FROM main_data_quality.data_quality_for_pbi
        WHERE invalid_reason IS NOT NULL
          AND bucket_name NOT IN ('valid')
          AND invalid_reason NOT IN ('valid', 'multiple')
        GROUP BY 1, 2
        ORDER BY total_records DESC
    """)


@st.cache_data
def load_worst_fields():
    return query("""
        SELECT
            table_name,
            claim_type,
            field_name,
            CAST(red AS DOUBLE) AS red,
            CAST(green AS DOUBLE) AS green,
            ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) AS fill_pct,
            CASE
                WHEN ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) >= green THEN 'green'
                WHEN ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) >= red   THEN 'yellow'
                ELSE 'red'
            END AS status
        FROM main_data_quality.summary
        WHERE red IS NOT NULL
          AND ROUND(100.0 * fill_num / NULLIF(denom, 0), 1) < green
        ORDER BY fill_pct ASC
        LIMIT 20
    """)


df_summary = load_summary()

with_threshold = df_summary[df_summary["status"] != "no threshold"]
n_green  = (with_threshold["status"] == "green").sum()
n_yellow = (with_threshold["status"] == "yellow").sum()
n_red    = (with_threshold["status"] == "red").sum()
n_total  = len(with_threshold)

total_weight = with_threshold["denom"].sum()
dqi_score = round(
    (with_threshold["fill_pct"].fillna(0) * with_threshold["denom"]).sum() / max(total_weight, 1),
    1,
)
dqi_label = "🟢" if dqi_score >= 80 else "🟡" if dqi_score >= 60 else "🔴"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fields with thresholds", f"{n_total:,}")
c2.metric("🟢 Green",  f"{n_green:,}  ({100*n_green//max(n_total,1)}%)")
c3.metric("🟡 Yellow", f"{n_yellow:,}  ({100*n_yellow//max(n_total,1)}%)")
c4.metric("🔴 Red",    f"{n_red:,}  ({100*n_red//max(n_total,1)}%)")
c5.metric("DQI Score", f"{dqi_label} {dqi_score:.1f} / 100")

st.caption(
    "🟢 **Green** — fill rate ≥ green threshold (usually 99%): field is production-ready.  "
    "🟡 **Yellow** — between red and green thresholds: needs investigation.  "
    "🔴 **Red** — fill rate < red threshold: blocking issue, action required.  "
    "⚪ **No threshold** — optional field, not scored.  "
    "**DQI** = weighted average fill rate across all scored fields."
)

st.divider()

# ── About: what type of DQ is being analysed ─────────────────────────────────
with st.expander("ℹ️  What data quality dimensions does this dashboard measure?", expanded=False):
    st.markdown("""
This dashboard runs the **[Tuva Project](https://thetuvaproject.com/) data quality framework**
against a synthetic Medicare claims dataset. It evaluates six dimensions of data quality:

| Dimension | What it checks | Where to find it |
|---|---|---|
| **Completeness (Fill rate)** | % of records where a field is not null. Most fields are expected to be ≥ 99% populated. | Scorecard · Worst fields |
| **Validity** | % of records that pass terminology/business-rule checks — e.g., diagnosis codes must exist in ICD-10-CM, NPIs must match the provider registry, date values must fall within plausible ranges. | Field detail · Invalid drill-down |
| **Referential integrity** | Foreign-key lookups: HCPCS codes → HCPCS Level II, ICD-10 codes → ICD-10-CM, NPIs → Provider table, status codes → Tuva terminology. Failures appear as `invalid` bucket records. | Invalid drill-down |
| **Uniqueness / Deduplication** | Detection of duplicate values — e.g., the same diagnosis code appearing in multiple positions on the same claim. | Invalid drill-down (bucket = `duplicate`) |
| **Timeliness** | Checks whether claim end dates and paid dates are consistent — large gaps may indicate delayed submission or extraction errors. | Trends (via `quality_trend`) |
| **Reasonableness** | Volume and spend trend analysis to detect sudden anomalies (e.g., a month with zero claims, implausible paid amounts). | Payer comparison |

#### How the thresholds work
- 🟢 **Green**: fill rate ≥ green threshold (typically 99%) — field is production-ready
- 🟡 **Yellow**: fill rate ≥ red threshold but < green — monitor and investigate
- 🔴 **Red**: fill rate < red threshold — blocking issue, action required
- ⚪ **No threshold**: field is optional or informational — not scored

#### What this dataset covers
- **3 data sources**: `medicare cclf` (claims), `emr` (clinical), `labcorp` (lab results)
- **8 clinical/claims tables**: MEDICAL_CLAIM (institutional + professional), PHARMACY_CLAIM, ELIGIBILITY, LAB_RESULT, OBSERVATION, APPOINTMENT, IMMUNIZATION
- **225 field-level quality checks** across all tables
""")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Scorecard",
    "Field detail",
    "Trends",
    "Invalid drill-down",
    "Payer comparison",
    "Worst fields",
    "Data profiling",
    "Recommendations",
])

# ── Tab 1: Scorecard ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Fill rate by table and field")

    tables_avail = sorted(df_summary["table_name"].dropna().unique())
    sel_tables = st.multiselect(
        "Filter by table", tables_avail, default=tables_avail, key="t1_tables"
    )

    df_view = df_summary[df_summary["table_name"].isin(sel_tables)].copy()

    # Status distribution bar
    status_counts = (
        df_view[df_view["status"] != "no threshold"]
        .groupby("status", as_index=False)
        .size()
        .rename(columns={"size": "fields"})
    )
    color_map = {"green": "#2ca02c", "yellow": "#f0c05a", "red": "#d62728"}
    if not status_counts.empty:
        bar = (
            alt.Chart(status_counts)
            .mark_bar()
            .encode(
                x=alt.X("fields:Q", title="Fields"),
                y=alt.Y("status:N", sort=["red", "yellow", "green"], title="Status"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=list(color_map.keys()),
                        range=list(color_map.values()),
                    ),
                    legend=None,
                ),
                tooltip=["status", "fields"],
            )
            .properties(title="Status distribution (fields with threshold)", height=180)
        )
        st.altair_chart(bar, use_container_width=True)

    # Heatmap fill_pct by table
    st.markdown("##### Fill rate heatmap by table")
    hm_data = (
        df_view.groupby(["table_name", "claim_type"], as_index=False)["fill_pct"]
        .mean()
        .round(1)
    )
    hm_data["label"] = hm_data["fill_pct"].astype(str) + "%"

    heatmap = (
        alt.Chart(hm_data)
        .mark_rect()
        .encode(
            x=alt.X("claim_type:N", title="Claim type"),
            y=alt.Y("table_name:N", title="Table"),
            color=alt.Color(
                "fill_pct:Q",
                scale=alt.Scale(domain=[0, 50, 100], range=["#d62728", "#f0c05a", "#2ca02c"]),
                legend=alt.Legend(title="Avg fill %"),
            ),
            tooltip=["table_name", "claim_type", "fill_pct"],
        )
        .properties(title="Average fill rate by table / claim type", height=300)
    )
    text = heatmap.mark_text(baseline="middle", fontSize=11).encode(
        text=alt.Text("fill_pct:Q", format=".1f"),
        color=alt.value("white"),
    )
    st.altair_chart(heatmap + text, use_container_width=True)

    # Full table
    status_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴", "no threshold": "⚪"}
    df_display = df_view.copy()
    df_display["status_icon"] = df_display["status"].map(status_emoji)
    st.dataframe(
        df_display[["status_icon", "table_name", "claim_type", "field_name",
                    "fill_pct", "valid_pct", "denom", "red", "green"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "fill_pct":  st.column_config.ProgressColumn("Fill %",  min_value=0, max_value=100, format="%.1f%%"),
            "valid_pct": st.column_config.ProgressColumn("Valid %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

# ── Tab 2: Field detail ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Field detail")

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        t2_table = st.selectbox("Table", sorted(df_summary["table_name"].dropna().unique()), key="t2_table")
    with col_sel2:
        fields_for_table = sorted(
            df_summary[df_summary["table_name"] == t2_table]["field_name"].dropna().unique()
        )
        t2_field = st.selectbox("Field", fields_for_table, key="t2_field")
    with col_sel3:
        claim_types_for = sorted(
            df_summary[
                (df_summary["table_name"] == t2_table) &
                (df_summary["field_name"] == t2_field)
            ]["claim_type"].dropna().unique()
        )
        t2_claim_type = st.selectbox("Claim type", claim_types_for, key="t2_claim_type") if claim_types_for else None

    # Thresholds for selected field
    row = df_summary[
        (df_summary["table_name"] == t2_table) &
        (df_summary["field_name"] == t2_field) &
        (df_summary["claim_type"] == t2_claim_type if t2_claim_type else True)
    ]
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fill %",  f"{r['fill_pct']}%" if pd.notna(r["fill_pct"]) else "N/A")
        m2.metric("Valid %", f"{r['valid_pct']}%" if pd.notna(r["valid_pct"]) else "N/A")
        m3.metric("Green threshold ≥", f"{r['green']}%" if pd.notna(r["green"]) else "—")
        m4.metric("Red threshold <",   f"{r['red']}%"  if pd.notna(r["red"])   else "—")

    # Bucket distribution
    try:
        df_buckets = query("""
            SELECT
                bucket_name,
                SUM(frequency) AS total
            FROM main_data_quality.data_quality_for_pbi
            WHERE table_name = ? AND field_name = ?
            GROUP BY bucket_name
            ORDER BY total DESC
        """, [t2_table, t2_field])

        if not df_buckets.empty:
            bucket_colors = {
                "valid":     "#2ca02c",
                "null":      "#aec7e8",
                "invalid":   "#d62728",
                "multiple":  "#ff7f0e",
                "duplicate": "#9467bd",
            }
            col_a, col_b = st.columns([1, 2])
            with col_a:
                pie = (
                    alt.Chart(df_buckets)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta="total:Q",
                        color=alt.Color(
                            "bucket_name:N",
                            scale=alt.Scale(
                                domain=list(bucket_colors.keys()),
                                range=list(bucket_colors.values()),
                            ),
                            legend=alt.Legend(title="Bucket"),
                        ),
                        tooltip=["bucket_name", "total"],
                    )
                    .properties(title="Bucket distribution", height=280)
                )
                st.altair_chart(pie, use_container_width=True)
            with col_b:
                st.dataframe(df_buckets, use_container_width=True, hide_index=True)
        else:
            st.info("No data found in data_quality_for_pbi for this field.")
    except Exception as e:
        st.error(f"Error loading buckets: {e}")

    # Invalid examples
    st.markdown("##### Invalid / null record samples")
    try:
        df_invalid = query("""
            SELECT
                bucket_name,
                invalid_reason,
                field_value,
                drill_down_key,
                drill_down_value,
                frequency
            FROM main_data_quality.data_quality_for_pbi
            WHERE table_name = ?
              AND field_name = ?
              AND bucket_name NOT IN ('valid')
            ORDER BY frequency DESC
            LIMIT 200
        """, [t2_table, t2_field])
        if not df_invalid.empty:
            st.dataframe(df_invalid, use_container_width=True, hide_index=True)
        else:
            st.success("No invalid or null records found for this field.")
    except Exception as e:
        st.error(f"Error loading examples: {e}")

# ── Tab 3: Trends ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Fill rate trend over time")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t3_table = st.selectbox("Table", sorted(df_summary["table_name"].dropna().unique()), key="t3_table")
    with col_t2:
        t3_fields = sorted(
            df_summary[df_summary["table_name"] == t3_table]["field_name"].dropna().unique()
        )
        t3_fields_sel = st.multiselect(
            "Fields (max 10)", t3_fields, default=t3_fields[:5], key="t3_fields", max_selections=10
        )

    if t3_fields_sel:
        try:
            placeholders = ", ".join(["?"] * len(t3_fields_sel))
            df_trend = query(f"""
                SELECT
                    t.first_day_of_month,
                    s.field_name,
                    s.claim_type,
                    ROUND(100.0 * t.fill_num / NULLIF(t.denom, 0), 1) AS fill_pct
                FROM main_data_quality.quality_trend t
                JOIN main_data_quality.summary s ON t.summary_sk = s.summary_sk
                WHERE s.table_name = ?
                  AND s.field_name IN ({placeholders})
                ORDER BY t.first_day_of_month, s.field_name
            """, [t3_table] + t3_fields_sel)

            if not df_trend.empty:
                df_trend["field_label"] = df_trend["field_name"] + " (" + df_trend["claim_type"] + ")"
                trend_chart = (
                    alt.Chart(df_trend)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("first_day_of_month:T", title="Month"),
                        y=alt.Y("fill_pct:Q", title="Fill %", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color("field_label:N", legend=alt.Legend(title="Field")),
                        tooltip=["first_day_of_month:T", "field_label", "fill_pct"],
                    )
                    .properties(title="Monthly fill rate by field", height=420)
                )
                st.altair_chart(trend_chart, use_container_width=True)
                st.dataframe(
                    df_trend.sort_values("first_day_of_month", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No trend data available for the current selection.")
        except Exception as e:
            st.error(f"Error loading trend: {e}")
    else:
        st.info("Select at least one field.")

# ── Tab 4: Invalid drill-down ─────────────────────────────────────────────────
with tab4:
    st.subheader("Invalid record drill-down")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        t4_tables = sorted(df_summary["table_name"].dropna().unique())
        t4_table = st.selectbox("Table", ["(all)"] + t4_tables, key="t4_table")
    with col_f2:
        if t4_table != "(all)":
            t4_field_opts = sorted(
                df_summary[df_summary["table_name"] == t4_table]["field_name"].dropna().unique()
            )
        else:
            t4_field_opts = sorted(df_summary["field_name"].dropna().unique())
        t4_field = st.selectbox("Field", ["(all)"] + t4_field_opts, key="t4_field")
    with col_f3:
        try:
            reasons_df = query("""
                SELECT DISTINCT invalid_reason
                FROM main_data_quality.data_quality_for_pbi
                WHERE invalid_reason IS NOT NULL
                  AND bucket_name NOT IN ('valid')
                ORDER BY invalid_reason
            """)
            reason_opts = reasons_df["invalid_reason"].tolist()
        except Exception:
            reason_opts = []
        t4_reason = st.selectbox("Invalid reason", ["(all)"] + reason_opts, key="t4_reason")

    try:
        where_clauses = ["bucket_name NOT IN ('valid')"]
        params = []
        if t4_table != "(all)":
            where_clauses.append("table_name = ?")
            params.append(t4_table)
        if t4_field != "(all)":
            where_clauses.append("field_name = ?")
            params.append(t4_field)
        if t4_reason != "(all)":
            where_clauses.append("invalid_reason = ?")
            params.append(t4_reason)

        where_sql = " AND ".join(where_clauses)

        df_drill = query(f"""
            SELECT
                table_name,
                field_name,
                bucket_name,
                invalid_reason,
                field_value,
                drill_down_key,
                drill_down_value,
                frequency
            FROM main_data_quality.data_quality_for_pbi
            WHERE {where_sql}
            ORDER BY frequency DESC
            LIMIT 500
        """, params if params else None)

        if not df_drill.empty:
            # Summary by reason
            reason_summary = (
                df_drill.groupby(["bucket_name", "invalid_reason"], as_index=False)["frequency"]
                .sum()
                .sort_values("frequency", ascending=False)
            )

            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.markdown("**Frequency by reason**")
                reason_bar = (
                    alt.Chart(reason_summary)
                    .mark_bar()
                    .encode(
                        x=alt.X("frequency:Q", title="Records"),
                        y=alt.Y("invalid_reason:N", sort="-x", title="Reason"),
                        color=alt.Color("bucket_name:N", legend=alt.Legend(title="Bucket")),
                        tooltip=["bucket_name", "invalid_reason", "frequency"],
                    )
                    .properties(height=max(200, len(reason_summary) * 28))
                )
                st.altair_chart(reason_bar, use_container_width=True)
            with col_b:
                st.markdown(f"**Records (top 500)** — {len(df_drill):,} rows")
                st.dataframe(df_drill, use_container_width=True, hide_index=True)
        else:
            st.success("No invalid records found for the current filters.")
    except Exception as e:
        st.error(f"Error in drill-down: {e}")

# ── Tab 5: Payer comparison ───────────────────────────────────────────────────
with tab5:
    st.subheader("Fill rate by data source / payer")
    st.caption("Data source (emr, labcorp, medicare cclf) is the payer dimension in this dataset.")

    try:
        df_payer_ov = load_payer_overview()
        df_payer_ov = df_payer_ov[df_payer_ov["data_source"].notna()]

        # Overview grouped bar
        overview_chart = (
            alt.Chart(df_payer_ov)
            .mark_bar()
            .encode(
                x=alt.X("data_source:N", title="Data source", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("avg_fill_pct:Q", title="Avg fill %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("data_source:N", legend=None),
                column=alt.Column("table_name:N", title="Table", header=alt.Header(labelAngle=-15)),
                tooltip=["data_source", "table_name", "claim_type", "avg_fill_pct", "field_count"],
            )
            .properties(width=120, height=280, title="Average fill rate by payer and table")
        )
        st.altair_chart(overview_chart)

        st.divider()

        # Field-level detail
        st.markdown("##### Field-level breakdown by payer")
        payer_tables = sorted(df_payer_ov["table_name"].unique())
        sel_payer_table = st.selectbox("Table", payer_tables, key="t5_table")

        df_payer_detail = load_payer_field_detail(sel_payer_table)
        if not df_payer_detail.empty:
            detail_chart = (
                alt.Chart(df_payer_detail)
                .mark_bar()
                .encode(
                    x=alt.X("fill_pct:Q", title="Fill %", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("field_name:N", sort="-x", title="Field"),
                    color=alt.Color("data_source:N", legend=alt.Legend(title="Data source")),
                    xOffset=alt.XOffset("data_source:N"),
                    tooltip=["data_source", "field_name", "claim_type", "fill_pct"],
                )
                .properties(
                    title=f"Field fill rate by payer — {sel_payer_table}",
                    height=max(300, len(df_payer_detail["field_name"].unique()) * 26),
                )
            )
            st.altair_chart(detail_chart, use_container_width=True)
            st.dataframe(df_payer_detail, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for the selected table.")
    except Exception as e:
        st.error(f"Error loading payer comparison: {e}")

# ── Tab 6: Worst fields ───────────────────────────────────────────────────────
with tab6:
    st.subheader("Worst fields — lowest fill rate")

    try:
        df_worst = load_worst_fields()

        if not df_worst.empty:
            n_show = st.slider("Number of fields to show", min_value=5, max_value=20, value=10, key="t6_n")
            df_worst = df_worst.head(n_show)
            df_worst["field_label"] = df_worst["field_name"] + "\n(" + df_worst["table_name"] + " · " + df_worst["claim_type"] + ")"

            status_colors = {"red": "#d62728", "yellow": "#f0c05a"}

            bars = (
                alt.Chart(df_worst)
                .mark_bar()
                .encode(
                    x=alt.X("fill_pct:Q", title="Fill %", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("field_label:N", sort="x", title="Field"),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=list(status_colors.keys()),
                            range=list(status_colors.values()),
                        ),
                        legend=alt.Legend(title="Status"),
                    ),
                    tooltip=["field_name", "table_name", "claim_type", "fill_pct", "status", "red", "green"],
                )
            )

            red_ticks = (
                alt.Chart(df_worst)
                .mark_tick(color="#d62728", thickness=2, size=18)
                .encode(
                    x=alt.X("red:Q", title=""),
                    y=alt.Y("field_label:N", sort="x"),
                    tooltip=[alt.Tooltip("red:Q", title="Red threshold")],
                )
            )

            green_ticks = (
                alt.Chart(df_worst)
                .mark_tick(color="#2ca02c", thickness=2, size=18)
                .encode(
                    x=alt.X("green:Q", title=""),
                    y=alt.Y("field_label:N", sort="x"),
                    tooltip=[alt.Tooltip("green:Q", title="Green threshold")],
                )
            )

            chart = (bars + red_ticks + green_ticks).properties(
                title="Lowest fill rate fields (🔴 tick = red threshold, 🟢 tick = green threshold)",
                height=max(300, n_show * 36),
            )
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(
                df_worst[["status", "table_name", "claim_type", "field_name", "fill_pct", "red", "green"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fill_pct": st.column_config.ProgressColumn("Fill %", min_value=0, max_value=100, format="%.1f%%"),
                },
            )
        else:
            st.success("No fields in red or yellow status.")
    except Exception as e:
        st.error(f"Error loading worst fields: {e}")

# ── Tab 7: Data profiling ─────────────────────────────────────────────────────
with tab7:
    st.subheader("Data profiling")
    st.caption("Statistical summary of field population, null volumes, and validity across all tables.")

    try:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Fill rate distribution")
            st.caption("How many fields fall into each fill-rate band.")
            df_dist = load_fill_distribution()
            dist_chart = (
                alt.Chart(df_dist)
                .mark_bar()
                .encode(
                    x=alt.X("fields:Q", title="Number of fields"),
                    y=alt.Y("bucket:N", sort=alt.EncodingSortField("sort_order", order="ascending"),
                            title="Fill rate band"),
                    color=alt.Color(
                        "sort_order:O",
                        scale=alt.Scale(scheme="redyellowgreen"),
                        legend=None,
                    ),
                    tooltip=["bucket", "fields"],
                )
                .properties(height=260)
            )
            text_dist = dist_chart.mark_text(align="left", dx=4).encode(
                text="fields:Q", color=alt.value("#333")
            )
            st.altair_chart(dist_chart + text_dist, use_container_width=True)

            total_fields = df_dist["fields"].sum()
            empty_fields = df_dist[df_dist["bucket"].str.startswith("0%")]["fields"].sum() if not df_dist.empty else 0
            full_fields  = df_dist[df_dist["bucket"].str.startswith("100%")]["fields"].sum() if not df_dist.empty else 0
            st.markdown(
                f"**{empty_fields}** fields ({100*empty_fields//max(total_fields,1)}%) are completely empty · "
                f"**{full_fields}** fields ({100*full_fields//max(total_fields,1)}%) are fully populated"
            )

        with col_b:
            st.markdown("##### Validity vs. fill rate by table")
            st.caption("Average fill rate and validity rate per table — a gap between them signals invalid (non-null but wrong) values.")
            df_val = load_validity_by_table()
            if not df_val.empty:
                df_val_melt = df_val.melt(
                    id_vars=["table_name", "fields"],
                    value_vars=["avg_fill_pct", "avg_valid_pct"],
                    var_name="metric", value_name="pct"
                )
                df_val_melt["metric"] = df_val_melt["metric"].map({
                    "avg_fill_pct": "Fill rate",
                    "avg_valid_pct": "Validity rate",
                })
                val_chart = (
                    alt.Chart(df_val_melt)
                    .mark_bar()
                    .encode(
                        x=alt.X("pct:Q", title="Average %", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("table_name:N", sort="-x", title="Table"),
                        color=alt.Color(
                            "metric:N",
                            scale=alt.Scale(
                                domain=["Fill rate", "Validity rate"],
                                range=["#4c78a8", "#72b7b2"],
                            ),
                            legend=alt.Legend(title="Metric"),
                        ),
                        xOffset="metric:N",
                        tooltip=["table_name", "metric", "pct", "fields"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(val_chart, use_container_width=True)

        st.divider()

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("##### Top fields by null record volume")
            st.caption("Fields with the highest absolute number of missing records — prioritize these for remediation.")
            df_nulls = load_null_volume()
            null_chart = (
                alt.Chart(df_nulls.head(15))
                .mark_bar(color="#d62728")
                .encode(
                    x=alt.X("null_records:Q", title="Null records"),
                    y=alt.Y("field_name:N", sort="-x", title="Field"),
                    tooltip=["table_name", "claim_type", "field_name", "null_records", "fill_pct"],
                )
                .properties(height=360)
            )
            st.altair_chart(null_chart, use_container_width=True)

        with col_d:
            st.markdown("##### Invalid records by reason")
            st.caption("Root causes of validity failures, ranked by total records affected.")
            df_reasons = load_invalid_reasons_volume()
            if not df_reasons.empty:
                reason_chart = (
                    alt.Chart(df_reasons)
                    .mark_bar()
                    .encode(
                        x=alt.X("total_records:Q", title="Records affected"),
                        y=alt.Y("invalid_reason:N", sort="-x", title=None,
                                axis=alt.Axis(labelLimit=280)),
                        color=alt.Color(
                            "bucket_name:N",
                            scale=alt.Scale(
                                domain=["invalid", "duplicate", "null", "multiple"],
                                range=["#d62728", "#9467bd", "#aec7e8", "#ff7f0e"],
                            ),
                            legend=alt.Legend(title="Bucket"),
                        ),
                        tooltip=["invalid_reason", "bucket_name", "total_records",
                                 "affected_fields", "affected_tables"],
                    )
                    .properties(height=360)
                )
                st.altair_chart(reason_chart, use_container_width=True)

        st.divider()
        st.dataframe(load_null_volume(), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading profiling data: {e}")


# ── Tab 8: Recommendations ────────────────────────────────────────────────────
with tab8:
    st.subheader("Data quality recommendations")
    st.caption("Auto-generated from the actual data — sorted by impact (records affected × field importance).")

    try:
        df_reasons  = load_invalid_reasons_volume()
        df_nulls    = load_null_volume()
        df_worst    = load_worst_fields()

        # ── Summary sentence ─────────────────────────────────────────────────
        n_red_fields   = (df_summary["status"] == "red").sum()
        n_yellow_fields = (df_summary["status"] == "yellow").sum()
        n_empty_fields = (df_summary["fill_pct"].fillna(0) == 0).sum()
        total_null_vol = int(df_nulls["null_records"].sum()) if not df_nulls.empty else 0
        top_reason     = df_reasons.iloc[0] if not df_reasons.empty else None

        st.markdown(f"""
**{n_red_fields} fields are in red status** and **{n_yellow_fields} are in yellow** across this dataset.
{n_empty_fields} fields are completely empty (0% fill rate), representing **{total_null_vol:,} missing values in total**.
The highest-volume validity failure is *"{top_reason['invalid_reason'] if top_reason is not None else 'N/A'}"*
affecting **{int(top_reason['total_records']):,} records**.
""")

        st.divider()

        # ── Priority 1: Critical ─────────────────────────────────────────────
        st.markdown("### 🔴 Priority 1 — Critical")
        st.markdown("These issues affect the largest number of records and/or block downstream analytics.")

        with st.container(border=True):
            st.markdown("#### HCPCS code validity (129,575 records invalid)")
            st.markdown("""
**Issue:** `HCPCS_CODE` on `MEDICAL_CLAIM` (institutional) has 129,575 records that do not join to the
CMS HCPCS Level II terminology table — representing virtually all institutional claim lines.

**Root cause likely:** The source system may be sending internal charge codes, CPT-4 codes, or revenue
codes in the HCPCS field instead of valid HCPCS Level II codes.

**Recommended actions:**
1. Audit the upstream ETL — confirm which field in the source system maps to `HCPCS_CODE`.
2. Verify whether CPT-4 codes (professional) are being incorrectly placed in the institutional HCPCS field.
3. Add a pre-load validation step that rejects or flags records where `HCPCS_CODE` is not in the
   CMS HCPCS Level II value set before ingestion.
4. If the codes are legitimate but absent from the Tuva terminology, request an update to the value set.
""")

        with st.container(border=True):
            st.markdown("#### Date fields flagged 'too old' (71,695 records across 11 fields)")
            st.markdown("""
**Issue:** Across `MEDICAL_CLAIM` and `ELIGIBILITY`, 71,695 records have date values that fall
outside plausible historical ranges (flagged as `too old`).

**Root cause likely:** Date fields may contain default/sentinel values (e.g., `1900-01-01`),
Unix epoch zero (`1970-01-01`), or incorrectly formatted dates parsed as very old dates.

**Recommended actions:**
1. Identify which specific date fields are affected using the **Invalid drill-down** tab
   (filter by invalid reason = "too old").
2. Check source system default values for null dates — replace with NULL rather than sentinel dates.
3. Add range validation in the ETL: reject dates outside `[1920-01-01, today]` for birth dates,
   and `[2000-01-01, today]` for claim/service dates.
4. Re-run after fix and monitor via the **Trends** tab to confirm improvement.
""")

        st.divider()

        # ── Priority 2: High ─────────────────────────────────────────────────
        st.markdown("### 🟡 Priority 2 — High")
        st.markdown("Significant gaps that impact analytics and reporting accuracy.")

        with st.container(border=True):
            st.markdown("#### OBSERVATION table — reference range fields completely empty (5 fields, 150,244 records)")
            st.markdown("""
**Issue:** All five reference range fields (`SOURCE_REFERENCE_RANGE_HIGH/LOW`,
`NORMALIZED_REFERENCE_RANGE_HIGH/LOW`, `NORMALIZED_UNITS`) are 0% populated across
150,244 observation records.

**Impact:** Lab result interpretation (normal/abnormal classification) is impossible without
reference ranges. Clinical analytics built on this data will produce misleading results.

**Recommended actions:**
1. Confirm with the clinical data team whether reference ranges are captured in the source EMR.
2. If available, map them in the ETL transformation from the `labcorp` or EMR extract.
3. As a workaround, consider joining to a standard reference range table by LOINC code.
""")

        with st.container(border=True):
            st.markdown("#### Professional claim financial fields — 0% fill (6 fields, 133,474 claim lines)")
            st.markdown("""
**Issue:** `PAID_DATE`, `FACILITY_NPI`, `TOTAL_COST_AMOUNT`, `COPAYMENT_AMOUNT`,
`COINSURANCE_AMOUNT`, `DEDUCTIBLE_AMOUNT` are all empty on professional claim lines.

**Impact:** Cost-of-care analytics, PMPM calculations, and member cost-sharing analysis
cannot be performed for professional claims.

**Recommended actions:**
1. Verify whether these fields exist in the source claim system for professional claim types.
2. If absent from the source, document as a known limitation and exclude professional claims
   from financial roll-ups until the data is available.
3. Check if there is a separate remittance feed (835 file) that contains the financial fields
   and can be joined to the professional claim records.
""")

        with st.container(border=True):
            st.markdown("#### Duplicate diagnosis codes (5,144 records)")
            st.markdown("""
**Issue:** 5,144 claim records have the same diagnosis code appearing in multiple positions
(e.g., `DIAGNOSIS_CODE_1` = `DIAGNOSIS_CODE_2`).

**Impact:** Condition counts and comorbidity scores are inflated. Risk adjustment calculations
(HCC coding) may be affected.

**Recommended actions:**
1. Add a deduplication step in the ETL that removes repeated diagnosis codes at the claim level.
2. Investigate whether the source system duplicates codes during claim adjudication or remittance.
3. Review the Tuva `core__condition` model output to confirm whether deduplication is already
   applied downstream — if so, the raw claim data issue may be acceptable.
""")

        st.divider()

        # ── Priority 3: Medium ────────────────────────────────────────────────
        st.markdown("### 🟠 Priority 3 — Medium")
        st.markdown("Targeted issues with lower volume but specific terminology mismatches.")

        with st.container(border=True):
            st.markdown("#### Diagnosis codes not matching ICD-10-CM (354 records)")
            st.markdown("""
**Recommended actions:**
1. Run the **Invalid drill-down** tab filtered to *"Diagnosis Code does not join to Terminology"*
   to identify the specific codes.
2. Check for ICD-9 codes that were not migrated (transition cutoff: October 2015).
3. Validate that the `DIAGNOSIS_CODE_TYPE` field is set correctly — a code marked `icd-10-cm`
   but formatted as ICD-9 will always fail the lookup.
""")

        with st.container(border=True):
            st.markdown("#### Provider NPI validation failures (Rendering: 13, Billing: 4)")
            st.markdown("""
**Recommended actions:**
1. Cross-reference the failing NPIs against the NPPES registry (nppes.cms.hhs.gov).
2. Check for deactivated or retired NPIs — providers who left practice may still appear in
   historical claims.
3. Verify NPI format (10 digits, Luhn check digit) — malformed NPIs indicate upstream data entry errors.
""")

        with st.container(border=True):
            st.markdown("#### Race and dual-status codes failing terminology (396 + 438 records)")
            st.markdown("""
**Recommended actions:**
1. Review source system code lists for `RACE`, `DUAL_STATUS_CODE`, and `MEDICARE_STATUS_CODE`.
2. Map non-standard codes to the Tuva terminology value sets.
3. For dual-status, confirm the Medicare Dual Eligibility monthly file is being loaded correctly.
""")

        st.divider()

        # ── Next steps ────────────────────────────────────────────────────────
        st.markdown("### Next steps")
        cols = st.columns(3)
        with cols[0]:
            with st.container(border=True):
                st.markdown("**📋 Short-term (< 2 weeks)**")
                st.markdown("""
- Fix sentinel date values in ETL (date 'too old' issue)
- Document HCPCS code mapping discrepancy
- Add diagnosis code type validation
""")
        with cols[1]:
            with st.container(border=True):
                st.markdown("**🔧 Medium-term (1–2 months)**")
                st.markdown("""
- Add reference range mapping for OBSERVATION table
- Implement pre-load HCPCS validation
- Add diagnosis deduplication step
- Source professional claim financial fields
""")
        with cols[2]:
            with st.container(border=True):
                st.markdown("**📈 Ongoing monitoring**")
                st.markdown("""
- Track DQI score monthly (current: {:.0f}/100)
- Re-run after each ETL change and compare via Trends tab
- Set alerts when any field drops below red threshold
""".format(dqi_score))

        # ── Download recommendations ──────────────────────────────────────────
        st.divider()
        recs = [
            {"priority": "Critical", "issue": "HCPCS code validity", "records_affected": 129575,
             "table": "MEDICAL_CLAIM", "action": "Audit ETL HCPCS field mapping; add pre-load validation against CMS HCPCS Level II"},
            {"priority": "Critical", "issue": "Date fields too old", "records_affected": 71695,
             "table": "MEDICAL_CLAIM / ELIGIBILITY", "action": "Replace sentinel dates with NULL; add date range validation [1920, today]"},
            {"priority": "High", "issue": "OBSERVATION reference ranges 0% fill", "records_affected": 150244,
             "table": "OBSERVATION", "action": "Map reference range fields from EMR extract or join to LOINC reference ranges"},
            {"priority": "High", "issue": "Professional claim financial fields 0% fill", "records_affected": 133474,
             "table": "MEDICAL_CLAIM (professional)", "action": "Source financial fields from 835 remittance file or document as known gap"},
            {"priority": "High", "issue": "Duplicate diagnosis codes", "records_affected": 5144,
             "table": "MEDICAL_CLAIM", "action": "Add deduplication step in ETL; review source system adjudication logic"},
            {"priority": "Medium", "issue": "Diagnosis codes not in ICD-10-CM", "records_affected": 354,
             "table": "MEDICAL_CLAIM", "action": "Validate DIAGNOSIS_CODE_TYPE; check for un-migrated ICD-9 codes"},
            {"priority": "Medium", "issue": "Provider NPI validation failures", "records_affected": 17,
             "table": "MEDICAL_CLAIM", "action": "Cross-reference against NPPES registry; check for deactivated NPIs"},
            {"priority": "Medium", "issue": "Race / dual-status code mismatches", "records_affected": 834,
             "table": "ELIGIBILITY", "action": "Map source codes to Tuva terminology value sets; validate dual eligibility file load"},
        ]
        df_recs = pd.DataFrame(recs)
        st.download_button(
            label="⬇ Download recommendations as CSV",
            data=df_recs.to_csv(index=False).encode("utf-8"),
            file_name="dq_recommendations.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

st.divider()
st.caption("Data: Tuva synthetic dataset · DuckDB 1.5 · dbt 1.11 · Tuva Project 0.17.2")
