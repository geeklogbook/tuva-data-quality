import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DB_PATH = "/home/j/Documents/geeklogbook/tuva-data-quality/tuva.duckdb"

FEATURE_NAMES = ["patient_age", "sex_encoded", "total_paid", "conditions", "claim_lines"]


@st.cache_data
def load_hcc_summary():
    con = duckdb.connect(DB_PATH, read_only=True)
    return con.execute("""
        SELECT person_id, payer, patient_sex, patient_birth_date, patient_age, suspecting_gaps
        FROM main_hcc_suspecting.summary
        WHERE patient_age IS NOT NULL
        ORDER BY suspecting_gaps DESC
    """).df()


@st.cache_data
def load_ml_dataset():
    con = duckdb.connect(DB_PATH, read_only=True)
    return con.execute("""
        SELECT
            h.person_id,
            h.patient_age,
            h.patient_sex,
            h.suspecting_gaps,
            COALESCE(c.claim_lines, 0)  AS claim_lines,
            COALESCE(c.total_paid, 0.0) AS total_paid,
            COALESCE(cond.conditions, 0) AS conditions
        FROM main_hcc_suspecting.summary h
        LEFT JOIN (
            SELECT person_id,
                   COUNT(*)            AS claim_lines,
                   SUM(paid_amount)    AS total_paid
            FROM main_input_layer.medical_claim
            GROUP BY person_id
        ) c ON h.person_id = c.person_id
        LEFT JOIN (
            SELECT person_id, COUNT(DISTINCT normalized_code) AS conditions
            FROM main_core.condition
            GROUP BY person_id
        ) cond ON h.person_id = cond.person_id
        WHERE h.patient_age IS NOT NULL
    """).df()


@st.cache_data
def load_patient_conditions(person_id):
    con = duckdb.connect(DB_PATH, read_only=True)
    return con.execute("""
        SELECT normalized_code, normalized_description, condition_type, recorded_date
        FROM main_core.condition
        WHERE person_id = ?
        ORDER BY recorded_date DESC
        LIMIT 15
    """, [person_id]).df()


@st.cache_data
def load_patient_hcc_suspects(person_id):
    con = duckdb.connect(DB_PATH, read_only=True)
    return con.execute("""
        SELECT hcc_code, hcc_description, reason, contributing_factor, suspect_date
        FROM main_hcc_suspecting.list
        WHERE person_id = ?
        ORDER BY suspect_date DESC
        LIMIT 10
    """, [person_id]).df()


@st.cache_resource
def train_model():
    df = load_ml_dataset()
    if df.empty:
        return None

    median_gaps = df["suspecting_gaps"].median()
    df = df.copy()
    df["high_risk"]    = (df["suspecting_gaps"] >= median_gaps).astype(int)
    df["sex_encoded"]  = (df["patient_sex"].str.lower() == "male").astype(int)

    X = df[FEATURE_NAMES].fillna(0)
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    importances = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model":        clf,
        "accuracy":     acc,
        "importances":  importances,
        "cm":           cm,
        "median_gaps":  median_gaps,
        "X_full":       df[FEATURE_NAMES + ["person_id"]].copy(),
    }


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 HCC Risk & Condition Predictor")
st.caption("Medicare patients · HCC suspecting gaps · Random Forest · Synthetic dataset v0.15/0.16")

FEATURE_LABELS = {
    "claim_lines": "Number of claim lines",
    "conditions":  "Number of distinct conditions",
    "total_paid":  "Total paid amount",
    "patient_age": "Patient age",
    "sex_encoded": "Patient sex (male)",
}

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient risk overview",
    "Condition predictor (ML)",
    "Individual patient lookup",
    "Recommendations & export",
])

# ── Tab 1: Patient risk overview ──────────────────────────────────────────────
with tab1:
    st.subheader("HCC suspecting gaps by patient")

    df_hcc = load_hcc_summary()
    if df_hcc.empty:
        st.info("No HCC suspecting data available.")
    else:
        age_min = int(df_hcc["patient_age"].min())
        age_max = int(df_hcc["patient_age"].max())
        age_range = st.slider(
            "Filter by age range",
            min_value=age_min, max_value=age_max,
            value=(age_min, age_max), key="t1_age",
        )

        df_view = df_hcc[
            df_hcc["patient_age"].between(age_range[0], age_range[1])
        ]

        # Derived columns needed for both charts
        df_view = df_view.copy()
        df_view["age_group"] = pd.cut(
            df_view["patient_age"],
            bins=[0, 64, 69, 74, 79, 84, 89, 200],
            labels=["<65", "65–69", "70–74", "75–79", "80–84", "85–89", "90+"],
        ).astype(str)
        df_view["risk_band"] = pd.cut(
            df_view["suspecting_gaps"],
            bins=[-1, 4, 14, 29, 999],
            labels=["Low (0–4)", "Medium (5–14)", "High (15–29)", "Very high (30+)"],
        ).astype(str)

        median_gaps = df_view["suspecting_gaps"].median()
        n_high = (df_view["suspecting_gaps"] >= 15).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patients shown",      f"{len(df_view):,}")
        m2.metric("Median gaps / patient", f"{median_gaps:.0f}")
        m3.metric("High-risk patients (≥15 gaps)", f"{n_high:,}")
        m4.metric("% high-risk",         f"{100*n_high//max(len(df_view),1)}%")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Avg suspecting gaps by age group and sex")
            st.caption("Which demographic cohorts have the most unaddressed HCC conditions?")
            df_age = (
                df_view.groupby(["age_group", "patient_sex"], as_index=False)["suspecting_gaps"]
                .mean()
                .round(1)
            )
            df_age.columns = ["age_group", "patient_sex", "avg_gaps"]
            age_chart = (
                alt.Chart(df_age)
                .mark_bar()
                .encode(
                    x=alt.X("age_group:N", title="Age group",
                            sort=["<65","65–69","70–74","75–79","80–84","85–89","90+"]),
                    y=alt.Y("avg_gaps:Q", title="Avg suspecting gaps"),
                    color=alt.Color(
                        "patient_sex:N",
                        scale=alt.Scale(domain=["male","female"], range=["#4c78a8","#f58518"]),
                        legend=alt.Legend(title="Sex"),
                    ),
                    xOffset="patient_sex:N",
                    tooltip=["age_group", "patient_sex",
                             alt.Tooltip("avg_gaps:Q", title="Avg gaps", format=".1f")],
                )
                .properties(height=320)
            )
            st.altair_chart(age_chart, use_container_width=True)
            best_group = df_age.loc[df_age["avg_gaps"].idxmax()]
            st.caption(
                f"Highest-risk cohort: **{best_group['age_group']} {best_group['patient_sex']}** "
                f"with avg **{best_group['avg_gaps']:.1f} gaps** per patient."
            )

        with col_b:
            st.markdown("##### How many patients are at each risk level?")
            st.caption("Distribution of suspecting gaps — shows where to concentrate outreach effort.")
            df_band = (
                df_view.groupby(["risk_band", "patient_sex"], as_index=False)
                .size()
                .rename(columns={"size": "patients"})
            )
            band_order = ["Low (0–4)", "Medium (5–14)", "High (15–29)", "Very high (30+)"]
            band_colors = {"Low (0–4)": "#2ca02c", "Medium (5–14)": "#f0c05a",
                           "High (15–29)": "#ff7f0e", "Very high (30+)": "#d62728"}
            band_chart = (
                alt.Chart(df_band)
                .mark_bar()
                .encode(
                    x=alt.X("risk_band:N", title="Risk band", sort=band_order),
                    y=alt.Y("patients:Q", title="Number of patients"),
                    color=alt.Color(
                        "risk_band:N",
                        scale=alt.Scale(
                            domain=list(band_colors.keys()),
                            range=list(band_colors.values()),
                        ),
                        legend=None,
                    ),
                    column=alt.Column("patient_sex:N", title="Sex",
                                      header=alt.Header(labelAngle=0)),
                    tooltip=["risk_band", "patient_sex", "patients"],
                )
                .properties(width=180, height=320)
            )
            st.altair_chart(band_chart)
            n_very_high = (df_view["suspecting_gaps"] >= 30).sum()
            st.caption(
                f"**{n_very_high} patients** have 30+ suspecting gaps — "
                f"immediate priority for care management outreach."
            )

        st.markdown("##### Top 10 patients by suspecting gaps")
        st.dataframe(
            df_view.head(10)[["person_id", "payer", "patient_sex", "patient_age", "suspecting_gaps"]],
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 2: ML model ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest — high-risk classifier")

    result = train_model()
    if result is None:
        st.info("Not enough data to train a model.")
    else:
        acc      = result["accuracy"]
        imps     = result["importances"]
        cm       = result["cm"]
        median_g = result["median_gaps"]

        st.caption(
            f"Binary target: **high_risk = 1** if suspecting_gaps ≥ {median_g:.0f} (dataset median). "
            f"Features: {', '.join(FEATURE_NAMES)}. 80/20 train/test split, 200 trees."
        )

        col_acc, col_blank = st.columns([1, 3])
        col_acc.metric("Test accuracy", f"{acc:.1%}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Feature importances**")
            fi_chart = (
                alt.Chart(imps)
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title="Importance", scale=alt.Scale(domain=[0, imps["importance"].max() * 1.1])),
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    color=alt.Color("importance:Q",
                        scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
                )
                .properties(height=240)
            )
            st.altair_chart(fi_chart, use_container_width=True)

        with col_b:
            st.markdown("**Confusion matrix**")
            labels = ["Low risk (0)", "High risk (1)"]
            cm_df = pd.DataFrame([
                {"actual": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
                for i in range(2)
                for j in range(2)
            ])
            total = cm_df["count"].sum()
            cm_df["pct"] = (cm_df["count"] / total * 100).round(1)
            cm_df["label"] = cm_df["count"].astype(str) + "\n(" + cm_df["pct"].astype(str) + "%)"

            heatmap = (
                alt.Chart(cm_df)
                .mark_rect()
                .encode(
                    x=alt.X("predicted:N", title="Predicted"),
                    y=alt.Y("actual:N", title="Actual"),
                    color=alt.Color("count:Q",
                        scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["actual", "predicted", "count", "pct"],
                )
                .properties(height=240)
            )
            cm_text = heatmap.mark_text(fontSize=14).encode(
                text="label:N",
                color=alt.value("white"),
            )
            st.altair_chart(heatmap + cm_text, use_container_width=True)

# ── Tab 3: Individual patient lookup ─────────────────────────────────────────
with tab3:
    st.subheader("Individual patient risk profile")

    df_hcc_all = load_hcc_summary()
    if df_hcc_all.empty:
        st.info("No patient data available.")
    else:
        person_ids = sorted(df_hcc_all["person_id"].unique())
        sel_pid = st.selectbox("Select patient ID", person_ids, key="t3_pid")

        row = df_hcc_all[df_hcc_all["person_id"] == sel_pid].iloc[0]

        # Demographics
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Age",             row["patient_age"])
        d2.metric("Sex",             row["patient_sex"].title() if pd.notna(row["patient_sex"]) else "—")
        d3.metric("Payer",           row["payer"].title() if pd.notna(row["payer"]) else "—")
        d4.metric("Suspecting gaps", row["suspecting_gaps"])

        # Risk score from trained model
        result = train_model()
        if result is not None:
            model  = result["model"]
            X_full = result["X_full"]

            patient_row = X_full[X_full["person_id"] == sel_pid]
            if not patient_row.empty:
                feat_vec = patient_row[FEATURE_NAMES].fillna(0).values
                prob = model.predict_proba(feat_vec)[0][1]

                st.markdown("---")
                st.markdown("##### Risk score")
                if prob < 0.4:
                    st.success(f"🟢 Low Risk — predicted probability: **{prob:.0%}**")
                elif prob < 0.7:
                    st.warning(f"🟡 Medium Risk — predicted probability: **{prob:.0%}**")
                else:
                    st.error(f"🔴 High Risk — predicted probability: **{prob:.0%}**")

                r_feat = patient_row[FEATURE_NAMES].iloc[0]
                f1, f2, f3 = st.columns(3)
                f1.metric("Total paid",   f"${r_feat['total_paid']:,.0f}")
                f2.metric("Claim lines",  f"{int(r_feat['claim_lines']):,}")
                f3.metric("Conditions",   f"{int(r_feat['conditions'])}")

        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### HCC suspects")
            df_suspects = load_patient_hcc_suspects(sel_pid)
            if df_suspects.empty:
                st.info("No HCC suspects found for this patient.")
            else:
                st.dataframe(df_suspects, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("##### Top conditions")
            df_cond = load_patient_conditions(sel_pid)
            if df_cond.empty:
                st.info("No conditions found for this patient.")
            else:
                st.dataframe(df_cond, use_container_width=True, hide_index=True)

# ── Tab 4: Recommendations & export ──────────────────────────────────────────
with tab4:
    st.subheader("Recommendations & export")

    result = train_model()
    if result is None:
        st.info("Model not available — cannot generate recommendations.")
    else:
        model      = result["model"]
        X_full     = result["X_full"]
        imps       = result["importances"]
        acc        = result["accuracy"]
        median_g   = result["median_gaps"]
        df_hcc_rec = load_hcc_summary()

        # ── Score every patient ───────────────────────────────────────────────
        X_all    = X_full[FEATURE_NAMES].fillna(0)
        probs    = model.predict_proba(X_all)[:, 1]
        df_scored = X_full[["person_id"]].copy()
        df_scored["risk_probability"] = probs.round(3)
        df_scored = df_scored.merge(
            df_hcc_rec[["person_id", "patient_age", "patient_sex", "payer", "suspecting_gaps"]],
            on="person_id", how="left",
        ).sort_values("risk_probability", ascending=False).reset_index(drop=True)
        df_scored["risk_tier"] = pd.cut(
            df_scored["risk_probability"],
            bins=[-0.001, 0.4, 0.7, 1.001],
            labels=["Low", "Medium", "High"],
        )

        n_high   = (df_scored["risk_tier"] == "High").sum()
        n_medium = (df_scored["risk_tier"] == "Medium").sum()
        n_low    = (df_scored["risk_tier"] == "Low").sum()
        n_total  = len(df_scored)
        top_feat = imps.iloc[0]
        top2_feat = imps.iloc[1]

        # ── Key insights ─────────────────────────────────────────────────────
        st.markdown("### Key findings")
        st.markdown(f"""
**{n_high} out of {n_total} patients ({100*n_high//n_total}%) are classified as high risk.**
These patients have suspecting gaps well above the dataset median of **{median_g:.0f} gaps** and
represent the population most likely to have unrecognized HCC conditions that could affect
risk adjustment and care management.

**The model explains {acc:.0%} of cases correctly** using 5 clinical and claims features.
The two strongest predictors are:

1. **{FEATURE_LABELS[top_feat['feature']]}** — accounts for {top_feat['importance']:.0%} of the model's decisions.
   Patients with higher {top_feat['feature'].replace('_', ' ')} tend to have more suspecting gaps,
   suggesting they are more complex clinically and may have more unaddressed conditions.

2. **{FEATURE_LABELS[top2_feat['feature']]}** — accounts for {top2_feat['importance']:.0%} of decisions.
   This reinforces that **utilization and clinical complexity are the primary drivers of HCC risk**
   in this population, ahead of demographics like age or sex.

**Sex has minimal predictive power** ({imps[imps['feature']=='sex_encoded']['importance'].values[0]:.0%}),
meaning risk is driven by clinical history, not patient demographics.
""")

        st.divider()

        # ── Recommended actions ───────────────────────────────────────────────
        st.markdown("### Recommended actions")

        col_h, col_m, col_l = st.columns(3)
        with col_h:
            with st.container(border=True):
                st.markdown("#### 🔴 High risk")
                st.markdown(f"**{n_high} patients**")
                st.markdown("""
- Schedule HCC gap-closure reviews
- Prioritize for care management outreach
- Validate open HCC codes before annual risk adjustment submission
- Review top conditions for coding completeness
""")
        with col_m:
            with st.container(border=True):
                st.markdown("#### 🟡 Medium risk")
                st.markdown(f"**{n_medium} patients**")
                st.markdown("""
- Flag for next routine visit follow-up
- Run suspect-condition report and share with PCP
- Monitor claim volume trends — rising utilization may signal deterioration
""")
        with col_l:
            with st.container(border=True):
                st.markdown("#### 🟢 Low risk")
                st.markdown(f"**{n_low} patients**")
                st.markdown("""
- No immediate action required
- Re-score quarterly as new claims arrive
- Watch for sudden spikes in conditions or paid amount
""")

        st.divider()

        # ── Exportable ranked patient table ──────────────────────────────────
        st.markdown("### Ranked patient list")
        st.caption("All patients scored by the model, sorted by risk probability descending.")

        tier_filter = st.multiselect(
            "Filter by risk tier",
            ["High", "Medium", "Low"],
            default=["High", "Medium"],
            key="rec_tier",
        )
        df_export = df_scored[df_scored["risk_tier"].isin(tier_filter)].copy()
        df_export.index = range(1, len(df_export) + 1)

        st.dataframe(
            df_export[["person_id", "patient_age", "patient_sex", "payer",
                        "suspecting_gaps", "risk_probability", "risk_tier"]],
            use_container_width=True,
            column_config={
                "risk_probability": st.column_config.ProgressColumn(
                    "Risk probability", min_value=0, max_value=1, format="%.0%%"
                ),
            },
        )

        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download {len(df_export)} patients as CSV",
            data=csv_bytes,
            file_name="hcc_risk_scores.csv",
            mime="text/csv",
        )
