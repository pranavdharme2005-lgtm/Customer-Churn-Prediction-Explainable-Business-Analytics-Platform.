"""
Customer Churn Prediction & Explainable Business Analytics Platform (dashboard/app.py)
-------------------------------------------------------------------------------------
Streamlit Web Application combining ML Predictions, SHAP Explainable AI,
Unsupervised Customer Segmentation, What-If Scenario Simulations, and Business Recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_model_pipeline,
    load_segmentation_artifacts,
    load_customer_data,
    load_feature_importance,
    load_what_if_scenarios,
    predict_live_churn,
    prepare_single_customer_features
)
from components import (
    render_custom_css,
    render_non_causal_banner,
    render_kpi_card,
    create_risk_distribution_chart,
    create_probability_histogram,
    create_segment_churn_chart,
    create_feature_importance_chart,
    create_what_if_comparison_chart
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn Analytics & Recommendation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom CSS
render_custom_css()

# Load Cached Models & Datasets
pipeline = load_model_pipeline()
kmeans_model, scaler = load_segmentation_artifacts()
df_master = load_customer_data()
df_importance = load_feature_importance()
df_what_if = load_what_if_scenarios()

# Sidebar Navigation Header
st.sidebar.markdown("## 📊 Churn Analytics Platform")
st.sidebar.markdown("*Predict • Explain • Segment • Recommend*")
st.sidebar.divider()

# Navigation Radio Options
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Executive Overview",
        "2. Customer Risk Analysis",
        "3. Customer Segmentation",
        "4. Explainable AI (SHAP)",
        "5. What-If Scenario Analysis",
        "6. Business Recommendations",
        "7. Customer Risk Predictor",
        "8. Model & Dataset Info"
    ]
)

st.sidebar.divider()

# Global Sidebar Filters (Applicable to Tables & Analytics)
st.sidebar.markdown("### 🎛️ Global Data Filters")
if not df_master.empty:
    risk_filter = st.sidebar.multiselect(
        "Filter by Risk Level",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"]
    )
    
    seg_filter = st.sidebar.multiselect(
        "Filter by Customer Segment",
        options=sorted(df_master['Segment_Name'].dropna().unique()),
        default=sorted(df_master['Segment_Name'].dropna().unique())
    )

    contract_filter = st.sidebar.multiselect(
        "Filter by Contract Type",
        options=sorted(df_master['Contract'].dropna().unique()),
        default=sorted(df_master['Contract'].dropna().unique())
    )

    # Filter Master DataFrame
    df_filtered = df_master[
        (df_master['risk_level'].isin(risk_filter)) &
        (df_master['Segment_Name'].isin(seg_filter)) &
        (df_master['Contract'].isin(contract_filter))
    ].copy()
else:
    df_filtered = pd.DataFrame()

# Footer in Sidebar
st.sidebar.caption("Customer Churn Analytics Platform v1.0 | Stage 12 Dashboard")

# ==============================================================================
# SECTION 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_selection == "1. Executive Overview":
    st.markdown('<div class="header-title">Executive Churn Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">High-level KPIs, risk distribution, and portfolio churn analytics</div>', unsafe_allow_html=True)
    
    render_non_causal_banner()

    if df_master.empty:
        st.error("No dataset available. Please ensure Stage 9 & 11 outputs are generated.")
    else:
        total_customers = len(df_master)
        high_risk_count = (df_master['risk_level'] == 'High Risk').sum()
        med_risk_count = (df_master['risk_level'] == 'Medium Risk').sum()
        low_risk_count = (df_master['risk_level'] == 'Low Risk').sum()
        overall_churn_rate = (df_master['Churn'] == 'Yes').mean()
        avg_churn_prob = df_master['churn_probability'].mean()
        avg_monthly_charges = df_master['MonthlyCharges'].mean()
        avg_tenure = df_master['tenure'].mean()

        # Render KPI Cards in Columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Total Accounts", f"{total_customers:,}", "Active Customer Base", "#1f77b4")
        with c2:
            render_kpi_card("High-Risk Accounts", f"{high_risk_count:,}", f"{high_risk_count/total_customers*100:.1f}% of portfolio", "#d62728")
        with c3:
            render_kpi_card("Medium-Risk Accounts", f"{med_risk_count:,}", f"{med_risk_count/total_customers*100:.1f}% of portfolio", "#ff7f0e")
        with c4:
            render_kpi_card("Low-Risk Accounts", f"{low_risk_count:,}", f"{low_risk_count/total_customers*100:.1f}% of portfolio", "#2ca02c")

        st.markdown("<br>", unsafe_allow_html=True)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            render_kpi_card("Observed Churn Rate", f"{overall_churn_rate*100:.1f}%", "Historical benchmark", "#9467bd")
        with c6:
            render_kpi_card("Avg Churn Probability", f"{avg_churn_prob*100:.1f}%", "Model estimated mean", "#8c564b")
        with c7:
            render_kpi_card("Avg Monthly Bill", f"${avg_monthly_charges:.2f}", "Per account / month", "#e377c2")
        with c8:
            render_kpi_card("Avg Tenure", f"{avg_tenure:.1f} mos", "Account longevity", "#7f7f7f")

        st.divider()

        # Visual Analytics Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_pie = create_risk_distribution_chart(df_filtered)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_chart2:
            fig_hist = create_probability_histogram(df_filtered)
            st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# SECTION 2: CUSTOMER RISK ANALYSIS
# ==============================================================================
elif nav_selection == "2. Customer Risk Analysis":
    st.markdown('<div class="header-title">Customer Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Prioritized high-risk customer queues and filterable account tables</div>', unsafe_allow_html=True)

    if df_filtered.empty:
        st.warning("No customers match the current sidebar filter selections.")
    else:
        st.markdown(f"### 📋 Filtered Accounts ({len(df_filtered):,} accounts found)")

        # Search Bar
        search_id = st.text_input("🔍 Search by Customer ID (e.g. CUST-00006)", "")
        
        df_display = df_filtered.copy()
        if search_id.strip():
            df_display = df_display[df_display['customer_id'].str.contains(search_id.strip(), case=False, na=False)]

        # Format and display table
        display_cols = [
            'customer_id', 'churn_probability', 'risk_level', 'Segment_Name',
            'Contract', 'tenure', 'MonthlyCharges', 'InternetService', 'priority', 'recommendation'
        ]
        
        df_table = df_display[display_cols].sort_values(by='churn_probability', ascending=False)
        
        st.dataframe(
            df_table.style.format({
                'churn_probability': '{:.2%}',
                'MonthlyCharges': '${:.2f}',
                'tenure': '{:.0f} mos'
            }),
            use_container_width=True,
            height=450
        )

        st.divider()
        st.markdown("### 🚨 High-Priority Retention Queue (Top 10 High-Risk Accounts)")
        top_high_risk = df_master[df_master['risk_level'] == 'High Risk'].sort_values(by='churn_probability', ascending=False).head(10)
        st.dataframe(
            top_high_risk[['customer_id', 'churn_probability', 'risk_level', 'Segment_Name', 'key_risk_factors', 'recommendation_category', 'priority']].style.format({
                'churn_probability': '{:.2%}'
            }),
            use_container_width=True
        )

# ==============================================================================
# SECTION 3: CUSTOMER SEGMENTATION
# ==============================================================================
elif nav_selection == "3. Customer Segmentation":
    st.markdown('<div class="header-title">Unsupervised Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Behavioral customer archetypes generated via K-Means Clustering (Stage 9)</div>', unsafe_allow_html=True)

    if not df_master.empty:
        # Segment Profiles Summary Table
        seg_summary = df_master.groupby('Segment_Name').agg(
            Customer_Count=('customer_id', 'count'),
            Avg_Tenure=('tenure', 'mean'),
            Avg_Monthly_Bill=('MonthlyCharges', 'mean'),
            Avg_Total_Spend=('TotalCharges', 'mean'),
            Avg_Service_Count=('ServiceCount', 'mean'),
            Avg_Churn_Probability=('churn_probability', 'mean'),
            Observed_Churn_Rate=('Churn', lambda x: (x == 'Yes').mean())
        ).reset_index()

        seg_summary['Percentage'] = (seg_summary['Customer_Count'] / len(df_master) * 100).round(1)

        st.markdown("### 📊 Segment Statistics & Benchmark Metrics")
        st.dataframe(
            seg_summary[['Segment_Name', 'Customer_Count', 'Percentage', 'Avg_Tenure', 'Avg_Monthly_Bill', 'Avg_Total_Spend', 'Avg_Service_Count', 'Observed_Churn_Rate', 'Avg_Churn_Probability']].style.format({
                'Percentage': '{:.1f}%',
                'Avg_Tenure': '{:.1f} mos',
                'Avg_Monthly_Bill': '${:.2f}',
                'Avg_Total_Spend': '${:.2f}',
                'Avg_Service_Count': '{:.1f}',
                'Observed_Churn_Rate': '{:.1%}',
                'Avg_Churn_Probability': '{:.1%}'
            }),
            use_container_width=True
        )

        st.divider()

        # Visual Charts
        col_seg1, col_seg2 = st.columns(2)
        with col_seg1:
            fig_seg_churn = create_segment_churn_chart(df_master)
            st.plotly_chart(fig_seg_churn, use_container_width=True)
        with col_seg2:
            fig_scatter = px.scatter(
                df_filtered,
                x="tenure",
                y="MonthlyCharges",
                color="Segment_Name",
                opacity=0.6,
                title="<b>Tenure vs Monthly Charges by Segment</b>",
                labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# SECTION 4: EXPLAINABLE AI (SHAP)
# ==============================================================================
elif nav_selection == "4. Explainable AI (SHAP)":
    st.markdown('<div class="header-title">Explainable AI & Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Global and local model interpretability powered by SHAP (Stage 8)</div>', unsafe_allow_html=True)

    render_non_causal_banner()

    col_exp1, col_exp2 = st.columns([1, 1])

    with col_exp1:
        st.markdown("### 🌐 Global Feature Importance")
        if not df_importance.empty:
            fig_imp = create_feature_importance_chart(df_importance, top_n=10)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance CSV not found.")

    with col_exp2:
        st.markdown("### 👤 Local Individual Customer SHAP Lookup")
        if not df_master.empty:
            selected_cust_id = st.selectbox(
                "Select Customer ID for Local Explanation",
                options=df_master['customer_id'].head(100).tolist(),
                index=0
            )
            
            cust_data = df_master[df_master['customer_id'] == selected_cust_id].iloc[0]

            st.markdown(f"**Customer ID**: `{cust_data['customer_id']}`")
            st.markdown(f"**Churn Probability**: `{cust_data['churn_probability']*100:.1f}%`")
            st.markdown(f"**Risk Level**: `{cust_data['risk_level']}`")
            st.markdown(f"**Segment**: `{cust_data['Segment_Name']}`")
            st.markdown(f"**Key Risk Factors**: `{cust_data['key_risk_factors']}`")

            st.success(f"**Human-Readable Explanation**: {cust_data['reason']}")

# ==============================================================================
# SECTION 5: WHAT-IF SCENARIO ANALYSIS
# ==============================================================================
elif nav_selection == "5. What-If Scenario Analysis":
    st.markdown('<div class="header-title">What-If Churn Analysis & Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Simulate hypothetical customer attribute changes and observe model probability response</div>', unsafe_allow_html=True)

    render_non_causal_banner()

    if not df_master.empty:
        col_select, col_sim = st.columns([1, 2])

        with col_select:
            st.markdown("### 1. Select Customer & View Baseline")
            cust_id_whatif = st.selectbox(
                "Choose Customer for Simulation",
                options=df_master['customer_id'].head(100).tolist(),
                index=0
            )
            base_cust = df_master[df_master['customer_id'] == cust_id_whatif].iloc[0].to_dict()

            base_res = predict_live_churn(base_cust, pipeline)
            base_prob = base_res["churn_probability"]

            st.metric("Baseline Churn Probability", f"{base_prob*100:.1f}%", delta=None)
            st.markdown(f"**Baseline Risk Level**: `{base_res['risk_level']}`")
            st.markdown(f"**Current Contract**: `{base_cust.get('Contract')}`")
            st.markdown(f"**Current Tenure**: `{base_cust.get('tenure')} months`")
            st.markdown(f"**Current Monthly Bill**: `${base_cust.get('MonthlyCharges'):.2f}`")

        with col_sim:
            st.markdown("### 2. Tweak Scenario Attributes")

            new_tenure = st.slider("Hypothetical Tenure (Months)", 1, 72, int(base_cust.get('tenure', 12)))
            new_charges = st.slider("Hypothetical Monthly Bill ($)", 18.0, 120.0, float(base_cust.get('MonthlyCharges', 65.0)))
            new_contract = st.selectbox("Hypothetical Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(base_cust.get('Contract', 'Month-to-month')))
            new_tech = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(base_cust.get('TechSupport', 'No')))
            new_sec = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(base_cust.get('OnlineSecurity', 'No')))
            new_pay = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(base_cust.get('PaymentMethod', 'Electronic check')))

            # Construct Scenario Dict
            scenario_cust = base_cust.copy()
            scenario_cust['tenure'] = new_tenure
            scenario_cust['MonthlyCharges'] = new_charges
            scenario_cust['TotalCharges'] = new_tenure * new_charges
            scenario_cust['Contract'] = new_contract
            scenario_cust['TechSupport'] = new_tech
            scenario_cust['OnlineSecurity'] = new_sec
            scenario_cust['PaymentMethod'] = new_pay

            # Run Scenario Prediction
            sc_res = predict_live_churn(scenario_cust, pipeline)
            sc_prob = sc_res["churn_probability"]
            delta_pp = (sc_prob - base_prob) * 100

            st.divider()
            
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.metric("Baseline Risk", f"{base_prob*100:.1f}%")
            with c_m2:
                st.metric("Scenario Risk", f"{sc_prob*100:.1f}%")
            with c_m3:
                st.metric("Risk Delta", f"{delta_pp:+.1f} pp", delta_color="inverse")

            fig_gauge = create_what_if_comparison_chart(base_prob, sc_prob)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================================================================
# SECTION 6: BUSINESS RECOMMENDATIONS
# ==============================================================================
elif nav_selection == "6. Business Recommendations":
    st.markdown('<div class="header-title">Business Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Prescriptive decision-support directives derived from Stage 11 rules</div>', unsafe_allow_html=True)

    render_non_causal_banner()

    if not df_master.empty:
        st.markdown("### 💡 Search Customer Recommendation Card")
        rec_cust_id = st.selectbox("Select Customer ID", options=df_master['customer_id'].head(100).tolist(), index=0)
        rec_row = df_master[df_master['customer_id'] == rec_cust_id].iloc[0]

        r_col1, r_col2 = st.columns([1, 2])
        with r_col1:
            st.markdown(f"**Customer ID**: `{rec_row['customer_id']}`")
            st.markdown(f"**Risk Level**: `{rec_row['risk_level']}`")
            st.markdown(f"**Priority Level**: `{rec_row['priority']}`")
            st.markdown(f"**Category**: `{rec_row['recommendation_category']}`")
            st.markdown(f"**Segment**: `{rec_row['Segment_Name']}`")

        with r_col2:
            st.success(f"**Action Directive**: {rec_row['recommendation']}")
            st.info(f"**Business Rationale**: {rec_row['reason']}")
            st.warning(f"**Scenario Impact Context**: {rec_row['hypothetical_intervention_impact']}")

# ==============================================================================
# SECTION 7: CUSTOMER RISK PREDICTOR
# ==============================================================================
elif nav_selection == "7. Customer Risk Predictor":
    st.markdown('<div class="header-title">Live Customer Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Input custom customer parameters to evaluate real-time churn risk</div>', unsafe_allow_html=True)

    with st.form("custom_predict_form"):
        st.markdown("#### Input Custom Account Parameters")
        
        f1, f2, f3 = st.columns(3)
        with f1:
            in_tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)
            in_monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0)
            in_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

        with f2:
            in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            in_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

        with f3:
            in_pay = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            in_senior = st.selectbox("Senior Citizen", [0, 1])
            in_paper = st.selectbox("Paperless Billing", ["Yes", "No"])

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk")

    if submit_btn:
        custom_cust = {
            'gender': 'Male',
            'SeniorCitizen': in_senior,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': in_tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': in_internet,
            'OnlineSecurity': in_sec,
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': in_tech,
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': in_contract,
            'PaperlessBilling': in_paper,
            'PaymentMethod': in_pay,
            'MonthlyCharges': in_monthly,
            'TotalCharges': in_tenure * in_monthly
        }

        res = predict_live_churn(custom_cust, pipeline)
        
        st.divider()
        st.markdown("### 📊 Prediction Result")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.metric("Predicted Churn Probability", f"{res['churn_probability']*100:.1f}%")
        with p_c2:
            st.markdown(f"**Assigned Risk Tier**: `{res['risk_level']}`")

# ==============================================================================
# SECTION 8: MODEL & DATASET INFO
# ==============================================================================
elif nav_selection == "8. Model & Dataset Info":
    st.markdown('<div class="header-title">Model & Dataset Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Performance benchmarks and evaluation metrics from Stages 6 & 7</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi_card("Accuracy", "76.30%", "Overall Test Correctness", "#1f77b4")
    with m2:
        render_kpi_card("Precision", "53.77%", "Positive Predictive Value", "#ff7f0e")
    with m3:
        render_kpi_card("Recall", "76.20%", "Sensitivity / Churn Caught", "#2ca02c")
    with m4:
        render_kpi_card("F1-Score", "0.6305", "Harmonic Mean", "#d62728")
    with m5:
        render_kpi_card("ROC-AUC", "0.8394", "Area Under ROC Curve", "#9467bd")

    st.divider()
    st.markdown("### 📋 Final Model Specification")
    st.markdown("""
    - **Selected Algorithm**: Tuned Random Forest Classifier (`max_depth=12`, `n_estimators=150`, `class_weight='balanced'`)
    - **Optimization Strategy**: 5-Fold `StratifiedKFold` Cross-Validation via `GridSearchCV` & `RandomizedSearchCV`.
    - **Evaluation Dataset**: 1,409 untouched test customer accounts (20% holdout).
    - **Key Business Objective**: Maximize **Recall (76.20%)** to catch as many churners as possible while maintaining strong ROC-AUC (0.8394).
    """)
