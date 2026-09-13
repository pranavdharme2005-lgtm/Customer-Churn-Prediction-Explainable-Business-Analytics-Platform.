"""
Dashboard Visual Components Module (dashboard/components.py)
--------------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Custom CSS styling, KPI metric cards, non-causal alerts,
             and interactive Plotly chart generators for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_custom_css():
    """Injects custom CSS styling for modern UI aesthetics and clean glassmorphism KPI cards."""
    st.markdown("""
        <style>
        /* Modern font & body styling */
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Custom Header Styling */
        .header-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1f77b4 0%, #00b4d8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .header-subtitle {
            font-size: 1.05rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }
        
        /* KPI Metric Cards */
        .kpi-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #212529;
            margin: 0.3rem 0;
        }
        .kpi-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #6c757d;
        }
        .kpi-subtext {
            font-size: 0.78rem;
            color: #adb5bd;
        }
        
        /* Badges */
        .badge-high {
            background-color: #ffebe9;
            color: #d62728;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .badge-med {
            background-color: #fff4e6;
            color: #ff7f0e;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .badge-low {
            background-color: #e6f4ea;
            color: #2ca02c;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        /* Alert Box */
        .disclaimer-box {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 0.9rem 1.2rem;
            border-radius: 4px;
            margin-bottom: 1.2rem;
            font-size: 0.88rem;
            color: #495057;
        }
        </style>
    """, unsafe_allow_html=True)

def render_non_causal_banner(text: str = None):
    """Renders a non-causal decision-support alert banner."""
    default_text = (
        "**Decision-Support & Non-Causal Notice**: Models and scenario simulations measure statistical "
        "associations and probability sensitivity under hypothetical inputs. They do **not** prove physical "
        "causation, and recommendations serve as decision-support guidance for human account managers."
    )
    st.info(text if text else default_text)

def render_kpi_card(label: str, value: str, subtext: str = "", accent_color: str = "#1f77b4"):
    """Renders a styled KPI metric card in HTML."""
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 4px solid {accent_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def create_risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a Plotly donut chart of Risk Level distribution."""
    risk_counts = df['risk_level'].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        hole=0.55,
        color=risk_counts.index,
        color_discrete_map={
            "Low Risk": "#2ca02c",
            "Medium Risk": "#ff7f0e",
            "High Risk": "#d62728"
        },
        title="<b>Customer Breakdown by Risk Tier</b>"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_probability_histogram(df: pd.DataFrame) -> go.Figure:
    """Creates a Plotly histogram of churn probabilities with threshold lines."""
    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=40,
        color="risk_level",
        color_discrete_map={
            "Low Risk": "#2ca02c",
            "Medium Risk": "#ff7f0e",
            "High Risk": "#d62728"
        },
        title="<b>Distribution of Predicted Churn Probabilities</b>",
        labels={"churn_probability": "Predicted Churn Probability", "count": "Customer Count"}
    )
    fig.add_vline(x=0.35, line_dash="dash", line_color="#ff7f0e", annotation_text="Medium Risk (0.35)")
    fig.add_vline(x=0.60, line_dash="dash", line_color="#d62728", annotation_text="High Risk (0.60)")
    fig.update_layout(
        xaxis_tickformat=".0%",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def create_segment_churn_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a Plotly bar chart comparing average churn probability by Customer Segment."""
    seg_df = df.groupby('Segment_Name')['churn_probability'].mean().reset_index().sort_values(by='churn_probability', ascending=False)
    fig = px.bar(
        seg_df,
        x="churn_probability",
        y="Segment_Name",
        orientation="h",
        color="churn_probability",
        color_continuous_scale="Reds",
        title="<b>Average Churn Probability by Segment</b>",
        labels={"churn_probability": "Avg Churn Prob", "Segment_Name": ""}
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

def create_feature_importance_chart(df_imp: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Creates a Plotly horizontal bar chart of top SHAP / native feature importances."""
    if df_imp.empty:
        return go.Figure()
    top_df = df_imp.head(top_n).sort_values(by="Importance", ascending=True)
    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title=f"<b>Top {top_n} Global Feature Drivers</b>",
        labels={"Importance": "Feature Contribution / Importance", "Feature": ""}
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

def create_what_if_comparison_chart(baseline_prob: float, scenario_prob: float) -> go.Figure:
    """Creates a Plotly indicator gauge comparing baseline vs scenario churn probability."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=scenario_prob * 100,
        number={'suffix': '%', 'valueformat': '.1f'},
        delta={'reference': baseline_prob * 100, 'relative': False, 'valueformat': '+.1f', 'suffix': ' pp'},
        title={'text': "<b>Scenario Churn Risk</b><br><span style='font-size:0.8em;color:gray;'>vs Baseline Risk</span>"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        height=220,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig
