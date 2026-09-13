"""
Dashboard Utilities & Data Loading Module (dashboard/utils.py)
----------------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Cached data loading, model pipeline management, feature transformation,
             and prediction utilities for the Streamlit web dashboard.
"""

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Robust portable project root determination
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model_pipeline(pipeline_rel_path: str = "models/final_churn_pipeline.joblib"):
    """Loads and caches the trained final candidate churn prediction pipeline using portable paths."""
    pipeline_path = BASE_DIR / pipeline_rel_path
    if not pipeline_path.exists():
        st.error(f"Pipeline artifact not found at: {pipeline_path}")
        return None
    return joblib.load(pipeline_path)

@st.cache_resource
def load_segmentation_artifacts(
    model_rel_path: str = "models/customer_segmentation_model.joblib",
    scaler_rel_path: str = "models/segmentation_scaler.joblib"
):
    """Loads and caches K-Means clustering model and feature scaler using portable paths."""
    model_path = BASE_DIR / model_rel_path
    scaler_path = BASE_DIR / scaler_rel_path
    
    kmeans_model = joblib.load(model_path) if model_path.exists() else None
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return kmeans_model, scaler

@st.cache_data
def load_customer_data(
    cleaned_rel_csv: str = "data/processed/cleaned_customer_churn.csv",
    segments_rel_csv: str = "data/processed/customer_segments.csv",
    recommendations_rel_csv: str = "reports/customer_recommendations.csv"
) -> pd.DataFrame:
    """
    Loads, merges, and caches full customer master dataset combining raw attributes,
    K-Means segments, recommendations, risk levels, and probabilities using portable paths.
    """
    cleaned_csv = BASE_DIR / cleaned_rel_csv
    segments_csv = BASE_DIR / segments_rel_csv
    recommendations_csv = BASE_DIR / recommendations_rel_csv

    if not cleaned_csv.exists() or not segments_csv.exists():
        st.error(f"Missing core CSV datasets at {cleaned_csv} or {segments_csv}!")
        return pd.DataFrame()

    df_cleaned = pd.read_csv(cleaned_csv)
    df_seg = pd.read_csv(segments_csv)

    df_merged = df_cleaned.copy()
    df_merged['customer_id'] = df_seg['customer_id']
    df_merged['Cluster_ID'] = df_seg['Cluster_ID']
    df_merged['Segment_Name'] = df_seg['Segment_Name']
    df_merged['ServiceCount'] = df_seg['ServiceCount'] if 'ServiceCount' in df_seg.columns else 0
    df_merged['AverageMonthlySpend'] = df_seg['AverageMonthlySpend'] if 'AverageMonthlySpend' in df_seg.columns else df_merged['MonthlyCharges']

    if recommendations_csv.exists():
        df_recs = pd.read_csv(recommendations_csv)
        df_merged['churn_probability'] = df_recs['churn_probability']
        df_merged['risk_level'] = df_recs['risk_level']
        df_merged['key_risk_factors'] = df_recs['key_risk_factors']
        df_merged['recommendation_category'] = df_recs['recommendation_category']
        df_merged['recommendation'] = df_recs['recommendation']
        df_merged['priority'] = df_recs['priority']
        df_merged['reason'] = df_recs['reason']
        df_merged['hypothetical_intervention_impact'] = df_recs['hypothetical_intervention_impact']
    else:
        df_merged['churn_probability'] = 0.50
        df_merged['risk_level'] = "Medium Risk"

    return df_merged

@st.cache_data
def load_feature_importance(importance_rel_csv: str = "reports/feature_importance.csv") -> pd.DataFrame:
    """Loads and caches SHAP & native feature importances using portable paths."""
    importance_csv = BASE_DIR / importance_rel_csv
    if importance_csv.exists():
        return pd.read_csv(importance_csv)
    return pd.DataFrame()

@st.cache_data
def load_what_if_scenarios(what_if_rel_csv: str = "reports/what_if_results.csv") -> pd.DataFrame:
    """Loads and caches saved What-If scenario results using portable paths."""
    what_if_csv = BASE_DIR / what_if_rel_csv
    if what_if_csv.exists():
        return pd.read_csv(what_if_csv)
    return pd.DataFrame()

def prepare_single_customer_features(cust_dict: dict) -> pd.DataFrame:
    """
    Consistently computes engineered features (AverageMonthlySpend, ServiceCount, IsFirstYear)
    for a single customer dictionary input.
    """
    df_single = pd.DataFrame([cust_dict])

    df_single['AverageMonthlySpend'] = np.where(
        df_single['tenure'] == 0,
        df_single['MonthlyCharges'],
        df_single['TotalCharges'] / np.maximum(df_single['tenure'], 1)
    )

    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PhoneService', 'MultipleLines'
    ]
    df_single['ServiceCount'] = df_single[service_cols].apply(
        lambda row: sum(1 for val in row if str(val).strip() in ['Yes', 'Two lines']), axis=1
    )

    df_single['IsFirstYear'] = (df_single['tenure'] <= 12).astype(int)

    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
        'TotalCharges', 'AverageMonthlySpend', 'ServiceCount', 'IsFirstYear'
    ]
    return df_single[feature_order]

def predict_live_churn(cust_dict: dict, pipeline) -> dict:
    """
    Generates real-time churn probability, non-churn probability, and risk level for any custom input.
    """
    if pipeline is None:
        return {"churn_probability": 0.5, "risk_level": "Unknown", "predicted_class": 0}
        
    df_formatted = prepare_single_customer_features(cust_dict)
    probs = pipeline.predict_proba(df_formatted)[0]
    non_churn_prob, churn_prob = float(probs[0]), float(probs[1])

    if churn_prob >= 0.60:
        risk_level = "High Risk"
        pred_class = 1
    elif churn_prob >= 0.35:
        risk_level = "Medium Risk"
        pred_class = 1
    else:
        risk_level = "Low Risk"
        pred_class = 0

    return {
        "churn_probability": churn_prob,
        "non_churn_probability": non_churn_prob,
        "risk_level": risk_level,
        "predicted_class": pred_class
    }
