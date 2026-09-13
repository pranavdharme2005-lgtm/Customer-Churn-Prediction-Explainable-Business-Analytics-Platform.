# Stage 14: Final Project Executive Summary

This executive summary provides a high-level overview of the complete **Customer Churn Prediction & Explainable Business Analytics Platform**.

---

## 📌 Executive Summary
Customer churn is one of the most critical financial metrics for subscription-based businesses. Acquiring new customers costs up to **5x to 25x more** than retaining existing ones.

This platform bridges predictive machine learning, explainable AI, unsupervised segmentation, scenario simulation, and prescriptive business recommendations to deliver an end-to-end decision-support solution for retention management.

---

## 📊 Core Performance Metrics
- **Final Model**: Tuned Random Forest Classifier (`models/final_churn_pipeline.joblib`)
- **Evaluation Holdout Set**: 1,409 untouched customer accounts (20% test split)
- **Accuracy**: **76.30%**
- **Precision**: **53.77%**
- **Recall (Sensitivity)**: **76.20%** (Catches 285 out of 374 churned test accounts)
- **F1-Score**: **0.6305**
- **ROC-AUC**: **0.8394**

---

## 🔑 Key Analytics & Business Discoveries
1. **Contract Type Impact**: Month-to-month contracts exhibit a **42.71%** observed churn rate vs **11.27%** (1-Year) and **2.83%** (2-Year). SHAP analysis confirms `Contract_Month-to-month` is the #1 global churn driver (Mean |SHAP| = 0.0697).
2. **Onboarding Risk Window**: First-year customers (**0–12 months**) churn at **47.44%**, dropping to **9.51%** for customers over 4 years.
3. **High-Risk Segment Identification**: K-Means clustering ($K=4$) identified **Segment 2: High-Spend At-Risk Onboarders** (34.3% of customers) with an observed **46.38% churn rate**.
4. **Retention Interventions**: Scenario simulations indicate that upgrading a high-risk account to a 1-Year contract reduces estimated churn risk by **~27 percentage points**.

---

## 🛠️ Complete Repository Artifact Map
- **Cleaned Data**: `data/processed/cleaned_customer_churn.csv`
- **SQLite Database**: `data/customer_churn.db`
- **Tuned Model Pipeline**: `models/final_churn_pipeline.joblib`
- **Segmentation Model**: `models/customer_segmentation_model.joblib`
- **Streamlit Dashboard**: `dashboard/app.py`
- **Customer Recommendations**: `reports/customer_recommendations.csv`
- **Scenario Simulation Results**: `reports/what_if_results.csv`
- **SHAP Importances**: `reports/feature_importance.csv`
- **Documentation & Interview Package**: `reports/*.md`
