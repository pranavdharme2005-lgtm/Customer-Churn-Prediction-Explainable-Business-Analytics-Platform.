# Customer Churn Prediction & Explainable Business Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-green.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning, Explainable AI (XAI), Unsupervised Customer Segmentation, and Prescriptive Business Analytics platform designed to predict telecom customer churn, explain model drivers, simulate retention scenarios, and operationalize recommendations via an interactive Streamlit web dashboard.

---

## 🌐 Live Demo & Quick Start

### 🚀 Live Demo
`Live Demo: To be added after deployment`

### 💻 Run Locally in 3 Steps
```bash
# 1. Clone repository & navigate to project root
cd customer-churn-prediction

# 2. Install pinned production dependencies
pip install -r requirements.txt

# 3. Launch interactive Streamlit Web Application
streamlit run dashboard/app.py
```
The application will launch automatically at `http://localhost:8501`.

---

## 📌 1. Project Overview

Customer churn (the percentage of subscribers who stop doing business with a company) is one of the most critical metrics for subscription-based industries. Acquiring a new customer can cost up to **5x to 25x more** than retaining an existing one. Predicting which customers are likely to churn allows telecommunications providers to proactively intervene, protect revenue, and boost customer lifetime value (CLV).

However, **prediction alone is not enough**. A raw machine learning probability score (e.g., `87.2%`) tells business managers *who* is likely to leave, but fails to explain *why* or suggest *what operational action* to take. Black-box models create distrust among stakeholders and fail to deliver actionable retention strategies.

This platform bridges predictive machine learning and executive decision support. By synthesizing **tuned Random Forest classification**, **SHAP explainable AI**, **K-Means customer segmentation**, and **What-If scenario simulation**, the platform converts raw predictive scores into human-readable, non-causal business recommendations deployed inside an 8-tab interactive Streamlit web application.

---

## 🎯 2. Business Problem & Project Objectives

### Business Problem
Telecommunications providers face fierce market competition, low switching costs, and high customer acquisition expenses. Traditional retention strategies rely on reactive off-boarding surveys or blanket promotional discounts sent to low-risk customers—wasting marketing capital.

### Core Objectives
1. **Predict Churn Risk**: Build and evaluate classification models to identify high-risk accounts with high Sensitivity/Recall.
2. **Optimize Performance**: Apply cross-validated hyperparameter tuning to maximize model Recall while maintaining strong ROC-AUC.
3. **Explain Predictions (XAI)**: Utilize SHAP to identify global churn drivers and automated local feature attributions.
4. **Segment Customers**: Group customer accounts into natural behavioral cohorts using unsupervised K-Means clustering.
5. **Simulate Scenarios**: Build a What-If scenario engine evaluating risk deltas under hypothetical feature updates.
6. **Generate Prescriptive Recommendations**: Rule-based engine translating model signals into prioritized retention directives.
7. **Deploy Interactive Interface**: Build a responsive Streamlit dashboard for real-time customer risk lookups and executive reporting.

---

## 📊 3. Dataset Specification

| Dataset Property | Value / Description |
|---|---|
| **Source** | IBM Telco Customer Churn Public Dataset |
| **Total Records** | **7,043 customer accounts** |
| **Raw Attributes** | **21 features** (15 categorical, 5 numerical, 1 target) |
| **Target Variable** | `Churn` (Binary: `No` = 0 [73.46%], `Yes` = 1 [26.54%]) |
| **Numerical Features** | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Categorical Features** | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Engineered Features** | `AverageMonthlySpend` (`TotalCharges / tenure`), `ServiceCount` (0-8 add-ons), `IsFirstYear` (`tenure <= 12`) |

---

## 🛠️ 4. Technology Stack

- **Programming Language**: Python 3.13+
- **Data Manipulation & Analysis**: Pandas, NumPy
- **SQL & Relational Storage**: SQLite3 (`data/customer_churn.db`)
- **Data Visualization**: Matplotlib, Seaborn, Plotly Express, Plotly Graph Objects
- **Machine Learning & Pipeline**: Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`)
- **Model PERSISTENCE**: Joblib (`models/final_churn_pipeline.joblib`)
- **Explainable AI (XAI)**: SHAP (`shap.TreeExplainer`)
- **Unsupervised Learning**: Scikit-Learn K-Means Clustering & PCA
- **Web Dashboard**: Streamlit 1.30+
- **Version Control & Deployment**: Git, GitHub, Streamlit Community Cloud

---

## 🔄 5. End-to-End Workflow Architecture

```text
               Customer Dataset (IBM Telco CSV)
                              ↓
                    Stage 2: Data Cleaning
                              ↓
              Stage 3-4: EDA & SQL Business Analysis
                              ↓
              Stage 5: Feature Engineering & Preprocessing
                              ↓
               Stage 6-7: ML Training & Hyperparameter Tuning
                              ↓
                  Final Churn Model Pipeline
                              ↓
      ┌───────────────────┬───────────────┬──────────────────────┐
      ↓                   ↓               ↓                      ↓
 Stage 8: XAI       Stage 9: Segs    Stage 10: What-If   Stage 11: Rec Engine
 (SHAP Drivers)     (K-Means K=4)    (Scenario Deltas)   (Decision Support)
      └───────────────────┴───────────────┴──────────────────────┘
                                  ↓
                     Stage 12: Streamlit Dashboard
                                  ↓
                     Stage 13: Deployed Web App
```

---

## 🤖 6. Machine Learning Model Evaluation & Optimization

Evaluated on an untouched 20% holdout test dataset (**1,409 customer accounts**).

### Baseline vs Tuned Model Performance Summary

| Model Version | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression - Baseline** | 73.31% | 49.83% | **79.14%** | 0.6116 | 0.8411 |
| **Logistic Regression - Tuned** | 75.16% | 52.18% | 76.74% | 0.6212 | 0.8410 |
| **Decision Tree - Baseline (Depth=5)** | 75.44% | 52.60% | 75.67% | 0.6206 | 0.8330 |
| **Random Forest - Baseline** | 76.37% | 54.92% | 61.23% | 0.5790 | 0.8192 |
| **Random Forest - Tuned (Selected Final Model)** | **76.30%** | **53.77%** | **76.20%** | **0.6305** | **0.8394** |
| **Gradient Boosting - Baseline** | 80.27% | 66.44% | 51.87% | 0.5826 | 0.8436 |
| **Gradient Boosting - Tuned** | 80.27% | 66.44% | 51.87% | 0.5826 | 0.8436 |

### Selected Final Model Justification
- **Model**: **Tuned Random Forest Classifier** (`max_depth=12`, `n_estimators=150`, `class_weight='balanced'`)
- **Rationale**: Delivered a **+14.97% boost in Recall** over baseline Random Forest (catching **285 out of 374 test churners**), achieving peak **F1-Score (0.6305)** and strong **ROC-AUC (0.8394)**.

---

## 🔍 7. Explainable AI & SHAP Interpretability

Applied `shap.TreeExplainer` across all test accounts to decompose model predictions.

### Top 5 Global Churn Drivers (SHAP Mean |SHAP| Values)
1. **`Contract_Month-to-month` (Mean |SHAP| = 0.0697)**: Flexible short-term contracts are the #1 driver of high predicted churn probability.
2. **`tenure` (Mean |SHAP| = 0.0443)**: Short customer tenure strongly elevates predicted churn risk.
3. **`InternetService_Fiber optic` (Mean |SHAP| = 0.0413)**: Fiber optic subscription correlates with higher predicted churn.
4. **`TechSupport_No` (Mean |SHAP| = 0.0330)**: Lacking technical support elevates churn risk.
5. **`OnlineSecurity_No` (Mean |SHAP| = 0.0309)**: Lacking digital security protection elevates churn risk.

---

## 🧩 8. Unsupervised Customer Segmentation (K-Means)

Standardized features using `StandardScaler` and evaluated $K \in [2, 8]$ using Elbow (Inertia) and Silhouette Analysis. Selected **$K = 4$** (Silhouette Score = **0.4261**).

| Segment ID | Segment Name | Customer Count | % Total | Avg Tenure | Avg Monthly Bill | Avg Total Spend | Observed Churn Rate |
|---|---|---|---|---|---|---|---|
| **Segment 2** | **High-Spend At-Risk Onboarders** | **2,417** | **34.3%** | 15.6 mos | $78.27 | $1,213.00 | **46.38%** (Highest Risk) |
| **Segment 1** | **New Low-Spend Basic Customers** | **1,610** | **22.9%** | 10.8 mos | $29.80 | $298.18 | **23.85%** |
| **Segment 0** | **High-Value Long-Term Power Users** | **1,996** | **28.3%** | 58.5 mos | $93.08 | $5,433.42 | **15.73%** |
| **Segment 3** | **Loyal Low-Spend Long-Termers** | **1,020** | **14.5%** | 54.9 mos | $32.51 | $1,763.96 | **4.90%** (Lowest Risk) |

---

## 🎛️ 9. What-If Scenario Analysis

Evaluated hypothetical customer changes against the saved pipeline (`models/final_churn_pipeline.joblib`) while dynamically synchronizing feature dependencies (`TotalCharges = tenure * MonthlyCharges`).

### High-Risk Customer #6 Simulation Example (Baseline Risk: 97.03%)
- **Scenario C2 (2-Year Contract Upgrade)**: Churn probability drops to **62.82%** (Risk Delta: **-34.21%**).
- **Scenario C1 (1-Year Contract Upgrade)**: Churn probability drops to **70.04%** (Risk Delta: **-26.99%**).
- **Scenario A2 (Tenure +40m)**: Churn probability drops to **70.35%** (Risk Delta: **-26.68%**).
- **Scenario D1 (Add Tech Support)**: Churn probability drops to **86.87%** (Risk Delta: **-10.16%**).
- **Scenario G1 (Combined Retention Bundle)**: Churn probability drops to **62.93%** (Total Risk Reduction: **-34.11%**).

---

## 💼 10. Business Recommendation Engine

Synthesized risk probabilities, SHAP drivers, customer segments, and What-If deltas into actionable retention directives.

### Risk Tier Breakdown & Prioritization
- **High Risk** ($\ge 60\%$ Probability): **2,110 customers (29.96%)** $\to$ **Priority 1 Outreach** (Contract review & billing credits).
- **Medium Risk** ($35\% \le P < 60\%$): **1,330 customers (18.88%)** $\to$ **Priority 2 Outreach** (Automated contract upgrade promos & security add-ons).
- **Low Risk** ($< 35\%$ Probability): **3,603 customers (51.16%)** $\to$ **Priority 3** (Standard loyalty tracking).

---

## 📱 11. Streamlit Interactive Web Dashboard

The web application (`dashboard/app.py`) features 8 navigation sections:
1. **Executive Overview**: Portfolio KPI cards, risk distribution donut charts, and probability histograms.
2. **Customer Risk Analysis**: Filterable account table (by risk tier, segment, contract, search ID) and priority queue.
3. **Customer Segmentation**: Segment profiles, size distributions, tenure vs monthly charges scatter plot, and PCA projections.
4. **Explainable AI (SHAP)**: Global feature importance chart and local customer SHAP lookup.
5. **What-If Scenario Analysis**: Interactive simulator with sliders/dropdowns for real-time risk delta calculations.
6. **Business Recommendations**: Personalized action cards, priority badges, and business rationales.
7. **Customer Risk Predictor**: Form allowing custom account inputs to generate real-time risk predictions.
8. **Model & Dataset Info**: Performance benchmarks and dataset summary statistics.

---

## 📁 12. Complete Repository File Structure

```text
customer-churn-prediction/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/
│   │   ├── cleaned_customer_churn.csv
│   │   ├── customer_segments.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── customer_churn.db
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── load_data_to_sql.py
│   ├── run_sql_analysis.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   ├── tune_models.py
│   ├── explain_model.py
│   ├── customer_segmentation.py
│   ├── what_if_analysis.py
│   └── recommendation_engine.py
│
├── models/
│   ├── final_churn_pipeline.joblib
│   ├── final_churn_model.joblib
│   ├── preprocessor.joblib
│   ├── customer_segmentation_model.joblib
│   └── segmentation_scaler.joblib
│
├── dashboard/
│   ├── app.py
│   ├── utils.py
│   └── components.py
│
├── reports/
│   ├── data_cleaning_report.md
│   ├── eda_summary.md
│   ├── sql_business_analysis.md
│   ├── feature_engineering.md
│   ├── model_evaluation.md
│   ├── model_optimization.md
│   ├── explainability.md
│   ├── customer_segmentation.md
│   ├── what_if_analysis.md
│   ├── business_recommendations.md
│   ├── deployment_guide.md
│   ├── project_production_readiness.md
│   ├── resume_project_description.md
│   ├── project_interview_explanation.md
│   ├── final_interview_questions.md
│   ├── skills_used.md
│   ├── project_step_by_step.md
│   ├── final_project_checklist.md
│   ├── final_project_summary.md
│   ├── customer_recommendations.csv
│   ├── feature_importance.csv
│   ├── what_if_results.csv
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💡 13. Results & Business Insights Summary

1. **Contract Type is the Primary Churn Driver**: Customers on month-to-month contracts churn at **42.71%**, compared to 11.27% for 1-year and 2.83% for 2-year contracts.
2. **First-Year Vulnerability Window**: Customers in their first 12 months churn at **47.44%**, dropping to **9.51%** after 4 years.
3. **High-Risk Segment Target**: Segment 2 ("High-Spend At-Risk Onboarders") contains 34.3% of customers and exhibits a **46.38% churn rate**. Recommending contract migration incentives to this cohort yields the highest business ROI.

---

## ⚠️ 14. Limitations & Responsible AI Guidelines

- **Predictive Sensitivity vs Causation**: What-If scenario simulations measure model probability sensitivity, **not physical causation**. Recommendations provide decision support for human account managers.
- **Batch CSV Processing**: The project uses batch CSV snapshots; production deployment could integrate real-time database streaming (e.g. Snowflake/Kafka).
- **Data Privacy**: Built using public anonymized IBM Telco data containing zero PII.

---

## 🚀 15. Future Enhancements

- Integration of Causal Forest / Uplift Modeling to estimate Individual Treatment Effect (ITE).
- Real-time model drift monitoring via Evidently AI.
- Automated Docker containerization (`Dockerfile`) for Kubernetes deployment.

---

## 📜 License & Citation
This project is released under the **MIT License**. Dataset courtesy of the IBM Telco Customer Churn public dataset.
