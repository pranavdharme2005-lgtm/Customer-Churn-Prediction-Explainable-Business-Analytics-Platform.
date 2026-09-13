# Stage 14: Step-by-Step Project Walkthrough (Stages 1 – 14)

This walkthrough documents the complete end-to-end execution of the **Customer Churn Prediction & Explainable Business Analytics Platform** across all 14 project stages.

---

## 📌 Stage-by-Stage Execution Walkthrough

### 🔹 Stage 1: Setup & Dataset Preparation
- **What We Did**: Created standard directory structure (`data/`, `notebooks/`, `src/`, `models/`, `dashboard/`, `reports/`). Ingested IBM Telco Customer Churn dataset (7,043 rows x 21 columns) into `data/raw/`.
- **Key Output**: Initialized environment, `requirements.txt`, `.gitignore`, and `src/load_data.py`.

### 🔹 Stage 2: Data Understanding & Data Cleaning
- **What We Did**: Audited missing values; discovered 11 hidden whitespace strings (`" "`) in `TotalCharges` for `tenure == 0` customers. Imputed `0.0` and converted column to `float64`. Removed non-predictive string hash `customerID`.
- **Key Output**: Saved cleaned master CSV to `data/processed/cleaned_customer_churn.csv` and report `reports/data_cleaning_report.md`.

### 🔹 Stage 3: Exploratory Data Analysis (EDA)
- **What We Did**: Answered 10 core business questions linking churn to contract type, tenure, payment method, and add-on services. Analyzed tenure groups, monthly charge tiers, correlation matrices, and IQR outliers.
- **Key Output**: Notebook `notebooks/01_exploratory_data_analysis.ipynb` and report `reports/eda_summary.md`.

### 🔹 Stage 4: SQL Business Analysis
- **What We Did**: Created SQLite database `data/customer_churn.db` and populated table `customers`. Executed SQL queries using `CASE` and `HAVING` to discover that Month-to-month + Fiber Optic + No Tech Support + Electronic Check customers churn at a peak **62.92%** rate.
- **Key Output**: `src/sql_analysis.sql`, `src/run_sql_analysis.py`, CSV exports in `reports/sql_results/`, and report `reports/sql_business_analysis.md`.

### 🔹 Stage 5: Feature Engineering & Preprocessing
- **What We Did**: Engineered domain features (`AverageMonthlySpend`, `ServiceCount`, `IsFirstYear`). Conducted leakage-free 80/20 train-test split (`5,634` train / `1,409` test). Fitted Scikit-learn `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) strictly on training data.
- **Key Output**: Saved preprocessor `models/preprocessor.joblib`, processed train/test datasets, and report `reports/feature_engineering.md`.

### 🔹 Stage 6: Machine Learning Model Development & Baseline Evaluation
- **What We Did**: Trained 4 baseline algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting). Evaluated on 1,409 untouched test accounts.
- **Key Output**: Comparison CSV `reports/model_comparison.csv` and report `reports/model_evaluation.md`.

### 🔹 Stage 7: Hyperparameter Tuning & Model Optimization
- **What We Did**: Performed 5-Fold `StratifiedKFold` CV tuning using `GridSearchCV` on training data. Tuned Random Forest achieved **76.20% Recall** (+14.97% boost), **0.6305 F1-Score**, and **0.8394 ROC-AUC**.
- **Key Output**: Saved final candidate model `models/final_churn_model.joblib`, full pipeline `models/final_churn_pipeline.joblib`, and report `reports/model_optimization.md`.

### 🔹 Stage 8: Explainable AI & Model Interpretability (SHAP)
- **What We Did**: Calculated native Gini feature importances and fitted `shap.TreeExplainer`. Extracted top global churn drivers (`Contract_Month-to-month` mean |SHAP| = 0.0697, `tenure`, `Fiber Optic`). Automated local natural language feature attribution.
- **Key Output**: Feature importances `reports/feature_importance.csv`, plots in `reports/figures/`, and report `reports/explainability.md`.

### 🔹 Stage 9: Customer Segmentation via Unsupervised Learning
- **What We Did**: Standardized behavioral metrics and trained K-Means clustering ($K=4$, Silhouette Score = **0.4261**). Profiled 4 customer segments, identifying Segment 2 ("High-Spend At-Risk Onboarders") with a **46.38% observed churn rate**.
- **Key Output**: Segmentation model `models/customer_segmentation_model.joblib`, updated dataset `data/processed/customer_segments.csv`, and report `reports/customer_segmentation.md`.

### 🔹 Stage 10: What-If Churn Analysis & Scenario Simulation
- **What We Did**: Built a non-causal scenario simulator using `models/final_churn_pipeline.joblib`. Evaluated risk deltas under hypothetical feature changes, maintaining dynamic feature dependencies.
- **Key Output**: Simulation results `reports/what_if_results.csv`, figures, and report `reports/what_if_analysis.md`.

### 🔹 Stage 11: Business Recommendation Engine
- **What We Did**: Built a rule-based decision-support engine synthesizing probabilities, SHAP drivers, customer segments, and What-If scenario impact into risk tiers (`High`, `Med`, `Low`) and actionable retention directives.
- **Key Output**: Recommendations dataset `reports/customer_recommendations.csv`, figures, and report `reports/business_recommendations.md`.

### 🔹 Stage 12: Professional Interactive Streamlit Dashboard
- **What We Did**: Developed an 8-tab Streamlit dashboard (`dashboard/app.py`, `dashboard/utils.py`, `dashboard/components.py`) featuring KPI metric cards, Plotly charts, Live Predictor, What-If simulator, and XAI lookup.
- **Key Output**: Fully interactive dashboard tested on local server (`http://localhost:8501`).

### 🔹 Stage 13: Deployment & Productionization
- **What We Did**: Refactored paths using `pathlib.Path`, configured `.streamlit/config.toml`, updated `.gitignore`, pinned `requirements.txt`, and conducted security audits.
- **Key Output**: `reports/deployment_guide.md`, `reports/project_production_readiness.md`, and updated `README.md`.

### 🔹 Stage 14: Portfolio, Documentation & Interview Package
- **What We Did**: Polished `README.md`, created resume bullet points, 3 verbal interview responses, 30 comprehensive interview Q&A pairs, skills matrix, step-by-step walkthrough, and final checklist.
- **Key Output**: Full interview preparation package in `reports/`.
