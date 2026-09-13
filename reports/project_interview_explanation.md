# Stage 14: Project Interview Explanation Guide

This guide provides structured 30-second, 1-minute, and 3-minute verbal responses for answering **"Tell me about your Customer Churn project"** in Data Science, Data Analyst, and Machine Learning interviews.

---

## ⚡ 1. The 30-Second Elevator Pitch

> *"I built an end-to-end Customer Churn Analytics Platform using Python on 7,043 telecom accounts. I handled data cleaning, SQL business queries, and feature engineering before building a Scikit-learn machine learning pipeline. My tuned Random Forest model achieved a 76.2% Recall and 0.8394 ROC-AUC. To make predictions actionable, I integrated SHAP for explainable AI, K-Means for customer segmentation, and built a What-If scenario simulator. Finally, I deployed everything into an interactive Streamlit web dashboard that translates risk scores into automated business recommendations."*

---

## ⏱️ 2. The 1-Minute Standard Interview Answer

> *"In subscription businesses, acquiring new customers costs up to 5x to 25x more than retaining existing ones. I wanted to build a platform that doesn't just predict who will leave, but explains why and suggests retention actions.*
> 
> *I worked with the IBM Telco dataset containing over 7,000 customer records. After cleaning missing values and exploring data via SQL and Seaborn, I engineered key features like Average Monthly Spend and Service Adoption. To prevent data leakage, I used Scikit-learn ColumnTransformers inside stratified cross-validation pipelines.*
> 
> *I evaluated four classification algorithms and selected a Tuned Random Forest model because it boosted Recall to 76.20% with an 0.8394 ROC-AUC. I then used SHAP to extract top churn drivers—like month-to-month contracts and lack of tech support—and used K-Means clustering to discover four distinct customer segments.*
> 
> *Finally, I created a What-If scenario simulator and deployed the entire workflow into a responsive Streamlit dashboard with real-time risk predictors and automated business recommendations."*

---

## 🧠 3. The 3-Minute Deep Technical Explanation

> *"The goal of this project was to build a comprehensive Customer Churn Prediction and Business Intelligence Platform. The workflow spans 13 distinct stages:
> 
> **Data Cleaning & SQL Analysis**: I audited the IBM Telco dataset (7,043 rows x 21 columns). I resolved hidden whitespace missing values in `TotalCharges` by imputing 0.0 for new customers (`tenure = 0`) and removed non-predictive string hashes (`customerID`). I populated a SQLite database and executed SQL queries using `CASE` statements to identify peak churn risk combinations—such as Month-to-month + Fiber Optic + No Tech Support + Electronic Check—which churned at 62.92%.
> 
> **Feature Engineering & Preprocessing**: I engineered domain features including `AverageMonthlySpend`, `ServiceCount`, and `IsFirstYear`. I set up an 80/20 stratified train-test split (`5,634` train / `1,409` test) and fitted a Scikit-learn `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) strictly on training data to prevent target leakage.
> 
> **Model Training & Optimization**: I trained Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting. I evaluated models on the untouched test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Through 5-Fold `StratifiedKFold` cross-validation using `GridSearchCV`, I optimized a Random Forest classifier (`max_depth=12`, `n_estimators=150`, `class_weight='balanced'`), boosting Recall from 61.23% to 76.20% (+14.97% gain) with an 0.8394 ROC-AUC.
> 
> **Explainable AI & Unsupervised Segmentation**: To open the black box, I used `shap.TreeExplainer` to identify global churn drivers (`Contract_Month-to-month` mean |SHAP| = 0.0697, `tenure`, `Fiber Optic`). I built local natural language explanations converting SHAP feature contributions into clear business sentences. Next, I applied standardized K-Means clustering ($K=4$, Silhouette Score = 0.4261) to segment the portfolio into 4 behavioral cohorts, profiling a high-risk onboarding segment with a 46.38% observed churn rate.
> 
> **Scenario Simulation & Deployment**: I built a What-If scenario simulator that re-computes dynamic feature dependencies and evaluates predicted risk deltas under hypothetical input changes—such as showing that upgrading to a 1-Year contract reduces estimated churn risk for a high-risk account by ~27 percentage points. Finally, I built a rule-based Business Recommendation Engine and deployed the entire solution into an 8-tab interactive Streamlit dashboard (`dashboard/app.py`) optimized with `@st.cache_data` and `@st.cache_resource`."*
