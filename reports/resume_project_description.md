# Customer Churn Prediction & Explainable Business Analytics Platform — Resume Guide

This document provides a professional, highly targeted resume title, one-line summary, and resume bullet points tailored for Data Science, Machine Learning, and Data Analytics job applications.

---

## 📌 Project Title (Choose One)
- **Customer Churn Prediction & Explainable Business Analytics Platform** *(Recommended)*
- **End-to-End Customer Churn Analytics Platform: ML, SHAP XAI & Streamlit Dashboard**
- **Customer Retention Analytics Platform (Predictive ML, K-Means & SHAP)**

---

## 💡 One-Line Description
> *Engineered an end-to-end Machine Learning and Explainable AI platform on 7,000+ telecom accounts, optimizing a Random Forest classifier to 76.2% Recall (0.8394 ROC-AUC), segmenting customers via K-Means, and deploying an interactive Streamlit decision-support dashboard.*

---

## 📄 Resume Bullet Points (4–5 Bullet Points)

- **Predictive ML Modeling & Optimization**: Built and evaluated 4 classification models on 7,043 telecom records; tuned a Random Forest classifier using 5-Fold Stratified Cross-Validation (`GridSearchCV`), boosting Recall by **+14.97%** to **76.20%** and achieving an **0.8394 ROC-AUC**.
- **Data Engineering & SQL Analytics**: Cleaned raw data, handled missing values, engineered 3 domain features (`AverageMonthlySpend`, `ServiceCount`, `IsFirstYear`), and constructed a SQLite database executing complex SQL queries to identify peak churn risk segments (62.92% churn rate).
- **Explainable AI (SHAP) & Model Interpretability**: Applied `shap.TreeExplainer` to extract global feature drivers (`Contract`, `tenure`, `Fiber Optic`, `TechSupport`) and automated natural language feature attribution to convert black-box predictions into transparent customer explanations.
- **Unsupervised Customer Segmentation**: Grouped customer portfolio into 4 distinct behavioral cohorts using standardized K-Means clustering ($K=4$, Silhouette Score = **0.4261**), profiling a high-risk segment with an observed **46.38% churn rate**.
- **What-If Simulations & Prescriptive Analytics**: Designed a non-causal scenario simulation module evaluating risk deltas under hypothetical input changes (contract upgrades, bill discounts) and deployed a responsive 8-tab Streamlit dashboard with real-time risk predictors.

---

## 🚫 What NOT to Claim on Your Resume
- **Do NOT claim multi-node real-time streaming infrastructure**: (The project uses batch CSV & SQLite processing, not Apache Spark/Kafka clusters).
- **Do NOT claim causal proof**: (The project measures statistical sensitivity and predictive model risk deltas, not physical A/B test causal guarantees).
- **Do NOT claim millions of live users**: (Frame it as an end-to-end production-ready portfolio platform).
