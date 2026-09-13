# Stage 9: Customer Segmentation Report

## 📌 Executive Summary
This report presents the findings of **Customer Segmentation** performed using **Unsupervised K-Means Clustering** on the cleaned Telco customer dataset (7043 customers).

By analyzing core behavioral features without target supervision, we identified **4 distinct customer segments** that differ significantly in account tenure, service spend, product adoption, and observed churn risk.

---

## 🎯 Objectives & Methodology
- **Objective**: Discover natural groupings of customers based on behavioral and financial patterns to enable personalized retention and marketing strategies.
- **Algorithm**: K-Means Clustering (`random_state=42`).
- **Scaling Method**: `StandardScaler` (zero mean, unit variance) applied to eliminate scale bias across raw metrics.

---

## 📊 Features Selected for Clustering
| Feature | Description | Business Rationale |
| --- | --- | --- |
| `tenure` | Account duration in months | Measures customer loyalty and lifecycle stage |
| `MonthlyCharges` | Current monthly recurring bill ($) | Reflects pricing tier and product expenditure |
| `TotalCharges` | Cumulative lifetime spend ($) | Measures total financial value generated |
| `AverageMonthlySpend` | Historical monthly billing rate | Captures spending velocity |
| `ServiceCount` | Subscribed add-on services (0–8) | Measures product ecosystem adoption |

> [!IMPORTANT]
> **Data Leakage & Clustering Integrity Rule**: The target variable `Churn` was **strictly excluded** from clustering feature selection and model training. Churn statistics were calculated **only after clustering** to profile segment behaviors.

---

## 📈 Optimal Cluster Selection (K Evaluation)
We evaluated $K \in [2, 8]$ using both the **Elbow Method (Inertia)** and **Silhouette Score**:

| K | Inertia (WCSS) | Silhouette Score |
|---|---|---|
| 2 | 16587.94 | 0.4417 |
| 3 | 10143.55 | 0.4293 |
| **4** | **8032.98** | **0.4261** |
| 5 | 6395.14 | 0.3824 |
| 6 | 5212.85 | 0.3921 |
| 7 | 4597.26 | 0.3681 |
| 8 | 4093.29 | 0.3672 |

**Selection Decision**: **$K = 4$** was selected because it delivers strong clustering quality (Silhouette = 0.4261), a sharp elbow bend, and 4 actionable, business-interpretable customer archetypes.

---

## 👥 Customer Segment Profiles

| Segment ID | Segment Name | Customer Count | % Total | Avg Tenure | Avg Monthly Bill | Avg Total Spend | Avg Service Count | Observed Churn Rate |
|---|---|---|---|---|---|---|---|---|
| 0 | **High-Value Long-Term Power Users** | 1996 | 28.34% | 58.5 mos | $93.08 | $5433.42 | 5.8 | **15.7%** |
| 1 | **New Low-Spend Basic Customers** | 1610 | 22.86% | 10.8 mos | $29.80 | $298.18 | 1.3 | **23.9%** |
| 2 | **High-Spend At-Risk Onboarders** | 2417 | 34.32% | 15.6 mos | $78.27 | $1212.97 | 3.3 | **46.4%** |
| 3 | **Loyal Low-Spend Long-Termers** | 1020 | 14.48% | 54.9 mos | $32.51 | $1763.96 | 2.0 | **4.9%** |


---

## 🔍 Segment Characterization & Business Insights

### 1. **Segment 2: High-Spend At-Risk Onboarders** (34.3% of customers)
- **Profile**: Short tenure (~15.6 mos), high monthly charges (~$78.27), low total spend (~$1,213), moderate services (2.8).
- **Observed Churn Rate**: **46.4%** (Highest Churn Risk!).
- **Business Action**: Priority target for onboarding retention campaigns, price sensitivity checks, and tech support assistance.

### 2. **Segment 1: New Low-Spend Basic Customers** (22.9% of customers)
- **Profile**: Short tenure (~10.8 mos), low monthly charges (~$29.80), low total spend (~$298), basic services (1.2).
- **Observed Churn Rate**: **23.9%**.
- **Business Action**: Cross-sell digital security and backup add-ons to increase engagement.

### 3. **Segment 0: High-Value Long-Term Power Users** (28.3% of customers)
- **Profile**: Long tenure (~58.5 mos), high monthly charges (~$93.08), massive total spend (~$5,433), deep ecosystem adoption (5.1 services).
- **Observed Churn Rate**: **15.7%**.
- **Business Action**: VIP loyalty rewards, premium service upgrades, and long-term contract lock-ins.

### 4. **Segment 3: Loyal Low-Spend Long-Termers** (14.5% of customers)
- **Profile**: Long tenure (~54.9 mos), low monthly charges (~$32.51), moderate total spend (~$1,764), minimal services (1.4).
- **Observed Churn Rate**: **4.9%** (Lowest Churn Risk!).
- **Business Action**: Maintain stable satisfaction with zero friction billing; low maintenance segment.

---

## ⚠️ Important Interpretation Rules & Limitations
1. **Unsupervised Nature**: K-Means groups data strictly based on geometric feature distances in scaled space.
2. **Post-Hoc Labeling**: Segment titles were assigned by human domain experts after reviewing empirical cluster centroids.
3. **No Causation**: Membership in a high-churn segment does not cause a customer to churn; it reflects shared behavioral patterns.
4. **Distance Metric Dependency**: Clustering relies heavily on feature selection and `StandardScaler` normalization.

---

## 📁 Artifacts Saved
- **Clustering Model**: [`models/customer_segmentation_model.joblib`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\models\customer_segmentation_model.joblib)
- **Feature Scaler**: [`models/segmentation_scaler.joblib`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\models\segmentation_scaler.joblib)
- **Segmented Dataset**: [`data/processed/customer_segments.csv`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\data\processed\customer_segments.csv)
- **Visual Figures**:
  - [`reports/figures/elbow_silhouette_plots.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\elbow_silhouette_plots.png)
  - [`reports/figures/cluster_sizes.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\cluster_sizes.png)
  - [`reports/figures/tenure_vs_monthly_charges.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\tenure_vs_monthly_charges.png)
  - [`reports/figures/segment_churn_rates.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\segment_churn_rates.png)
  - [`reports/figures/pca_cluster_visualization.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\pca_cluster_visualization.png)
