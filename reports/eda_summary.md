# Exploratory Data Analysis (EDA) Summary Report

## 📌 Executive Overview
This report synthesizes the empirical findings from **Stage 3: Exploratory Data Analysis (EDA)** of the **Customer Churn Prediction & Explainable Business Analytics Platform**. The analysis examined **7,043 customer records** across **20 cleaned features** to identify core factors driving customer attrition.

---

## 📊 1. Dataset & Target Distribution Summary

- **Total Customer Accounts**: `7,043`
- **Cleaned Features**: `20` (4 numerical, 16 categorical)
- **Target Variable**: `Churn` (`No` = Retained, `Yes` = Churned)
- **Class Breakdown**:
  - **Retained (`No`)**: **5,174** customers (**73.46%**)
  - **Churned (`Yes`)**: **1,869** customers (**26.54%**)
- **Target Imbalance Evaluation**:
  - The dataset presents a mild class imbalance (~3:1 ratio).
  - *Machine Learning Impact*: Standard accuracy will be skewed. Future model development in Stage 4/5 must prioritize **ROC-AUC, Precision, Recall, F1-Score**, and employ class-weighting or SMOTE resampling.

---

## 📈 2. Numerical & Categorical Features Overview

### Numerical Summaries:
| Feature | Mean | Median | Min | Max | Standard Deviation |
|---|---|---|---|---|---|
| `tenure` (months) | 32.37 | 29.00 | 0 | 72 | 24.56 |
| `MonthlyCharges` ($) | $64.76 | $70.35 | $18.25 | $118.75 | $30.09 |
| `TotalCharges` ($) | $2,279.73 | $1,394.55 | $0.00 | $8,684.80 | $2,266.79 |

### Categorical Distribution Highlights:
- **Contract Types**: Month-to-month (55.02%), Two year (24.07%), One year (20.91%).
- **Internet Service**: Fiber optic (43.96%), DSL (34.37%), No Internet (21.67%).
- **Payment Method**: Electronic check (33.58%), Mailed check (22.89%), Bank transfer (21.92%), Credit card (21.61%).

---

## 🔍 3. Churn-Focused Bivariate Findings (Q1–Q10)

| Question / Feature | Key Finding & Churn Rates |
|---|---|
| **Q1. Contract Type** | **Month-to-month**: **42.71%** churn rate vs **11.27%** (1-Year) and **2.83%** (2-Year). Short commitments are the #1 risk factor. |
| **Q2. Tenure Duration** | New customers (**0–12 months**) churn at **47.44%**. Churn drops below **10%** after 48 months. |
| **Q3. Monthly Charges** | Median monthly bill for churned customers is **$79.65** vs **$64.42** for retained customers. High monthly costs increase churn. |
| **Q4. Internet Service** | **Fiber optic** users churn at **41.89%** vs **18.96%** for DSL and **7.40%** for non-internet users. |
| **Q5. Tech Support** | Customers **without Tech Support** churn at **41.64%** vs **15.17%** for those with Tech Support. |
| **Q6. Online Security** | Customers **without Online Security** churn at **41.77%** vs **14.61%** for those with Online Security. |
| **Q7. Payment Method** | **Electronic check** users churn at **45.29%** vs ~15–19% for automated bank transfer / credit card / mailed check. |
| **Q8. Paperless Billing** | Customers with **Paperless Billing** churn at **33.57%** vs **16.33%** for manual paper billing. |
| **Q9. Senior Citizens** | **Senior Citizens** churn at **41.68%** vs **23.61%** for non-seniors. |
| **Q10. Multiple Lines** | Multiple lines show a **28.61%** churn rate vs **25.04%** for single lines. |

---

## 📐 4. Tenure & Monthly Charge Binned Analysis

### Tenure Groups:
- **0–12 months**: **47.44%** churn rate (2,186 customers) — *High Risk Onboarding Window*
- **13–24 months**: **28.71%** churn rate (1,024 customers)
- **25–48 months**: **20.39%** churn rate (1,594 customers)
- **49–72 months**: **9.51%** churn rate (2,239 customers) — *High Retention Loyalty Group*

### Monthly Charge Tiers:
- **$18–$35 (Low)**: **10.89%** churn rate (1,735 customers)
- **$35–$65 (Med-Low)**: **23.14%** churn rate (1,409 customers)
- **$65–$90 (Med-High)**: **36.30%** churn rate (2,160 customers) — *Highest Risk Price Bracket*
- **$90–$120 (High)**: **32.78%** churn rate (1,739 customers)

---

## 🔗 5. Correlation & Outlier Observations

### Correlation Matrix Findings:
- **`tenure` & `TotalCharges` (+0.826)**: Strong positive linear relationship (longer tenure accumulates higher revenue).
- **`tenure` & `Churn` (-0.352)**: Strongest negative linear relationship with churn (longer tenure = lower risk).
- **`MonthlyCharges` & `Churn` (+0.193)**: Positive correlation with churn.
- **Correlation vs. Causation**: While tenure is negatively correlated with churn, tenure itself is an outcome of customer satisfaction, not a direct causal mechanism.

### Outlier Assessment:
- **Statistical Outliers**: **0** mathematical outliers detected using standard $1.5 \times IQR$ bounds on `tenure`, `MonthlyCharges`, and `TotalCharges`.
- **Handling Decision**: All extreme numerical values represent genuine long-tenure customers or high-tier plans. **No records were deleted**.

---

## 💡 6. Key Business Insights

1. **Incentivize Long-Term Contracts**: Month-to-Month customers churn at 42.71%. Shifting customers to 1-Year or 2-Year contracts via discount incentives will dramatically reduce churn.
2. **Focus Retention on the First 12 Months**: Nearly 50% of new customers churn in their first year. Implementing early-onboarding check-ins and loyalty rewards during months 1–12 will protect revenue.
3. **Audit Fiber Optic Pricing & Service Quality**: Fiber optic subscribers churn at 41.89% (over double DSL churn), coinciding with high monthly charges ($65–$90). Investigating service outages and adjusting package pricing is recommended.
4. **Bundle Support & Security Add-Ons**: Customers without Tech Support (41.64% churn) or Online Security (41.77% churn) leave at nearly triple the rate of those with add-ons. Offering discounted support/security bundles can serve as retention anchors.
5. **Promote Automatic Payment Enrollment**: Electronic check users churn at 45.29%. Transitioning customers to automatic bank transfers or credit card auto-pay can lower churn by over 25 percentage points.
6. **Senior Citizen Tailored Engagement**: Senior citizens exhibit a 41.68% churn rate. Dedicated support lines or simplified billing packages could reduce senior attrition.

---

## ⚠️ 7. Limitations of Analysis
- **Observational Correlation**: Cross-sectional data shows correlations but cannot establish strict causal relationships.
- **Lack of Usage Metrics**: Dataset does not include actual data usage (GB consumed), customer service call logs, or net promoter scores (NPS).
- **Class Imbalance**: Imbalanced target requires careful model evaluation in Stage 4.
