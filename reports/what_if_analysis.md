# Stage 10: What-If Churn & Scenario Analysis Report

## 📌 Executive Summary
This report presents the findings of the **What-If Churn Analysis (Scenario & Sensitivity Analysis)** performed on representative customer profiles using the **Tuned Random Forest Pipeline** saved during Stage 7 (`models/final_churn_pipeline.joblib`).

What-If analysis enables customer retention teams to simulate hypothetical changes in customer attributes (e.g. contract type, tenure, billing discounts, and add-on services) and quantify how the model's estimated churn probability responds.

---

## 🎯 Objectives & Methodology
- **Objective**: Quantify sensitivity of churn probability predictions to hypothetical customer changes.
- **Model Used**: Tuned Random Forest Classifier Pipeline (`models/final_churn_pipeline.joblib`).
- **Feature Consistency**: All inputs passed through the exact `ColumnTransformer` (StandardScaler + OneHotEncoder) without manual encoding or target leakage. Re-calculated `AverageMonthlySpend`, `ServiceCount`, and `IsFirstYear` dynamically for all scenarios.

> [!IMPORTANT]
> **Non-Causal Interpretation Rule**: This analysis measures **model prediction sensitivity under modified inputs**, NOT causal inference. Decreasing predicted churn probability in a hypothetical scenario does **not prove** that making the change will force a real-world customer to stay.

---

## 👤 Representative Customer Baseline Profiles

| Customer Profile | Customer ID | Baseline Churn Prob | Predicted Class | Actual Churn Status | Key Characteristics |
|---|---|---|---|---|---|
| **High Risk** | Customer #6 | **97.03%** | High Risk (1) | Yes | Month-to-month, Tenure = 8 mos, Bill = $99.65, Fiber Optic, No Tech Support |
| **Medium Risk** | Customer #34 | **52.04%** | High Risk (1) | No | Month-to-month, Tenure = 1 mo, Bill = $49.65, DSL, Yes Security |
| **Low Risk** | Customer #4 | **4.26%** | Low Risk (0) | No | One year, Tenure = 45 mos, Bill = $42.30, DSL, Yes Support & Security |

---

## 📊 Scenario Simulation Results (High-Risk Account #6)

| Scenario Name | Changed Feature(s) | Original Value | New Value | Baseline Prob | Scenario Prob | Risk Delta | Risk Direction |
|---|---|---|---|---|---|---|---|
| **Baseline (Current Profile)** | None | Baseline | Baseline | 97.03% | **97.03%** | **+0.00%** | No Change |
| **Scenario A1: Tenure Increase (+16m)** | tenure | 8 | 24 | 97.03% | **84.44%** | **-12.59%** | Decreased Risk |
| **Scenario A2: Tenure Increase (+40m)** | tenure | 8 | 48 | 97.03% | **70.35%** | **-26.68%** | Decreased Risk |
| **Scenario B1: Monthly Bill Discount ($75)** | MonthlyCharges | 99.65 | 75.0 | 97.03% | **84.47%** | **-12.56%** | Decreased Risk |
| **Scenario C1: Contract Upgrade (1-Year)** | Contract | Month-to-month | One year | 97.03% | **70.04%** | **-26.99%** | Decreased Risk |
| **Scenario C2: Contract Upgrade (2-Year)** | Contract | Month-to-month | Two year | 97.03% | **62.82%** | **-34.21%** | Decreased Risk |
| **Scenario D1: Add Tech Support** | TechSupport | No | Yes | 97.03% | **86.87%** | **-10.16%** | Decreased Risk |
| **Scenario E1: Add Online Security** | OnlineSecurity | No | Yes | 97.03% | **91.52%** | **-5.51%** | Decreased Risk |
| **Scenario F1: Switch Payment (Auto Credit Card)** | PaymentMethod | Electronic check | Credit card (automatic) | 97.03% | **94.80%** | **-2.23%** | Decreased Risk |
| **Scenario G1: Combined Retention Bundle** | Contract, TechSupport, OnlineSecurity, PaymentMethod | Month-to-month, No, No, Electronic check | One year, Yes, Yes, Bank transfer (automatic) | 97.03% | **62.93%** | **-34.11%** | Decreased Risk |


---

## 🔍 Key Findings & Business Interpretations

### 1. **Contract Upgrades Yield the Largest Risk Reduction**
- Upgrading Customer #6 from `Month-to-month` to a **2-Year Contract** reduced predicted churn probability from **97.03% to 62.82%** (a **-34.21% risk reduction**).
- Upgrading to a **1-Year Contract** reduced predicted churn probability to **70.04%** (a **-26.99% risk reduction**).

### 2. **Technical Support Add-on Stabilizes High-Risk Accounts**
- Adding `TechSupport = Yes` reduced predicted churn probability from **97.03% to 86.87%** (a **-10.16% risk reduction**).

### 3. **Tenure Growth Reduces Model Risk Perception**
- Increasing tenure from 8 to 48 months reduced predicted churn risk to **70.35%** (a **-26.68% risk reduction**).

### 4. **Combined Retention Bundling**
- Combining a **1-Year Contract + Tech Support + Online Security + Automatic Payment** lowered churn probability from **97.03% to 62.93%** (a **-34.11% total risk reduction**).

---

## ⚠️ Limitations of What-If Scenario Analysis
1. **Model Boundedness**: What-If analysis evaluates the model's learned associations, not human psychological behavior.
2. **Co-varying Unobserved Variables**: Changing `Contract` in real life might require pricing or promotional commitments that are unobserved in the model.
3. **No Causal Guarantee**: The results reflect statistical sensitivity, not guarantee of retention success.

---

## 📁 Artifacts Saved
- **Scenario Results CSV**: [`reports/what_if_results.csv`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\what_if_results.csv)
- **Baseline vs Scenario Chart**: [`reports/figures/what_if_baseline_vs_scenarios.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\what_if_baseline_vs_scenarios.png)
- **Scenario Risk Deltas Chart**: [`reports/figures/what_if_scenario_deltas.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\what_if_scenario_deltas.png)
