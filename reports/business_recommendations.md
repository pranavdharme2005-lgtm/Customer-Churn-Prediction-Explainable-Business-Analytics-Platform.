# Stage 11: Business Recommendation Engine Report

## 📌 Executive Summary
This report presents the **Business Recommendation Engine** built for the Customer Churn Analytics Platform.

By synthesizing model predictions, **SHAP explainability drivers** (Stage 8), **unsupervised customer segments** (Stage 9), and **What-If scenario simulations** (Stage 10), the engine translates raw churn probability scores into practical, human-readable retention plays.

---

## 🏗️ Recommendation Engine Architecture

```
[ Model Predictions ] + [ SHAP Drivers ] + [ Customer Segments ] + [ What-If Simulations ]
                                  │
                                  ▼
                   [ Rule-Based Recommendation Engine ]
                                  │
                                  ▼
            [ Actionable Non-Causal Decision Support Output ]
```

---

## 📊 Risk Level Classification Benchmarks

| Risk Level | Probability Threshold | Customer Count | % Total | Primary Business Focus |
|---|---|---|---|---|
| **High Risk** | $\ge 60\%$ ($\ge 0.60$) | **2110** | **30.0%** | Urgent proactive retention intervention & account review |
| **Medium Risk** | $35\% \le P < 60\%$ | **1330** | **18.9%** | Automated engagement & contract upgrade promotions |
| **Low Risk** | $< 35\%$ ($< 0.35$) | **3603** | **51.2%** | Standard loyalty tracking & annual appreciation |

---

## 🔝 Top 5 Highest-Risk Customer Priorities

| Customer ID | Churn Prob | Risk Level | Customer Segment | Key Risk Factors | Category | Recommended Action | Priority |
|---|---|---|---|---|---|---|---|
| **CUST-04801** | **97.3%** | High Risk | High-Spend At-Risk Onboarders | Flexible Month-to-Month Contract; Short Tenure (<= 12 months); No Tech Support Service | Contract Review | Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract). | **High** |
| **CUST-01977** | **97.3%** | High Risk | High-Spend At-Risk Onboarders | Flexible Month-to-Month Contract; Short Tenure (<= 12 months); No Tech Support Service | Contract Review | Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract). | **High** |
| **CUST-06233** | **97.3%** | High Risk | High-Spend At-Risk Onboarders | Flexible Month-to-Month Contract; Short Tenure (<= 12 months); No Tech Support Service | Contract Review | Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract). | **High** |
| **CUST-00006** | **97.0%** | High Risk | High-Spend At-Risk Onboarders | Flexible Month-to-Month Contract; Short Tenure (<= 12 months); No Tech Support Service | Contract Review | Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract). | **High** |
| **CUST-02578** | **97.0%** | High Risk | High-Spend At-Risk Onboarders | Flexible Month-to-Month Contract; Short Tenure (<= 12 months); No Tech Support Service | Contract Review | Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract). | **High** |


---

## 🎯 Recommendation Rule Logic

### Rule 1 — Contract Review (High Risk + Month-to-Month Contract)
- **Condition**: Churn Probability $\ge 0.60$ and `Contract == 'Month-to-month'`.
- **Recommendation**: Offer contract migration incentives (10% bill discount or streaming add-on for 1-Year or 2-Year commitment).
- **What-If Simulation Context**: Scenario simulations indicate upgrading to a 1-Year contract reduces estimated churn probability by ~27 percentage points.

### Rule 2 — Plan & Price Review (High Risk + High Charges + Short Tenure)
- **Condition**: Churn Probability $\ge 0.60$, `MonthlyCharges > $70`, and `tenure <= 12 months`.
- **Recommendation**: Conduct account pricing review; suggest optimized plan tier packages or temporary promotional bill credits.
- **What-If Simulation Context**: Scenario simulations estimate a $15 bill discount reduces estimated churn probability by ~8 to 12 percentage points.

### Rule 3 — Technical Support (High Risk + No Tech Support)
- **Condition**: Churn Probability $\ge 0.60$ and `TechSupport == 'No'`.
- **Recommendation**: Initiate proactive tech support outreach; offer 3 months of complimentary premium technical assistance.
- **What-If Simulation Context**: Scenario simulations estimate adding Tech Support reduces estimated churn risk by ~10 percentage points.

---

## 🛡️ Responsible AI & Non-Causal Guidance
1. **Decision Support, Not Guaranteed Prevention**: Recommendations provide structured guidance for human account managers. They do not guarantee that a customer will be retained.
2. **Correlation vs Causation**: A feature associated with high churn does not physically cause the customer to leave.
3. **Privacy & Human-in-the-Loop**: Customer interventions must comply with data privacy policies and involve human judgment for sensitive outreach.

---

## 📁 Artifacts Saved
- **Recommendation CSV**: [`reports/customer_recommendations.csv`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\customer_recommendations.csv)
- **Interview Guide**: [`reports/business_recommendation_interview_questions.md`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\business_recommendation_interview_questions.md)
- **Visual Figures**:
  - [`reports/figures/risk_level_distribution.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\risk_level_distribution.png)
  - [`reports/figures/avg_prob_by_recommendation_category.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\avg_prob_by_recommendation_category.png)
  - [`reports/figures/high_risk_by_segment.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\high_risk_by_segment.png)
  - [`reports/figures/top_risk_drivers_distribution.png`](file:///C:\Users\prana\.gemini\antigravity\scratch\jarvis_assistant\customer-churn-prediction\reports\figures\top_risk_drivers_distribution.png)
