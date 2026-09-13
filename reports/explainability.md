# Stage 8: Explainable AI & Model Interpretability Report

## 📌 Executive Summary
This report documents the **Explainable AI (XAI)** methodology applied to the **Tuned Random Forest** candidate model saved during **Stage 7**.

Using **SHAP (SHapley Additive exPlanations)** and tree-native feature importances, we explain the global feature drivers across the dataset as well as local individual prediction drivers for specific customer accounts.

---

## 🔍 Why Explainable AI is Critical for Churn Management
1. **Trust & Verification**: Black-box models cannot be deployed safely without understanding why they assign high churn risk.
2. **Actionable Retention Strategies**: Identifying *why* a customer is at risk allows customer success teams to offer targeted incentives (e.g. offering a discount to a customer churn-driven by price vs offering tech support to a customer churn-driven by service friction).
3. **Regulatory & Business Compliance**: Auditing feature drivers prevents discriminatory model biases.

---

## 📊 Global Feature Importance Ranking (Top 10 SHAP Features)

| Feature | Mean_Abs_SHAP |
| --- | --- |
| Contract_Month-to-month | 0.06968298527259362 |
| tenure | 0.044320044904374196 |
| InternetService_Fiber optic | 0.0412609037696919 |
| TechSupport_No | 0.03303650130788846 |
| OnlineSecurity_No | 0.03094613317629251 |
| TotalCharges | 0.03024473562146132 |
| IsFirstYear | 0.028225201291851684 |
| Contract_Two year | 0.028165699338998762 |
| PaymentMethod_Electronic check | 0.023893906174142065 |
| MonthlyCharges | 0.018911114076750474 |

### Key Global Observations:
1. **`Contract_Month-to-month` (Mean |SHAP| = 0.0697)**: The single strongest driver of high predicted churn probability.
2. **`tenure` (Mean |SHAP| = 0.0443)**: Low tenure strongly increases predicted churn risk, while high tenure lowers predicted churn risk.
3. **`InternetService_Fiber optic` (Mean |SHAP| = 0.0413)**: Fiber optic subscription consistently pushes churn risk higher.
4. **`TechSupport_No` & `OnlineSecurity_No`**: Absence of add-on support and security services elevates predicted churn risk.

---

## 👤 Individual Customer Local Explanations

### Case 1: High Risk Account (Test Customer #627)
- **Predicted Churn Probability**: **96.57%**
- **Predicted Status**: High Churn Risk (1)
- **Account Context**: Contract = `Month-to-month`, Tenure = `3 months`, Monthly Charges = `$96.6`, Internet = `Fiber optic`

#### Factors Increasing Predicted Churn Risk:
- Having a flexible Month-to-Month contract contributed (+0.0492) towards a increased predicted churn probability.
- Customer tenure duration contributed (+0.0478) towards a increased predicted churn probability.
- Subscribing to Fiber Optic internet service contributed (+0.0449) towards a increased predicted churn probability.
- Being in the first-year onboarding period (tenure <= 12 months) contributed (+0.0314) towards a increased predicted churn probability.

#### Factors Decreasing Predicted Churn Risk:
- Senior citizen account status contributed (-0.0017) towards a decreased predicted churn probability.
- Feature 'PhoneService_No' contributed (-0.0007) towards a decreased predicted churn probability.
- Feature 'MultipleLines_No phone service' contributed (-0.0007) towards a decreased predicted churn probability.
- Feature 'PhoneService_Yes' contributed (-0.0005) towards a decreased predicted churn probability.

---

### Case 2: Low Risk Account (Test Customer #1008)
- **Predicted Churn Probability**: **0.01%**
- **Predicted Status**: Low Risk / Retained (0)
- **Account Context**: Contract = `Two year`, Tenure = `72 months`, Monthly Charges = `$88.7`, Internet = `DSL`

#### Factors Decreasing Predicted Churn Risk:
- Having a flexible Month-to-Month contract contributed (-0.0801) towards a decreased predicted churn probability.
- Having a long-term 2-Year contract commitment contributed (-0.0743) towards a decreased predicted churn probability.
- Customer tenure duration contributed (-0.0685) towards a decreased predicted churn probability.
- Cumulative total charges billed contributed (-0.0434) towards a decreased predicted churn probability.

---

## ⚠️ Distinction Between Association & Causation
- **Statistical Association**: SHAP measures feature contribution towards the *model's statistical probability calculation*.
- **No Direct Causation**: Having a month-to-month contract is *associated* with higher predicted churn, but it does not independently *cause* a customer to leave. External unobserved factors (such as competitor discounts or poor service quality) drive the behavior.

---

## 📁 Generated Artifacts & Visualizations
- **Feature Importance CSV**: [`reports/feature_importance.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/feature_importance.csv)
- **Native Feature Importance Chart**: [`reports/figures/feature_importance_native.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/feature_importance_native.png)
- **SHAP Bar Plot**: [`reports/figures/shap_bar_plot.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/shap_bar_plot.png)
- **SHAP Summary Plot**: [`reports/figures/shap_summary_plot.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/shap_summary_plot.png)
