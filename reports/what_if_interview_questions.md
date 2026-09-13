# Stage 10: What-If Churn Analysis — Data Science Interview Guide

This guide provides technical and conceptual answers to key data science interview questions covering **What-If Analysis**, **Scenario Simulation**, **Sensitivity Analysis**, and **Causal vs Predictive Modeling**.

---

### Q1: What is What-If Analysis in Machine Learning?
**Answer:**  
What-If Analysis (also known as Scenario or Sensitivity Analysis) is a decision-support technique where hypothetical changes are applied to input feature values of a trained machine learning model to observe and quantify how the model's output prediction (e.g. churn probability) changes relative to the baseline profile.

---

### Q2: Why did you add What-If Analysis to this Customer Churn project?
**Answer:**  
Supervised machine learning models output static predictions for a customer's current state. Adding What-If Analysis empowers customer success and retention teams to simulate proactive intervention strategies (e.g., offering a 1-Year contract upgrade or adding free Tech Support) and evaluate which hypothetical intervention yields the highest estimated reduction in churn risk before committing marketing capital.

---

### Q3: How did you generate scenario predictions without modifying the underlying model?
**Answer:**  
We created a reusable simulation function `run_what_if_analysis()` that:
1. Accepts a customer baseline record as a dictionary.
2. Applies specific hypothetical feature updates (e.g., changing `Contract` from `Month-to-month` to `One year`).
3. Re-computes dynamic engineered features (`AverageMonthlySpend`, `ServiceCount`, `IsFirstYear`) consistently.
4. Passes the modified feature vector through the exact saved `ColumnTransformer` + Random Forest pipeline artifact (`models/final_churn_pipeline.joblib`) via `predict_proba()`.
5. Compares the baseline probability $P_{\text{base}}$ to scenario probability $P_{\text{scenario}}$.

---

### Q4: Why did you use the saved ML pipeline instead of retraining the model for scenarios?
**Answer:**  
Retraining the model on scenario data would alter the learned decision boundaries, corrupt model evaluation consistency, and introduce catastrophic data leakage. The goal of What-If Analysis is to measure how the **already trained, validated production model** responds to hypothetical customer changes.

---

### Q5: What is the fundamental difference between Predictive Sensitivity and Causal Inference?
**Answer:**  
- **Predictive Sensitivity**: Measures how the model's mathematical output formula changes when an input variable is altered ($P(Y \mid \text{do}(X))$ is approximated by model inference $P(Y \mid X)$).
- **Causal Inference**: Proves whether changing $X$ in the real world physically *causes* outcome $Y$ to change (accounting for unobserved confounders via A/B testing or Causal DAGs).
What-If Analysis measures predictive sensitivity. Decreasing predicted churn probability in a scenario does **not guarantee** that forcing a contract change will prevent a real-world customer from leaving.

---

### Q6: How did you prevent unrealistic or impossible hypothetical scenarios?
**Answer:**  
1. **Domain Bounds**: Numerical features were restricted to empirical training distributions (e.g., tenure $\in [1, 72]$ months, monthly charges within observed plan tiers).
2. **Categorical Audit**: Only valid categorical levels present in the `OneHotEncoder` vocabulary (e.g., `One year`, `Two year`) were permitted.
3. **Feature Interdependence**: Dependent engineered features were dynamically synchronized (e.g., modifying `tenure` automatically updated `TotalCharges = tenure * MonthlyCharges` and `IsFirstYear = int(tenure <= 12)`).

---

### Q7: How does What-If Analysis help business stakeholders and retention teams?
**Answer:**  
- **Intervention Prioritization**: Allows account managers to compare retention plays (e.g. $10 billing discount vs free Tech Support) to see which play achieves a larger risk reduction for a specific customer profile.
- **Sensitivity Auditing**: Helps data scientists verify that the model exhibits domain-logical behavior (e.g. longer tenure and longer contract commitments decrease predicted churn risk).
- **Interactive Decision Support**: Serves as the analytical core for interactive dashboard simulators.

---

### Q8: How did you calculate Probability Change and Risk Direction?
**Answer:**  
- **Probability Change ($\Delta P$)**:
  $$\Delta P = P_{\text{scenario}} - P_{\text{baseline}}$$
- **Absolute Change**: $|\Delta P|$
- **Risk Direction**:
  - $\Delta P < -0.001 \implies$ **Decreased Risk**
  - $\Delta P > +0.001 \implies$ **Increased Risk**
  - $|\Delta P| \le 0.001 \implies$ **No Significant Change**

---

### Q9: What are the main limitations of What-If Scenario Analysis?
**Answer:**  
1. **Model Boundedness**: It evaluates the model's learned associations, which may fail if real-world macro conditions shift.
2. **Unobserved Confounders**: Real-world customer decisions depend on unobserved factors like competitor pricing or customer service satisfaction.
3. **Correlation vs Causation**: High-churn features might be symptoms rather than root causes of dissatisfaction.

---

### Q10: How would you integrate this feature into a web dashboard (e.g. Streamlit)?
**Answer:**  
In a Streamlit dashboard:
1. Load `models/final_churn_pipeline.joblib`.
2. Allow users to select a customer ID from a dropdown to view their baseline risk score and current attributes.
3. Provide interactive UI widgets (sliders for `tenure` and `MonthlyCharges`, dropdowns for `Contract` and `TechSupport`).
4. On widget change, trigger `run_what_if_analysis()`, re-compute predictions instantly, and render a dynamic Plotly gauge and comparison bar chart showing real-time risk delta.
