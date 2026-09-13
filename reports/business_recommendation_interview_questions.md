# Stage 11: Business Recommendation Engine — Data Science Interview Guide

This guide provides technical and conceptual answers to key data science interview questions covering **Recommendation Engine Design**, **Explainable AI Integration**, **Customer Segmentation Context**, and **Responsible AI / Non-Causal Decision Support**.

---

### Q1: What is a Business Recommendation Engine in the context of Customer Churn?
**Answer:**  
A Business Recommendation Engine is a decision-support layer that converts raw machine learning churn probabilities, SHAP explainability feature drivers, customer segments, and scenario simulations into structured, human-readable action items for business stakeholders. Rather than simply declaring *who* is likely to leave, it provides *why* they are at risk and *what specific retention action* an account manager should consider.

---

### Q2: Why did you add a Recommendation Engine to this Customer Churn project?
**Answer:**  
In real-world business settings, raw ML probability scores (e.g. `0.872`) are difficult for non-technical customer success representatives to operationalize. Adding a recommendation engine bridges the gap between predictive data science and business operations by assigning clear risk levels, prioritizing high-risk accounts, and suggesting tailored retention plays (e.g. contract migration, pricing reviews, or tech support outreach).

---

### Q3: How are Risk Levels calculated and configured?
**Answer:**  
Risk levels are categorized into 3 distinct tiers using configurable probability boundaries:
- **High Risk**: $\text{Churn Probability} \ge 0.60$ (60%+)
- **Medium Risk**: $0.35 \le \text{Churn Probability} < 0.60$ (35% to 60%)
- **Low Risk**: $\text{Churn Probability} < 0.35$ (< 35%)
These thresholds prioritize customer success resources towards the most vulnerable accounts.

---

### Q4: How did you incorporate SHAP model explainability into recommendation generation?
**Answer:**  
Rather than using arbitrary domain guesses, we used the top global SHAP feature drivers identified in Stage 8 (`Contract_Month-to-month`, `tenure`, `InternetService_Fiber optic`, `TechSupport_No`, `OnlineSecurity_No`, `PaymentMethod_Electronic check`, `MonthlyCharges`). The recommendation engine audits each customer's specific profile against these empirical SHAP drivers to generate personalized risk factors (e.g., flagging "Flexible Month-to-Month Contract" and "No Tech Support Service").

---

### Q5: How did Customer Segmentation (Stage 9) enhance the recommendation engine?
**Answer:**  
Customer Segmentation provided macro behavioral archetypes. For example, if a high-risk customer belongs to **Segment 2: High-Spend At-Risk Onboarders** (a segment with a 46.4% observed churn rate), the recommendation engine tailors outreach specifically for early onboarding price sensitivity rather than treating them like a long-term power user.

---

### Q6: How did What-If Scenario Analysis (Stage 10) support recommendations?
**Answer:**  
What-If analysis provided empirical model sensitivity estimates for recommended actions. When recommending a contract upgrade, the engine can attach scenario context: *"Scenario simulations estimate that upgrading to a 1-Year contract reduces estimated churn risk by ~27 percentage points."* This allows account managers to compare estimated risk deltas across competing retention plays.

---

### Q7: What is the difference between a Churn Prediction and a Business Recommendation?
**Answer:**  
- **Churn Prediction**: A quantitative model score estimating the mathematical probability $P(Y=1 \mid X)$ that a customer will churn.
- **Business Recommendation**: A prescriptive decision-support directive ($A$) advising human managers on an operational action based on the prediction, risk factors, and business constraints.

---

### Q8: Why can't recommendations be considered guaranteed causal interventions?
**Answer:**  
Machine learning models learn **statistical correlations**, not **physical causal mechanisms**. Observing that `Contract_Month-to-month` correlates with high churn does not guarantee that forcing a customer into a 1-Year contract will cause them to stay in real life (they may face unobserved issues like competitor offers or service outages). Recommendations must be explicitly framed as decision-support suggestions, not causal guarantees.

---

### Q9: How can a telecommunications company operationalize this system?
**Answer:**  
1. **Daily Operational Feed**: Export `reports/customer_recommendations.csv` into CRM tools (e.g., Salesforce / HubSpot).
2. **Prioritized Queue**: Filter by `Priority == High` and `Risk_Level == High Risk` to auto-assign high-value accounts to Customer Success reps.
3. **Tailored Scripting**: Reps use the `key_risk_factors` and `recommendation` text to guide customer retention phone calls and offer targeted discounts or free add-ons.

---

### Q10: What are the main Responsible AI considerations in recommendation design?
**Answer:**  
- **Non-Causal Framing**: Explicitly warning users that recommendations do not guarantee retention.
- **Human-in-the-Loop**: Ensuring human managers review recommendations before taking significant action.
- **Fairness & Non-Discrimination**: Verifying that demographic attributes (e.g. `gender`) are not used as risk drivers or recommendation criteria.
- **Data Privacy**: Protecting customer PII during data processing and export.

---

### Q11: How would you improve this recommendation engine in a production setup?
**Answer:**  
1. **Constraint Optimization / Linear Programming**: Incorporate financial customer lifetime value (CLV) and retention budget constraints to maximize overall ROI.
2. **Reinforcement Learning from Human Feedback (RLHF)**: Track which recommendations human reps select and whether real-world retention calls succeed, feeding outcome labels back into the recommendation engine.
3. **Causal Forest / Uplift Modeling**: Train dedicated Uplift Models to estimate incremental treatment effect (ITE) for specific marketing promotions.

---

### Q12: How will this Recommendation Engine be integrated into the final Web Dashboard?
**Answer:**  
In the Streamlit dashboard:
- **Executive Summary Tab**: Render pie charts of risk levels and recommendation categories.
- **Customer Lookup Tab**: Allow reps to select any Customer ID to display their risk badge, top SHAP drivers, segment badge, and instant recommendation card with What-If scenario sliders.
- **Filterable Export Table**: Display a searchable data table of prioritized high-risk accounts with one-click CSV export.
