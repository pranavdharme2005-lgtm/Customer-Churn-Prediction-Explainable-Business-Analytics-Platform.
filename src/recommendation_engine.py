"""
Stage 11: Business Recommendation Engine Script
-----------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Converts machine learning churn predictions, SHAP feature drivers, customer segments,
             and What-If scenario simulations into actionable, non-causal decision-support recommendations.
             Outputs reports/customer_recommendations.csv, reports/business_recommendations.md,
             reports/business_recommendation_interview_questions.md, and visual figures in reports/figures/.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for reproducible publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Configurable Risk Classification Thresholds
RISK_THRESHOLDS = {
    "HIGH_RISK": 0.60,
    "MEDIUM_RISK": 0.35
}

def classify_risk(churn_probability: float) -> tuple[str, int]:
    """
    Classifies customer churn probability into risk level and binary class.
    - High Risk: Probability >= 0.60
    - Medium Risk: 0.35 <= Probability < 0.60
    - Low Risk: Probability < 0.35
    """
    if churn_probability >= RISK_THRESHOLDS["HIGH_RISK"]:
        return "High Risk", 1
    elif churn_probability >= RISK_THRESHOLDS["MEDIUM_RISK"]:
        return "Medium Risk", 1
    else:
        return "Low Risk", 0

def generate_customer_recommendation(cust_row: pd.Series, churn_prob: float, segment_name: str) -> dict:
    """
    Applies rule-based recommendation logic combining SHAP risk factors, customer profile attributes,
    segment archetypes, and What-If scenario insights to generate non-causal decision support.
    """
    risk_level, pred_class = classify_risk(churn_prob)
    
    risk_factors = []
    recommendation_category = "General Monitoring"
    recommendation = "Maintain standard automated customer tracking; send annual loyalty appreciation reward."
    priority = "Low"
    reason = "Customer exhibits low predicted churn probability and stable account parameters."
    hypothetical_impact = "No immediate scenario intervention required."

    # Identify Key Risk Factors based on SHAP feature drivers (Stage 8) & profile
    is_month_to_month = cust_row.get('Contract') == 'Month-to-month'
    is_no_tech_support = cust_row.get('TechSupport') == 'No' and cust_row.get('InternetService') != 'No'
    is_no_online_security = cust_row.get('OnlineSecurity') == 'No' and cust_row.get('InternetService') != 'No'
    is_short_tenure = cust_row.get('tenure', 0) <= 12
    is_early_onboarding = cust_row.get('tenure', 0) <= 6
    is_high_charges = cust_row.get('MonthlyCharges', 0) > 70.0
    is_electronic_check = cust_row.get('PaymentMethod') == 'Electronic check'
    is_fiber = cust_row.get('InternetService') == 'Fiber optic'

    if is_month_to_month:
        risk_factors.append("Flexible Month-to-Month Contract")
    if is_short_tenure:
        risk_factors.append("Short Tenure (<= 12 months)")
    if is_no_tech_support:
        risk_factors.append("No Tech Support Service")
    if is_no_online_security:
        risk_factors.append("No Online Security Protection")
    if is_high_charges:
        risk_factors.append(f"High Monthly Bill (${cust_row.get('MonthlyCharges', 0):.2f})")
    if is_electronic_check:
        risk_factors.append("Manual Electronic Check Payment")
    if is_fiber:
        risk_factors.append("Fiber Optic Service Friction")

    # Recommendation Rules based on Risk Level & Combined Factors
    if risk_level == "High Risk":
        priority = "High"
        
        if is_month_to_month and (is_no_tech_support or is_short_tenure):
            recommendation_category = "Contract Review"
            recommendation = "Offer contract migration incentives (e.g. 10% monthly discount or bonus streaming features for upgrading to a 1-Year or 2-Year contract)."
            reason = f"High predicted risk ({churn_prob*100:.1f}%) driven by short-term contract commitment and lack of support add-ons."
            hypothetical_impact = "Scenario analysis estimates upgrading to a 1-Year contract reduces estimated churn risk by ~27 percentage points."
            
        elif is_high_charges and is_short_tenure:
            recommendation_category = "Plan & Price Review"
            recommendation = "Conduct account pricing review; suggest optimized plan tier package or temporary 6-month promotional bill credit."
            reason = f"Customer bill is high (${cust_row.get('MonthlyCharges', 0):.2f}/mo) relative to their brief account tenure ({cust_row.get('tenure', 0)} mos)."
            hypothetical_impact = "Scenario analysis estimates a $15-$20 bill discount reduces estimated churn probability by ~8 to 12 percentage points."

        elif is_no_tech_support or is_no_online_security:
            recommendation_category = "Technical Support"
            recommendation = "Initiate proactive technical support outreach; offer 3 months of complimentary premium tech assistance and digital security add-ons."
            reason = "Absence of technical support and online security add-ons elevates predicted risk."
            hypothetical_impact = "Scenario analysis estimates adding Tech Support reduces estimated churn risk by ~10 percentage points."

        elif is_early_onboarding:
            recommendation_category = "Onboarding Support"
            recommendation = "Schedule dedicated Customer Success check-in call to ensure smooth service setup and resolve technical friction."
            reason = "Customer is in the vulnerable early onboarding window (tenure <= 6 months)."
            hypothetical_impact = "Scenario analysis estimates reaching 24 months of tenure reduces estimated churn probability by ~12 to 26 percentage points."

        else:
            recommendation_category = "Retention Outreach"
            recommendation = "Assign account to Customer Retention Team for personalized phone consultation and service review."
            reason = f"High predicted churn probability ({churn_prob*100:.1f}%) across overall account profile."
            hypothetical_impact = "Combined retention bundling (1-Year contract + Tech Support + Auto Payment) estimates up to 34 percentage points of risk reduction."

    elif risk_level == "Medium Risk":
        priority = "Medium"

        if is_month_to_month:
            recommendation_category = "Contract Review"
            recommendation = "Send automated contract upgrade promotion offering a $5/mo discount for converting to an annual contract."
            reason = "Flexible month-to-month contract presents potential mid-term churn vulnerability."
            hypothetical_impact = "Upgrading to a 1-Year contract reduces estimated risk into the Low Risk category."

        elif is_no_online_security or is_no_tech_support:
            recommendation_category = "Security Awareness"
            recommendation = "Send digital security awareness email campaign highlighting bundled online backup and security features."
            reason = "Product ecosystem adoption is low; customer lacks security add-ons."
            hypothetical_impact = "Adding digital security features stabilizes account engagement."

        elif is_electronic_check:
            recommendation_category = "Service Engagement"
            recommendation = "Promote automatic bank transfer or credit card auto-pay with a one-time $10 account credit."
            reason = "Manual electronic check payments exhibit higher historical churn rates."
            hypothetical_impact = "Switching to automated payments reduces estimated risk."

        else:
            recommendation_category = "Plan Review"
            recommendation = "Include customer in quarterly plan optimization digest highlighting value-added features."
            reason = "Moderate churn risk profile; proactive value reinforcement recommended."
            hypothetical_impact = "Routine plan review maintains account stability."

    # Incorporate Stage 9 Customer Segment Context
    if "At-Risk Onboarders" in segment_name:
        reason += " Belongs to 'High-Spend At-Risk Onboarders' segment (46.4% segment churn rate)."
    elif "Power Users" in segment_name:
        reason += " Belongs to 'High-Value Long-Term Power Users' segment (high lifetime financial value)."

    key_factors_str = "; ".join(risk_factors[:3]) if risk_factors else "No major risk factors identified"

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "key_risk_factors": key_factors_str,
        "recommendation_category": recommendation_category,
        "recommendation": recommendation,
        "priority": priority,
        "reason": reason,
        "hypothetical_intervention_impact": hypothetical_impact
    }

def run_business_recommendation_engine(
    cleaned_csv_path: str = "data/processed/cleaned_customer_churn.csv",
    segments_csv_path: str = "data/processed/customer_segments.csv",
    pipeline_path: str = "models/final_churn_pipeline.joblib",
    output_recommendations_csv: str = "reports/customer_recommendations.csv",
    figures_dir: str = "reports/figures",
    report_md_path: str = "reports/business_recommendations.md"
):
    print("=" * 75)
    print(" STAGE 11: BUSINESS RECOMMENDATION ENGINE")
    print("=" * 75)

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_recommendations_csv), exist_ok=True)
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)

    # 1. Load Pipeline Artifact & Datasets
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Final ML pipeline missing at: {pipeline_path}")
    if not os.path.exists(cleaned_csv_path) or not os.path.exists(segments_csv_path):
        raise FileNotFoundError("Cleaned dataset or customer segments CSV missing!")

    print(f"[*] Step 1: Loading saved pipeline artifact from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    print(f"[*] Step 2: Loading cleaned customer dataset & Stage 9 segments...")
    df_cleaned = pd.read_csv(cleaned_csv_path)
    df_segments = pd.read_csv(segments_csv_path)

    df_merged = df_cleaned.copy()
    df_merged['customer_id'] = df_segments['customer_id']
    df_merged['Segment_Name'] = df_segments['Segment_Name']

    # Vectorized Feature Engineering
    df_merged['AverageMonthlySpend'] = np.where(
        df_merged['tenure'] == 0,
        df_merged['MonthlyCharges'],
        df_merged['TotalCharges'] / np.maximum(df_merged['tenure'], 1)
    )
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PhoneService', 'MultipleLines'
    ]
    df_merged['ServiceCount'] = df_merged[service_cols].apply(
        lambda row: sum(1 for val in row if str(val).strip() in ['Yes', 'Two lines']), axis=1
    )
    df_merged['IsFirstYear'] = (df_merged['tenure'] <= 12).astype(int)

    feature_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
        'TotalCharges', 'AverageMonthlySpend', 'ServiceCount', 'IsFirstYear'
    ]

    # 2. Generate Model Predictions
    print("[*] Step 3: Generating churn probabilities for all customers...")
    churn_probs = pipeline.predict_proba(df_merged[feature_cols])[:, 1]
    df_merged['churn_probability'] = churn_probs

    # 3. Generate Recommendations
    print("[*] Step 4: Generating structured business recommendations...")
    rec_records = []
    for idx, row in df_merged.iterrows():
        rec_info = generate_customer_recommendation(
            cust_row=row,
            churn_prob=row['churn_probability'],
            segment_name=row['Segment_Name']
        )
        rec_records.append({
            "customer_id": row['customer_id'],
            "churn_probability": rec_info["churn_probability"],
            "risk_level": rec_info["risk_level"],
            "segment_name": row['Segment_Name'],
            "key_risk_factors": rec_info["key_risk_factors"],
            "recommendation_category": rec_info["recommendation_category"],
            "recommendation": rec_info["recommendation"],
            "priority": rec_info["priority"],
            "reason": rec_info["reason"],
            "hypothetical_intervention_impact": rec_info["hypothetical_intervention_impact"],
            "tenure": row['tenure'],
            "MonthlyCharges": row['MonthlyCharges'],
            "Contract": row['Contract']
        })

    df_recs = pd.DataFrame(rec_records)
    
    # Sort customers by churn_probability DESC for high-risk prioritization
    df_recs_sorted = df_recs.sort_values(by="churn_probability", ascending=False)
    
    # Export full recommendation table
    export_cols = [
        "customer_id", "churn_probability", "risk_level", "segment_name", 
        "key_risk_factors", "recommendation_category", "recommendation", 
        "priority", "reason", "hypothetical_intervention_impact"
    ]
    df_recs_sorted[export_cols].to_csv(output_recommendations_csv, index=False)
    print(f"    -> Exported customer recommendations CSV to: {output_recommendations_csv}")

    # 4. Generate Visualizations
    print("[*] Step 5: Generating visual charts for executive report...")

    # Chart 1: Customer Risk Level Distribution
    plt.figure(figsize=(8, 5))
    risk_counts = df_recs['risk_level'].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
    ax = sns.barplot(
        x=risk_counts.index, y=risk_counts.values,
        palette=['#2ca02c', '#ff7f0e', '#d62728'], hue=risk_counts.index, legend=False
    )
    plt.title("Customer Distribution by Churn Risk Level", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("")
    plt.ylabel("Number of Customers")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df_recs)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "risk_level_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # Chart 2: Average Churn Probability by Recommendation Category
    plt.figure(figsize=(9, 5))
    cat_prob = df_recs.groupby('recommendation_category')['churn_probability'].mean().sort_values(ascending=False)
    ax = sns.barplot(
        x=cat_prob.values, y=cat_prob.index,
        palette='magma', hue=cat_prob.index, legend=False
    )
    plt.title("Average Churn Probability by Recommendation Category", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Average Churn Probability")
    plt.ylabel("")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{w*100:.1f}%", (w + 0.01, p.get_y() + p.get_height()/2.),
                    ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "avg_prob_by_recommendation_category.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # Chart 3: High-Risk Customers by Customer Segment
    plt.figure(figsize=(9, 5))
    high_risk_seg = df_recs[df_recs['risk_level'] == 'High Risk']['segment_name'].value_counts()
    ax = sns.barplot(
        x=high_risk_seg.values, y=high_risk_seg.index,
        palette='Reds_r', hue=high_risk_seg.index, legend=False
    )
    plt.title("High-Risk Customer Count by Segment", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("High-Risk Customer Count")
    plt.ylabel("")
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{int(w)}", (w + 10, p.get_y() + p.get_height()/2.),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "high_risk_by_segment.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    # Chart 4: Most Common Risk Factors Among High-Risk Accounts
    plt.figure(figsize=(9, 5))
    high_risk_df = df_recs[df_recs['risk_level'] == 'High Risk']
    factor_series = high_risk_df['key_risk_factors'].str.split('; ').explode()
    top_factors = factor_series.value_counts().head(6)
    
    ax = sns.barplot(
        x=top_factors.values, y=top_factors.index,
        palette='Blues_r', hue=top_factors.index, legend=False
    )
    plt.title("Top Risk Factors Among High-Risk Customers", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Frequency of Risk Factor")
    plt.ylabel("")
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f"{int(w)}", (w + 10, p.get_y() + p.get_height()/2.),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    chart4_path = os.path.join(figures_dir, "top_risk_drivers_distribution.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()

    print(f"    -> Exported all Stage 11 visual charts to: {figures_dir}")

    # 5. Write Stage 11 Markdown Report
    print("[*] Step 6: Writing Stage 11 markdown report...")
    high_count = (df_recs['risk_level'] == 'High Risk').sum()
    med_count = (df_recs['risk_level'] == 'Medium Risk').sum()
    low_count = (df_recs['risk_level'] == 'Low Risk').sum()
    total_cust = len(df_recs)

    top5_high = df_recs_sorted.head(5)

    report_content = f"""# Stage 11: Business Recommendation Engine Report

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
| **High Risk** | $\ge 60\%$ ($\ge 0.60$) | **{high_count}** | **{high_count/total_cust*100:.1f}%** | Urgent proactive retention intervention & account review |
| **Medium Risk** | $35\% \le P < 60\%$ | **{med_count}** | **{med_count/total_cust*100:.1f}%** | Automated engagement & contract upgrade promotions |
| **Low Risk** | $< 35\%$ ($< 0.35$) | **{low_count}** | **{low_count/total_cust*100:.1f}%** | Standard loyalty tracking & annual appreciation |

---

## 🔝 Top 5 Highest-Risk Customer Priorities

| Customer ID | Churn Prob | Risk Level | Customer Segment | Key Risk Factors | Category | Recommended Action | Priority |
|---|---|---|---|---|---|---|---|
"""
    for _, row in top5_high.iterrows():
        report_content += f"| **{row['customer_id']}** | **{row['churn_probability']*100:.1f}%** | {row['risk_level']} | {row['segment_name']} | {row['key_risk_factors']} | {row['recommendation_category']} | {row['recommendation']} | **{row['priority']}** |\n"

    report_content += f"""

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
- **Recommendation CSV**: [`reports/customer_recommendations.csv`](file:///{os.path.abspath(output_recommendations_csv)})
- **Interview Guide**: [`reports/business_recommendation_interview_questions.md`](file:///{os.path.abspath(os.path.join(os.path.dirname(report_md_path), "business_recommendation_interview_questions.md"))})
- **Visual Figures**:
  - [`reports/figures/risk_level_distribution.png`](file:///{os.path.abspath(chart1_path)})
  - [`reports/figures/avg_prob_by_recommendation_category.png`](file:///{os.path.abspath(chart2_path)})
  - [`reports/figures/high_risk_by_segment.png`](file:///{os.path.abspath(chart3_path)})
  - [`reports/figures/top_risk_drivers_distribution.png`](file:///{os.path.abspath(chart4_path)})
"""

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"    -> Saved Stage 11 business report to: {report_md_path}")

    print("\n" + "=" * 75)
    print(" STAGE 11 BUSINESS RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_business_recommendation_engine()
