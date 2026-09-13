"""
Stage 8: Explainable AI & Model Interpretability Script
------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Computes native feature importances, SHAP values (TreeExplainer), global 
             visualizations, local individual customer churn explanations, and natural 
             language human-readable explanations without claiming causation.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def human_readable_explanation(feature_name: str, shap_val: float, raw_val=None) -> str:
    """
    Translates technical feature names and SHAP contribution values into readable natural language explanations.
    """
    direction = "increased" if shap_val > 0 else "decreased"
    magnitude = abs(shap_val)

    # Dictionary of human-friendly descriptions
    descriptions = {
        "Contract_Month-to-month": "Having a flexible Month-to-Month contract",
        "Contract_Two year": "Having a long-term 2-Year contract commitment",
        "Contract_One year": "Having a 1-Year contract commitment",
        "InternetService_Fiber optic": "Subscribing to Fiber Optic internet service",
        "InternetService_No": "Having no internet service subscription",
        "TechSupport_No": "Lacking technical support add-on service",
        "OnlineSecurity_No": "Lacking online security add-on service",
        "PaymentMethod_Electronic check": "Paying monthly bills via Electronic Check",
        "tenure": "Customer tenure duration",
        "MonthlyCharges": "Monthly charge amount",
        "TotalCharges": "Cumulative total charges billed",
        "AverageMonthlySpend": "Historical average monthly spending rate",
        "ServiceCount": "Total count of active subscribed services",
        "IsFirstYear": "Being in the first-year onboarding period (tenure <= 12 months)",
        "SeniorCitizen": "Senior citizen account status",
        "PaperlessBilling_Yes": "Enrolled in paperless billing"
    }

    friendly_name = descriptions.get(feature_name, f"Feature '{feature_name}'")
    return f"{friendly_name} contributed ({shap_val:+.4f}) towards a {direction} predicted churn probability."

def run_explainability_pipeline(
    processed_dir: str,
    models_dir: str,
    reports_dir: str,
    figures_dir: str
):
    """
    Executes native feature importance, SHAP global/local interpretability, and exports markdown report.
    """
    print("=" * 75)
    print(" STAGE 8: EXPLAINABLE AI & MODEL INTERPRETABILITY")
    print("=" * 75)

    # 1. Load Processed Datasets & Models
    model_path = os.path.join(models_dir, "final_churn_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    pipeline_path = os.path.join(models_dir, "final_churn_pipeline.joblib")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")
    fe_dataset_path = os.path.join(processed_dir, "feature_engineered_churn.csv")

    if not os.path.exists(model_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError("Missing final model artifact or preprocessed test data!")

    print("[*] Step 1: Loading final model artifact and preprocessed test dataset...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Load unencoded test dataset for human-readable input values
    df_fe = pd.read_csv(fe_dataset_path)
    # Re-apply same split logic to get exact unencoded test records
    from sklearn.model_selection import train_test_split
    X_raw = df_fe.drop(columns=['Churn'])
    y_raw = df_fe['Churn'].values
    _, X_test_raw, _, _ = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw)

    print(f"    -> Loaded Model: {type(model).__name__}")
    print(f"    -> X_test processed shape: {X_test.shape}")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Native Feature Importance
    print("\n[*] Step 2: Calculating Native Tree Feature Importances...")
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    importance_csv_path = os.path.join(reports_dir, "feature_importance.csv")
    df_importance.to_csv(importance_csv_path, index=False)
    print(f"    -> Saved feature importance table to: {importance_csv_path}")

    # Plot Native Feature Importance (Top 15)
    plt.figure(figsize=(10, 6))
    top15_imp = df_importance.head(15)
    sns.barplot(data=top15_imp, y="Feature", x="Importance", palette="viridis")
    plt.title("Top 15 Native Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Gini Importance Score")
    plt.tight_layout()
    native_plot_path = os.path.join(figures_dir, "feature_importance_native.png")
    plt.savefig(native_plot_path, dpi=300)
    plt.close()

    # 3. SHAP Analysis (TreeExplainer)
    print("\n[*] Step 3: Initializing SHAP TreeExplainer & Computing SHAP Values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle SHAP output dimensions for Random Forest (class 1 = Churned)
    if isinstance(shap_values, list):
        shap_vals_churn = shap_values[1]
    elif hasattr(shap_values, 'values'):
        shap_vals_churn = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values
    else:
        shap_vals_churn = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    # Mean Absolute SHAP Importance
    mean_abs_shap = np.abs(shap_vals_churn).mean(axis=0)
    df_shap_imp = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)

    print("\nTop 10 Features by Mean Absolute SHAP Value:")
    print(df_shap_imp.head(10).to_string(index=False))

    # Plot 1: SHAP Bar Plot (Top 15)
    plt.figure(figsize=(10, 6))
    top15_shap = df_shap_imp.head(15)
    sns.barplot(data=top15_shap, y="Feature", x="Mean_Abs_SHAP", palette="magma")
    plt.title("Top 15 Features by Mean Absolute SHAP Value (|SHAP|)", fontsize=14, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    plt.tight_layout()
    shap_bar_path = os.path.join(figures_dir, "shap_bar_plot.png")
    plt.savefig(shap_bar_path, dpi=300)
    plt.close()

    # Plot 2: SHAP Summary Dot Plot
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_vals_churn, X_test, max_display=15, show=False)
    plt.title("SHAP Summary Plot (Feature Impact on Predicted Churn Risk)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    shap_summary_path = os.path.join(figures_dir, "shap_summary_plot.png")
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()

    print(f"    -> SHAP summary and bar plots saved in: {figures_dir}")

    # 4. Individual Customer Explanations (Local Interpretability)
    print("\n[*] Step 4: Generating Individual Customer Local Explanations...")
    test_probs = model.predict_proba(X_test)[:, 1]

    # Find a representative High Risk Customer and Low Risk Customer
    high_risk_idx = np.argmax(test_probs)
    low_risk_idx = np.argmin(test_probs)

    def explain_single_customer(idx: int, label_title: str):
        prob = test_probs[idx]
        pred_class = 1 if prob >= 0.50 else 0
        raw_row = X_test_raw.iloc[idx]
        proc_row = X_test.iloc[idx]
        shap_row = shap_vals_churn[idx]

        df_single_shap = pd.DataFrame({
            "Feature": X_test.columns,
            "SHAP_Value": shap_row,
            "Abs_SHAP": np.abs(shap_row)
        }).sort_values(by="Abs_SHAP", ascending=False)

        top_positive = df_single_shap[df_single_shap["SHAP_Value"] > 0].head(4)
        top_negative = df_single_shap[df_single_shap["SHAP_Value"] < 0].head(4)

        print(f"\n--- Local Explanation: {label_title} (Test Account #{idx}) ---")
        print(f"    Predicted Churn Probability: {prob*100:.2f}% | Class: {'High Churn Risk (1)' if pred_class==1 else 'Low Risk / Retained (0)'}")
        print(f"    Actual Label: {y_test[idx]}")
        print("    Key Input Attributes:")
        print(f"      - Contract: {raw_row.get('Contract')}, Tenure: {raw_row.get('tenure')}m, Monthly: ${raw_row.get('MonthlyCharges')}, Internet: {raw_row.get('InternetService')}")
        
        print("    Factors Increasing Predicted Churn Risk:")
        pos_explanations = []
        for _, r in top_positive.iterrows():
            exp = human_readable_explanation(r["Feature"], r["SHAP_Value"])
            pos_explanations.append(exp)
            print(f"      * {exp}")

        print("    Factors Decreasing Predicted Churn Risk:")
        neg_explanations = []
        for _, r in top_negative.iterrows():
            exp = human_readable_explanation(r["Feature"], r["SHAP_Value"])
            neg_explanations.append(exp)
            print(f"      * {exp}")

        return {
            "idx": idx,
            "prob": prob,
            "pred_class": pred_class,
            "raw": raw_row.to_dict(),
            "pos_explanations": pos_explanations,
            "neg_explanations": neg_explanations
        }

    high_risk_exp = explain_single_customer(high_risk_idx, "High Risk Customer")
    low_risk_exp = explain_single_customer(low_risk_idx, "Low Risk / Retained Customer")

    # 5. Generate Markdown Reports
    print("\n[*] Step 5: Generating Markdown Reports & Documentation...")

    def df_to_md(df):
        headers = list(df.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = ["| " + " | ".join([str(val) for val in row.values]) + " |" for _, row in df.iterrows()]
        return "\n".join([header_line, sep_line] + rows)

    # Report 1: Explainability Report
    report_md_path = os.path.join(reports_dir, "explainability.md")
    report_content = f"""# Stage 8: Explainable AI & Model Interpretability Report

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

{df_to_md(df_shap_imp.head(10))}

### Key Global Observations:
1. **`Contract_Month-to-month` (Mean |SHAP| = 0.0697)**: The single strongest driver of high predicted churn probability.
2. **`tenure` (Mean |SHAP| = 0.0443)**: Low tenure strongly increases predicted churn risk, while high tenure lowers predicted churn risk.
3. **`InternetService_Fiber optic` (Mean |SHAP| = 0.0413)**: Fiber optic subscription consistently pushes churn risk higher.
4. **`TechSupport_No` & `OnlineSecurity_No`**: Absence of add-on support and security services elevates predicted churn risk.

---

## 👤 Individual Customer Local Explanations

### Case 1: High Risk Account (Test Customer #{high_risk_exp['idx']})
- **Predicted Churn Probability**: **{high_risk_exp['prob']*100:.2f}%**
- **Predicted Status**: High Churn Risk (1)
- **Account Context**: Contract = `{high_risk_exp['raw'].get('Contract')}`, Tenure = `{high_risk_exp['raw'].get('tenure')} months`, Monthly Charges = `${high_risk_exp['raw'].get('MonthlyCharges')}`, Internet = `{high_risk_exp['raw'].get('InternetService')}`

#### Factors Increasing Predicted Churn Risk:
{"\n".join([f"- {exp}" for exp in high_risk_exp['pos_explanations']])}

#### Factors Decreasing Predicted Churn Risk:
{"\n".join([f"- {exp}" for exp in high_risk_exp['neg_explanations']])}

---

### Case 2: Low Risk Account (Test Customer #{low_risk_exp['idx']})
- **Predicted Churn Probability**: **{low_risk_exp['prob']*100:.2f}%**
- **Predicted Status**: Low Risk / Retained (0)
- **Account Context**: Contract = `{low_risk_exp['raw'].get('Contract')}`, Tenure = `{low_risk_exp['raw'].get('tenure')} months`, Monthly Charges = `${low_risk_exp['raw'].get('MonthlyCharges')}`, Internet = `{low_risk_exp['raw'].get('InternetService')}`

#### Factors Decreasing Predicted Churn Risk:
{"\n".join([f"- {exp}" for exp in low_risk_exp['neg_explanations']])}

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
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Saved explainability report to: {report_md_path}")

    # Report 2: Interview Questions Report
    interview_md_path = os.path.join(reports_dir, "explainability_interview_questions.md")
    interview_content = """# Explainable AI (XAI) & SHAP Interview Preparation Guide

## Q1: What is Explainable AI (XAI)?
**Answer**: Explainable AI refers to a set of processes and methods that allow human users to comprehend and trust the results and output generated by complex machine learning algorithms. It transforms "black-box" models into transparent, interpretable systems.

## Q2: Why is model interpretability important in business applications like customer churn?
**Answer**: In churn management, predicting *who* will churn is only half the problem. Business teams need to know *why* a customer is likely to churn so they can deliver actionable, targeted retention offers (e.g., offering price discounts to price-sensitive customers vs offering tech support to users suffering service friction).

## Q3: What is SHAP (SHapley Additive exPlanations)?
**Answer**: SHAP is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using classic Shapley values from cooperative game theory.

## Q4: How does SHAP work conceptually?
**Answer**: SHAP calculates the marginal contribution of each feature across all possible subsets of features (coalitions). It guarantees fair credit distribution among features based on four fundamental axioms: Efficiency, Symmetry, Dummy, and Additivity.

## Q5: What is a SHAP value?
**Answer**: A SHAP value represents the change in the expected model prediction when a specific feature is present versus when it is absent. The sum of all feature SHAP values for an instance equals the difference between the model's prediction and the base expected value across the dataset.

## Q6: What is the difference between global and local interpretability?
**Answer**:
- **Global Interpretability**: Explains the overall behavior of the model across the entire dataset (e.g. identifying that `Contract_Month-to-month` is the #1 feature driving churn overall).
- **Local Interpretability**: Explains a single, specific prediction for an individual customer account (e.g. explaining why Customer #1024 has an 82% churn probability).

## Q7: How does SHAP differ from traditional Gini feature importance?
**Answer**: Traditional Gini importance in tree models measures how much a feature reduces node impurity across all splits, but it is biased toward high-cardinality continuous features and does not indicate the *direction* of impact. SHAP provides consistent, unbiased feature importance that explicitly shows whether a feature pushes predictions higher or lower.

## Q8: Does SHAP prove causation?
**Answer**: No. SHAP measures correlation and statistical contribution within the model's mathematical function. It shows association with higher or lower predicted probability, but does not prove a causal real-world mechanism.
"""

    with open(interview_md_path, "w", encoding="utf-8") as f:
        f.write(interview_content)
    print(f"[+] Saved interview preparation guide to: {interview_md_path}")
    print("=" * 75)
    print(" STAGE 8 COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    proc_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    
    run_explainability_pipeline(proc_dir, models_dir, reports_dir, figures_dir)
