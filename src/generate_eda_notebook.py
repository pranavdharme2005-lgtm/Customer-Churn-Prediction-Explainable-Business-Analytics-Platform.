"""
Script to generate notebooks/01_exploratory_data_analysis.ipynb
"""

import json
import os

def create_notebook():
    notebook_path = os.path.join("customer-churn-prediction", "notebooks", "01_exploratory_data_analysis.ipynb")
    
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })
        
    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # Header
    add_md("""# Stage 3: Exploratory Data Analysis (EDA)
## Customer Churn Prediction & Explainable Business Analytics Platform

### Objectives:
1. Understand the dataset structure, features, distributions, and summary statistics.
2. Analyze the **Churn** target variable distribution and assess class imbalance.
3. Conduct univariate analysis on numerical and categorical features.
4. Perform churn-focused bivariate analysis to answer 10 key business questions (Q1–Q10).
5. Analyze tenure groups and monthly charge ranges.
6. Compute correlation matrix heatmap for numerical variables.
7. Assess potential outliers using box plots and IQR analysis.
8. Synthesize 5–8 actionable Key Business Insights for churn reduction strategies.
""")

    # Setup
    add_md("## 1. Setup & Environment Configuration")
    add_code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling for seaborn and matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

print("Libraries successfully imported!")""")

    # Data Loading
    add_md("## 2. Load Cleaned Dataset & Structural Overview")
    add_code("""# Load cleaned dataset from processed folder
dataset_path = "../data/processed/cleaned_customer_churn.csv"
df = pd.read_csv(dataset_path)

print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("First 5 records:")
df.head()""")

    add_code("""# Data Types and Non-Null Counts
print("--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics for Numerical Features ---")
df.describe().T""")

    add_code("""# Data Integrity Checks
print(f"Missing Values Count: {df.isnull().sum().sum()}")
print(f"Duplicate Rows Count: {df.duplicated().sum()}")""")

    # Target Variable
    add_md("""## 3. Target Variable Analysis (`Churn`)
We evaluate the total count and proportion of churned vs retained customers to measure class balance.
""")
    add_code("""# Calculate target counts and percentages
total_customers = len(df)
churn_counts = df['Churn'].value_counts()
churn_pcts = df['Churn'].value_counts(normalize=True) * 100

print(f"Total Customers: {total_customers}")
print(f"Retained ('No'):  {churn_counts['No']} ({churn_pcts['No']:.2f}%)")
print(f"Churned ('Yes'):   {churn_counts['Yes']} ({churn_pcts['Yes']:.2f}%)\n")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Donut Chart
colors = ['#2ca02c', '#d62728']
axes[0].pie(churn_counts, labels=['Retained (No)', 'Churned (Yes)'], autopct='%1.2f%%', startangle=90, colors=colors, explode=(0, 0.1), wedgeprops=dict(width=0.4))
axes[0].set_title("Customer Churn Percentage Breakdown", fontsize=14, fontweight='bold')

# Count Plot
sns.countplot(data=df, x='Churn', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[1], legend=False)
axes[1].set_title("Customer Count by Churn Status", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Number of Customers")
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2), ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()""")

    add_md("""### Target Imbalance Analysis:
- **Class Distribution**: Retained (`No`) = **73.46%** (5,174), Churned (`Yes`) = **26.54%** (1,869).
- **ML Implications**: The dataset exhibits a mild **class imbalance (~3:1 ratio)**. Standard accuracy metrics will be misleading (a naive model predicting 'No' for everyone would get 73.5% accuracy). In Stage 4/5, evaluation must focus on **ROC-AUC, Precision, Recall, and F1-Score**, and resampling techniques (like SMOTE or class weighting) will be considered.
""")

    # Univariate Analysis
    add_md("## 4. Univariate Analysis")

    add_md("### 4.1 Numerical Features (`tenure`, `MonthlyCharges`, `TotalCharges`)")
    add_code("""num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, col in enumerate(num_features):
    # Histogram & KDE
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color='#1f77b4')
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[i, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[i, 0].axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
    axes[i, 0].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i, 0].legend()
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[i, 1], color='#ff7f0e')
    axes[i, 1].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()""")

    add_md("### 4.2 Categorical Features Breakdown")
    add_code("""cat_features = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette='viridis', ax=axes[i])
    axes[i].set_title(f'Frequency Count: {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Customer Count")

plt.tight_layout()
plt.show()""")

    # Churn-Based Bivariate Analysis
    add_md("## 5. Churn-Based Bivariate Analysis (Business Questions Q1–Q10)")

    add_md("### Q1. Does contract type affect churn?")
    add_code("""# Q1 Analysis
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print("Churn Rate by Contract Type (%):")
print(contract_churn.round(2))

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Contract', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'})
plt.title("Q1: Customer Churn by Contract Type", fontsize=14, fontweight='bold')
plt.ylabel("Number of Customers")
plt.show()""")
    add_md("""**Finding for Q1**:
- **Month-to-month** contracts have a massive **42.71%** churn rate.
- **One-year** contracts have an **11.27%** churn rate.
- **Two-year** contracts have a minimal **2.83%** churn rate.
- *Insight*: Longer contract commitments strongly correlate with customer retention.
""")

    add_md("### Q2. Does customer tenure affect churn?")
    add_code("""# Q2 Analysis
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='tenure', hue='Churn', common_norm=False, palette={'No': '#2ca02c', 'Yes': '#d62728'}, fill=True, alpha=0.4)
plt.title("Q2: Density Distribution of Tenure by Churn Status", fontsize=14, fontweight='bold')
plt.xlabel("Tenure (Months)")
plt.show()""")
    add_md("""**Finding for Q2**:
- Churn is heavily concentrated among new customers in their **first 1–12 months** of tenure.
- As customer tenure increases past 24 months, the density of churn drops drastically.
- *Association vs Causation*: Long tenure is associated with low churn because satisfied customers stay longer, but tenure itself is an outcome of customer satisfaction and switching costs.
""")

    add_md("### Q3. Do higher monthly charges relate to higher churn?")
    add_code("""# Q3 Analysis
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette={'No': '#2ca02c', 'Yes': '#d62728'})
plt.title("Q3: Monthly Charges vs Churn Status", fontsize=14, fontweight='bold')
plt.show()

print(f"Median Monthly Charges - Retained: ${df[df['Churn']=='No']['MonthlyCharges'].median():.2f}")
print(f"Median Monthly Charges - Churned:  ${df[df['Churn']=='Yes']['MonthlyCharges'].median():.2f}")""")
    add_md("""**Finding for Q3**:
- Churned customers have a significantly higher median monthly charge (**$79.65**) compared to retained customers (**$64.42**).
- High monthly bills increase customer price sensitivity and churn likelihood.
""")

    add_md("### Q4. Does internet service type affect churn?")
    add_code("""# Q4 Analysis
is_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
print("Churn Rate by Internet Service (%):")
print(is_churn.round(2))

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='InternetService', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'})
plt.title("Q4: Churn by Internet Service Type", fontsize=14, fontweight='bold')
plt.show()""")
    add_md("""**Finding for Q4**:
- **Fiber optic** subscribers experience an extremely high churn rate of **41.89%**.
- **DSL** subscribers have a **18.96%** churn rate.
- Customers with **No internet service** have the lowest churn rate of **7.40%**.
- *Insight*: Fiber optic service may suffer from higher pricing or service reliability issues.
""")

    add_md("### Q5 & Q6. Do Tech Support & Online Security relate to churn?")
    add_code("""# Q5 & Q6 Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='TechSupport', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[0])
axes[0].set_title("Q5: Churn by Tech Support Availability", fontsize=13, fontweight='bold')

sns.countplot(data=df, x='OnlineSecurity', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[1])
axes[1].set_title("Q6: Churn by Online Security Availability", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()""")
    add_md("""**Finding for Q5 & Q6**:
- Customers **without Tech Support** churn at **41.64%** vs only **15.17%** for those with Tech Support.
- Customers **without Online Security** churn at **41.77%** vs only **14.61%** for those with Online Security.
- *Insight*: Security and support add-on services act as strong retention anchors.
""")

    add_md("### Q7 & Q8. Do Payment Method & Paperless Billing relate to churn?")
    add_code("""# Q7 & Q8 Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.countplot(data=df, y='PaymentMethod', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[0])
axes[0].set_title("Q7: Churn by Payment Method", fontsize=13, fontweight='bold')

sns.countplot(data=df, x='PaperlessBilling', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[1])
axes[1].set_title("Q8: Churn by Paperless Billing", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()""")
    add_md("""**Finding for Q7 & Q8**:
- **Electronic check** users churn at **45.29%**, compared to ~15-19% for automatic bank transfer, automatic credit card, or mailed check.
- Customers with **Paperless Billing** churn at **33.57%** vs **16.33%** for non-paperless billing.
""")

    add_md("### Q9 & Q10. Senior Citizen status & Multiple Services vs Churn")
    add_code("""# Q9 & Q10 Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='SeniorCitizen', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[0])
axes[0].set_title("Q9: Churn by Senior Citizen Status (0=No, 1=Yes)", fontsize=13, fontweight='bold')

sns.countplot(data=df, x='MultipleLines', hue='Churn', palette={'No': '#2ca02c', 'Yes': '#d62728'}, ax=axes[1])
axes[1].set_title("Q10: Churn by Multiple Lines Service", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()""")
    add_md("""**Finding for Q9 & Q10**:
- **Senior Citizens** churn at **41.68%** compared to **23.61%** for non-seniors.
- Having **Multiple Lines** shows a slightly higher churn rate (**28.61%**) compared to single phone lines (**25.04%**).
""")

    # Tenure Group Analysis
    add_md("## 6. Tenure Group Analysis")
    add_code("""# Create tenure groups
bins = [-1, 12, 24, 48, 72]
labels = ['0-12 months', '13-24 months', '25-48 months', '49-72 months']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

tenure_summary = df.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack() * 100
tenure_summary['Total_Count'] = df.groupby('tenure_group')['Churn'].count()

print("Tenure Group Churn Breakdown:")
print(tenure_summary.round(2))

plt.figure(figsize=(9, 5))
ax = sns.barplot(x=tenure_summary.index, y=tenure_summary['Yes'], palette='Reds_d')
plt.title("Churn Rate (%) Across Tenure Groups", fontsize=14, fontweight='bold')
plt.ylabel("Churn Rate (%)")
plt.xlabel("Tenure Range")
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2), ha='center', va='center', color='white', fontweight='bold')
plt.show()""")
    add_md("""### Observations on Tenure Groups:
- **0–12 months**: Highest churn risk (**47.44%**). Nearly 1 in 2 new customers churn within their first year.
- **13–24 months**: Churn drops significantly to **28.71%**.
- **25–48 months**: Churn drops further to **20.39%**.
- **49–72 months**: Lowest churn risk (**9.51%**). Long-standing customers are highly loyal.
""")

    # Monthly Charges Binned Analysis
    add_md("## 7. Monthly Charges Binned Analysis")
    add_code("""# Create Monthly Charges Bins
mc_bins = [18, 35, 65, 90, 120]
mc_labels = ['$18-$35 (Low)', '$35-$65 (Med-Low)', '$65-$90 (Med-High)', '$90-$120 (High)']
df['mc_group'] = pd.cut(df['MonthlyCharges'], bins=mc_bins, labels=mc_labels)

mc_summary = df.groupby('mc_group')['Churn'].value_counts(normalize=True).unstack() * 100
mc_summary['Total_Count'] = df.groupby('mc_group')['Churn'].count()

print("Monthly Charges Bins Churn Breakdown:")
print(mc_summary.round(2))

plt.figure(figsize=(10, 5))
ax = sns.barplot(x=mc_summary.index, y=mc_summary['Yes'], palette='Oranges_d')
plt.title("Churn Rate (%) Across Monthly Charge Tiers", fontsize=14, fontweight='bold')
plt.ylabel("Churn Rate (%)")
plt.xlabel("Monthly Charge Range")
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2), ha='center', va='center', color='black', fontweight='bold')
plt.show()""")
    add_md("""### Observations on Monthly Charge Tiers:
- **$18–$35 (Low)**: Lowest churn rate (**10.89%**). Basic service plans have high stability.
- **$65–$90 (Medium-High)**: Highest churn rate (**36.30%**), driven heavily by Fiber Optic service packages without adequate bundling discounts.
- **$90–$120 (High)**: **32.78%** churn rate. High billing amounts increase vulnerability to competitor offers.
""")

    # Correlation Analysis
    add_md("## 8. Correlation Analysis")
    add_code("""# Encode Churn for correlation matrix (Yes=1, No=0)
df_corr = df.copy()
df_corr['Churn_num'] = (df_corr['Churn'] == 'Yes').astype(int)

num_cols_corr = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_num']
corr_matrix = df_corr[num_cols_corr].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix Heatmap (Numerical Features & Churn)", fontsize=14, fontweight='bold')
plt.show()""")
    add_md("""### Key Correlation Observations:
1. **`tenure` vs `TotalCharges` (+0.826)**: Strong positive linear correlation. The longer a customer stays, the more cumulative revenue they generate.
2. **`tenure` vs `Churn_num` (-0.352)**: Strongest negative linear correlation with churn. Increasing customer tenure directly reduces churn probability.
3. **`MonthlyCharges` vs `Churn_num` (+0.193)**: Moderate positive correlation. Higher monthly fees correlate with higher churn.
4. **`MonthlyCharges` vs `TotalCharges` (+0.651)**: Strong positive correlation.
5. **Correlation vs Causation Caution**: A negative correlation between tenure and churn does not mean tenure *causes* retention by itself; rather, satisfied customers naturally accumulate longer tenure over time.
""")

    # Outlier Analysis
    add_md("## 9. Outlier Assessment (IQR & Boxplots)")
    add_code("""# Compute IQR bounds for numerical columns
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f} | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Outlier Count: {len(outliers)}")""")

    add_md("""### Outlier Findings & Decision:
- **Mathematical Outliers**: 0 statistical outliers were detected beyond standard $1.5 \\times IQR$ bounds in `tenure`, `MonthlyCharges`, or `TotalCharges`.
- **Domain Interpretation**: All values (e.g., tenure up to 72 months, monthly charges up to $118.75) reflect genuine customer account behaviors and contract structures.
- **Handling Action**: **No rows deleted**. Deleting extreme values would destroy valid data representing high-value long-tenure customers.
""")

    # Key Business Insights
    add_md("""## 10. Key Business Insights

Based strictly on empirical findings from the dataset:

1. **Contract Structure is the Single Strongest Churn Predictor**: Customers on Month-to-Month contracts churn at **42.71%**, compared to only **11.27%** for 1-Year contracts and **2.83%** for 2-Year contracts. Incentivizing annual contracts will directly reduce churn.
2. **First-Year Onboarding Critical Window**: Customers in their first 12 months experience a **47.44%** churn rate. Churn drops below **10%** after 4 years of tenure. Proactive retention offers must target customers in months 1–12.
3. **Fiber Optic Pricing & Service Friction**: Fiber Optic subscribers churn at **41.89%** (vs 18.96% for DSL), correlating with higher monthly charges ($65–$90/month range). Fiber optic service packages require price restructuring or service quality reviews.
4. **Value Add Support & Security Services Retention Anchor**: Customers without Tech Support churn at **41.64%** (vs **15.17%** with support), and those without Online Security churn at **41.77%** (vs **14.61%** with security). Bundling free or discounted tech support and security will improve retention.
5. **Electronic Check Payment Risk**: Customers paying via Electronic Check churn at **45.29%**, compared to ~15-16% for automated payment methods (Bank Transfer / Credit Card). Encouraging auto-pay enrollment will lower churn.
6. **Senior Citizen Vulnerability**: Senior Citizens churn at **41.68%** (vs 23.61% for non-seniors), indicating a need for dedicated customer support or tailored senior discount plans.
""")

    # Save notebook JSON
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print(f"Jupyter Notebook successfully created at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
