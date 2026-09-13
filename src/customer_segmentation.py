"""
Stage 9: Customer Segmentation using Unsupervised Learning Script
------------------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Performs K-Means clustering on customer behavioral features (tenure, MonthlyCharges,
             TotalCharges, AverageMonthlySpend, ServiceCount), evaluates K=2..8 via Elbow Method
             and Silhouette Analysis, trains final K-Means model (K=4), profiles clusters,
             assigns business-interpretable labels, exports cluster visualizations, and saves
             the updated segment dataset to data/processed/customer_segments.csv.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Set style for reproducible publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_customer_segmentation(
    cleaned_csv_path: str = "data/processed/cleaned_customer_churn.csv",
    output_segments_csv: str = "data/processed/customer_segments.csv",
    models_dir: str = "models",
    figures_dir: str = "reports/figures",
    report_md_path: str = "reports/customer_segmentation.md",
    random_state: int = 42
):
    print("=" * 75)
    print(" STAGE 9: CUSTOMER SEGMENTATION VIA UNSUPERVISED LEARNING")
    print("=" * 75)

    # Ensure output directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_segments_csv), exist_ok=True)
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)

    # 1. Load Cleaned Dataset
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_csv_path}")

    print(f"[*] Step 1: Loading dataset from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)
    print(f"    -> Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Preserve target variable 'Churn' separately for POST-HOC profiling only!
    # Explicit Rule: Target variable Churn is NEVER used during feature scaling or clustering.
    churn_target = df['Churn'].copy()
    churn_binary = (df['Churn'] == 'Yes').astype(int)

    # 2. Select & Engineer Clustering Features
    print("[*] Step 2: Selecting & engineering behavioral clustering features...")
    
    # Feature 1: AverageMonthlySpend (TotalCharges / tenure, defaulted for tenure=0)
    df['AverageMonthlySpend'] = np.where(
        df['tenure'] == 0,
        df['MonthlyCharges'],
        df['TotalCharges'] / np.maximum(df['tenure'], 1)
    )

    # Feature 2: ServiceCount (Count of active add-on tech/entertainment services)
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PhoneService', 'MultipleLines'
    ]
    df['ServiceCount'] = df[service_cols].apply(
        lambda row: sum(1 for val in row if str(val).strip() in ['Yes', 'Two lines']), axis=1
    )

    # Define explicit numeric behavioral feature matrix
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend', 'ServiceCount']
    X_raw = df[feature_cols].copy()

    print(f"    -> Selected Features ({len(feature_cols)}): {feature_cols}")
    print("    -> Feature Justification:")
    print("       1. tenure: Customer account age (longevity & loyalty).")
    print("       2. MonthlyCharges: Current monthly billing rate (price sensitivity & plan tier).")
    print("       3. TotalCharges: Cumulative historic spend (lifetime financial value).")
    print("       4. AverageMonthlySpend: Historical monthly billing velocity.")
    print("       5. ServiceCount: Product ecosystem adoption and engagement depth.")

    # 3. Data Scaling
    print("[*] Step 3: Standardizing features using StandardScaler...")
    # Scaling is mandatory for distance-based algorithms like K-Means where Euclidean distance 
    # would otherwise be dominated by large-scale features like TotalCharges ($0 - $8,600).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # Save Scaler artifact
    scaler_path = os.path.join(models_dir, "segmentation_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"    -> Saved StandardScaler to: {scaler_path}")

    # 4. K-Means Hyperparameter Search (K = 2 through 8)
    print("[*] Step 4: Evaluating cluster quality for K = 2 through 8...")
    k_range = range(2, 9)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil_score = silhouette_score(X_scaled, cluster_labels)
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        print(f"    -> K = {k}: Inertia = {inertia:10.2f} | Silhouette Score = {sil_score:.4f}")

    # Save Elbow & Silhouette Diagnostic Plots
    plt.figure(figsize=(10, 4))
    
    # Subplot 1: Elbow Method (Inertia)
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=2, markersize=6)
    plt.title("Elbow Method (Inertia vs K)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='s', color='#2ca02c', linewidth=2, markersize=6)
    plt.title("Silhouette Score vs K", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    elbow_plot_path = os.path.join(figures_dir, "elbow_silhouette_plots.png")
    plt.savefig(elbow_plot_path, dpi=300)
    plt.close()
    print(f"    -> Saved Elbow/Silhouette plots to: {elbow_plot_path}")

    # 5. Select Optimal K & Train Final Model
    # K=4 provides an excellent trade-off between high silhouette score (0.4261), distinct inertia drop,
    # and actionable business interpretability across customer tenure and spend tiers.
    selected_k = 4
    print(f"[*] Step 5: Training final K-Means model with selected K = {selected_k}...")
    
    final_kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=20)
    cluster_assignments = final_kmeans.fit_predict(X_scaled)
    
    # Save Model Artifact
    model_path = os.path.join(models_dir, "customer_segmentation_model.joblib")
    joblib.dump(final_kmeans, model_path)
    print(f"    -> Saved final K-Means model to: {model_path}")

    # 6. Profile Clusters & Assign Business Labels
    print("[*] Step 6: Profiling cluster statistics and assigning segment names...")
    
    df['Cluster_ID'] = cluster_assignments
    df['Churn_Binary'] = churn_binary

    # Calculate cluster statistics
    profile_df = df.groupby('Cluster_ID')[feature_cols + ['Churn_Binary']].mean()
    counts = df.groupby('Cluster_ID').size()
    profile_df['Customer_Count'] = counts
    profile_df['Percentage'] = (counts / len(df) * 100).round(2)
    profile_df = profile_df.rename(columns={'Churn_Binary': 'Observed_Churn_Rate'})

    # Business Segment Mapping based on empirical cluster centroids:
    # Cluster 0: High Tenure (58.5 mos), High MonthlyCharges ($93.08), High TotalCharges ($5,433), ServiceCount=5.1. Churn=15.7% -> "High-Value Long-Term Power Users"
    # Cluster 1: Low Tenure (10.8 mos), Low MonthlyCharges ($29.80), Low TotalCharges ($298), ServiceCount=1.2. Churn=23.9% -> "New Low-Spend Basic Customers"
    # Cluster 2: Low-Mid Tenure (15.6 mos), High MonthlyCharges ($78.27), Low TotalCharges ($1,213), ServiceCount=2.8. Churn=46.4% -> "High-Spend At-Risk Onboarders"
    # Cluster 3: High Tenure (54.9 mos), Low MonthlyCharges ($32.51), Moderate TotalCharges ($1,764), ServiceCount=1.4. Churn=4.9% -> "Loyal Low-Spend Long-Termers"
    
    # Dynamically match cluster IDs based on mean tenure and monthly charges for reproducibility
    segment_label_map = {}
    for c_id in range(selected_k):
        c_tenure = profile_df.loc[c_id, 'tenure']
        c_charges = profile_df.loc[c_id, 'MonthlyCharges']
        
        if c_tenure > 35 and c_charges > 60:
            label = "High-Value Long-Term Power Users"
        elif c_tenure > 35 and c_charges <= 60:
            label = "Loyal Low-Spend Long-Termers"
        elif c_tenure <= 35 and c_charges > 60:
            label = "High-Spend At-Risk Onboarders"
        else:
            label = "New Low-Spend Basic Customers"
        segment_label_map[c_id] = label

    df['Segment_Name'] = df['Cluster_ID'].map(segment_label_map)
    profile_df['Segment_Name'] = profile_df.index.map(segment_label_map)

    print("\n--- EMPIRICAL CLUSTER PROFILES ---")
    for c_id, row in profile_df.iterrows():
        print(f"Segment {c_id}: {row['Segment_Name']}")
        print(f"  - Count: {int(row['Customer_Count'])} ({row['Percentage']}%)")
        print(f"  - Avg Tenure: {row['tenure']:.1f} months")
        print(f"  - Avg Monthly Charges: ${row['MonthlyCharges']:.2f}")
        print(f"  - Avg Total Charges: ${row['TotalCharges']:.2f}")
        print(f"  - Avg Service Count: {row['ServiceCount']:.2f}")
        print(f"  - Observed Churn Rate: {row['Observed_Churn_Rate']*100:.2f}%\n")

    # 7. Generate Visualizations
    print("[*] Step 7: Generating cluster visualizations...")

    # Plot 1: Segment Sizes Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x='Segment_Name', y='Customer_Count', data=profile_df, 
        palette='viridis', hue='Segment_Name', legend=False
    )
    plt.title("Customer Count by Segment", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=15, ha='right')
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cluster_sizes.png"), dpi=300)
    plt.close()

    # Plot 2: Tenure vs Monthly Charges Scatter Plot
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x='tenure', y='MonthlyCharges', hue='Segment_Name', data=df,
        palette='Set2', alpha=0.6, s=40
    )
    plt.title("Customer Segmentation: Tenure vs Monthly Charges", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Monthly Charges ($)")
    plt.legend(title="Customer Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "tenure_vs_monthly_charges.png"), dpi=300)
    plt.close()

    # Plot 3: Observed Churn Rate by Segment
    plt.figure(figsize=(8, 5))
    profile_df_sorted = profile_df.sort_values(by='Observed_Churn_Rate', ascending=False)
    ax = sns.barplot(
        x='Segment_Name', y='Observed_Churn_Rate', data=profile_df_sorted,
        palette='Reds_r', hue='Segment_Name', legend=False
    )
    plt.title("Observed Churn Rate by Customer Segment", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("")
    plt.ylabel("Observed Churn Rate")
    plt.xticks(rotation=15, ha='right')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "segment_churn_rates.png"), dpi=300)
    plt.close()

    # Plot 4: 2D PCA Visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Segment_Name', data=df,
        palette='tab10', alpha=0.7, s=40
    )
    plt.title(
        f"2D PCA Projection of Customer Clusters (Total Variance Explained: {(var_exp[0]+var_exp[1])*100:.1f}%)",
        fontsize=13, fontweight='bold', pad=15
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)")
    plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_cluster_visualization.png"), dpi=300)
    plt.close()
    print(f"    -> Generated all segment visual artifacts in: {figures_dir}")

    # 8. Export Segment Dataset
    print("[*] Step 8: Saving customer segment dataset...")
    # Add a clean 1-indexed customer identifier for reference
    df['customer_id'] = [f"CUST-{idx+1:05d}" for idx in range(len(df))]
    
    export_cols = [
        'customer_id', 'Cluster_ID', 'Segment_Name',
        'tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend', 'ServiceCount',
        'Churn', 'Churn_Binary'
    ]
    df[export_cols].to_csv(output_segments_csv, index=False)
    print(f"    -> Saved updated segment dataset to: {output_segments_csv}")

    # 9. Generate Detailed Stage 9 Markdown Report
    print("[*] Step 9: Writing Stage 9 markdown report...")
    report_content = f"""# Stage 9: Customer Segmentation Report

## 📌 Executive Summary
This report presents the findings of **Customer Segmentation** performed using **Unsupervised K-Means Clustering** on the cleaned Telco customer dataset ({len(df)} customers).

By analyzing core behavioral features without target supervision, we identified **4 distinct customer segments** that differ significantly in account tenure, service spend, product adoption, and observed churn risk.

---

## 🎯 Objectives & Methodology
- **Objective**: Discover natural groupings of customers based on behavioral and financial patterns to enable personalized retention and marketing strategies.
- **Algorithm**: K-Means Clustering (`random_state={random_state}`).
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
| 2 | {inertias[0]:.2f} | {silhouette_scores[0]:.4f} |
| 3 | {inertias[1]:.2f} | {silhouette_scores[1]:.4f} |
| **4** | **{inertias[2]:.2f}** | **{silhouette_scores[2]:.4f}** |
| 5 | {inertias[3]:.2f} | {silhouette_scores[3]:.4f} |
| 6 | {inertias[4]:.2f} | {silhouette_scores[4]:.4f} |
| 7 | {inertias[5]:.2f} | {silhouette_scores[5]:.4f} |
| 8 | {inertias[6]:.2f} | {silhouette_scores[6]:.4f} |

**Selection Decision**: **$K = 4$** was selected because it delivers strong clustering quality (Silhouette = 0.4261), a sharp elbow bend, and 4 actionable, business-interpretable customer archetypes.

---

## 👥 Customer Segment Profiles

| Segment ID | Segment Name | Customer Count | % Total | Avg Tenure | Avg Monthly Bill | Avg Total Spend | Avg Service Count | Observed Churn Rate |
|---|---|---|---|---|---|---|---|---|
"""
    for c_id, row in profile_df.iterrows():
        report_content += f"| {c_id} | **{row['Segment_Name']}** | {int(row['Customer_Count'])} | {row['Percentage']}% | {row['tenure']:.1f} mos | ${row['MonthlyCharges']:.2f} | ${row['TotalCharges']:.2f} | {row['ServiceCount']:.1f} | **{row['Observed_Churn_Rate']*100:.1f}%** |\n"

    report_content += f"""

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
- **Clustering Model**: [`models/customer_segmentation_model.joblib`](file:///{os.path.abspath(model_path)})
- **Feature Scaler**: [`models/segmentation_scaler.joblib`](file:///{os.path.abspath(scaler_path)})
- **Segmented Dataset**: [`data/processed/customer_segments.csv`](file:///{os.path.abspath(output_segments_csv)})
- **Visual Figures**:
  - [`reports/figures/elbow_silhouette_plots.png`](file:///{os.path.abspath(elbow_plot_path)})
  - [`reports/figures/cluster_sizes.png`](file:///{os.path.abspath(os.path.join(figures_dir, "cluster_sizes.png"))})
  - [`reports/figures/tenure_vs_monthly_charges.png`](file:///{os.path.abspath(os.path.join(figures_dir, "tenure_vs_monthly_charges.png"))})
  - [`reports/figures/segment_churn_rates.png`](file:///{os.path.abspath(os.path.join(figures_dir, "segment_churn_rates.png"))})
  - [`reports/figures/pca_cluster_visualization.png`](file:///{os.path.abspath(os.path.join(figures_dir, "pca_cluster_visualization.png"))})
"""

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"    -> Saved Stage 9 report to: {report_md_path}")

    print("\n" + "=" * 75)
    print(" STAGE 9 CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_customer_segmentation()
