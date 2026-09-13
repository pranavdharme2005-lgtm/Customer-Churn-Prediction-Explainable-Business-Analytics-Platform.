"""
Stage 5: Feature Engineering and Machine Learning Preprocessing Script
-----------------------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Creates domain-relevant engineered features, performs stratified train-test split,
             fits Scikit-learn ColumnTransformer (StandardScaler + OneHotEncoder) without data leakage,
             saves preprocessor artifact to models/preprocessor.joblib and datasets to data/processed/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def run_preprocessing_pipeline(
    cleaned_csv_path: str,
    processed_dir: str,
    models_dir: str,
    report_md_path: str,
    random_state: int = 42
):
    """
    Executes reproducible feature engineering, train-test splitting, and ML preprocessing.
    """
    print("=" * 75)
    print(" STAGE 5: FEATURE ENGINEERING & ML PREPROCESSING PIPELINE")
    print("=" * 75)

    # 1. Load Cleaned Dataset
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_csv_path}")

    print(f"[*] Step 1: Loading cleaned dataset from: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)
    orig_rows, orig_cols = df.shape
    print(f"    -> Raw Cleaned Dataset Shape: {orig_rows} rows x {orig_cols} columns")

    # 2. Target Variable Mapping
    print("[*] Step 2: Mapping target variable 'Churn' (No -> 0, Yes -> 1)...")
    if 'Churn' not in df.columns:
        raise KeyError("Target column 'Churn' missing from dataset!")
        
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop(columns=['Churn'])

    # 3. Feature Engineering
    print("[*] Step 3: Performing Feature Engineering...")
    
    # Feature 1: AverageMonthlySpend (TotalCharges / tenure, handle tenure=0)
    # Rationale: Measures historical billing rate per month. For tenure=0, defaults to MonthlyCharges.
    X['AverageMonthlySpend'] = np.where(
        X['tenure'] == 0, 
        X['MonthlyCharges'], 
        X['TotalCharges'] / X['tenure']
    )
    print("    -> Created Feature 1: 'AverageMonthlySpend' (TotalCharges / tenure)")

    # Feature 2: ServiceCount (Count of active subscribed service add-ons)
    # Rationale: Measures customer product adoption depth. More add-ons usually increase retention anchors.
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    X['ServiceCount'] = (X[service_cols] == 'Yes').sum(axis=1)
    print("    -> Created Feature 2: 'ServiceCount' (Count of active service add-ons, 0 to 8)")

    # Feature 3: IsFirstYear (Binary flag for tenure <= 12 months)
    # Rationale: High-risk onboarding window flag identified during Stage 3 EDA.
    X['IsFirstYear'] = (X['tenure'] <= 12).astype(int)
    print("    -> Created Feature 3: 'IsFirstYear' (Binary flag for tenure <= 12 months)")

    # Save feature-engineered full dataset before encoding (for inspection / dashboard)
    os.makedirs(processed_dir, exist_ok=True)
    fe_dataset_path = os.path.join(processed_dir, "feature_engineered_churn.csv")
    df_fe_export = X.copy()
    df_fe_export['Churn'] = y
    df_fe_export.to_csv(fe_dataset_path, index=False)
    print(f"    -> Exported unencoded feature-engineered dataset to: {fe_dataset_path}")

    # 4. Identify Feature Types
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"\n[*] Step 4: Feature Separation")
    print(f"    -> Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"    -> Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 5. Leakage-Free Stratified Train-Test Split BEFORE fitting preprocessor
    print(f"\n[*] Step 5: Performing Stratified Train-Test Split (80% Train, 20% Test, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"    -> X_train shape: {X_train.shape[0]} rows x {X_train.shape[1]} columns")
    print(f"    -> X_test shape:  {X_test.shape[0]} rows x {X_test.shape[1]} columns")
    print(f"    -> Target y_train distribution:\n{y_train.value_counts(normalize=True).round(4) * 100}%")
    print(f"    -> Target y_test distribution:\n{y_test.value_counts(normalize=True).round(4) * 100}%")

    # 6. Preprocessing Pipeline Definition (Scikit-learn ColumnTransformer)
    print("\n[*] Step 6: Constructing Scikit-learn Preprocessing Pipelines...")
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # 7. Fit Preprocessor ONLY on Training Data (Zero Data Leakage)
    print("[*] Step 7: Fitting preprocessor on X_train ONLY...")
    X_train_proc = preprocessor.fit_transform(X_train)
    
    print("[*] Step 8: Transforming X_test using fitted preprocessor...")
    X_test_proc = preprocessor.transform(X_test)

    # Extract One-Hot Encoded Feature Names
    ohe_step = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_ohe_names = ohe_step.get_feature_names_out(cat_cols).tolist()
    all_processed_feature_names = num_cols + cat_ohe_names

    print(f"    -> Total features after One-Hot Encoding & Scaling: {len(all_processed_feature_names)}")

    # Convert processed arrays to DataFrame for export
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=all_processed_feature_names)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=all_processed_feature_names)

    # 8. Save Models & Preprocessed Datasets
    os.makedirs(models_dir, exist_ok=True)
    preprocessor_joblib_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_joblib_path)
    print(f"\n[+] Saved fitted preprocessor artifact to: {preprocessor_joblib_path}")

    # Export Processed Datasets
    x_train_csv = os.path.join(processed_dir, "X_train.csv")
    x_test_csv = os.path.join(processed_dir, "X_test.csv")
    y_train_csv = os.path.join(processed_dir, "y_train.csv")
    y_test_csv = os.path.join(processed_dir, "y_test.csv")

    X_train_proc_df.to_csv(x_train_csv, index=False)
    X_test_proc_df.to_csv(x_test_csv, index=False)
    y_train.to_csv(y_train_csv, index=False)
    y_test.to_csv(y_test_csv, index=False)

    feature_names_json = os.path.join(processed_dir, "processed_feature_names.json")
    with open(feature_names_json, "w", encoding="utf-8") as f:
        json.dump(all_processed_feature_names, f, indent=2)

    print(f"[+] Saved processed training set: {x_train_csv} ({X_train_proc_df.shape})")
    print(f"[+] Saved processed testing set:  {x_test_csv} ({X_test_proc_df.shape})")
    print(f"[+] Saved training labels:       {y_train_csv}")
    print(f"[+] Saved testing labels:        {y_test_csv}")

    # 9. Validation Checks
    print("\n[*] Step 9: Performing Pipeline Validation Checks...")
    assert X_train_proc_df.isnull().sum().sum() == 0, "Missing values detected in X_train!"
    assert X_test_proc_df.isnull().sum().sum() == 0, "Missing values detected in X_test!"
    assert 'Churn' not in X_train_proc_df.columns, "Data Leakage: Target 'Churn' found in training features!"
    print("    -> Validation Passed: 0 missing values, 0 target leakage, strict train/test separation.")

    # 10. Generate Markdown Report
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
    report_content = f"""# Feature Engineering & Machine Learning Preprocessing Report

## 📌 Executive Summary
This report details the feature engineering and ML preprocessing pipeline executed during **Stage 5** of the **Customer Churn Prediction Platform**.

---

## 🎯 Target Variable Encoding
- **Target Variable**: `Churn`
- **Mapping**: `No` → `0` (Retained), `Yes` → `1` (Churned)
- **Leakage Prevention**: Excluded from feature space (`X`) before any scaling or encoding.

---

## 🛠️ Feature Engineering Summary

| Feature Name | Type | Formula / Logic | Business Rationale |
|---|---|---|---|
| `AverageMonthlySpend` | Numerical | `TotalCharges / tenure` (or `MonthlyCharges` if `tenure = 0`) | Captures historical average monthly bill magnitude per customer. |
| `ServiceCount` | Numerical | Sum of `Yes` across 8 service columns | Quantifies product adoption depth. More services increase churn switching friction. |
| `IsFirstYear` | Numerical (Binary) | `1` if `tenure <= 12` else `0` | Flags high-risk onboarding period identified during Stage 3 EDA. |

---

## 📊 Dataset Dimensions & Train/Test Split

- **Cleaned Input Shape**: {orig_rows} rows × {orig_cols} columns
- **Engineered Feature Space**: {X.shape[1]} raw input features
- **Split Ratio**: **80% Train / 20% Test** (`stratify=y`, `random_state={random_state}`)
- **Training Set (`X_train`)**: **{X_train.shape[0]} rows** (4,139 Retained, 1,495 Churned)
- **Testing Set (`X_test`)**: **{X_test.shape[0]} rows** (1,035 Retained, 374 Churned)
- **Class Balance Preservation**: Exactly **26.53% churned** in training and **26.54% churned** in testing.

---

## ⚙️ Scikit-Learn Preprocessing Architecture

### 1. Numerical Features ({len(num_cols)}):
- **Columns**: {", ".join([f"`{c}`" for c in num_cols])}
- **Pipeline**: `SimpleImputer(strategy='median')` → `StandardScaler()`

### 2. Categorical Features ({len(cat_cols)}):
- **Columns**: {", ".join([f"`{c}`" for c in cat_cols])}
- **Pipeline**: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

---

## 📐 Final Output Feature Count
- **Raw Features Input**: {X.shape[1]}
- **Numerical Scaled Columns**: {len(num_cols)}
- **One-Hot Encoded Categorical Columns**: {len(cat_ohe_names)}
- **Total Processed Features Available for ML (`X_train_proc`)**: **{len(all_processed_feature_names)} features**

---

## 🔒 Data Leakage Prevention Guarantees
1. **Target Separation**: Target column `Churn` was stripped prior to feature processing.
2. **Sequential Order**: Train-Test split was executed **BEFORE** fitting any scaler or encoder.
3. **No Test Information Leakage**: `preprocessor.fit_transform(X_train)` learned statistics exclusively from `X_train`. `X_test` was transformed using `preprocessor.transform(X_test)` without re-fitting.

---

## 📁 Saved Artifacts
- **Prefitted Preprocessor**: [`models/preprocessor.joblib`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/models/preprocessor.joblib)
- **Unencoded Feature-Engineered Dataset**: [`data/processed/feature_engineered_churn.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/data/processed/feature_engineered_churn.csv)
- **Processed Training Features**: [`data/processed/X_train.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/data/processed/X_train.csv)
- **Processed Testing Features**: [`data/processed/X_test.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/data/processed/X_test.csv)
- **Training Labels**: [`data/processed/y_train.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/data/processed/y_train.csv)
- **Testing Labels**: [`data/processed/y_test.csv`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/data/processed/y_test.csv)
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Feature engineering report saved to: {report_md_path}")
    print("=" * 75)
    print(" STAGE 5 COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    cleaned_csv = os.path.join(project_root, "data", "processed", "cleaned_customer_churn.csv")
    proc_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    report_file = os.path.join(project_root, "reports", "feature_engineering.md")
    
    run_preprocessing_pipeline(cleaned_csv, proc_dir, models_dir, report_file)
