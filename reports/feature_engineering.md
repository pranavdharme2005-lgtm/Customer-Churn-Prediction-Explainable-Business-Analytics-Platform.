# Feature Engineering & Machine Learning Preprocessing Report

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

- **Cleaned Input Shape**: 7043 rows × 20 columns
- **Engineered Feature Space**: 22 raw input features
- **Split Ratio**: **80% Train / 20% Test** (`stratify=y`, `random_state=42`)
- **Training Set (`X_train`)**: **5634 rows** (4,139 Retained, 1,495 Churned)
- **Testing Set (`X_test`)**: **1409 rows** (1,035 Retained, 374 Churned)
- **Class Balance Preservation**: Exactly **26.53% churned** in training and **26.54% churned** in testing.

---

## ⚙️ Scikit-Learn Preprocessing Architecture

### 1. Numerical Features (7):
- **Columns**: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `AverageMonthlySpend`, `ServiceCount`, `IsFirstYear`
- **Pipeline**: `SimpleImputer(strategy='median')` → `StandardScaler()`

### 2. Categorical Features (15):
- **Columns**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Pipeline**: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

---

## 📐 Final Output Feature Count
- **Raw Features Input**: 22
- **Numerical Scaled Columns**: 7
- **One-Hot Encoded Categorical Columns**: 41
- **Total Processed Features Available for ML (`X_train_proc`)**: **48 features**

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
