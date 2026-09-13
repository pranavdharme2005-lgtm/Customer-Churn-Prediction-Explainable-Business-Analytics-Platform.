# Data Cleaning & Quality Assessment Report

## 📌 Executive Summary
This report documents the data quality assessment and data cleaning operations performed on the raw IBM Telco Customer Churn dataset during **Stage 2** of the project.

---

## 📊 Dataset Dimensions Comparison
- **Original Raw Dataset Size**: 7043 rows × 21 columns
- **Final Cleaned Dataset Size**: 7043 rows × 20 columns
- **Removed Columns**: 1 (`customerID`)
- **Removed Rows**: 0 (No exact duplicate rows existed)

---

## 🔍 Data Quality Problems Identified & Solutions Applied

### 1. Hidden Missing Values in `TotalCharges`
- **Problem**: The `TotalCharges` column was parsed as an `object` (string) data type because it contained **11 whitespace string entries (`" "`)**.
- **Root Cause**: All 11 records belong to new customers with a `tenure` of `0` months who have not yet completed a billing cycle.
- **Solution**: Replaced blank strings with `0.0` (since zero months tenure equals zero total charges accumulated) and cast the column to `float64`.

### 2. Non-Predictive Primary Key Hash (`customerID`)
- **Problem**: The `customerID` column contains unique alphanumeric string hashes (e.g., `7590-VHVEG`).
- **Root Cause**: Unique primary keys add 7,043 unique categorical levels without any predictive or statistical relationship to customer churn.
- **Solution**: Dropped `customerID` from the analytical dataset to avoid overfitting and redundant memory usage.

### 3. Duplicate Rows
- **Assessment**: Evaluated all 7,043 rows for exact duplication.
- **Result**: `0` duplicate rows found.

### 4. Categorical Consistency & Range Validation
- **Assessment**: Checked categorical values across all text columns for casing anomalies, typos, or unexpected categories.
- **Result**: All categorical features are well-standardized (`gender`, `Contract`, `PaymentMethod`, `InternetService`, etc.).
- **Numerical Ranges**:
  - `tenure`: 0 to 72 months (Valid range)
  - `MonthlyCharges`: $18.25 to $118.75 (Valid positive range)
  - `TotalCharges`: $0.00 to $8,684.80 (Valid non-negative range)
  - `SeniorCitizen`: Binary (0 or 1)

---

## 🎯 Target Variable Distribution (`Churn`)
- **Target Feature**: `Churn`
- **No (Retained)**: 5174 customers (73.46%)
- **Yes (Churned)**: 1869 customers (26.54%)
- **Class Balance Status**: Mild class imbalance (~26.54% positive churn class).

---

## 📋 Final Schema Breakdown

### Numerical Columns (4):
`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`

### Categorical Columns (16):
`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `Churn`

---

## ✅ Final Data Quality Status
- **Missing Values**: 0
- **Duplicate Rows**: 0
- **Status**: Cleaned, fully structured, and ready for Stage 3 (Exploratory Data Analysis & Feature Engineering).
