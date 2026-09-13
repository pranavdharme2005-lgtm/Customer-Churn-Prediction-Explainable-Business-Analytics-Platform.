"""
Stage 1: Basic Data Loading and Initial Overview Script
------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Loads the raw IBM Telco Customer Churn dataset, inspects its structure, 
             data types, missing values, and duplicate rows.
"""

import os
import pandas as pd
import numpy as np

def load_and_inspect_data(file_path: str):
    """
    Loads customer churn dataset and prints key diagnostic info.
    
    Parameters:
        file_path (str): Path to the CSV dataset file.
    """
    print("=" * 70)
    print(" STAGE 1: CUSTOMER CHURN DATASET INITIAL INSPECTION")
    print("=" * 70)
    
    # 1. Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
        
    print(f"[*] Loading dataset from: {file_path}\n")
    
    # 2. Load dataset using Pandas
    df = pd.read_csv(file_path)
    
    # 3. First 5 Rows
    print("--- 1. FIRST 5 ROWS ---")
    print(df.head())
    print("\n" + "-" * 70)
    
    # 4. Number of Rows and Columns
    rows, cols = df.shape
    print("--- 2. DATASET DIMENSIONS ---")
    print(f"Total Rows (Customers): {rows}")
    print(f"Total Columns (Features): {cols}")
    print("\n" + "-" * 70)
    
    # 5. Column Names
    print("--- 3. COLUMN NAMES ---")
    for i, col in enumerate(df.columns, 1):
        print(f" {i:2d}. {col}")
    print("\n" + "-" * 70)
    
    # 6. Data Types
    print("--- 4. DATA TYPES ---")
    print(df.dtypes)
    print("\n" + "-" * 70)
    
    # 7. Missing Value Counts
    print("--- 5. MISSING VALUE COUNTS ---")
    standard_nulls = df.isnull().sum()
    print("Standard Null / NaN Counts:")
    print(standard_nulls)
    
    # Check for hidden missing values (e.g. whitespace strings in numerical columns like TotalCharges)
    print("\nHidden Missing Values Check (Empty or whitespace strings):")
    hidden_missing = {}
    for col in df.columns:
        # Check count of string values that are purely whitespace
        blank_count = (df[col].astype(str).str.strip() == '').sum()
        if blank_count > 0:
            hidden_missing[col] = blank_count
            print(f"  -> Column '{col}' contains {blank_count} whitespace/empty string entry(ies).")
    
    if not hidden_missing:
        print("  -> No whitespace missing values found.")
        
    print("\n" + "-" * 70)
    
    # 8. Duplicate Row Count
    duplicate_count = df.duplicated().sum()
    print("--- 6. DUPLICATE ROWS COUNT ---")
    print(f"Total Duplicate Rows: {duplicate_count}")
    print("\n" + "=" * 70)
    print(" Stage 1 Data Loading and Overview Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    # Path relative to project root or current working script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_path = os.path.join(project_root, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    load_and_inspect_data(dataset_path)
