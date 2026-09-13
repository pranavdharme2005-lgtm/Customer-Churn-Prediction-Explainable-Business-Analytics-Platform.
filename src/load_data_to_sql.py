"""
Stage 4: Load Cleaned Dataset into SQLite Database
--------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Reads data/processed/cleaned_customer_churn.csv and populates the SQLite 
             database at data/customer_churn.db inside table 'customers'.
"""

import os
import sqlite3
import pandas as pd

def load_csv_to_sqlite(csv_path: str, db_path: str, table_name: str = "customers"):
    """
    Loads cleaned customer churn CSV into a SQLite table safely.

    Parameters:
        csv_path (str): Path to cleaned CSV file.
        db_path (str): Path to SQLite database file.
        table_name (str): Table name in SQLite database.
    """
    print("=" * 75)
    print(" STAGE 4: LOADING DATA INTO SQLITE DATABASE")
    print("=" * 75)

    # 1. Verify CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned CSV not found at: {csv_path}")

    print(f"[*] Step 1: Reading cleaned CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    csv_rows, csv_cols = df.shape
    print(f"    -> CSV Shape: {csv_rows} rows x {csv_cols} columns")

    # 2. Connect to SQLite database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    print(f"[*] Step 2: Connecting to SQLite database at: {db_path}")
    conn = sqlite3.connect(db_path)

    # 3. Load DataFrame into SQLite table
    # Using if_exists='replace' ensures reruns safely overwrite/recreate the table without duplicates
    print(f"[*] Step 3: Writing data to table '{table_name}'...")
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # 4. Verify Row Count in SQL
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    db_row_count = cursor.fetchone()[0]
    print(f"[*] Step 4: Verification")
    print(f"    -> Table '{table_name}' row count: {db_row_count}")

    # 5. Display Table Structure (Column Names & Types)
    print("\n[*] Table Schema / Structure:")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    for col in columns_info:
        cid, name, dtype, notnull, dflt_value, pk = col
        print(f"    - Column #{cid+1:2d}: {name:<20} | Type: {dtype}")

    conn.close()
    
    assert db_row_count == csv_rows, f"Mismatch! CSV rows ({csv_rows}) != DB rows ({db_row_count})"
    print("\n[+] Success: Cleaned dataset loaded into SQLite table 'customers' cleanly!")
    print("=" * 75)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    csv_file = os.path.join(project_root, "data", "processed", "cleaned_customer_churn.csv")
    db_file = os.path.join(project_root, "data", "customer_churn.db")
    
    load_csv_to_sqlite(csv_file, db_file)
