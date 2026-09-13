# SQL Business Analysis Report

## 📌 Executive Overview
This report presents the database queries and business analysis executed on the **SQLite database (`data/customer_churn.db`)** within the `customers` table during **Stage 4** of the project.

All SQL query results have been cross-validated against the Stage 3 Python/EDA results to ensure **100% data consistency**.

---

## 📊 1. Basic Dataset Metrics (Q1 – Q6)

| Metric | SQL Approach | Result | Business Interpretation |
|---|---|---|---|
| **Total Customers (Q1)** | `SELECT COUNT(*)` | **7,043** | Complete customer population in dataset. |
| **Churned Customers (Q2)** | `WHERE Churn = 'Yes'` | **1,869** | Customers who have canceled their service. |
| **Retained Customers (Q3)** | `WHERE Churn = 'No'` | **5,174** | Active, retained customer accounts. |
| **Overall Churn Rate (Q4)** | `ROUND(100.0 * SUM(CASE WHEN Churn='Yes'...)/COUNT(*), 2)` | **26.54%** | Over 1 in 4 customers churn overall. |
| **Avg Monthly Charge (Q5)** | `SELECT ROUND(AVG(MonthlyCharges), 2)` | **$64.76** | Benchmark monthly revenue per account. |
| **Avg Customer Tenure (Q6)** | `SELECT ROUND(AVG(tenure), 2)` | **32.37 months** | Average customer lifetime duration. |

---

## 📜 2. Contract Analysis (Q7 – Q9)

### SQL Query:
```sql
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
```

### Results Table:
| Contract | total_customers | churned_customers | retained_customers | churn_rate_pct |
| --- | --- | --- | --- | --- |
| Month-to-month | 3875 | 1655 | 2220 | 42.71 |
| One year | 1473 | 166 | 1307 | 11.27 |
| Two year | 1695 | 48 | 1647 | 2.83 |

### Business Interpretation:
- **Month-to-month contracts present the highest risk**: **42.71%** churn rate (1,655 churned customers out of 3,875).
- **Long-term contracts act as strong retention anchors**: 1-Year contracts drop churn to **11.27%**, and 2-Year contracts drop churn to **2.83%**.

---

## ⏱️ 3. Customer Tenure Group Analysis

### SQL Query:
```sql
SELECT 
    CASE 
        WHEN tenure BETWEEN 0 AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure) ASC;
```

### Results Table:
| tenure_group | total_customers | churned_customers | retained_customers | churn_rate_pct |
| --- | --- | --- | --- | --- |
| 0-12 months | 2186 | 1037 | 1149 | 47.44 |
| 13-24 months | 1024 | 294 | 730 | 28.71 |
| 25-48 months | 1594 | 325 | 1269 | 20.39 |
| 49+ months | 2239 | 213 | 2026 | 9.51 |

### Business Interpretation:
- **First-Year Vulnerability**: **47.44%** of new customers churn in their first 12 months.
- **Loyalty Maturity**: Once tenure exceeds 48 months (4 years), churn drops to **9.51%**. Proactive retention efforts must target months 1–12.

---

## 💳 4. Monthly Charges & Payment Method Analysis

### Monthly Charge Tiers:
| monthly_charges_tier | total_customers | churned_customers | retained_customers | churn_rate_pct |
| --- | --- | --- | --- | --- |
| $18-$35 (Low) | 1735 | 189 | 1546 | 10.89 |
| $35-$65 (Medium-Low) | 1409 | 326 | 1083 | 23.14 |
| $65-$90 (Medium-High) | 2160 | 784 | 1376 | 36.3 |
| $90-$120 (High) | 1739 | 570 | 1169 | 32.78 |

### Payment Method Analysis:
| PaymentMethod | total_customers | churned_customers | churn_rate_pct |
| --- | --- | --- | --- |
| Electronic check | 2365 | 1071 | 45.29 |
| Mailed check | 1612 | 308 | 19.11 |
| Bank transfer (automatic) | 1544 | 258 | 16.71 |
| Credit card (automatic) | 1522 | 232 | 15.24 |

### Business Interpretation:
- Customers paying between **$65 and $90/month** exhibit the highest churn rate (**36.30%**).
- **Electronic Check users** have an alarming **45.29%** churn rate compared to ~15-16% for automatic bank transfer or credit card payments.

---

## 🚨 5. High-Risk Customer Profile Combinations

### SQL Query (Using CTE and HAVING clause):
```sql
WITH FeatureCombinations AS (
    SELECT 
        Contract, InternetService, TechSupport, PaymentMethod,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
        ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY Contract, InternetService, TechSupport, PaymentMethod
    HAVING COUNT(*) >= 50
)
SELECT * FROM FeatureCombinations ORDER BY churn_rate_pct DESC;
```

### Top 5 High-Risk Profiles:
| Contract | InternetService | TechSupport | PaymentMethod | total_customers | churned_customers | churn_rate_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Month-to-month | Fiber optic | No | Electronic check | 1138 | 716 | 62.92 |
| Month-to-month | Fiber optic | No | Mailed check | 164 | 87 | 53.05 |
| Month-to-month | Fiber optic | No | Bank transfer (automatic) | 260 | 127 | 48.85 |
| Month-to-month | DSL | No | Electronic check | 365 | 162 | 44.38 |
| Month-to-month | Fiber optic | No | Credit card (automatic) | 234 | 103 | 44.02 |

### Business Interpretation:
- Customers on **Month-to-month contracts + Fiber Optic internet + No Tech Support + Electronic Check payment** churn at **62.92%** (716 out of 1,138 customers). This represents the single highest-risk segment in the company.

---

## ✅ 6. Data Validation against Stage 3 Python Results

| Feature / Metric | Stage 3 Python/EDA Result | Stage 4 SQL Result | Status |
|---|---|---|---|
| Total Customers | 7,043 | 7,043 | **Matched 100%** |
| Churned Customers | 1,869 (26.54%) | 1,869 (26.54%) | **Matched 100%** |
| Month-to-Month Churn Rate | 42.71% | 42.71% | **Matched 100%** |
| 0-12m Tenure Churn Rate | 47.44% | 47.44% | **Matched 100%** |
| Fiber Optic Churn Rate | 41.89% | 41.89% | **Matched 100%** |
| Electronic Check Churn Rate | 45.29% | 45.29% | **Matched 100%** |

---

## 📁 7. Exported CSV Summary Files
The following exported query results are stored in [`reports/sql_results/`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/sql_results/):
- `contract_churn_analysis.csv`
- `tenure_churn_analysis.csv`
- `service_churn_analysis.csv`
- `payment_method_analysis.csv`
- `high_risk_combinations.csv`
