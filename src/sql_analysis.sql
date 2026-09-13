-- ============================================================================
-- STAGE 4: SQL BUSINESS ANALYSIS QUERIES
-- Project: Customer Churn Prediction & Explainable Business Analytics Platform
-- Database: customer_churn.db
-- Table: customers
-- ============================================================================

-- ----------------------------------------------------------------------------
-- SECTION 1: BASIC SQL ANALYSIS (Q1 - Q6)
-- ----------------------------------------------------------------------------

-- Q1. How many total customers are there?
SELECT COUNT(*) AS total_customers
FROM customers;

-- Q2. How many customers churned?
SELECT COUNT(*) AS churned_customers
FROM customers
WHERE Churn = 'Yes';

-- Q3. How many customers did not churn?
SELECT COUNT(*) AS retained_customers
FROM customers
WHERE Churn = 'No';

-- Q4. What is the overall churn rate?
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- Q5. What is the average monthly charge?
SELECT ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers;

-- Q6. What is the average customer tenure?
SELECT ROUND(AVG(tenure), 2) AS avg_tenure_months
FROM customers;


-- ----------------------------------------------------------------------------
-- SECTION 2: CONTRACT ANALYSIS (Q7 - Q9)
-- ----------------------------------------------------------------------------

-- Q7, Q8, Q9. Contract distribution, churn count, and churn rate sorted descending
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- SECTION 3: CUSTOMER TENURE GROUP ANALYSIS
-- ----------------------------------------------------------------------------

-- Tenure Binned Analysis using CASE statement
SELECT 
    CASE 
        WHEN tenure BETWEEN 0 AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure) ASC;


-- ----------------------------------------------------------------------------
-- SECTION 4: MONTHLY CHARGES BINNED ANALYSIS
-- ----------------------------------------------------------------------------

-- Monthly Charges Binned Analysis using CASE statement
SELECT 
    CASE 
        WHEN MonthlyCharges BETWEEN 18 AND 35 THEN '$18-$35 (Low)'
        WHEN MonthlyCharges > 35 AND MonthlyCharges <= 65 THEN '$35-$65 (Medium-Low)'
        WHEN MonthlyCharges > 65 AND MonthlyCharges <= 90 THEN '$65-$90 (Medium-High)'
        ELSE '$90-$120 (High)'
    END AS monthly_charges_tier,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY monthly_charges_tier
ORDER BY MIN(MonthlyCharges) ASC;


-- ----------------------------------------------------------------------------
-- SECTION 5: SERVICE ANALYSIS
-- ----------------------------------------------------------------------------

-- 5.1 Churn by Internet Service Type
SELECT 
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 5.2 Churn by Tech Support Availability
SELECT 
    TechSupport,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- 5.3 Churn by Online Security Availability
SELECT 
    OnlineSecurity,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY OnlineSecurity
ORDER BY churn_rate_pct DESC;

-- 5.4 Churn by Payment Method
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 5.5 Churn by Paperless Billing
SELECT 
    PaperlessBilling,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaperlessBilling
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- SECTION 6: HIGH-RISK CUSTOMER COMBINATION ANALYSIS
-- ----------------------------------------------------------------------------

-- Multi-feature combination analysis using CTE and HAVING clause
WITH FeatureCombinations AS (
    SELECT 
        Contract,
        InternetService,
        TechSupport,
        PaymentMethod,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
        ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY Contract, InternetService, TechSupport, PaymentMethod
    HAVING COUNT(*) >= 50 -- Filter out insignificant sample sizes
)
SELECT *
FROM FeatureCombinations
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- SECTION 7: TOP CUSTOMER ANALYSIS & SEGMENTATION
-- ----------------------------------------------------------------------------

-- 7.1 Customers with highest monthly charges (Top 10)
SELECT Contract, InternetService, tenure, MonthlyCharges, TotalCharges, Churn
FROM customers
ORDER BY MonthlyCharges DESC
LIMIT 10;

-- 7.2 Customers with longest tenure (Top 10)
SELECT Contract, InternetService, tenure, MonthlyCharges, TotalCharges, Churn
FROM customers
ORDER BY tenure DESC
LIMIT 10;

-- 7.3 Customers with high monthly charges ($80+) and short tenure (<= 12 months)
SELECT 
    Contract,
    InternetService,
    COUNT(*) AS high_charge_new_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE MonthlyCharges >= 80 AND tenure <= 12
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;
