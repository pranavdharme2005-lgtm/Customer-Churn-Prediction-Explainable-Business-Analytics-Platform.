# Stage 14: Final Comprehensive Interview Guide (30 Q&A Pairs)

This document contains 30 technically correct, project-specific interview questions and answers divided into **Python**, **Pandas**, **SQL**, **Machine Learning**, **Explainable AI**, **Streamlit/Deployment**, and **Business Analytics**.

---

## 🐍 Python (5 Questions)

### Q1: How did you structure your Python codebase for modularity and reproducibility?
**Answer:**  
I divided the codebase into functional modules under `src/` (`data_loading`, `clean_data`, `preprocess_data`, `train_models`, `tune_models`, `explain_model`, `customer_segmentation`, `what_if_analysis`, `recommendation_engine`) and `dashboard/` (`app.py`, `utils.py`, `components.py`). Each script uses modular functions with clear parameter signatures, explicit return types, reproducible random seeds (`random_state=42`), and portable path handling (`pathlib.Path`).

### Q2: How did you use `pathlib.Path` to resolve file paths portably across operating systems?
**Answer:**  
In `dashboard/utils.py` and pipeline scripts, I declared `BASE_DIR = Path(__file__).resolve().parent.parent`. All relative paths (e.g. `BASE_DIR / "models" / "final_churn_pipeline.joblib"`) resolve dynamically relative to the project root, preventing OS-specific path errors when deploying to Linux servers.

### Q3: How did you preserve target data types when exporting and loading CSVs?
**Answer:**  
During Stage 2 cleaning, `TotalCharges` contained blank strings (`" "`). I used `df['TotalCharges'].replace(' ', np.nan)` and imputed `0.0` for customers with `tenure == 0`, before explicitly casting to `float64` via `astype(float)`. When reading CSVs back, pandas loads `TotalCharges` correctly as continuous numeric float data.

### Q4: What Python data structures were used to represent What-If scenarios and customer records?
**Answer:**  
Customer records were handled as Python dictionaries (`cust_dict`) for single-sample manipulation, and as pandas DataFrames for vectorized batch inference. Scenarios were configured as lists of dictionary objects containing scenario names and target feature update key-value pairs (e.g., `{"scenario_name": "Contract Upgrade", "updates": {"Contract": "One year"}}`).

### Q5: How did you handle errors gracefully in your Python scripts?
**Answer:**  
I implemented defensive file checking using `os.path.exists()` and `Path.exists()`, raising clear `FileNotFoundError` or `KeyError` messages if required preprocessed artifacts were missing, while wrapping Streamlit UI rendering in fallback error blocks to prevent stack traces from displaying to business users.

---

## 🐼 Pandas / Data Cleaning (5 Questions)

### Q6: What missing values existed in the IBM Telco dataset and how did you resolve them?
**Answer:**  
The dataset contained 11 hidden whitespace strings (`" "`) in `TotalCharges`. Inspection revealed that all 11 records belonged to new customers with `tenure == 0` months who had not yet received a monthly bill. I imputed `0.0` for `TotalCharges` (matching their billed tenure) and converted the column to `float64`.

### Q7: Why did you drop the `customerID` column during data cleaning?
**Answer:**  
`customerID` consists of 7,043 unique alphanumeric string hashes (e.g., `7590-VHVEG`). Primary key identifiers add high cardinality with zero statistical predictive value, leading to severe overfitting in tree-based models and unnecessary memory overhead.

### Q8: What feature engineering operations did you perform in Pandas?
**Answer:**  
I engineered 3 domain features:
1. `AverageMonthlySpend`: `TotalCharges / tenure` (or `MonthlyCharges` if tenure=0) to capture historical monthly billing pace.
2. `ServiceCount`: Vectorized sum of active subscribed add-ons out of 8 potential services.
3. `IsFirstYear`: Binary indicator (`1` if `tenure <= 12`, else `0`).

### Q9: How did you implement One-Hot Encoding without data leakage?
**Answer:**  
I used Scikit-learn's `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` inside a `ColumnTransformer`. I split raw data into training (`5,634` rows) and testing (`1,409` rows) **before** fitting the encoder, fitting strictly on `X_train` to ensure test categories did not leak into training vocabulary.

### Q10: How did you perform outlier assessment on numerical features?
**Answer:**  
I analyzed IQR bounds ($Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}$) for `tenure`, `MonthlyCharges`, and `TotalCharges`. 0 mathematical outliers were found; genuine high-spend and long-tenure telecom customer profiles were preserved.

---

## 🗄️ SQL (5 Questions)

### Q11: How did you integrate SQL into a Python Data Science pipeline?
**Answer:**  
In Stage 4 (`src/load_data_to_sql.py`), I used Python's native `sqlite3` library to create a lightweight database `data/customer_churn.db`. I loaded the cleaned Pandas DataFrame directly into a SQLite table named `customers` via `df.to_sql()`.

### Q12: How did you calculate overall churn rate in SQL?
**Answer:**  
```sql
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers;
```
Result: 7,043 total customers, 1,869 churned, **26.54% overall churn rate**.

### Q13: How did you group customer tenure into analytical bins using SQL CASE statements?
**Answer:**  
```sql
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        ELSE '49+ Months'
    END AS tenure_group,
    COUNT(*) AS customer_count,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;
```
Result: 0–12 Months group churned at **47.44%**, dropping to **9.51%** for 49+ Months.

### Q14: How did you identify high-risk customer feature combinations using SQL?
**Answer:**  
Using SQL `GROUP BY` and `HAVING COUNT(*) > 50`, I identified that customers combining **Month-to-month Contract + Fiber Optic Internet + No Tech Support + Electronic Check Payment** churned at a peak **62.92%** rate.

### Q15: Why use SQLite instead of running all aggregation in Pandas?
**Answer:**  
SQL is the universal language of business data stores. Demonstrating SQLite integration proves proficiency in writing production SQL queries, aggregating database tables, and transferring query results directly back into analytics reports and DataFrames.

---

## 🤖 Machine Learning (8 Questions)

### Q16: What type of machine learning problem is customer churn prediction?
**Answer:**  
Supervised binary classification ($y \in \{0, 1\}$), where $0$ represents a retained customer and $1$ represents a churned customer.

### Q17: What baseline classification models did you evaluate in Stage 6?
**Answer:**  
I trained and compared four algorithms: Logistic Regression (Balanced), Decision Tree (Depth=5, Balanced), Random Forest (Balanced), and Gradient Boosting.

### Q18: Why did you choose Recall and ROC-AUC as primary optimization metrics over Accuracy?
**Answer:**  
The dataset exhibits a 3:1 class imbalance (73.5% retained vs 26.5% churned). A dummy classifier predicting "No Churn" achieves 73.5% accuracy but catches 0% of churners. In telecom churn, **Recall** measures the percentage of actual churners correctly caught by the model. Maximize Recall minimizes costly false negatives (undetected churners).

### Q19: How did hyperparameter tuning improve model performance in Stage 7?
**Answer:**  
Using 5-Fold `StratifiedKFold` cross-validation via `GridSearchCV` on training data, I tuned the **Random Forest** classifier (`max_depth=12`, `n_estimators=150`, `class_weight='balanced'`). This boosted test set **Recall from 61.23% to 76.20% (+14.97% gain)** and achieved an **F1-Score of 0.6305** and **ROC-AUC of 0.8394**.

### Q20: How did you prevent data leakage during hyperparameter tuning?
**Answer:**  
1. Separated test data ($20\%$ holdout) before any preprocessor fitting or tuning.
2. Performed cross-validation strictly within `X_train`.
3. Applied preprocessing transforms inside `Pipeline` steps so fold validation data was scaled strictly on fold training data.

### Q21: What is a Confusion Matrix and what were your final model results?
**Answer:**  
A Confusion Matrix summarizes True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). On 1,409 test accounts:
- **True Positives (TP)**: 285 churners caught (Recall = 76.20%)
- **False Negatives (FN)**: 89 missed churners
- **True Negatives (TN)**: 789 retained customers correctly identified
- **False Positives (FP)**: 246 false alarms

### Q22: What is the ROC-AUC metric and what does 0.8394 signify?
**Answer:**  
ROC-AUC measures the Area Under the Receiver Operating Characteristic Curve (plotting True Positive Rate vs False Positive Rate across all classification thresholds). A ROC-AUC of **0.8394** means there is an **83.94% probability** that the model ranks a randomly chosen churned customer higher in risk than a randomly chosen retained customer.

### Q23: Why save the full `Pipeline` object instead of just the trained model weights?
**Answer:**  
Saving `models/final_churn_pipeline.joblib` packages both the fitted `ColumnTransformer` (scaling + encoding parameters) and the trained `RandomForestClassifier` into a single artifact. At inference time, raw data dictionaries can be passed directly to `pipeline.predict_proba()` without manually recreating preprocessing steps.

---

## 🔍 Explainable AI (3 Questions)

### Q24: What is SHAP and how does TreeExplainer work?
**Answer:**  
SHAP (SHapley Additive exPlanations) uses game theory to calculate the marginal contribution of each feature to a model's prediction. `TreeExplainer` is an optimized algorithm for tree ensembles (like Random Forest) that calculates exact feature attributions by evaluating conditional expectations across tree paths in $O(TLD^2)$ time.

### Q25: What were the top 3 global feature drivers of churn in your model?
**Answer:**  
1. **`Contract_Month-to-month` (Mean |SHAP| = 0.0697)**: Short-term flexible contracts are the single largest driver of high predicted churn risk.
2. **`tenure` (Mean |SHAP| = 0.0443)**: Low tenure strongly increases predicted churn probability.
3. **`InternetService_Fiber optic` (Mean |SHAP| = 0.0413)**: Fiber optic subscription correlates with elevated churn risk.

### Q26: How did you convert local SHAP contributions into human-readable business explanations?
**Answer:**  
I implemented an automated rule parser (`explain_model.py`) that extracts feature attribution vectors, filters factors by positive risk contribution, and formats them into plain English: *"Flexible Month-to-Month contract contributed (+0.0492) towards an increased predicted churn probability."*

---

## 💻 Streamlit / Deployment (2 Questions)

### Q27: How did you optimize performance and caching in your Streamlit dashboard?
**Answer:**  
I used `@st.cache_resource` for loading the 8.6 MB model pipeline artifact into RAM once on boot, and `@st.cache_data` for reading CSV dataframes. This guarantees fast interactive responses (< 100ms) without model retraining on page refreshes.

### Q28: How did you ensure your Streamlit dashboard is deployment-ready?
**Answer:**  
I replaced absolute OS paths with `pathlib.Path` relative resolution, pinned dependencies in `requirements.txt`, created `.streamlit/config.toml`, configured `.gitignore` to preserve production model artifacts, and verified that `dashboard/app.py` boots cleanly.

---

## 💼 Business Analytics (2 Questions)

### Q29: What is the difference between Predictive Model Sensitivity and Causal Inference?
**Answer:**  
Predictive sensitivity measures how a model's output formula changes under modified inputs ($P(Y \mid X)$). Causal inference proves real-world physical cause-and-effect ($P(Y \mid \text{do}(X))$). Decreasing predicted churn probability in a What-If scenario does **not guarantee** that forcing a contract change will prevent a customer from leaving.

### Q30: How does your platform deliver measurable business value to a telecom company?
**Answer:**  
By prioritizing the top **2,110 High-Risk customers** (29.96% of portfolio), the platform focuses retention efforts where they matter most. Combining SHAP drivers, K-Means customer segments, and What-If scenario deltas enables customer success teams to replace generic calls with tailored retention plays—maximizing retention ROI and customer lifetime value.
