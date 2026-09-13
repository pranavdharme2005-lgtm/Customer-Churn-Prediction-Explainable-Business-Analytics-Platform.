# Stage 14: Technical Skills Demonstrated Matrix

This document provides a categorized matrix of technical, analytical, machine learning, and software engineering skills demonstrated throughout the **Customer Churn Prediction & Explainable Business Analytics Platform** project.

---

## 🛠️ Categorized Skills Matrix

### 💻 1. Programming & Software Engineering
- **Python 3.13+**: Advanced object-oriented and functional scripting, modular package structuring, exception handling.
- **Path Portability (`pathlib`)**: Cross-platform path handling using `Path(__file__).resolve().parent.parent`.
- **Package Management**: Dependency pinning, virtual environment configuration (`venv`), `requirements.txt` optimization.
- **Git & GitHub**: Version control, `.gitignore` configuration, clean commit structuring, branch management.

### 📊 2. Data Analysis & Wrangling
- **Pandas**: DataFrame manipulation, vectorized transformations, string cleaning, type casting, merging datasets.
- **NumPy**: Matrix operations, conditional arrays (`np.where`), zero-division handling (`np.maximum`).
- **Data Cleaning**: Imputing whitespace missing values, dropping non-predictive cardinality hashes (`customerID`), validating data integrity.

### 🗄️ 3. SQL & Database Analytics
- **SQLite3**: Database creation (`data/customer_churn.db`), Python SQL ingestion (`df.to_sql()`).
- **SQL Querying**: Complex queries using `CASE` statements, `HAVING`, `GROUP BY`, `ORDER BY`, `COUNT`, `SUM`, `AVG`.
- **Cohort Analysis**: Categorizing tenure groups and calculating segment churn percentages in relational SQL.

### 🎨 4. Data Visualization
- **Matplotlib & Seaborn**: Univariate histograms, bivariate bar charts, correlation heatmaps, box plots, publication styling.
- **Plotly Express & Graph Objects**: Interactive donut charts, probability histograms, scatter plots, horizontal bar charts, indicator gauges.

### 🤖 5. Machine Learning & Optimization
- **Scikit-learn**: Classification models (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`).
- **Leakage-Free Preprocessing**: `ColumnTransformer`, `StandardScaler`, `OneHotEncoder(handle_unknown='ignore')`, `SimpleImputer`.
- **Model Optimization**: 5-Fold `StratifiedKFold` Cross-Validation, `GridSearchCV`, `RandomizedSearchCV`.
- **Model Evaluation**: Metrics selection (Recall 76.20%, Precision 53.77%, F1 0.6305, ROC-AUC 0.8394, Confusion Matrix analysis).
- **Model Persistence**: Packaging pipelines with `joblib`.

### 🔍 6. Explainable AI (XAI) & Interpretability
- **SHAP (SHapley Additive exPlanations)**: `shap.TreeExplainer` global summary plots and feature importance rankings.
- **Local Feature Attribution**: Translating numeric SHAP contribution vectors into human-readable business explanations.

### 🧩 7. Unsupervised Machine Learning
- **K-Means Clustering**: Standardized feature distance clustering ($K=4$, Silhouette Score = 0.4261).
- **Cluster Diagnostics**: Elbow Method (Inertia plot) and Silhouette Analysis.
- **PCA Visualization**: 2D Principal Component Analysis projection for multi-dimensional cluster visualization.

### 🎛️ 8. Business Analytics & Prescriptive Modeling
- **What-If Scenario Simulation**: Quantifying probability deltas ($\Delta P$) under hypothetical feature updates.
- **Feature Dependency Synchronization**: Re-calculating dependent variables (`TotalCharges = tenure * MonthlyCharges`).
- **Prescriptive Decision Support**: Designing transparent, non-causal recommendation rules linking XAI, risk tiers, and customer segments.

### 🌐 9. Dashboard & Web Development
- **Streamlit**: Multi-page web dashboard (`dashboard/app.py`), sidebar navigation, custom HTML KPI metric cards.
- **Performance Optimization**: Caching models with `@st.cache_resource` and datasets with `@st.cache_data`.
- **Form Controls & Inputs**: Interactive sliders, dropdowns, forms, and custom risk predictors.
