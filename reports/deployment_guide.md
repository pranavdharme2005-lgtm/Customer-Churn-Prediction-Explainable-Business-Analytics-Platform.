# Stage 13: End-to-End Deployment & Productionization Guide

This guide details the step-by-step instructions for running the **Customer Churn Analytics & Recommendation Platform** locally and deploying it to **Streamlit Community Cloud**.

---

## 💻 1. Local Environment Setup & Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed
- Git installed

### Step-by-Step Local Launch
```bash
# 1. Clone or navigate to the project root directory
cd customer-churn-prediction

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install pinned production dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit Web Application
streamlit run dashboard/app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ 2. Streamlit Community Cloud Deployment Steps

### Step 1: Push Repository to GitHub
1. Initialize Git repository (if not already initialized):
   ```bash
   git init
   git add .
   git commit -m "Complete Customer Churn Prediction & Analytics Platform (Stages 1-13)"
   ```
2. Create a public repository on GitHub (e.g. `customer-churn-analytics-platform`).
3. Push local commits to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/customer-churn-analytics-platform.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Connect Repository to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"** (or **"Deploy an app"**).
3. Fill in the deployment form:
   - **Repository**: `YOUR_USERNAME/customer-churn-analytics-platform`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Advanced Settings (Optional):
   - Select **Python 3.11** or **Python 3.12** for maximum wheel compatibility.
5. Click **"Deploy!"**.

### Step 3: Monitor Build & Verify Live App
- Streamlit Cloud will parse `requirements.txt`, install dependencies, load `models/final_churn_pipeline.joblib`, and launch the web server.
- Upon completion, your live URL will be generated (e.g. `https://customer-churn-platform.streamlit.app/`).

---

## ⚠️ 3. Common Deployment Errors & Troubleshooting

### Issue 1: `FileNotFoundError` during data or model loading
- **Root Cause**: Absolute local OS paths (like `C:\Users\...`) breaking on Linux cloud servers.
- **Resolution**: Use portable `Path(__file__).resolve().parent.parent` path resolution (implemented in `dashboard/utils.py`).

### Issue 2: `ModuleNotFoundError` on Cloud Server
- **Root Cause**: Missing dependency in `requirements.txt`.
- **Resolution**: Ensure all packages (`streamlit`, `plotly`, `scikit-learn`, `joblib`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`) are pinned in `requirements.txt`.

### Issue 3: Memory Limit Exceeded (OOM Kills)
- **Root Cause**: Retraining ML models or calculating large SHAP matrices on every page reload.
- **Resolution**: Caching model pipelines via `@st.cache_resource` and dataframes via `@st.cache_data`.

---

## 🛡️ 4. Security, Privacy & Secret Management
- **No Credentials Required**: This application relies strictly on local static model artifacts and public datasets; no external database connection strings or API keys are required.
- **Data Privacy**: Built using the public IBM Telco Customer Churn anonymized dataset containing 0 real PII (Personally Identifiable Information).
