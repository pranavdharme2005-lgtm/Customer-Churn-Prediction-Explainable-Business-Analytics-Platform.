# Stage 13: Deployment & Productionization — Data Science Interview Guide

This guide provides technical and conceptual answers to key data science interview questions covering **ML Deployment**, **Streamlit Cloud**, **Path Portability**, **Docker Containerization**, and **Production Monitoring**.

---

### Q1: How did you prepare and deploy your Customer Churn platform?
**Answer:**  
I prepared the platform by standardizing all data and model paths using `pathlib.Path` relative resolution, pinning production dependencies in `requirements.txt`, configuring Streamlit server parameters in `.streamlit/config.toml`, and structuring `.gitignore` to track runtime model artifacts (`models/final_churn_pipeline.joblib`). For deployment, the repository is connected to Streamlit Community Cloud with `dashboard/app.py` configured as the main entry point.

---

### Q2: Why did you choose Streamlit for the application interface?
**Answer:**  
Streamlit enables pythonic data applications without requiring complex custom JavaScript/React frontends. It natively integrates with Scikit-learn model pipelines, Pandas DataFrames, and interactive Plotly charts, making it ideal for prototyping executive analytics dashboards and decision-support tools.

---

### Q3: What is Streamlit Community Cloud and how does it work?
**Answer:**  
Streamlit Community Cloud is a platform-as-a-service (PaaS) that connects directly to a public GitHub repository. Upon push, it automatically spins up a Linux container, installs dependencies from `requirements.txt`, executes `streamlit run dashboard/app.py`, and exposes a public HTTPS endpoint.

---

### Q4: What is the main application entry point and why is it important?
**Answer:**  
The main entry point is `dashboard/app.py`. Having a single, explicit entry point prevents competing startup scripts, simplifies build configurations on hosting platforms, and ensures clean resource loading.

---

### Q5: How did you manage dependencies and ensure reproducibility?
**Answer:**  
I created a clean `requirements.txt` file containing only required runtime libraries with compatible version specifiers (e.g. `streamlit>=1.30.0`, `plotly>=5.18.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`, `shap>=0.42.0`). This prevents version mismatch errors between local development and cloud hosting servers.

---

### Q6: Why are relative portable paths (`pathlib.Path`) critical for production deployment?
**Answer:**  
Hardcoded absolute local paths (like `C:\Users\prana\...`) will immediately crash on Linux cloud servers (`FileNotFoundError`). Using `BASE_DIR = Path(__file__).resolve().parent.parent` ensures all model and dataset paths resolve correctly regardless of OS, directory depth, or user environment.

---

### Q7: How did you optimize dashboard startup time and performance?
**Answer:**  
1. **Resource Caching**: Wrapped heavy model pipeline loading in `@st.cache_resource` so `final_churn_pipeline.joblib` loads into RAM only once upon server startup.
2. **Data Caching**: Wrapped master CSV loading in `@st.cache_data`.
3. **No Retraining**: All inference runs on pre-trained saved artifacts, avoiding expensive training cycles on page reload.

---

### Q8: How did you audit and protect secrets and sensitive credentials?
**Answer:**  
I conducted a security scan ensuring `0` API keys, database credentials, or access tokens exist in the codebase. Because the platform relies on local static model artifacts and public datasets, no external secrets are required. If external APIs were needed, credentials would be managed securely via Streamlit Secrets (`.streamlit/secrets.toml` or OS environment variables) excluded from Git.

---

### Q9: What common deployment challenges did you encounter and how were they resolved?
**Answer:**  
- **Path Resolution**: Fixed OS-specific path issues by implementing `pathlib.Path`.
- **Git LFS vs Direct Commits**: Confirmed our tuned model pipeline size is **8.6 MB**, avoiding GitHub's 100 MB limit without requiring Git LFS overhead.
- **Module Import Errors**: Resolved Streamlit component import syntax to adhere to latest package versions.

---

### Q10: How would you deploy this platform in an enterprise production environment?
**Answer:**  
In an enterprise setting:
1. **API Layer**: Expose `models/final_churn_pipeline.joblib` via a FastAPI or Flask REST API running on Kubernetes (EKS/GKE) with auto-scaling.
2. **Database Integration**: Connect the API directly to an enterprise data warehouse (Snowflake, BigQuery, or PostgreSQL).
3. **Frontend**: Deploy the Streamlit/React dashboard on AWS ECS or GCP Cloud Run behind an enterprise API Gateway with SSO authentication.

---

### Q11: How would Docker containerization improve deployment reliability?
**Answer:**  
Docker packages the Python runtime, system OS dependencies, application code, and model artifacts into an immutable container image (`Dockerfile`). This guarantees 100% environment parity between local testing and production cloud clusters, eliminating "works on my machine" issues.

---

### Q12: How would you monitor the model and dashboard after deployment?
**Answer:**  
1. **Application Performance Monitoring (APM)**: Track response latency, uptime, and memory usage via Prometheus/Grafana or Datadog.
2. **Model Data Drift Monitoring**: Periodically compute Population Stability Index (PSI) and Wasserstein distance between training feature distributions and live inference inputs using tools like Evidently AI.
3. **Prediction Tracking**: Log live predictions and actual customer churn outcomes to track real-world precision and recall decay over time.
