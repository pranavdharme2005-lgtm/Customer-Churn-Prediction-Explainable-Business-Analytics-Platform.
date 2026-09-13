# Stage 6: Machine Learning Model Evaluation Report

## 📌 Executive Summary
This report presents the training, testing, and comparative evaluation of **4 classification algorithms** for customer churn prediction during **Stage 6**.

All models were evaluated on the **untouched 20% test dataset** (`1,409` customer accounts) preprocessed without data leakage.

---

## 📊 Model Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7331 | 0.4983 | 0.7914 | 0.6116 | 0.8411 |
| Decision Tree | 0.7544 | 0.526 | 0.7567 | 0.6206 | 0.833 |
| Random Forest | 0.7637 | 0.5492 | 0.6123 | 0.579 | 0.8192 |
| Gradient Boosting | 0.8027 | 0.6644 | 0.5187 | 0.5826 | 0.8436 |

---

## 🔍 Detailed Model Diagnostics & Interpretations

### 1. Logistic Regression (Baseline Model with `class_weight='balanced'`)
- **Accuracy**: 73.31% | **Precision**: 49.83% | **Recall**: **79.14%** | **F1 Score**: 0.6116 | **ROC-AUC**: **0.8411**
- **Confusion Matrix**: True Negatives (TN) = 737, False Positives (FP) = 298, False Negatives (FN) = 78, True Positives (TP) = 296
- **Key Advantage**: Achieves the **highest recall (79.14%)**, correctly identifying **296 out of 374 actual churners**. In churn management, missing a churner (FN) is far more costly than sending a retention offer to a loyal customer (FP).

### 2. Decision Tree (Max Depth = 5, `class_weight='balanced'`)
- **Accuracy**: 75.44% | **Precision**: 52.60% | **Recall**: 75.67% | **F1 Score**: **0.6206** | **ROC-AUC**: 0.8330
- **Key Advantage**: Achieves the **highest overall F1-score (0.6206)**, offering an excellent balance between precision and recall while remaining fully transparent and rule-interpretable.

### 3. Random Forest (100 Trees, `class_weight='balanced'`)
- **Accuracy**: 76.37% | **Precision**: 54.92% | **Recall**: 61.23% | **F1 Score**: 0.5790 | **ROC-AUC**: 0.8192
- **Key Advantage**: Good stability and precision, but lower recall compared to Logistic Regression under default parameters.

### 4. Gradient Boosting (100 Estimators)
- **Accuracy**: **80.27%** | **Precision**: **66.44%** | **Recall**: 51.87% | **F1 Score**: 0.5826 | **ROC-AUC**: **0.8436**
- **Key Advantage**: Highest overall accuracy and highest ROC-AUC score. However, its default decision threshold (0.50) yields a lower recall (51.87%), missing 180 churners. In future stages, adjusting its decision threshold can unlock peak recall.

---

## 🏆 Baseline Model Selection & Justification

- **Selected Model**: **Logistic Regression (Balanced)**
- **Selection Rationale**:
  1. **Peak Recall Performance**: In customer churn prediction, the primary business objective is catching at-risk customers before they cancel. Logistic Regression correctly detects **79.14% of churners**.
  2. **High Discriminative Ability**: Strong **ROC-AUC score of 0.8411**, proving excellent ranking capacity across probability thresholds.
  3. **Simplicity & Interpretability**: Provides well-calibrated risk probabilities and clear linear coefficients.

---

## 🖼️ Saved Visualizations
- **Metrics Comparison Plot**: [`reports/figures/model_metrics_comparison.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/model_metrics_comparison.png)
- **Confusion Matrices Grid**: [`reports/figures/confusion_matrices.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/confusion_matrices.png)
- **ROC Curves**: [`reports/figures/roc_curves.png`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/reports/figures/roc_curves.png)

---

## 📁 Model Artifacts Saved
- **Selected Baseline Model**: [`models/best_churn_model.joblib`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/models/best_churn_model.joblib)
- **End-to-End Pipeline (Preprocessor + Classifier)**: [`models/full_churn_pipeline.joblib`](file:///c:/Users/prana/.gemini/antigravity/scratch/jarvis_assistant/customer-churn-prediction/models/full_churn_pipeline.joblib)
