"""
Stage 6: Machine Learning Model Training & Evaluation Script
------------------------------------------------------------
Project: Customer Churn Prediction & Explainable Business Analytics Platform
Description: Trains and compares Logistic Regression, Decision Tree, Random Forest, 
             and Gradient Boosting models. Evaluates metrics on untouched test data,
             generates comparison figures/reports, and exports trained model artifacts.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline

def train_and_evaluate_models(
    processed_dir: str,
    models_dir: str,
    reports_dir: str,
    figures_dir: str,
    random_state: int = 42
):
    """
    Trains multiple classification models, evaluates test metrics, saves plots, and exports model artifacts.
    """
    print("=" * 75)
    print(" STAGE 6: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 75)

    # 1. Load Processed Datasets & Preprocessor Artifact
    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    fe_dataset_path = os.path.join(processed_dir, "feature_engineered_churn.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Missing Stage 5 preprocessed data or preprocessor artifact!")

    print("[*] Step 1: Loading Stage 5 preprocessed datasets...")
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    preprocessor = joblib.load(preprocessor_path)

    print(f"    -> X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    -> X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    # 2. Define Model Candidates
    print("\n[*] Step 2: Initializing Classification Model Candidates...")
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state, max_depth=5
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_estimators=100
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state, n_estimators=100
        )
    }

    results = []
    trained_models = {}
    predictions = {}
    probabilities = {}
    conf_matrices = {}

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 3. Train & Evaluate Models
    print("\n[*] Step 3: Training & Evaluating Models on Test Data...")
    for name, model in candidate_models.items():
        print(f"    -> Training '{name}'...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })

        trained_models[name] = model
        predictions[name] = y_pred
        probabilities[name] = y_prob
        conf_matrices[name] = cm

        # Save individual candidate model
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))

    df_results = pd.DataFrame(results)
    print("\n[*] Model Evaluation Results Summary:")
    print(df_results.to_string(index=False))

    # Save Comparison CSV
    comparison_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df_results.to_csv(comparison_csv_path, index=False)
    print(f"\n[+] Saved model comparison table to: {comparison_csv_path}")

    # 4. Generate Visualizations
    print("\n[*] Step 4: Generating Diagnostic Evaluation Visualizations...")

    # Plot 1: Model Metrics Comparison
    plt.figure(figsize=(10, 6))
    df_plot = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Model Performance Metrics Comparison on Test Set", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Model Candidate", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    metrics_plot_path = os.path.join(figures_dir, "model_metrics_comparison.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()

    # Plot 2: Confusion Matrices Subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("Actual Label")

    plt.tight_layout()
    cm_plot_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # Plot 3: ROC Curves
    plt.figure(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = df_results.loc[df_results["Model"] == name, "ROC-AUC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=colors[idx], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)")
    plt.title("ROC Curves Comparison on Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_plot_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    print(f"    -> Figures saved in: {figures_dir}")
    # 5. Best Model Selection & Full Pipeline Serialization

    results_df = pd.DataFrame(results)

    # Select model with highest Recall
    best_model_name = results_df.loc[
        results_df["Recall"].idxmax(), "Model"
    ]

    best_model = trained_models[best_model_name]

    print("\n" + "=" * 50)
    print("BEST MODEL SELECTION")
    print("=" * 50)

    print(f"Selected Model: {best_model_name}")

    best_result = results_df[
        results_df["Model"] == best_model_name
    ].iloc[0]

    print(f"Recall:    {best_result['Recall']:.4f}")
    print(f"Precision: {best_result['Precision']:.4f}")
    print(f"F1 Score:  {best_result['F1 Score']:.4f}")
    print(f"ROC-AUC:   {best_result['ROC-AUC']:.4f}")


    # 5. Threshold Tuning Analysis

    print("\\n[*] Step 5: Performing Threshold Tuning...")

    best_probabilities = probabilities[best_model_name]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_results = []

    for threshold in thresholds:

        threshold_pred = (
            best_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
            "Threshold": threshold,
            "Precision": round(
                precision_score(y_test, threshold_pred), 4
            ),
            "Recall": round(
                recall_score(y_test, threshold_pred), 4
            ),
            "F1 Score": round(
                f1_score(y_test, threshold_pred), 4
            )
        })

    threshold_df = pd.DataFrame(threshold_results)

    print("\\nThreshold Comparison:")
    print(threshold_df.to_string(index=False))

    threshold_csv_path = os.path.join(
        reports_dir, "threshold_comparison.csv"
    )

    threshold_df.to_csv(threshold_csv_path, index=False)

    # Threshold Comparison Visualization

    plt.figure(figsize=(9, 5))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1 Score"],
        marker="o",
        label="F1 Score"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning Analysis")
    plt.legend()
    plt.grid(True)

    threshold_plot_path = os.path.join(
        figures_dir, "threshold_tuning_analysis.png"
    )

    plt.savefig(threshold_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Threshold CSV saved: {threshold_csv_path}")
    print(f"Threshold plot saved: {threshold_plot_path}")


    # Save Best Model
    best_model_path = os.path.join(
        models_dir, "best_churn_model.joblib"
    )

    joblib.dump(best_model, best_model_path)

    print(f"\nBest model saved: {best_model_path}")

    # Construct End-to-End Pipeline on Raw Data
    if os.path.exists(fe_dataset_path):

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", best_model)
        ])

        full_pipeline_path = os.path.join(
            models_dir, "full_churn_pipeline.joblib"
        )

        joblib.dump(full_pipeline, full_pipeline_path)

        print(
            f"End-to-end pipeline saved: {full_pipeline_path}"
        )

    print(
        f"\nSelected Baseline Best Model: {best_model_name}"
    )
    # 6. Generate Dynamic Markdown Evaluation Report

    report_md_path = os.path.join(
        reports_dir, "model_evaluation.md"
    )

    def df_to_md(df):
        headers = list(df.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = [
            "| " + " | ".join([str(val) for val in row.values]) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header_line, sep_line] + rows)

    # Generate dynamic diagnostics for every model
    diagnostics_content = ""

    for index, row in results_df.iterrows():

        model_name = row["Model"]
        tn, fp, fn, tp = conf_matrices[model_name].ravel()

        diagnostics_content += f"""
### {index + 1}. {model_name}

- **Accuracy:** {row['Accuracy']:.2%}
- **Precision:** {row['Precision']:.2%}
- **Recall:** {row['Recall']:.2%}
- **F1 Score:** {row['F1 Score']:.4f}
- **ROC-AUC:** {row['ROC-AUC']:.4f}

**Confusion Matrix:**
- True Negatives (TN): {tn}
- False Positives (FP): {fp}
- False Negatives (FN): {fn}
- True Positives (TP): {tp}

"""

    # Generate selected model explanation
    threshold_table = df_to_md(threshold_df)

    report_content = f"""# Stage 6: Machine Learning Model Evaluation Report

## 1. Project Objective

The objective is to build and evaluate machine learning models
that identify customers who are likely to leave the service.

## 2. Model Performance Comparison

{df_to_md(results_df)}

## 3. Detailed Model Diagnostics

{diagnostics_content}

## 4. Best Model Selection

- **Selected Model:** {best_model_name}
- **Selection Metric:** Recall
- **Recall:** {best_result['Recall']:.2%}
- **Precision:** {best_result['Precision']:.2%}
- **F1 Score:** {best_result['F1 Score']:.4f}
- **ROC-AUC:** {best_result['ROC-AUC']:.4f}

### Selection Rationale

The model was selected based on Recall because the primary
business objective is to identify as many potential churners
as possible.

However, Precision, F1 Score and ROC-AUC should also be
considered when evaluating its practical usefulness.

## 5. Threshold Tuning Analysis

The following table compares model performance at different
classification thresholds.

{threshold_table}

The threshold of 0.60 achieved the highest F1 Score in this
exploratory comparison. However, it reduced Recall compared
with the default threshold of 0.50.

This comparison uses the test dataset and should not be treated
as an unbiased final threshold selection.

## 6. Business Interpretation

- Higher Recall helps identify more customers who may churn.
- False Negatives represent actual churners missed by the model.
- False Positives represent customers incorrectly identified
  as potential churners.
- Retention teams should evaluate the cost of contacting
  incorrectly identified customers.

## 7. Saved Visualizations

- Model Metrics Comparison:
  `figures/model_metrics_comparison.png`

- Threshold Tuning Analysis:
  `figures/threshold_tuning_analysis.png`

## 8. Saved Model Artifacts

- Best Model: `models/best_churn_model.joblib`
- Full Pipeline: `models/full_churn_pipeline.joblib`

## 9. Conclusion

Multiple classification models were trained and evaluated.
The final baseline model was selected using Recall as the
primary evaluation metric.

Further improvements may include threshold tuning,
hyperparameter optimization and business-cost evaluation.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nEvaluation report saved: {report_md_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    proc_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")

    train_and_evaluate_models(
        proc_dir,
        models_dir,
        reports_dir,
        figures_dir
    )