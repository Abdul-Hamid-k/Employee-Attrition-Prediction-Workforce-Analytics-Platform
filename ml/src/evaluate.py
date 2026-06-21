"""
evaluate.py
Comprehensive evaluation module for the Employee Attrition classification model.

Usage:
    python evaluate.py --model ml/saved_models/best_model.pkl \
                        --test-x ml/saved_models/X_test.csv \
                        --test-y ml/saved_models/y_test.csv \
                        --train-x ml/saved_models/X_train.csv \
                        --train-y ml/saved_models/y_train.csv \
                        --output ml/evaluation_report

Note: train.py should save X_train/X_test/y_train/y_test as CSVs after the
split so evaluation is reproducible independent of the training run.

Produces in --output:
    metrics.json               - all numeric metrics
    report.md                  - human-readable summary (paste into README)
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    calibration_curve.png
    feature_importance.png
    shap_summary.png           - only if shap is installed
"""

import argparse
import json
import os
from pathlib import Path
import sys
sys.path.append('..')

from src.config import ROOT

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def load_artifacts(model_path, x_path, y_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()
    return model, X_test, y_test


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "positive_class_rate": round(float(y_test.mean()), 4),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()


def plot_precision_recall_curve(y_test, y_proba, output_dir):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()

    # Find the threshold that maximizes F1 - usually better than the
    # default 0.5 cutoff, especially on imbalanced data like this.
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return round(best_threshold, 3), round(float(f1_scores[best_idx]), 4)


def plot_calibration_curve(y_test, y_proba, output_dir):
    # Critical for this project: the ROI/priority score multiplies
    # probability x replacement_cost, so the probabilities need to be
    # trustworthy, not just good at ranking.
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual fraction positive")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, output_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(6, 6))
    plt.barh(np.array(feature_names)[idx], importances[idx], color="#1D9E75")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()


def plot_shap_summary(model, X_sample, output_dir):
    try:
        import shap
    except ImportError:
        print("shap not installed - skipping SHAP summary plot (pip install shap)")
        return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # Handle both old (list) and new (array) SHAP output formats
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"SHAP summary skipped: {e}")


def compare_to_baseline(X_train, y_train, X_test, y_test):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    return {
        "baseline_accuracy": round(accuracy_score(y_test, y_pred_base), 4),
        "baseline_f1": round(f1_score(y_test, y_pred_base, zero_division=0), 4),
    }


def cross_validate_stability(model, X, y, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    return {
        "cv_f1_mean": round(float(f1_scores.mean()), 4),
        "cv_f1_std": round(float(f1_scores.std()), 4),
        "cv_f1_scores": [round(float(s), 4) for s in f1_scores],
    }


def write_report(metrics, baseline_metrics, best_threshold, best_f1, output_dir):
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Model evaluation report\n\n")
        f.write("## Test set metrics\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v} |\n")
        if baseline_metrics:
            f.write("\n## Baseline comparison (majority-class classifier)\n\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for k, v in baseline_metrics.items():
                f.write(f"| {k} | {v} |\n")
        f.write("\n## Optimal decision threshold (max F1)\n\n")
        f.write(f"- Threshold: {best_threshold}\n- F1 at this threshold: {best_f1}\n")
        f.write("\n## Plots generated\n\n")
        f.write("- confusion_matrix.png\n- roc_curve.png\n- precision_recall_curve.png\n")
        f.write("- calibration_curve.png\n- feature_importance.png\n- shap_summary.png (if available)\n")
    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the attrition classification model")
    parser.add_argument("--model", default=f"{ROOT}/ml/models/attrition_classification_model.pkl")
    parser.add_argument("--test-x", default=f"{ROOT}/ml/data/processed/X_test.csv")
    parser.add_argument("--test-y", default=f"{ROOT}/ml/data/processed/y_test.csv")
    parser.add_argument("--train-x", default=f"{ROOT}/ml/data/processed/X_train.csv",
                         help="Optional, needed for baseline comparison and CV")
    parser.add_argument("--train-y", default=f"{ROOT}/ml/models/y_train.csv")
    parser.add_argument("--output", default=f"{ROOT}/ml/reports/evaluation_report")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model, X_test, y_test = load_artifacts(args.model, args.test_x, args.test_y)

    metrics, y_pred, y_proba = compute_metrics(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, args.output)
    plot_roc_curve(y_test, y_proba, args.output)
    best_threshold, best_f1 = plot_precision_recall_curve(y_test, y_proba, args.output)
    plot_calibration_curve(y_test, y_proba, args.output)
    plot_feature_importance(model, X_test.columns.tolist(), args.output)

    if not args.skip_shap:
        sample = X_test.sample(min(200, len(X_test)), random_state=42)
        plot_shap_summary(model, sample, args.output)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

    baseline_metrics = {}
    if os.path.exists(args.train_x) and os.path.exists(args.train_y):
        X_train = pd.read_csv(args.train_x)
        y_train = pd.read_csv(args.train_y).squeeze()
        baseline_metrics = compare_to_baseline(X_train, y_train, X_test, y_test)
        print("\n=== Baseline comparison (majority-class predictor) ===")
        print(baseline_metrics)

        print("\n=== 5-fold cross-validation stability ===")
        cv_metrics = cross_validate_stability(model, X_train, y_train)
        print(cv_metrics)
        baseline_metrics.update(cv_metrics)

    all_metrics = {
        **metrics,
        **baseline_metrics,
        "best_threshold": best_threshold,
        "best_f1_at_threshold": best_f1,
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    write_report(metrics, baseline_metrics, best_threshold, best_f1, args.output)
    print(f"\nEvaluation complete. See {args.output}/report.md and metrics.json")


if __name__ == "__main__":
    main()