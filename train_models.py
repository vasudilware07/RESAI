"""
train_models.py - AI Hiring Prediction Models
===============================================
Unit 6 – Case Study: Hiring System Simulation

We treat IBM HR Attrition prediction as a proxy for a hiring decision model:
  - Attrition = 1 → "Not Hired" (high attrition risk)
  - Attrition = 0 → "Hired" (low attrition risk)

Models trained:
  1. Logistic Regression
  2. Random Forest
  3. Decision Tree

Evaluation: Accuracy, Precision, Recall, F1-Score
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from utils import log, save_plot, save_json, set_plot_style


# ─── Model definitions ──────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
}


def train_all_models(data):
    """Train all configured models and return dict of fitted models."""
    log("Training AI hiring prediction models...")
    X_train, y_train = data["X_train"], data["y_train"]
    models = {}

    for name, model in MODEL_CONFIGS.items():
        log(f"  Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
        log(f"  {name} trained successfully.")

    return models


def evaluate_models(models, data):
    """Evaluate each model and output metrics + confusion matrices."""
    log("Evaluating models...")
    set_plot_style()
    X_test, y_test = data["X_test"], data["y_test"]
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        results[name] = {"metrics": metrics, "y_pred": y_pred}

        log(f"  {name}: {metrics}")
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["Hired", "Not Hired"]))

    # ── Comparison bar chart ─────────────────────────────────────────
    _plot_model_comparison(results)

    # ── Confusion matrices ───────────────────────────────────────────
    _plot_confusion_matrices(models, data)

    # ── Save metrics JSON ────────────────────────────────────────────
    metrics_export = {n: r["metrics"] for n, r in results.items()}
    save_json(metrics_export, "model_metrics.json")

    return results


def _plot_model_comparison(results):
    """Bar chart comparing Accuracy / Precision / Recall / F1 across models."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.25

    for i, name in enumerate(model_names):
        values = [results[name]["metrics"][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison – AI Hiring System")
    ax.legend()
    save_plot(fig, "model_comparison.png")


def _plot_confusion_matrices(models, data):
    """Side-by-side confusion matrices for all models."""
    X_test, y_test = data["X_test"], data["y_test"]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Hired", "Not Hired"],
                    yticklabels=["Hired", "Not Hired"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices – Hiring Predictions", y=1.02, fontsize=14)
    save_plot(fig, "confusion_matrices.png")


def get_predictions(models, data):
    """Return predictions dict keyed by model name."""
    X_test = data["X_test"]
    return {name: model.predict(X_test) for name, model in models.items()}


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    data = run_preprocessing()
    models = train_all_models(data)
    results = evaluate_models(models, data)
