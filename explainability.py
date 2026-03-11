"""
explainability.py - Interpretability & Explainability Module
=============================================================
Unit 3 – Interpretability and Explainability

Implements:
  Intrinsic methods:
    • Decision Tree visualization (graphviz / text)
  Post-hoc methods:
    • SHAP values (TreeExplainer + summary plot)
    • Feature importance (Random Forest)
    • Partial Dependence Plots
  Individual explanations:
    • SHAP force-plot style for single candidates
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.tree import export_text, plot_tree
from sklearn.inspection import PartialDependenceDisplay

from utils import log, save_plot, save_json, set_plot_style, PLOTS_DIR

# Suppress SHAP warnings that clutter output
warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DECISION TREE VISUALIZATION (Intrinsic)
# ═════════════════════════════════════════════════════════════════════════════

def visualize_decision_tree(models, feature_names):
    """Render the Decision Tree structure."""
    log("Generating Decision Tree visualization...")
    dt = models.get("Decision Tree")
    if dt is None:
        log("No Decision Tree model found.", "WARN")
        return

    # Text representation
    tree_text = export_text(dt, feature_names=feature_names, max_depth=5)
    print("\n── Decision Tree Rules (depth ≤ 5) ──")
    print(tree_text[:2000])

    # Graphical plot
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(dt, feature_names=feature_names, class_names=["Hired", "Not Hired"],
              filled=True, rounded=True, ax=ax, max_depth=4, fontsize=8)
    ax.set_title("Decision Tree – Hiring Prediction (max depth 4 shown)", fontsize=14)
    save_plot(fig, "decision_tree.png")


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE (Random Forest)
# ═════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(models, feature_names):
    """Bar chart of top-20 feature importances from Random Forest."""
    log("Generating Feature Importance plot...")
    rf = models.get("Random Forest")
    if rf is None:
        log("No Random Forest model found.", "WARN")
        return None

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances – Random Forest (Hiring Model)")
    save_plot(fig, "feature_importance.png")

    return dict(zip([feature_names[i] for i in indices], importances[indices].round(4)))


# ═════════════════════════════════════════════════════════════════════════════
# 3. SHAP VALUES
# ═════════════════════════════════════════════════════════════════════════════

def compute_shap_values(models, data):
    """Compute SHAP values using TreeExplainer on the Random Forest model."""
    log("Computing SHAP values (this may take a moment)...")
    import shap

    rf = models.get("Random Forest")
    if rf is None:
        log("No Random Forest model found for SHAP analysis.", "WARN")
        return None

    X_test = data["X_test"]
    feature_names = data["feature_names"]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # Handle various SHAP output shapes:
    # - list of 2 arrays (older shap): use class 1
    # - 3D array (n_samples, n_features, n_classes): use class 1 slice
    # - 2D array: use as-is
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    # ── SHAP Summary Plot ─────────────────────────────────────────
    log("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot – Feature Impact on Hiring Decision")
    plt.tight_layout()
    save_plot(plt.gcf(), "shap_summary.png")

    # ── SHAP Bar Plot (mean absolute) ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    save_plot(plt.gcf(), "shap_bar.png")

    return {"explainer": explainer, "shap_values": shap_vals}


def explain_individual(models, data, shap_result, candidate_idx=0):
    """
    Explain the prediction for a single candidate using SHAP values.
    This demonstrates individual-level explainability.
    """
    log(f"Explaining prediction for candidate index {candidate_idx}...")
    import shap

    if shap_result is None:
        log("SHAP result not available.", "WARN")
        return

    rf = models["Random Forest"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]
    shap_vals = shap_result["shap_values"]

    candidate = X_test.iloc[candidate_idx]
    prediction = rf.predict(candidate.values.reshape(1, -1))[0]
    label = "Not Hired (High Attrition Risk)" if prediction == 1 else "Hired (Low Risk)"

    print(f"\n{'='*60}")
    print(f"Individual Explanation – Candidate #{candidate_idx}")
    print(f"Prediction: {label}")
    print(f"{'='*60}")

    # Top contributing features
    sv = shap_vals[candidate_idx]
    # Ensure sv is 1D (flatten if needed)
    sv = np.array(sv).flatten()
    top_idx = np.argsort(np.abs(sv))[-10:][::-1]
    explanation_data = []
    for i in top_idx:
        shap_val = float(sv[i])
        direction = "→ Not Hired" if shap_val > 0 else "→ Hired"
        explanation_data.append({
            "Feature": feature_names[i],
            "Value": round(float(candidate.iloc[i]), 3),
            "SHAP": round(shap_val, 4),
            "Direction": direction,
        })
        print(f"  {feature_names[i]}: SHAP={shap_val:.4f} ({direction})")

    # Waterfall-style plot
    fig, ax = plt.subplots(figsize=(10, 6))
    features_top = [feature_names[i] for i in top_idx]
    shap_top = [float(sv[i]) for i in top_idx]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_top]
    ax.barh(range(len(features_top)), shap_top, color=colors, edgecolor="black")
    ax.set_yticks(range(len(features_top)))
    ax.set_yticklabels(features_top)
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Individual Explanation – Candidate #{candidate_idx} ({label})")
    ax.axvline(x=0, color="black", linewidth=0.8)
    save_plot(fig, "individual_explanation.png")

    return explanation_data


# ═════════════════════════════════════════════════════════════════════════════
# 4. PARTIAL DEPENDENCE PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def partial_dependence_plots(models, data):
    """Generate Partial Dependence Plots for the top 4 features."""
    log("Generating Partial Dependence Plots...")
    rf = models.get("Random Forest")
    if rf is None:
        return

    X_test = data["X_test"]
    feature_names = data["feature_names"]

    # Top 4 by importance
    importances = rf.feature_importances_
    top4_idx = np.argsort(importances)[-4:][::-1]
    top4_features = [feature_names[i] for i in top4_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    PartialDependenceDisplay.from_estimator(
        rf, X_test, features=top4_idx, feature_names=feature_names, ax=ax
    )
    fig.suptitle("Partial Dependence Plots – Top 4 Features", fontsize=14)
    plt.tight_layout()
    save_plot(fig, "partial_dependence.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_explainability(models, data):
    """Execute the full explainability pipeline."""
    log("=" * 60)
    log("EXPLAINABILITY & INTERPRETABILITY ANALYSIS")
    log("=" * 60)

    set_plot_style()
    feature_names = data["feature_names"]

    # 1. Decision Tree
    visualize_decision_tree(models, feature_names)

    # 2. Feature importance
    importance_dict = plot_feature_importance(models, feature_names)

    # 3. SHAP
    shap_result = compute_shap_values(models, data)

    # 4. Individual explanation
    individual = explain_individual(models, data, shap_result, candidate_idx=0)

    # 5. Partial Dependence
    partial_dependence_plots(models, data)

    return {
        "feature_importance": importance_dict,
        "shap_result": shap_result,
        "individual_explanation": individual,
    }


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from train_models import train_all_models
    data = run_preprocessing()
    models = train_all_models(data)
    run_explainability(models, data)
