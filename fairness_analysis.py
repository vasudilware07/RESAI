"""
fairness_analysis.py - Fairness & Bias Detection Module
=========================================================
Unit 2 – Fairness and Bias

Implements:
  • Demographic Parity
  • Equal Opportunity
  • Disparate Impact
  • Statistical Parity Difference

Compares fairness across:
  • Gender (Male vs Female)
  • Age groups (Young vs Senior)

Uses: Fairlearn metrics + manual implementations for learning purposes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

from utils import log, save_plot, save_json, set_plot_style


# ═════════════════════════════════════════════════════════════════════════════
# CORE FAIRNESS METRICS (manual implementations for educational clarity)
# ═════════════════════════════════════════════════════════════════════════════

def demographic_parity(y_pred, sensitive):
    """
    Demographic Parity: P(ŷ=1 | G=0) vs P(ŷ=1 | G=1)
    A fair model should have similar positive-prediction rates across groups.
    """
    groups = np.unique(sensitive)
    rates = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = y_pred[mask].mean()
    return rates


def equal_opportunity(y_true, y_pred, sensitive):
    """
    Equal Opportunity: True Positive Rate should be equal across groups.
    TPR = P(ŷ=1 | y=1, G=g)
    """
    groups = np.unique(sensitive)
    tpr = {}
    for g in groups:
        mask = (sensitive == g) & (y_true == 1)
        if mask.sum() == 0:
            tpr[g] = 0.0
        else:
            tpr[g] = y_pred[mask].mean()
    return tpr


def disparate_impact(y_pred, sensitive, privileged=1, unprivileged=0):
    """
    Disparate Impact Ratio: P(ŷ=1 | G=unprivileged) / P(ŷ=1 | G=privileged)
    Ratio < 0.8 indicates adverse impact (80% rule).
    """
    rate_priv = y_pred[sensitive == privileged].mean()
    rate_unpriv = y_pred[sensitive == unprivileged].mean()
    if rate_priv == 0:
        return float("inf")
    return rate_unpriv / rate_priv


def statistical_parity_difference(y_pred, sensitive, privileged=1, unprivileged=0):
    """
    SPD = P(ŷ=1 | G=unprivileged) - P(ŷ=1 | G=privileged)
    Ideal value = 0.
    """
    rate_priv = y_pred[sensitive == privileged].mean()
    rate_unpriv = y_pred[sensitive == unprivileged].mean()
    return rate_unpriv - rate_priv


# ═════════════════════════════════════════════════════════════════════════════
# FAIRNESS ANALYSIS RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def analyze_fairness(models, data):
    """
    Run full fairness analysis for each model across gender and age.
    Returns a dict of results and generates plots.
    """
    log("Running Fairness & Bias Detection...")
    set_plot_style()

    X_test = data["X_test"]
    y_test = data["y_test"]
    sensitive_test = data["sensitive_test"]
    gender = sensitive_test["Gender"]

    # Create binary age group: Young (18-30) → 0, Senior (31+) → 1
    age_raw = sensitive_test.get("AgeGroup")
    if age_raw is not None:
        age_binary = (age_raw > 0).astype(int)  # 0=18-30, 1=31+
    else:
        age_binary = np.zeros(len(y_test))

    all_results = {}

    for model_name, model in models.items():
        log(f"  Analyzing fairness for: {model_name}")
        y_pred = model.predict(X_test)

        result = {}

        # ── Gender fairness ──────────────────────────────────────────
        result["gender"] = _compute_fairness_metrics(
            y_test, y_pred, gender,
            privileged=1, unprivileged=0,
            group_labels={0: "Female", 1: "Male"},
        )

        # ── Age fairness ─────────────────────────────────────────────
        result["age"] = _compute_fairness_metrics(
            y_test, y_pred, age_binary,
            privileged=1, unprivileged=0,
            group_labels={0: "Young (18-30)", 1: "Senior (31+)"},
        )

        # ── Fairlearn library metrics ────────────────────────────────
        result["fairlearn_gender"] = {
            "Demographic Parity Diff": round(
                demographic_parity_difference(y_test, y_pred, sensitive_features=gender), 4
            ),
            "Demographic Parity Ratio": round(
                demographic_parity_ratio(y_test, y_pred, sensitive_features=gender), 4
            ),
            "Equalized Odds Diff": round(
                equalized_odds_difference(y_test, y_pred, sensitive_features=gender), 4
            ),
        }

        all_results[model_name] = result
        _print_fairness(model_name, result)

    # ── Generate visualizations ──────────────────────────────────────
    _plot_fairness_comparison(all_results)
    _plot_bias_heatmap(all_results)
    _plot_group_fairness(all_results)

    # ── Save results ─────────────────────────────────────────────────
    # Convert for JSON serialization
    serializable = {}
    for mn, res in all_results.items():
        serializable[mn] = {}
        for key, val in res.items():
            serializable[mn][key] = {
                k: float(v) if isinstance(v, (np.floating,)) else v
                for k, v in val.items()
            }
    save_json(serializable, "fairness_results.json")

    return all_results


def _compute_fairness_metrics(y_true, y_pred, sensitive, privileged, unprivileged, group_labels):
    """Compute all four fairness metrics for a given sensitive attribute."""
    dp = demographic_parity(y_pred, sensitive)
    eo = equal_opportunity(y_true, y_pred, sensitive)
    di = disparate_impact(y_pred, sensitive, privileged, unprivileged)
    spd = statistical_parity_difference(y_pred, sensitive, privileged, unprivileged)

    return {
        f"Positive Rate ({group_labels[unprivileged]})": round(dp.get(unprivileged, 0), 4),
        f"Positive Rate ({group_labels[privileged]})": round(dp.get(privileged, 0), 4),
        f"TPR ({group_labels[unprivileged]})": round(eo.get(unprivileged, 0), 4),
        f"TPR ({group_labels[privileged]})": round(eo.get(privileged, 0), 4),
        "Disparate Impact Ratio": round(di, 4),
        "Statistical Parity Difference": round(spd, 4),
    }


def _print_fairness(model_name, result):
    """Pretty-print fairness metrics."""
    print(f"\n{'='*60}")
    print(f"Fairness Report: {model_name}")
    print(f"{'='*60}")
    for attr, metrics in result.items():
        print(f"\n  [{attr.upper()}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _plot_fairness_comparison(results):
    """Bar chart: Disparate Impact and SPD per model, by gender."""
    model_names = list(results.keys())
    di_vals = [results[m]["gender"]["Disparate Impact Ratio"] for m in model_names]
    spd_vals = [results[m]["gender"]["Statistical Parity Difference"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Disparate Impact
    colors = ["green" if 0.8 <= v <= 1.25 else "red" for v in di_vals]
    axes[0].bar(model_names, di_vals, color=colors, edgecolor="black")
    axes[0].axhline(y=0.8, color="orange", linestyle="--", label="80% threshold")
    axes[0].axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Ideal (1.0)")
    axes[0].set_title("Disparate Impact Ratio (Gender)")
    axes[0].set_ylabel("Ratio")
    axes[0].legend()

    # Statistical Parity Difference
    colors2 = ["green" if abs(v) < 0.1 else "red" for v in spd_vals]
    axes[1].bar(model_names, spd_vals, color=colors2, edgecolor="black")
    axes[1].axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Ideal (0)")
    axes[1].axhline(y=0.1, color="orange", linestyle="--", label="±0.1 threshold")
    axes[1].axhline(y=-0.1, color="orange", linestyle="--")
    axes[1].set_title("Statistical Parity Difference (Gender)")
    axes[1].set_ylabel("Difference")
    axes[1].legend()

    fig.suptitle("Fairness Metrics Comparison Across Models", fontsize=14)
    save_plot(fig, "fairness_comparison.png")


def _plot_bias_heatmap(results):
    """Heatmap of key fairness metrics across models and attributes."""
    rows = []
    for model_name, res in results.items():
        row = {
            "Model": model_name,
            "Gender DI": res["gender"]["Disparate Impact Ratio"],
            "Gender SPD": res["gender"]["Statistical Parity Difference"],
            "Age DI": res["age"]["Disparate Impact Ratio"],
            "Age SPD": res["age"]["Statistical Parity Difference"],
        }
        if "fairlearn_gender" in res:
            row["FL DP Diff"] = res["fairlearn_gender"]["Demographic Parity Diff"]
            row["FL EO Diff"] = res["fairlearn_gender"]["Equalized Odds Diff"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0, ax=ax,
                linewidths=1, linecolor="white")
    ax.set_title("Bias Heatmap – Fairness Metrics Overview")
    save_plot(fig, "bias_heatmap.png")


def _plot_group_fairness(results):
    """Grouped bar chart showing positive-prediction rates by gender for each model."""
    model_names = list(results.keys())
    female_rates = []
    male_rates = []

    for m in model_names:
        gender_data = results[m]["gender"]
        # Find keys dynamically
        for k, v in gender_data.items():
            if "Female" in k and "Positive Rate" in k:
                female_rates.append(v)
            elif "Male" in k and "Positive Rate" in k:
                male_rates.append(v)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width / 2, female_rates, width, label="Female", color="#e74c3c")
    ax.bar(x + width / 2, male_rates, width, label="Male", color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Positive Prediction Rate (Not Hired)")
    ax.set_title("Group Fairness: Positive Rate by Gender")
    ax.legend()

    for i, (f, m) in enumerate(zip(female_rates, male_rates)):
        ax.text(i - width / 2, f + 0.005, f"{f:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)

    save_plot(fig, "group_fairness_gender.png")


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from train_models import train_all_models
    data = run_preprocessing()
    models = train_all_models(data)
    analyze_fairness(models, data)
