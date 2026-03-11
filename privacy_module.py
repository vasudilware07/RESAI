"""
privacy_module.py - Privacy Preservation Module
=================================================
Unit 5 – Privacy Preservation

Implements:
  • Differential Privacy using IBM's diffprivlib
    - DP Logistic Regression with varying epsilon values
    - Compares accuracy and fairness vs non-private baseline

Shows:
  • Impact on accuracy
  • Impact on fairness
  • Privacy–accuracy–fairness tradeoff plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from utils import log, save_plot, save_json, set_plot_style


# ═════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def run_differential_privacy(data):
    """
    Train Logistic Regression with differential privacy at various
    epsilon levels and compare accuracy/fairness with the baseline.
    """
    log("=" * 60)
    log("PRIVACY PRESERVATION – DIFFERENTIAL PRIVACY")
    log("=" * 60)
    set_plot_style()

    X_train = data["X_train"].values
    y_train = data["y_train"]
    X_test = data["X_test"].values
    y_test = data["y_test"]
    gender_test = data["sensitive_test"]["Gender"]

    # Epsilon values: lower = more private, higher = more accurate
    epsilons = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    # ── Baseline (no privacy) ────────────────────────────────────────
    log("Training baseline (no DP)...")
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    baseline_spd = _compute_spd(baseline_pred, gender_test)

    log(f"Baseline – Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}, SPD: {baseline_spd:.4f}")

    # ── DP experiments ───────────────────────────────────────────────
    results = []
    dp_available = True

    try:
        from diffprivlib.models import LogisticRegression as DPLogisticRegression
        # Test that it actually works with current sklearn
        _test = DPLogisticRegression(epsilon=1.0)
        del _test
    except (ImportError, TypeError):
        log("diffprivlib incompatible with current sklearn. Using simulated DP.", "WARN")
        dp_available = False

    for eps in epsilons:
        log(f"  Training DP model with epsilon={eps}...")

        if dp_available:
            dp_model = DPLogisticRegression(epsilon=eps, max_iter=1000, random_state=42)
            dp_model.fit(X_train, y_train)
            dp_pred = dp_model.predict(X_test)
        else:
            # Fallback: simulate DP by adding Laplace noise to predictions
            dp_pred = _simulate_dp_predictions(baseline, X_test, eps)

        dp_acc = accuracy_score(y_test, dp_pred)
        dp_f1 = f1_score(y_test, dp_pred, zero_division=0)
        dp_spd = _compute_spd(dp_pred, gender_test)

        results.append({
            "epsilon": eps,
            "accuracy": round(dp_acc, 4),
            "f1_score": round(dp_f1, 4),
            "spd": round(dp_spd, 4),
        })
        log(f"    ε={eps}: Acc={dp_acc:.4f}, F1={dp_f1:.4f}, SPD={dp_spd:.4f}")

    # ── Visualizations ───────────────────────────────────────────────
    _plot_privacy_accuracy_tradeoff(results, baseline_acc, baseline_f1)
    _plot_privacy_fairness_tradeoff(results, baseline_spd)
    _plot_combined_tradeoff(results, baseline_acc, baseline_spd)

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "baseline": {
            "accuracy": round(baseline_acc, 4),
            "f1_score": round(baseline_f1, 4),
            "spd": round(baseline_spd, 4),
        },
        "comparison": results,
        "dp_method": "diffprivlib" if dp_available else "simulated_laplace_noise",
    }
    save_json(summary, "privacy_results.json")

    _print_privacy_summary(summary)

    return summary


def _compute_spd(y_pred, sensitive, privileged=1, unprivileged=0):
    """Statistical Parity Difference for gender."""
    rate_priv = y_pred[sensitive == privileged].mean()
    rate_unpriv = y_pred[sensitive == unprivileged].mean()
    return rate_unpriv - rate_priv


def _simulate_dp_predictions(model, X_test, epsilon):
    """Simulate DP by adding calibrated noise to model probability outputs."""
    proba = model.predict_proba(X_test)[:, 1]
    noise_scale = 1.0 / epsilon
    rng = np.random.default_rng(42)
    noisy_proba = proba + rng.laplace(0, noise_scale, size=len(proba))
    noisy_proba = np.clip(noisy_proba, 0, 1)
    return (noisy_proba >= 0.5).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _plot_privacy_accuracy_tradeoff(results, baseline_acc, baseline_f1):
    """Privacy (epsilon) vs Accuracy/F1 tradeoff plot."""
    epsilons = [r["epsilon"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    f1_scores = [r["f1_score"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilons, accuracies, "o-", label="Accuracy (DP)", color="#3498db", linewidth=2)
    ax.plot(epsilons, f1_scores, "s--", label="F1 Score (DP)", color="#e74c3c", linewidth=2)
    ax.axhline(y=baseline_acc, color="#3498db", linestyle=":", alpha=0.6, label=f"Baseline Acc ({baseline_acc:.3f})")
    ax.axhline(y=baseline_f1, color="#e74c3c", linestyle=":", alpha=0.6, label=f"Baseline F1 ({baseline_f1:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (ε) — lower = more private")
    ax.set_ylabel("Score")
    ax.set_title("Privacy vs Accuracy Tradeoff – Differential Privacy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "privacy_accuracy_tradeoff.png")


def _plot_privacy_fairness_tradeoff(results, baseline_spd):
    """Privacy vs Fairness (SPD) tradeoff."""
    epsilons = [r["epsilon"] for r in results]
    spd_vals = [abs(r["spd"]) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilons, spd_vals, "D-", color="#9b59b6", linewidth=2, label="|SPD| (DP)")
    ax.axhline(y=abs(baseline_spd), color="#9b59b6", linestyle=":", alpha=0.6,
               label=f"Baseline |SPD| ({abs(baseline_spd):.3f})")
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="Fairness threshold (0.1)")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("|Statistical Parity Difference|")
    ax.set_title("Privacy vs Fairness Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "privacy_fairness_tradeoff.png")


def _plot_combined_tradeoff(results, baseline_acc, baseline_spd):
    """Combined: Accuracy vs Fairness at different privacy levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        ax.scatter(r["accuracy"], abs(r["spd"]), s=100,
                   label=f"ε={r['epsilon']}", zorder=3)
        ax.annotate(f"ε={r['epsilon']}", (r["accuracy"], abs(r["spd"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.scatter(baseline_acc, abs(baseline_spd), s=200, marker="*", color="black",
               label="Baseline (no DP)", zorder=4)
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("|Statistical Parity Difference|")
    ax.set_title("Accuracy vs Fairness at Different Privacy Levels")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "accuracy_vs_fairness_privacy.png")


def _print_privacy_summary(summary):
    """Pretty-print privacy analysis summary."""
    print(f"\n{'='*60}")
    print("PRIVACY PRESERVATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nMethod: {summary['dp_method']}")
    print(f"\nBaseline (no DP):")
    print(f"  Accuracy: {summary['baseline']['accuracy']}")
    print(f"  F1 Score: {summary['baseline']['f1_score']}")
    print(f"  SPD: {summary['baseline']['spd']}")
    print(f"\nDifferential Privacy Results:")
    print(f"  {'Epsilon':<10} {'Accuracy':<12} {'F1':<12} {'SPD':<12}")
    print(f"  {'-'*46}")
    for r in summary["comparison"]:
        print(f"  {r['epsilon']:<10} {r['accuracy']:<12} {r['f1_score']:<12} {r['spd']:<12}")


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    data = run_preprocessing()
    run_differential_privacy(data)
