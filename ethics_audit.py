"""
ethics_audit.py - Ethics & Accountability Audit Module
========================================================
Unit 4 – Ethics and Accountability

Automatically generates an AI Audit Report that includes:
  • Bias findings summary
  • Ethical risk assessment
  • Model transparency evaluation
  • Fairness evaluation tables
  • Written explanations & recommendations
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import log, save_plot, save_text_report, save_json, set_plot_style, REPORTS_DIR


# ═════════════════════════════════════════════════════════════════════════════
# ETHICAL RISK ASSESSMENT
# ═════════════════════════════════════════════════════════════════════════════

ETHICAL_PRINCIPLES = {
    "Fairness": "The model should not discriminate based on protected attributes.",
    "Transparency": "Model decisions should be explainable and interpretable.",
    "Accountability": "Clear ownership of AI decisions and their consequences.",
    "Privacy": "Personal data must be protected and minimally used.",
    "Beneficence": "The AI system should actively benefit candidates and society.",
    "Non-maleficence": "The system must not cause harm to any group.",
}


def assess_ethical_risks(fairness_results, model_metrics):
    """Evaluate ethical risks based on fairness metrics and model behavior."""
    log("Assessing ethical risks...")
    risks = []

    for model_name, fair_res in fairness_results.items():
        gender = fair_res.get("gender", {})
        di = gender.get("Disparate Impact Ratio", 1.0)
        spd = gender.get("Statistical Parity Difference", 0.0)

        # Risk: Disparate Impact below 80%
        if di < 0.8:
            risks.append({
                "Model": model_name,
                "Risk": "HIGH",
                "Category": "Fairness – Disparate Impact",
                "Detail": f"DI ratio = {di:.3f} (below 0.8 threshold). "
                          f"The model shows adverse impact against the unprivileged gender group.",
                "Recommendation": "Apply bias mitigation (reweighting, threshold adjustment, or resampling).",
            })
        elif di < 1.0:
            risks.append({
                "Model": model_name,
                "Risk": "MEDIUM",
                "Category": "Fairness – Disparate Impact",
                "Detail": f"DI ratio = {di:.3f}. Mild imbalance detected.",
                "Recommendation": "Monitor in production; consider fairness constraints during training.",
            })
        else:
            risks.append({
                "Model": model_name,
                "Risk": "LOW",
                "Category": "Fairness – Disparate Impact",
                "Detail": f"DI ratio = {di:.3f}. Within acceptable range.",
                "Recommendation": "Continue monitoring.",
            })

        # Risk: Statistical Parity
        if abs(spd) > 0.1:
            risks.append({
                "Model": model_name,
                "Risk": "HIGH",
                "Category": "Fairness – Statistical Parity",
                "Detail": f"SPD = {spd:.3f}. Significant difference in positive prediction rates.",
                "Recommendation": "Investigate root cause in training data; apply fairness constraints.",
            })

    # Risk: Model transparency
    for model_name in fairness_results.keys():
        if model_name == "Decision Tree":
            risks.append({
                "Model": model_name,
                "Risk": "LOW",
                "Category": "Transparency",
                "Detail": "Decision Trees are inherently interpretable.",
                "Recommendation": "Good choice for high-stakes decisions.",
            })
        elif model_name == "Logistic Regression":
            risks.append({
                "Model": model_name,
                "Risk": "LOW",
                "Category": "Transparency",
                "Detail": "Logistic Regression coefficients are directly interpretable.",
                "Recommendation": "Document feature coefficients.",
            })
        else:
            risks.append({
                "Model": model_name,
                "Risk": "MEDIUM",
                "Category": "Transparency",
                "Detail": "Ensemble models (Random Forest) are not natively interpretable.",
                "Recommendation": "Use SHAP or LIME for post-hoc explanations.",
            })

    return risks


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_audit_report(fairness_results, model_metrics, risks, privacy_results=None):
    """Generate a comprehensive text-based AI Audit Report."""
    log("Generating Ethics & Accountability Audit Report...")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 70)
    lines.append("  RESPONSIBLE AI AUDIT REPORT")
    lines.append("  Bias Detection & Responsible AI in AI Hiring Systems")
    lines.append(f"  Generated: {timestamp}")
    lines.append("=" * 70)

    # ── Section 1: Executive Summary ─────────────────────────────────
    lines.append("\n1. EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append("This audit evaluates AI models used for hiring decisions (attrition")
    lines.append("prediction) for fairness, transparency, and ethical compliance.")
    lines.append(f"Models evaluated: {', '.join(fairness_results.keys())}")
    high_risks = [r for r in risks if r["Risk"] == "HIGH"]
    lines.append(f"High-risk findings: {len(high_risks)}")
    lines.append(f"Total risk items: {len(risks)}")

    # ── Section 2: Model Performance ─────────────────────────────────
    lines.append("\n\n2. MODEL PERFORMANCE")
    lines.append("-" * 40)
    if model_metrics:
        for model_name, metrics in model_metrics.items():
            if isinstance(metrics, dict) and "metrics" in metrics:
                m = metrics["metrics"]
            else:
                m = metrics
            lines.append(f"\n  {model_name}:")
            for k, v in m.items():
                lines.append(f"    {k}: {v}")

    # ── Section 3: Fairness Evaluation ───────────────────────────────
    lines.append("\n\n3. FAIRNESS EVALUATION")
    lines.append("-" * 40)
    for model_name, res in fairness_results.items():
        lines.append(f"\n  {model_name}:")
        for attr, metrics in res.items():
            lines.append(f"    [{attr}]")
            for k, v in metrics.items():
                lines.append(f"      {k}: {v}")

    # ── Section 4: Ethical Risk Assessment ────────────────────────────
    lines.append("\n\n4. ETHICAL RISK ASSESSMENT")
    lines.append("-" * 40)
    lines.append("\nEthical Principles Applied:")
    for principle, description in ETHICAL_PRINCIPLES.items():
        lines.append(f"  • {principle}: {description}")

    lines.append("\nIdentified Risks:")
    for i, risk in enumerate(risks, 1):
        lines.append(f"\n  Risk #{i}:")
        lines.append(f"    Model: {risk['Model']}")
        lines.append(f"    Severity: {risk['Risk']}")
        lines.append(f"    Category: {risk['Category']}")
        lines.append(f"    Detail: {risk['Detail']}")
        lines.append(f"    Recommendation: {risk['Recommendation']}")

    # ── Section 5: Privacy Assessment ────────────────────────────────
    if privacy_results:
        lines.append("\n\n5. PRIVACY ASSESSMENT")
        lines.append("-" * 40)
        lines.append("Differential Privacy was applied to evaluate the accuracy–privacy tradeoff.")
        if "comparison" in privacy_results:
            for item in privacy_results["comparison"]:
                lines.append(f"\n  Epsilon={item.get('epsilon', 'N/A')}:")
                lines.append(f"    Accuracy: {item.get('accuracy', 'N/A')}")

    # ── Section 6: Recommendations ───────────────────────────────────
    lines.append("\n\n6. RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append("  1. Implement bias mitigation techniques before deployment.")
    lines.append("  2. Use interpretable models (Decision Tree, Logistic Regression)")
    lines.append("     for high-stakes hiring decisions.")
    lines.append("  3. Apply differential privacy to protect candidate data.")
    lines.append("  4. Conduct regular fairness audits on production models.")
    lines.append("  5. Establish a human-in-the-loop review for edge cases.")
    lines.append("  6. Document all model decisions for accountability.")
    lines.append("  7. Provide candidates with explanations for decisions.")

    # ── Section 7: Compliance ────────────────────────────────────────
    lines.append("\n\n7. REGULATORY COMPLIANCE NOTES")
    lines.append("-" * 40)
    lines.append("  • EU AI Act: High-risk category (employment/recruitment)")
    lines.append("  • US EEOC: Must satisfy 4/5ths (80%) rule for disparate impact")
    lines.append("  • GDPR Article 22: Right to explanation for automated decisions")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF AUDIT REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    # Save
    save_text_report(report_text, "ethics_audit_report.txt")
    return report_text


# ═════════════════════════════════════════════════════════════════════════════
# RISK VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_risk_summary(risks):
    """Visualize risk distribution."""
    set_plot_style()
    df = pd.DataFrame(risks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Risk severity distribution
    severity_counts = df["Risk"].value_counts()
    colors_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    colors = [colors_map.get(s, "gray") for s in severity_counts.index]
    axes[0].bar(severity_counts.index, severity_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Risk Severity Distribution")
    axes[0].set_ylabel("Count")

    # Risk by category
    cat_counts = df.groupby(["Category", "Risk"]).size().unstack(fill_value=0)
    cat_counts.plot(kind="barh", stacked=True, ax=axes[1],
                    color=[colors_map.get(c, "gray") for c in cat_counts.columns])
    axes[1].set_title("Risks by Category")
    axes[1].set_xlabel("Count")

    fig.suptitle("Ethical Risk Assessment Summary", fontsize=14)
    save_plot(fig, "risk_summary.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_ethics_audit(fairness_results, model_metrics, privacy_results=None):
    """Full ethics audit pipeline."""
    log("=" * 60)
    log("ETHICS & ACCOUNTABILITY AUDIT")
    log("=" * 60)

    risks = assess_ethical_risks(fairness_results, model_metrics)
    report = generate_audit_report(fairness_results, model_metrics, risks, privacy_results)
    plot_risk_summary(risks)

    save_json(risks, "ethical_risks.json")

    return {
        "risks": risks,
        "report": report,
    }


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from train_models import train_all_models, evaluate_models
    from fairness_analysis import analyze_fairness

    data = run_preprocessing()
    models = train_all_models(data)
    model_metrics = evaluate_models(models, data)
    fairness_results = analyze_fairness(models, data)
    run_ethics_audit(fairness_results, model_metrics)
