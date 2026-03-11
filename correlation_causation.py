"""
correlation_causation.py - Correlation, Causation & Nudge Analysis
====================================================================
Advanced analytical module covering:

  1. CORRELATION ANALYSIS
     • Full feature correlation matrix (heatmap)
     • Sensitive attribute correlations with outcome
     • Feature–target correlation ranking
     • Multicollinearity check (VIF)

  2. CAUSATION ANALYSIS
     • Simpson's Paradox detection
     • Confounding variable identification
     • Stratified analysis (gender × department)
     • Proxy variable detection (features that act as proxies for protected attributes)

  3. NUDGE THEORY IN AI HIRING
     • Behavioral nudge framework for fairer decisions
     • Threshold nudging – adjusting decision boundaries per group
     • Counterfactual analysis – "what would change the decision?"
     • Fairness nudge impact assessment

These concepts bridge Units 1-4 of the Responsible AI syllabus.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils import log, save_plot, save_json, save_text_report, set_plot_style, DATASET_PATH


# ═════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def correlation_analysis(data):
    """Full correlation analysis between features, sensitive attributes, and target."""
    log("Running Correlation Analysis...")
    set_plot_style()

    df = data["dataframe"].copy()
    feature_names = data["feature_names"]

    # ── 1a. Full Correlation Matrix Heatmap ──────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.3, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix – All Numeric Features", fontsize=14)
    save_plot(fig, "correlation_matrix_full.png")

    # ── 1b. Target Correlation Ranking ───────────────────────────────
    if "Attrition" in numeric_cols:
        target_corr = corr_matrix["Attrition"].drop("Attrition").sort_values(key=abs, ascending=False)
    else:
        target_corr = pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_corr = target_corr.head(20)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_corr.values]
    ax.barh(range(len(top_corr)), top_corr.values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index)
    ax.set_xlabel("Pearson Correlation with Attrition (Hiring Decision)")
    ax.set_title("Top 20 Feature Correlations with Hiring Outcome")
    ax.axvline(x=0, color="black", linewidth=0.8)
    for i, v in enumerate(top_corr.values):
        ax.text(v + 0.005 if v > 0 else v - 0.005, i, f"{v:.3f}",
                va="center", ha="left" if v > 0 else "right", fontsize=9)
    save_plot(fig, "target_correlation_ranking.png")

    # ── 1c. Sensitive Attribute Correlations ─────────────────────────
    sensitive_cols = ["Gender_Binary", "Age"]
    sensitive_corrs = {}
    for col in sensitive_cols:
        if col in corr_matrix.columns:
            corrs = corr_matrix[col].drop(col, errors="ignore").sort_values(key=abs, ascending=False)
            sensitive_corrs[col] = corrs.head(10).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col in zip(axes, sensitive_cols):
        if col in corr_matrix.columns:
            corrs = corr_matrix[col].drop(col, errors="ignore").sort_values(key=abs, ascending=False).head(10)
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in corrs.values]
            ax.barh(range(len(corrs)), corrs.values, color=colors, edgecolor="black")
            ax.set_yticks(range(len(corrs)))
            ax.set_yticklabels(corrs.index)
            ax.set_xlabel("Correlation")
            ax.set_title(f"Top Correlations with {col}")
            ax.axvline(x=0, color="black", linewidth=0.8)
    plt.suptitle("Sensitive Attribute Correlations – Proxy Detection", fontsize=14)
    plt.tight_layout()
    save_plot(fig, "sensitive_attribute_correlations.png")

    # ── 1d. Multicollinearity (VIF approximation) ───────────────────
    log("Computing Variance Inflation Factors (VIF)...")
    from sklearn.linear_model import LinearRegression

    X_numeric = df[feature_names].select_dtypes(include=[np.number])
    # Use a subset to keep it fast
    cols_for_vif = X_numeric.columns[:20]  # Top 20 features
    vif_data = []
    for col in cols_for_vif:
        others = [c for c in cols_for_vif if c != col]
        if len(others) == 0:
            continue
        lr = LinearRegression()
        lr.fit(X_numeric[others], X_numeric[col])
        r_squared = lr.score(X_numeric[others], X_numeric[col])
        vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        vif_data.append({"Feature": col, "VIF": round(vif, 2), "R²": round(r_squared, 4)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71" for v in vif_df["VIF"]]
    ax.barh(range(len(vif_df)), vif_df["VIF"].values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(vif_df)))
    ax.set_yticklabels(vif_df["Feature"].values)
    ax.axvline(x=5, color="orange", linestyle="--", label="Moderate (VIF=5)")
    ax.axvline(x=10, color="red", linestyle="--", label="High (VIF=10)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("Multicollinearity Check (VIF) – Feature Redundancy")
    ax.legend()
    save_plot(fig, "vif_multicollinearity.png")

    print("\nCorrelation Analysis Complete:")
    print(f"  Top correlated with Attrition: {target_corr.index[0]} (r={target_corr.iloc[0]:.3f})")
    print(f"  High VIF features (>10): {list(vif_df[vif_df['VIF'] > 10]['Feature'])}")

    return {
        "target_correlations": target_corr.to_dict(),
        "sensitive_correlations": sensitive_corrs,
        "vif": vif_df.to_dict(orient="records"),
        "correlation_matrix": corr_matrix,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. CAUSATION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def causation_analysis(data):
    """
    Analyze causal relationships vs mere correlations.
    Includes Simpson's Paradox detection and proxy variable analysis.
    """
    log("Running Causation Analysis...")
    set_plot_style()

    df = data["dataframe"].copy()

    results = {}

    # ── 2a. Simpson's Paradox Detection ──────────────────────────────
    log("  Checking for Simpson's Paradox...")
    simpsons = _detect_simpsons_paradox(df)
    results["simpsons_paradox"] = simpsons

    # ── 2b. Stratified Analysis: Gender × Department ─────────────────
    log("  Stratified analysis: Gender × Department...")
    stratified = _stratified_analysis(df)
    results["stratified_analysis"] = stratified

    # ── 2c. Proxy Variable Detection ─────────────────────────────────
    log("  Detecting proxy variables for protected attributes...")
    proxies = _detect_proxy_variables(df, data["feature_names"])
    results["proxy_variables"] = proxies

    # ── 2d. Confounding Variable Analysis ────────────────────────────
    log("  Confounding variable analysis...")
    confounders = _confounding_analysis(df, data)
    results["confounders"] = confounders

    return results


def _detect_simpsons_paradox(df):
    """
    Simpson's Paradox: A trend that appears across groups reverses when
    the groups are combined. Check if gender-attrition relationship
    reverses when stratified by department.
    """
    # Overall attrition by gender
    overall = df.groupby("Gender_Binary")["Attrition"].mean()
    overall_diff = overall.get(0, 0) - overall.get(1, 0)  # Female - Male
    overall_direction = "Female > Male" if overall_diff > 0 else "Male > Female"

    # By department
    dept_col = "Department" if "Department" in df.columns else None
    paradox_found = False
    dept_results = []

    if dept_col is not None:
        for dept in df[dept_col].unique():
            subset = df[df[dept_col] == dept]
            dept_rates = subset.groupby("Gender_Binary")["Attrition"].mean()
            if len(dept_rates) == 2:
                diff = dept_rates.get(0, 0) - dept_rates.get(1, 0)
                direction = "Female > Male" if diff > 0 else "Male > Female"
                dept_results.append({
                    "department": int(dept),
                    "female_rate": round(dept_rates.get(0, 0), 4),
                    "male_rate": round(dept_rates.get(1, 0), 4),
                    "direction": direction,
                })
                if direction != overall_direction:
                    paradox_found = True

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    axes[0].bar(["Female", "Male"], [overall.get(0, 0), overall.get(1, 0)],
                color=["#e74c3c", "#3498db"], edgecolor="black")
    axes[0].set_title("Overall Attrition Rate by Gender")
    axes[0].set_ylabel("Attrition Rate")
    for i, v in enumerate([overall.get(0, 0), overall.get(1, 0)]):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)

    # By department
    if dept_results:
        depts = [f"Dept {d['department']}" for d in dept_results]
        female_bars = [d["female_rate"] for d in dept_results]
        male_bars = [d["male_rate"] for d in dept_results]

        x = np.arange(len(depts))
        width = 0.35
        axes[1].bar(x - width/2, female_bars, width, label="Female", color="#e74c3c")
        axes[1].bar(x + width/2, male_bars, width, label="Male", color="#3498db")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(depts)
        axes[1].set_title("Attrition by Gender × Department (Simpson's Paradox Check)")
        axes[1].set_ylabel("Attrition Rate")
        axes[1].legend()

    paradox_label = "DETECTED ⚠️" if paradox_found else "Not detected ✓"
    fig.suptitle(f"Simpson's Paradox Analysis — {paradox_label}", fontsize=14)
    save_plot(fig, "simpsons_paradox.png")

    summary = {
        "overall_direction": overall_direction,
        "paradox_detected": paradox_found,
        "department_breakdown": dept_results,
    }

    status = "DETECTED" if paradox_found else "NOT DETECTED"
    print(f"\n  Simpson's Paradox: {status}")
    print(f"  Overall: {overall_direction} (Female={overall.get(0,0):.3f}, Male={overall.get(1,0):.3f})")

    return summary


def _stratified_analysis(df):
    """Stratified attrition rates by Gender × JobLevel to uncover hidden patterns."""
    if "JobLevel" not in df.columns:
        return {}

    pivot = df.pivot_table(values="Attrition", index="JobLevel",
                           columns="Gender_Binary", aggfunc="mean")
    pivot.columns = ["Female", "Male"]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"], edgecolor="black")
    ax.set_title("Stratified Analysis: Attrition Rate by Gender × Job Level")
    ax.set_xlabel("Job Level")
    ax.set_ylabel("Attrition Rate")
    ax.legend(title="Gender")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    save_plot(fig, "stratified_gender_joblevel.png")

    return pivot.to_dict()


def _detect_proxy_variables(df, feature_names):
    """
    Proxy variables: Features that are highly correlated with protected
    attributes and could serve as indirect discrimination vectors.
    E.g., 'MaritalStatus' may proxy for Age; 'JobRole' may proxy for Gender.
    """
    sensitive = ["Gender_Binary", "Age"]
    numeric_df = df[feature_names].select_dtypes(include=[np.number])

    proxy_results = {}
    proxy_warnings = []

    for sens in sensitive:
        if sens not in numeric_df.columns:
            continue
        correlations = numeric_df.corr()[sens].drop(sens, errors="ignore")
        # Threshold: |r| > 0.3 = potential proxy
        potential_proxies = correlations[correlations.abs() > 0.3].sort_values(key=abs, ascending=False)

        proxy_results[sens] = {
            feat: round(float(corr), 4)
            for feat, corr in potential_proxies.items()
        }

        for feat, corr in potential_proxies.items():
            proxy_warnings.append({
                "protected_attribute": sens,
                "proxy_feature": feat,
                "correlation": round(float(corr), 4),
                "risk": "HIGH" if abs(corr) > 0.5 else "MEDIUM",
            })

    # Visualize proxy detection
    if proxy_warnings:
        fig, ax = plt.subplots(figsize=(10, max(4, len(proxy_warnings) * 0.5)))
        labels = [f"{p['proxy_feature']} → {p['protected_attribute']}" for p in proxy_warnings]
        values = [p["correlation"] for p in proxy_warnings]
        colors = ["#e74c3c" if abs(v) > 0.5 else "#f39c12" for v in values]
        ax.barh(range(len(labels)), values, color=colors, edgecolor="black")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title("Proxy Variable Detection – Indirect Discrimination Risk")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="|r|>0.5 HIGH")
        ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=0.3, color="orange", linestyle="--", alpha=0.5, label="|r|>0.3 MEDIUM")
        ax.axvline(x=-0.3, color="orange", linestyle="--", alpha=0.5)
        ax.legend()
        save_plot(fig, "proxy_variables.png")

    print(f"\n  Proxy variables detected: {len(proxy_warnings)}")
    for p in proxy_warnings:
        print(f"    {p['proxy_feature']} → {p['protected_attribute']} "
              f"(r={p['correlation']:.3f}, risk={p['risk']})")

    return {"proxies": proxy_warnings, "details": proxy_results}


def _confounding_analysis(df, data):
    """
    Check if the relationship between sensitive attributes and outcome
    changes when controlling for potential confounders (e.g., JobLevel,
    MonthlyIncome, TotalWorkingYears).
    """
    confounders = ["JobLevel", "MonthlyIncome", "TotalWorkingYears"]
    available = [c for c in confounders if c in df.columns]

    results = {}

    # Raw correlation: Gender → Attrition
    raw_corr = df["Gender_Binary"].corr(df["Attrition"])
    results["raw_gender_attrition_corr"] = round(raw_corr, 4)

    # After controlling for confounders (partial correlation approximation)
    if available:
        from sklearn.linear_model import LinearRegression

        # Regress out confounders from both Gender and Attrition
        X_conf = df[available].values
        lr1 = LinearRegression().fit(X_conf, df["Gender_Binary"])
        lr2 = LinearRegression().fit(X_conf, df["Attrition"])

        gender_residual = df["Gender_Binary"] - lr1.predict(X_conf)
        attrition_residual = df["Attrition"] - lr2.predict(X_conf)

        partial_corr = np.corrcoef(gender_residual, attrition_residual)[0, 1]
        results["partial_gender_attrition_corr"] = round(float(partial_corr), 4)
        results["confounders_controlled"] = available

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(["Raw Correlation", "Partial Correlation (controlled)"],
                    [raw_corr, partial_corr],
                    color=["#e74c3c", "#2ecc71"], edgecolor="black")
        axes[0].set_ylabel("Correlation Coefficient")
        axes[0].set_title("Gender → Attrition: Raw vs Controlled")
        axes[0].axhline(y=0, color="black", linewidth=0.8)
        for i, v in enumerate([raw_corr, partial_corr]):
            axes[0].text(i, v + 0.002 if v > 0 else v - 0.01, f"{v:.4f}", ha="center")

        # Change magnitude
        change = abs(raw_corr) - abs(partial_corr)
        pct_change = (change / abs(raw_corr)) * 100 if raw_corr != 0 else 0
        axes[1].bar(available,
                    [df[c].corr(df["Attrition"]) for c in available],
                    color="#9b59b6", edgecolor="black")
        axes[1].set_ylabel("Correlation with Attrition")
        axes[1].set_title("Confounder Correlations with Outcome")
        axes[1].axhline(y=0, color="black", linewidth=0.8)

        fig.suptitle(f"Confounding Analysis – Correlation change: {pct_change:.1f}%", fontsize=13)
        save_plot(fig, "confounding_analysis.png")

        print(f"\n  Confounding Analysis:")
        print(f"    Raw Gender→Attrition correlation: {raw_corr:.4f}")
        print(f"    Partial correlation (controlled): {partial_corr:.4f}")
        print(f"    Change: {pct_change:.1f}% — {'Significant confounding' if pct_change > 30 else 'Mild confounding'}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. NUDGE THEORY & COUNTERFACTUAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def nudge_analysis(models, data):
    """
    Nudge Theory in AI Hiring:
      - Threshold nudging: adjust decision boundaries for fairness
      - Counterfactual explanations: what minimal change flips the decision?
      - Fairness nudge impact assessment
    """
    log("Running Nudge & Counterfactual Analysis...")
    set_plot_style()

    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    gender_test = data["sensitive_test"]["Gender"]

    results = {}

    # ── 3a. Threshold Nudging ────────────────────────────────────────
    log("  Threshold nudging analysis...")
    threshold_results = _threshold_nudge(models, X_test, y_test, gender_test)
    results["threshold_nudge"] = threshold_results

    # ── 3b. Counterfactual Explanations ──────────────────────────────
    log("  Generating counterfactual explanations...")
    counterfactuals = _counterfactual_analysis(models, data)
    results["counterfactuals"] = counterfactuals

    # ── 3c. Nudge Impact Summary ─────────────────────────────────────
    log("  Computing nudge impact on fairness...")
    impact = _nudge_impact_summary(threshold_results)
    results["nudge_impact"] = impact

    return results


def _threshold_nudge(models, X_test, y_test, gender_test):
    """
    Nudge: Instead of a fixed 0.5 threshold, find the optimal threshold
    per group that balances accuracy and fairness.
    """
    # Use Logistic Regression (has predict_proba)
    lr = models.get("Logistic Regression")
    if lr is None:
        return {}

    proba = lr.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)

    threshold_data = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc = accuracy_score(y_test, preds)

        # SPD by gender
        rate_female = preds[gender_test == 0].mean()
        rate_male = preds[gender_test == 1].mean()
        spd = rate_female - rate_male

        threshold_data.append({
            "threshold": round(t, 2),
            "accuracy": round(acc, 4),
            "spd": round(spd, 4),
            "abs_spd": round(abs(spd), 4),
            "female_positive_rate": round(rate_female, 4),
            "male_positive_rate": round(rate_male, 4),
        })

    # Find optimal threshold (minimize |SPD| while keeping accuracy within 2% of max)
    df_t = pd.DataFrame(threshold_data)
    max_acc = df_t["accuracy"].max()
    acceptable = df_t[df_t["accuracy"] >= max_acc - 0.02]
    if len(acceptable) > 0:
        optimal_idx = acceptable["abs_spd"].idxmin()
        optimal = acceptable.loc[optimal_idx]
    else:
        optimal = df_t.loc[df_t["abs_spd"].idxmin()]

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df_t["threshold"], df_t["accuracy"], "o-", color="#3498db",
                 label="Accuracy", linewidth=2)
    axes[0].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
    axes[0].axvline(x=optimal["threshold"], color="green", linestyle="--",
                    label=f"Nudged ({optimal['threshold']})")
    axes[0].set_xlabel("Decision Threshold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Threshold vs Accuracy")
    axes[0].legend()

    axes[1].plot(df_t["threshold"], df_t["abs_spd"], "s-", color="#e74c3c",
                 label="|SPD|", linewidth=2)
    axes[1].axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="Fairness threshold")
    axes[1].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
    axes[1].axvline(x=optimal["threshold"], color="green", linestyle="--",
                    label=f"Nudged ({optimal['threshold']})")
    axes[1].set_xlabel("Decision Threshold")
    axes[1].set_ylabel("|Statistical Parity Difference|")
    axes[1].set_title("Threshold vs Fairness (|SPD|)")
    axes[1].legend()

    fig.suptitle("Nudge Theory: Threshold Adjustment for Fairer Hiring", fontsize=14)
    save_plot(fig, "threshold_nudge.png")

    # Default vs Nudged comparison
    default = df_t[df_t["threshold"] == 0.5].iloc[0] if 0.5 in df_t["threshold"].values else df_t.iloc[len(df_t)//2]

    fig2, ax = plt.subplots(figsize=(10, 5))
    labels = ["Default (t=0.5)", f"Nudged (t={optimal['threshold']})"]
    acc_vals = [default["accuracy"], optimal["accuracy"]]
    spd_vals = [default["abs_spd"], optimal["abs_spd"]]

    x = np.arange(2)
    width = 0.3
    ax.bar(x - width/2, acc_vals, width, label="Accuracy", color="#3498db", edgecolor="black")
    ax.bar(x + width/2, spd_vals, width, label="|SPD| (lower=fairer)", color="#e74c3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Nudge Impact: Default vs Optimized Threshold")
    ax.legend()
    for i, (a, s) in enumerate(zip(acc_vals, spd_vals)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=10)
        ax.text(i + width/2, s + 0.01, f"{s:.3f}", ha="center", fontsize=10)
    save_plot(fig2, "nudge_comparison.png")

    print(f"\n  Threshold Nudge Results:")
    print(f"    Default (t=0.5): Acc={default['accuracy']:.3f}, |SPD|={default['abs_spd']:.3f}")
    print(f"    Nudged  (t={optimal['threshold']}): Acc={optimal['accuracy']:.3f}, |SPD|={optimal['abs_spd']:.3f}")

    return {
        "all_thresholds": threshold_data,
        "optimal_threshold": round(float(optimal["threshold"]), 2),
        "default_accuracy": round(float(default["accuracy"]), 4),
        "nudged_accuracy": round(float(optimal["accuracy"]), 4),
        "default_spd": round(float(default["abs_spd"]), 4),
        "nudged_spd": round(float(optimal["abs_spd"]), 4),
    }


def _counterfactual_analysis(models, data):
    """
    Counterfactual Explanation: For candidates predicted as "Not Hired",
    find the minimum feature changes that would flip the decision.
    """
    rf = models.get("Random Forest")
    if rf is None:
        return []

    X_test = data["X_test"]
    feature_names = data["feature_names"]
    y_pred = rf.predict(X_test)

    # Find candidates predicted as "Not Hired" (1)
    not_hired_idx = np.where(y_pred == 1)[0]
    if len(not_hired_idx) == 0:
        return []

    counterfactuals = []

    # Analyze up to 5 candidates
    for idx in not_hired_idx[:5]:
        candidate = X_test.iloc[idx].copy()
        original_pred = rf.predict(candidate.values.reshape(1, -1))[0]

        changes = []
        # Try changing each feature slightly to flip the prediction
        for feat_idx, feat_name in enumerate(feature_names):
            for delta in [-0.5, -1.0, 0.5, 1.0]:
                modified = candidate.copy()
                modified.iloc[feat_idx] += delta
                new_pred = rf.predict(modified.values.reshape(1, -1))[0]
                if new_pred != original_pred:
                    changes.append({
                        "feature": feat_name,
                        "change": round(delta, 2),
                        "direction": "increase" if delta > 0 else "decrease",
                    })
                    break

        if changes:
            counterfactuals.append({
                "candidate_index": int(idx),
                "original_prediction": "Not Hired",
                "changes_to_flip": changes[:5],  # Top 5 actionable changes
            })

    # Visualize counterfactuals for first candidate
    if counterfactuals:
        cf = counterfactuals[0]
        changes = cf["changes_to_flip"]

        fig, ax = plt.subplots(figsize=(10, max(3, len(changes) * 0.6)))
        features = [c["feature"] for c in changes]
        deltas = [c["change"] for c in changes]
        colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
        ax.barh(range(len(features)), deltas, color=colors, edgecolor="black")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Change Needed (standardized units)")
        ax.set_title(f"Counterfactual: What Would Change the Decision?\n(Candidate #{cf['candidate_index']})")
        ax.axvline(x=0, color="black", linewidth=0.8)
        save_plot(fig, "counterfactual_explanation.png")

        print(f"\n  Counterfactual Example (Candidate #{cf['candidate_index']}):")
        for c in changes[:5]:
            print(f"    {c['direction'].capitalize()} {c['feature']} by {abs(c['change'])}"
                  f" → flips to 'Hired'")

    return counterfactuals


def _nudge_impact_summary(threshold_results):
    """Summarize the impact of nudging on the hiring system."""
    if not threshold_results:
        return {}

    acc_change = threshold_results.get("nudged_accuracy", 0) - threshold_results.get("default_accuracy", 0)
    spd_change = threshold_results.get("default_spd", 0) - threshold_results.get("nudged_spd", 0)

    impact = {
        "accuracy_change": round(acc_change, 4),
        "fairness_improvement": round(spd_change, 4),
        "optimal_threshold": threshold_results.get("optimal_threshold", 0.5),
        "recommendation": "",
    }

    if spd_change > 0 and abs(acc_change) < 0.02:
        impact["recommendation"] = (
            "RECOMMENDED: Threshold nudging improves fairness with minimal accuracy loss. "
            f"Nudge threshold from 0.5 to {impact['optimal_threshold']}."
        )
    elif spd_change > 0:
        impact["recommendation"] = (
            "TRADEOFF: Threshold nudging improves fairness but reduces accuracy. "
            "Consider whether fairness gain justifies the accuracy cost."
        )
    else:
        impact["recommendation"] = (
            "DEFAULT OK: The default threshold is already near-optimal for fairness."
        )

    print(f"\n  Nudge Impact Summary:")
    print(f"    Accuracy change: {acc_change:+.4f}")
    print(f"    Fairness improvement (|SPD| reduction): {spd_change:+.4f}")
    print(f"    {impact['recommendation']}")

    return impact


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_correlation_causation_nudge(models, data):
    """Execute the full correlation, causation, and nudge analysis pipeline."""
    log("=" * 60)
    log("CORRELATION, CAUSATION & NUDGE ANALYSIS")
    log("=" * 60)

    # 1. Correlation
    corr_results = correlation_analysis(data)

    # 2. Causation
    causation_results = causation_analysis(data)

    # 3. Nudge
    nudge_results = nudge_analysis(models, data)

    # Save combined results
    combined = {
        "correlation": {
            "target_correlations": {k: round(float(v), 4) for k, v in corr_results["target_correlations"].items()},
            "sensitive_correlations": corr_results["sensitive_correlations"],
            "vif": corr_results["vif"],
        },
        "causation": {
            "simpsons_paradox": causation_results.get("simpsons_paradox", {}),
            "proxy_variables": causation_results.get("proxy_variables", {}).get("proxies", []),
            "confounders": causation_results.get("confounders", {}),
        },
        "nudge": nudge_results,
    }
    save_json(combined, "correlation_causation_nudge.json")

    # Generate text report
    _generate_report(corr_results, causation_results, nudge_results)

    return combined


def _generate_report(corr_results, causation_results, nudge_results):
    """Generate a text report summarizing all findings."""
    lines = []
    lines.append("=" * 70)
    lines.append("  CORRELATION, CAUSATION & NUDGE ANALYSIS REPORT")
    lines.append("=" * 70)

    # Correlation
    lines.append("\n1. CORRELATION ANALYSIS")
    lines.append("-" * 40)
    tc = corr_results.get("target_correlations", {})
    top5 = list(tc.items())[:5]
    lines.append("  Top 5 features correlated with Hiring Decision:")
    for feat, corr in top5:
        lines.append(f"    {feat}: r = {corr:.4f}")

    vif = corr_results.get("vif", [])
    high_vif = [v for v in vif if v["VIF"] > 10]
    lines.append(f"\n  Multicollinearity: {len(high_vif)} features with VIF > 10")

    # Causation
    lines.append("\n\n2. CAUSATION ANALYSIS")
    lines.append("-" * 40)
    sp = causation_results.get("simpsons_paradox", {})
    paradox = sp.get("paradox_detected", False)
    lines.append(f"  Simpson's Paradox: {'DETECTED ⚠️' if paradox else 'Not detected ✓'}")

    proxies = causation_results.get("proxy_variables", {}).get("proxies", [])
    lines.append(f"  Proxy variables detected: {len(proxies)}")
    for p in proxies:
        lines.append(f"    {p['proxy_feature']} → {p['protected_attribute']} (r={p['correlation']:.3f})")

    conf = causation_results.get("confounders", {})
    lines.append(f"\n  Confounding Analysis:")
    lines.append(f"    Raw Gender→Attrition: r = {conf.get('raw_gender_attrition_corr', 'N/A')}")
    lines.append(f"    After controlling: r = {conf.get('partial_gender_attrition_corr', 'N/A')}")

    # Nudge
    lines.append("\n\n3. NUDGE ANALYSIS")
    lines.append("-" * 40)
    tn = nudge_results.get("threshold_nudge", {})
    lines.append(f"  Optimal threshold: {tn.get('optimal_threshold', 'N/A')}")
    lines.append(f"  Default accuracy: {tn.get('default_accuracy', 'N/A')}")
    lines.append(f"  Nudged accuracy: {tn.get('nudged_accuracy', 'N/A')}")
    lines.append(f"  Default |SPD|: {tn.get('default_spd', 'N/A')}")
    lines.append(f"  Nudged |SPD|: {tn.get('nudged_spd', 'N/A')}")

    impact = nudge_results.get("nudge_impact", {})
    lines.append(f"\n  Recommendation: {impact.get('recommendation', 'N/A')}")

    cfs = nudge_results.get("counterfactuals", [])
    if cfs:
        lines.append(f"\n  Counterfactual examples generated: {len(cfs)}")
        cf = cfs[0]
        lines.append(f"  Example (Candidate #{cf['candidate_index']}):")
        for c in cf.get("changes_to_flip", [])[:3]:
            lines.append(f"    {c['direction']} {c['feature']} → flips to 'Hired'")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)
    save_text_report(report, "correlation_causation_report.txt")


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from train_models import train_all_models
    data = run_preprocessing()
    models = train_all_models(data)
    run_correlation_causation_nudge(models, data)
