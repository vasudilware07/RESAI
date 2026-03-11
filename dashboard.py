"""
dashboard.py - Responsible AI Dashboard (Streamlit)
=====================================================
Unit 6 – Case Study: Interactive Dashboard

A dynamic dashboard showing:
  • Dataset overview & statistics
  • Model accuracy comparison
  • Bias detection results
  • Fairness metrics visualization
  • SHAP explanations
  • Ethical audit summary
  • Privacy vs accuracy tradeoffs

Run with:  streamlit run dashboard.py
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")


def load_json(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_image(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Responsible AI Hiring Dashboard",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("⚖️ Responsible AI – Bias Detection in AI Hiring Systems")
    st.markdown("**Academic Project** | IBM HR Analytics Dataset | Fairness · Explainability · Privacy")
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", [
        "📊 Dataset Overview",
        "🤖 Model Performance",
        "⚖️ Fairness & Bias",
        "🔍 Explainability",
        "� Correlation & Causation",
        "�📋 Ethics Audit",
        "🔒 Privacy Analysis",
        "🏗️ System Architecture",
    ])

    # Load data
    df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    model_metrics = load_json("model_metrics.json")
    fairness_results = load_json("fairness_results.json")
    privacy_results = load_json("privacy_results.json")
    ethical_risks = load_json("ethical_risks.json")
    ccn_results = load_json("correlation_causation_nudge.json")

    # ── PAGE: Dataset Overview ───────────────────────────────────────
    if page == "📊 Dataset Overview":
        st.header("📊 Dataset Overview")

        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Employees", len(df))
            col2.metric("Features", df.shape[1])
            attrition_rate = (df["Attrition"] == "Yes").mean() * 100
            col3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
            col4.metric("Gender Split", f"{(df['Gender']=='Male').sum()}M / {(df['Gender']=='Female').sum()}F")

            st.subheader("Sample Data")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Feature Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("Target Distribution")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                df["Attrition"].value_counts().plot.pie(
                    autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
                    ax=ax, startangle=90
                )
                ax.set_ylabel("")
                ax.set_title("Attrition Distribution")
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots()
                df["Gender"].value_counts().plot.bar(
                    color=["#3498db", "#e74c3c"], ax=ax, edgecolor="black"
                )
                ax.set_title("Gender Distribution")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                plt.close()

            st.subheader("Age Distribution by Attrition")
            fig, ax = plt.subplots(figsize=(10, 4))
            for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
                subset = df[df["Attrition"] == label]
                ax.hist(subset["Age"], bins=20, alpha=0.6, label=f"Attrition={label}", color=color)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            ax.legend()
            ax.set_title("Age Distribution by Attrition Status")
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Dataset not found. Run main.py first.")

    # ── PAGE: Model Performance ──────────────────────────────────────
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance Comparison")

        if model_metrics:
            df_metrics = pd.DataFrame(model_metrics).T
            st.dataframe(df_metrics.style.highlight_max(axis=0, color="#2ecc71"),
                         use_container_width=True)

            img = load_image("model_comparison.png")
            if img:
                st.image(img, caption="Model Comparison", use_container_width=True)

            img2 = load_image("confusion_matrices.png")
            if img2:
                st.image(img2, caption="Confusion Matrices", use_container_width=True)
        else:
            st.warning("No model metrics found. Run main.py first.")

    # ── PAGE: Fairness & Bias ────────────────────────────────────────
    elif page == "⚖️ Fairness & Bias":
        st.header("⚖️ Fairness & Bias Detection")
        st.markdown("""
        **Key Metrics:**
        - **Disparate Impact**: Ratio of positive rates (threshold: 0.8)
        - **Statistical Parity Difference**: Difference in positive rates (ideal: 0)
        - **Equal Opportunity**: Equal true-positive rates across groups
        - **Demographic Parity**: Equal prediction rates across groups
        """)

        if fairness_results:
            for model_name, res in fairness_results.items():
                with st.expander(f"📌 {model_name}", expanded=True):
                    for attr, metrics in res.items():
                        st.markdown(f"**{attr.upper()}**")
                        df_fair = pd.DataFrame([metrics])
                        st.dataframe(df_fair, use_container_width=True)

            for img_name in ["fairness_comparison.png", "bias_heatmap.png",
                             "group_fairness_gender.png"]:
                img = load_image(img_name)
                if img:
                    st.image(img, caption=img_name.replace("_", " ").replace(".png", "").title(),
                             use_container_width=True)
        else:
            st.warning("No fairness results. Run main.py first.")

    # ── PAGE: Explainability ─────────────────────────────────────────
    elif page == "🔍 Explainability":
        st.header("🔍 Interpretability & Explainability")
        st.markdown("""
        **Methods used:**
        - **Intrinsic**: Decision Tree visualization
        - **Post-hoc**: SHAP values, Feature importance, Partial Dependence Plots
        """)

        for img_name, caption in [
            ("feature_importance.png", "Feature Importance (Random Forest)"),
            ("shap_summary.png", "SHAP Summary Plot"),
            ("shap_bar.png", "SHAP Feature Importance (Mean |SHAP|)"),
            ("individual_explanation.png", "Individual Candidate Explanation"),
            ("decision_tree.png", "Decision Tree Visualization"),
            ("partial_dependence.png", "Partial Dependence Plots"),
        ]:
            img = load_image(img_name)
            if img:
                st.image(img, caption=caption, use_container_width=True)

    # ── PAGE: Correlation & Causation ────────────────────────────────
    elif page == "🔗 Correlation & Causation":
        st.header("🔗 Correlation, Causation & Nudge Analysis")

        # --- Correlation Section ---
        st.subheader("📊 Correlation Analysis")
        st.markdown("""
        Correlation measures the statistical association between variables.
        **Important**: Correlation ≠ Causation. A high correlation between a feature
        and attrition does not mean the feature *causes* attrition.
        """)

        for img_name, caption in [
            ("correlation_matrix_full.png", "Full Correlation Matrix Heatmap"),
            ("target_correlation_ranking.png", "Feature Correlations with Attrition (Target)"),
            ("sensitive_attribute_correlations.png", "Sensitive Attribute Correlations"),
            ("vif_multicollinearity.png", "Variance Inflation Factor (Multicollinearity Check)"),
        ]:
            img = load_image(img_name)
            if img:
                st.image(img, caption=caption, use_container_width=True)

        # Show top correlations from JSON
        if ccn_results and "correlation" in ccn_results:
            corr = ccn_results["correlation"]
            if "top_positive_correlations" in corr:
                st.markdown("**Top Positive Correlations with Attrition:**")
                pos_df = pd.DataFrame(corr["top_positive_correlations"])
                st.dataframe(pos_df, use_container_width=True)
            if "top_negative_correlations" in corr:
                st.markdown("**Top Negative Correlations with Attrition:**")
                neg_df = pd.DataFrame(corr["top_negative_correlations"])
                st.dataframe(neg_df, use_container_width=True)
            if "high_vif_features" in corr:
                st.warning(f"⚠️ {len(corr['high_vif_features'])} features have VIF > 5 (multicollinearity concern)")

        # --- Causation Section ---
        st.subheader("🔬 Causation Analysis")
        st.markdown("""
        **Causation** requires evidence beyond correlation: temporal precedence, mechanism,
        and ruling out confounders. We test for:
        - **Simpson's Paradox**: Where subgroup trends reverse the overall trend
        - **Proxy Variables**: Features that act as proxies for protected attributes
        - **Confounding Variables**: Hidden factors that drive spurious correlations
        """)

        for img_name, caption in [
            ("simpsons_paradox.png", "Simpson's Paradox Detection"),
            ("stratified_gender_joblevel.png", "Stratified Analysis: Gender × Job Level"),
            ("proxy_variables.png", "Proxy Variable Detection"),
            ("confounding_analysis.png", "Confounding Variable Analysis"),
        ]:
            img = load_image(img_name)
            if img:
                st.image(img, caption=caption, use_container_width=True)

        if ccn_results and "causation" in ccn_results:
            caus = ccn_results["causation"]
            if "simpsons_paradox" in caus:
                sp = caus["simpsons_paradox"]
                if sp.get("detected"):
                    st.error(f"⚠️ Simpson's Paradox detected in: {', '.join(sp.get('affected_groups', []))}")
                else:
                    st.success("✅ No Simpson's Paradox detected in analyzed subgroups.")
            if "proxy_variables" in caus:
                proxies = caus["proxy_variables"]
                if proxies:
                    names = [p["proxy_feature"] if isinstance(p, dict) else str(p) for p in proxies]
                    st.warning(f"⚠️ Potential proxy variables for protected attributes: {', '.join(names)}")
                else:
                    st.success("✅ No strong proxy variables detected.")

        # --- Nudge Theory Section ---
        st.subheader("🎯 Nudge Theory & Counterfactual Analysis")
        st.markdown("""
        **Nudge Theory** (Thaler & Sunstein, 2008) suggests small design changes can guide
        better decisions without restricting choice. In AI hiring:
        - **Threshold Nudging**: Adjusting decision thresholds to balance accuracy & fairness
        - **Counterfactual Explanations**: "What minimal change would flip the decision?"
        """)

        for img_name, caption in [
            ("threshold_nudge.png", "Decision Threshold Nudging Analysis"),
            ("nudge_comparison.png", "Nudge Impact: Before vs After"),
            ("counterfactual_explanation.png", "Counterfactual Explanation Example"),
        ]:
            img = load_image(img_name)
            if img:
                st.image(img, caption=caption, use_container_width=True)

        if ccn_results and "nudge" in ccn_results:
            nudge = ccn_results["nudge"]
            if "optimal_threshold" in nudge:
                opt = nudge["optimal_threshold"]
                st.metric("Optimal Threshold", f"{opt.get('threshold', 0.5):.3f}",
                         delta=f"{opt.get('threshold', 0.5) - 0.5:+.3f} from default 0.5")
            if "counterfactual_summary" in nudge:
                cf = nudge["counterfactual_summary"]
                st.info(f"Average features to change for decision flip: {cf.get('avg_features_changed', 'N/A')}")

        # Show full text report
        report_path = os.path.join(REPORTS_DIR, "correlation_causation_report.txt")
        if os.path.exists(report_path):
            with st.expander("📄 Full Correlation, Causation & Nudge Report"):
                with open(report_path) as f:
                    st.text(f.read())

    # ── PAGE: Ethics Audit ───────────────────────────────────────────
    elif page == "📋 Ethics Audit":
        st.header("📋 Ethics & Accountability Audit")

        # Show report
        report_path = os.path.join(REPORTS_DIR, "ethics_audit_report.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read()
            st.text(report_text)

        # Risk table
        if ethical_risks:
            st.subheader("Ethical Risk Assessment")
            df_risks = pd.DataFrame(ethical_risks)
            color_map = {"HIGH": "background-color: #ffcccc",
                         "MEDIUM": "background-color: #fff3cd",
                         "LOW": "background-color: #d4edda"}

            def color_risk(val):
                return color_map.get(val, "")

            st.dataframe(
                df_risks.style.applymap(color_risk, subset=["Risk"]),
                use_container_width=True,
            )

        img = load_image("risk_summary.png")
        if img:
            st.image(img, caption="Risk Summary", use_container_width=True)

    # ── PAGE: Privacy Analysis ───────────────────────────────────────
    elif page == "🔒 Privacy Analysis":
        st.header("🔒 Privacy Preservation – Differential Privacy")

        if privacy_results:
            st.subheader("Baseline (No Privacy)")
            baseline = privacy_results.get("baseline", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", baseline.get("accuracy", "N/A"))
            col2.metric("F1 Score", baseline.get("f1_score", "N/A"))
            col3.metric("SPD", baseline.get("spd", "N/A"))

            st.subheader("DP Comparison")
            comparison = privacy_results.get("comparison", [])
            if comparison:
                df_dp = pd.DataFrame(comparison)
                st.dataframe(df_dp, use_container_width=True)

            for img_name in ["privacy_accuracy_tradeoff.png",
                             "privacy_fairness_tradeoff.png",
                             "accuracy_vs_fairness_privacy.png"]:
                img = load_image(img_name)
                if img:
                    st.image(img, caption=img_name.replace("_", " ").replace(".png", "").title(),
                             use_container_width=True)
        else:
            st.warning("No privacy results. Run main.py first.")

    # ── PAGE: System Architecture ────────────────────────────────────
    elif page == "🏗️ System Architecture":
        st.header("🏗️ System Pipeline Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                    RESPONSIBLE AI PIPELINE                      │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
        │   │  Dataset      │───▶│  Data         │───▶│  Model       │     │
        │   │  (IBM HR)     │    │  Preprocessing│    │  Training    │     │
        │   └──────────────┘    └──────────────┘    └──────┬───────┘     │
        │                                                   │             │
        │          ┌────────────────────────────────────────┤             │
        │          │                    │                    │             │
        │          ▼                    ▼                    ▼             │
        │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
        │   │  Fairness &  │    │  Explain-     │    │  Privacy     │     │
        │   │  Bias        │    │  ability      │    │  Preserv.    │     │
        │   │  Detection   │    │  (SHAP, DT)   │    │  (Diff. Priv)│     │
        │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
        │          │                    │                    │             │
        │          └────────────────────┼────────────────────┘             │
        │                              │                                  │
        │                              ▼                                  │
        │                    ┌──────────────────┐                         │
        │                    │  Ethics &         │                         │
        │                    │  Accountability   │                         │
        │                    │  Audit            │                         │
        │                    └────────┬─────────┘                         │
        │                             │                                   │
        │                             ▼                                   │
        │                    ┌──────────────────┐                         │
        │                    │  Streamlit        │                         │
        │                    │  Dashboard        │                         │
        │                    └──────────────────┘                         │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        ```
        """)

        st.subheader("Modules")
        modules = {
            "data_preprocessing.py": "Load, clean, encode, feature-engineer the dataset",
            "train_models.py": "Train LR, RF, DT models; evaluate with metrics",
            "fairness_analysis.py": "Compute DI, SPD, DP, EO across gender & age",
            "explainability.py": "SHAP, feature importance, PDP, DT visualization",
            "ethics_audit.py": "Automated audit report with risk assessment",
            "privacy_module.py": "Differential privacy experiments & tradeoffs",
            "dashboard.py": "This interactive Streamlit dashboard",
        }
        for mod, desc in modules.items():
            st.markdown(f"- **{mod}**: {desc}")

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "🎓 **Responsible AI Academic Project** | "
        "Bias Detection in AI Hiring Systems | "
        "Built with Python, Scikit-learn, SHAP, Fairlearn, Streamlit"
    )


if __name__ == "__main__":
    main()
