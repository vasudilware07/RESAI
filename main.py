"""
main.py - Responsible AI Hiring Pipeline Orchestrator
=======================================================
Central script that runs the entire RAI pipeline end-to-end:

  1. Data Preprocessing
  2. Model Training & Evaluation
  3. Fairness & Bias Detection
  4. Explainability Analysis
  5. Privacy Preservation
  6. Ethics Audit Report

Usage:
    python main.py          # Run full pipeline
    python main.py --quick  # Skip SHAP (faster)
"""

import sys
import time
from utils import log


def main(quick=False):
    start = time.time()

    log("=" * 70)
    log("  RESPONSIBLE AI PIPELINE – Bias Detection in AI Hiring Systems")
    log("=" * 70)

    # ── Step 1: Data Preprocessing ───────────────────────────────────
    log("\n▶ STEP 1: Data Preprocessing")
    from data_preprocessing import run_preprocessing
    data = run_preprocessing()

    # ── Step 2: Model Training & Evaluation ──────────────────────────
    log("\n▶ STEP 2: Training AI Hiring Models")
    from train_models import train_all_models, evaluate_models
    models = train_all_models(data)
    model_results = evaluate_models(models, data)

    # ── Step 3: Fairness & Bias Detection ────────────────────────────
    log("\n▶ STEP 3: Fairness & Bias Detection")
    from fairness_analysis import analyze_fairness
    fairness_results = analyze_fairness(models, data)

    # ── Step 4: Explainability ───────────────────────────────────────
    log("\n▶ STEP 4: Interpretability & Explainability")
    from explainability import run_explainability
    explain_results = run_explainability(models, data)

    # ── Step 4.5: Correlation, Causation & Nudge Analysis ────────────
    log("\n▶ STEP 4.5: Correlation, Causation & Nudge Analysis")
    from correlation_causation import run_correlation_causation_nudge
    ccn_results = run_correlation_causation_nudge(models, data)

    # ── Step 5: Privacy Preservation ─────────────────────────────────
    log("\n▶ STEP 5: Privacy Preservation (Differential Privacy)")
    from privacy_module import run_differential_privacy
    privacy_results = run_differential_privacy(data)

    # ── Step 6: Ethics Audit ─────────────────────────────────────────
    log("\n▶ STEP 6: Ethics & Accountability Audit")
    from ethics_audit import run_ethics_audit
    audit_results = run_ethics_audit(fairness_results, model_results, privacy_results)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    log(f"\n{'='*70}")
    log(f"  PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    log(f"{'='*70}")
    log("  Output files saved to: outputs/plots/ and outputs/reports/")
    log("  To launch dashboard:   streamlit run dashboard.py")
    log(f"{'='*70}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
