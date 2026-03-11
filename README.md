# Responsible AI – Bias Detection in AI Hiring Systems

## Academic Project | IBM HR Analytics Dataset

---

## 📁 Project Structure

```
responsible_ai_hiring/
│
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── outputs/
│   ├── plots/          ← Generated visualizations
│   └── reports/        ← JSON reports & audit text
│
├── data_preprocessing.py   ← Data loading, cleaning, feature engineering
├── train_models.py         ← LR, RF, DT training & evaluation
├── fairness_analysis.py    ← Bias detection & fairness metrics
├── explainability.py       ← SHAP, feature importance, DT visualization
├── privacy_module.py       ← Differential privacy experiments
├── ethics_audit.py         ← Automated ethical audit report
├── dashboard.py            ← Streamlit interactive dashboard
├── utils.py                ← Shared utilities & paths
├── main.py                 ← Pipeline orchestrator
├── generate_ppt.py         ← PowerPoint presentation generator
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+ (recommended: 3.10 or 3.11)
- pip package manager

### Step 1: Install dependencies
```bash
cd responsible_ai_hiring
pip install -r requirements.txt
```

### Step 2: Run the complete pipeline
```bash
python main.py
```

This will:
1. Load & preprocess the IBM HR dataset
2. Train 3 models (Logistic Regression, Random Forest, Decision Tree)
3. Run fairness & bias analysis (Demographic Parity, Disparate Impact, etc.)
4. Generate explainability plots (SHAP, feature importance, PDP)
5. Run differential privacy experiments
6. Generate an ethics audit report

All outputs are saved in `outputs/plots/` and `outputs/reports/`.

### Step 3: Launch the interactive dashboard
```bash
streamlit run dashboard.py
```

### Step 4: Generate the PowerPoint presentation
```bash
python generate_ppt.py
```

---

## 📊 Syllabus Coverage

| Unit | Topic | Module |
|------|-------|--------|
| 1 | Introduction to Responsible AI | `main.py`, `data_preprocessing.py` |
| 2 | Fairness and Bias | `fairness_analysis.py` |
| 3 | Interpretability and Explainability | `explainability.py` |
| 4 | Ethics and Accountability | `ethics_audit.py` |
| 5 | Privacy Preservation | `privacy_module.py` |
| 6 | Case Study (Hiring Systems) | `dashboard.py`, full pipeline |

---

## 📈 Generated Outputs

### Plots
- `model_comparison.png` – Accuracy/Precision/Recall/F1 across models
- `confusion_matrices.png` – Confusion matrices for all models
- `fairness_comparison.png` – Disparate Impact & SPD comparison
- `bias_heatmap.png` – Heatmap of fairness metrics
- `group_fairness_gender.png` – Positive prediction rates by gender
- `feature_importance.png` – Random Forest feature importance
- `shap_summary.png` – SHAP summary plot
- `shap_bar.png` – SHAP bar plot
- `individual_explanation.png` – Single candidate explanation
- `decision_tree.png` – Decision Tree visualization
- `partial_dependence.png` – Partial Dependence Plots
- `privacy_accuracy_tradeoff.png` – Privacy vs Accuracy
- `privacy_fairness_tradeoff.png` – Privacy vs Fairness
- `accuracy_vs_fairness_privacy.png` – Combined tradeoff
- `risk_summary.png` – Ethical risk assessment

### Reports
- `model_metrics.json` – Model evaluation metrics
- `fairness_results.json` – Fairness analysis results
- `privacy_results.json` – Differential privacy comparison
- `ethical_risks.json` – Risk assessment data
- `ethics_audit_report.txt` – Full ethics audit report

---

## 🛠️ Key Libraries Used

| Library | Purpose |
|---------|---------|
| scikit-learn | ML models, preprocessing, evaluation |
| SHAP | Model explainability (post-hoc) |
| Fairlearn | Fairness metrics & analysis |
| diffprivlib | Differential privacy |
| Streamlit | Interactive dashboard |
| matplotlib / seaborn | Visualizations |
| python-pptx | PowerPoint generation |

---

## 👩‍🎓 For College Demonstration

1. Run `python main.py` to generate all results
2. Run `streamlit run dashboard.py` to show the interactive dashboard
3. Run `python generate_ppt.py` to create the presentation
4. Open `outputs/reports/ethics_audit_report.txt` for the written analysis
5. All plots in `outputs/plots/` can be included in reports
