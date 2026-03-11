"""
utils.py - Utility functions for the Responsible AI Hiring project.
Provides shared helpers for paths, plotting, and logging used across modules.
"""

import os
import json
import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Project paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

DATASET_PATH = os.path.join(DATASET_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Ensure output directories exist
for d in [OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Plot styling ────────────────────────────────────────────────────────────
def set_plot_style():
    """Apply a consistent, publication-quality plot style."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })

def save_plot(fig, filename):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] Plot saved: {path}")
    return path

# ─── JSON helpers ────────────────────────────────────────────────────────────
def save_json(data, filename):
    """Save a dict/list as JSON in the reports directory."""
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [✓] Report saved: {path}")
    return path

def save_text_report(text, filename):
    """Save a plain-text report."""
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [✓] Report saved: {path}")
    return path

# ─── Logging ─────────────────────────────────────────────────────────────────
def log(message, level="INFO"):
    """Simple timestamped console logger."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {message}")
