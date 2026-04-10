# src/10_premium_dashboard.py
# Premium dashboard generator (Random Forest) — aligned + no overlaps

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# -----------------------
# SETTINGS
# -----------------------
THRESHOLD = 0.80

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

# Preferred input files (based on your project)
X_TEST_PATH = os.path.join(PROJECT_ROOT, "outputs", "X_test.csv")
Y_TEST_PATH = os.path.join(PROJECT_ROOT, "outputs", "y_test.csv")

X_TRAIN_SMOTE_PATH = os.path.join(PROJECT_ROOT, "outputs", "X_train_smote.csv")
Y_TRAIN_SMOTE_PATH = os.path.join(PROJECT_ROOT, "outputs", "y_train_smote.csv")

MODEL_COMPARISON_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_results", "final_model_comparison.csv")

DATASET_NAME = "creditcard.csv (anonymized)"

# -----------------------
# SAFE LOADERS
# -----------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_series(path: str) -> pd.Series:
    df = load_csv(path)
    # accept either a column named Class or the first column
    if "Class" in df.columns:
        return df["Class"]
    return df.iloc[:, 0]

# -----------------------
# LOAD DATA
# -----------------------
X_test = load_csv(X_TEST_PATH)
y_test = load_series(Y_TEST_PATH)

# Train data: prefer SMOTE (as you already created)
X_train = load_csv(X_TRAIN_SMOTE_PATH)
y_train = load_series(Y_TRAIN_SMOTE_PATH)

# -----------------------
# TRAIN RANDOM FOREST
# -----------------------
# Keep it stable + strong, but not insanely slow
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

print("Training Random Forest (for dashboard)...")
rf.fit(X_train, y_train)

# -----------------------
# PREDICTIONS + METRICS
# -----------------------
proba = rf.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision_f = precision_score(y_test, y_pred, zero_division=0)
recall_f = recall_score(y_test, y_pred, zero_division=0)
f1_f = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, proba)

fpr, tpr, _ = roc_curve(y_test, proba)

test_size = len(y_test)
fraud_cases = int(y_test.sum())
fraud_rate = (fraud_cases / test_size) * 100

# -----------------------
# MODEL COMPARISON (F1)
# -----------------------
# Use your saved comparison if available, else fall back to just RF
comparison_df = None
if os.path.exists(MODEL_COMPARISON_PATH):
    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    # Expect columns like: Model, F1_Fraud
    # If your file uses different names, adjust here:
    if "F1_Fraud" not in comparison_df.columns and "F1" in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"F1": "F1_Fraud"})
else:
    comparison_df = pd.DataFrame(
        {"Model": ["Random Forest"], "F1_Fraud": [f1_f]}
    )

# Keep a nice order if present
order = ["Logistic Regression", "Random Forest", "XGBoost"]
comparison_df["Model"] = comparison_df["Model"].astype(str)
comparison_df["order"] = comparison_df["Model"].apply(lambda x: order.index(x) if x in order else 999)
comparison_df = comparison_df.sort_values("order").drop(columns=["order"])

# -----------------------
# FEATURE IMPORTANCE (Top 10)
# -----------------------
importances = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(10)
feat_names = importances.index.tolist()
feat_vals = importances.values

# -----------------------
# FIGURE LAYOUT (FIXED OVERLAPS)
# -----------------------
# Key fixes:
# - Larger figure height
# - Dedicated spacing between rows (hspace)
# - Titles use smaller fontsize + extra padding
# - Footer is separate, so it never crashes into plots
fig = plt.figure(figsize=(20, 12), dpi=160)

# 4 rows:
# Row0: KPI band
# Row1: Distribution + Confusion
# Row2: ROC + Probability
# Row3: Feature importance + Model comparison
gs = fig.add_gridspec(
    nrows=4, ncols=12,
    height_ratios=[1.25, 3.2, 3.0, 3.6],
    wspace=0.9, hspace=0.65
)

# ---- Title (top)
fig.suptitle(
    "Credit Card Fraud Detection — Premium Dashboard (Random Forest)",
    fontsize=20, fontweight="bold", y=0.98
)
fig.text(
    0.5, 0.952,
    f"Threshold = {THRESHOLD:.2f}   |   Test set size = {test_size:,} transactions   |   Data: {DATASET_NAME}",
    ha="center", va="center", fontsize=10
)

# ---- KPI band (row 0)
ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis("off")

# KPI cards (aligned, no overlap)
kpis = [
    ("Total Transactions", f"{test_size:,}"),
    ("Fraud Cases", f"{fraud_cases:,} ({fraud_rate:.3f}%)"),
    ("Precision (Fraud)", f"{precision_f:.4f}"),
    ("Recall (Fraud)", f"{recall_f:.4f}"),
    ("F1 (Fraud)", f"{f1_f:.4f}"),
    ("ROC-AUC", f"{roc_auc:.4f}"),
]

# Even spacing across the row
x_positions = np.linspace(0.03, 0.97, len(kpis))
for (label, value), x in zip(kpis, x_positions):
    ax_kpi.text(
        x, 0.5,
        f"{label}\n{value}",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", linewidth=1.2, facecolor="white", edgecolor="black")
    )

# ---- Row 1: Distribution (left) + Confusion matrix (right)
ax_dist = fig.add_subplot(gs[1, 0:8])
ax_cm = fig.add_subplot(gs[1, 8:12])

# Fraud vs Legit distribution (test set)
counts = pd.Series(y_test).value_counts().sort_index()
ax_dist.bar(["Legit (0)", "Fraud (1)"], [counts.get(0, 0), counts.get(1, 0)])
ax_dist.set_title("Fraud vs Legit Distribution (Test Set)", fontsize=12, fontweight="bold", pad=10)
ax_dist.set_ylabel("Count")
ax_dist.grid(True, axis="y", alpha=0.25)
ax_dist.text(
    0.98, 0.93, f"Fraud rate: {fraud_rate:.3f}%",
    transform=ax_dist.transAxes, ha="right", va="top",
    fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray")
)

# Confusion matrix heatmap (matplotlib only)
im = ax_cm.imshow(cm, interpolation="nearest")
ax_cm.set_title("Confusion Matrix (Threshold Applied)", fontsize=12, fontweight="bold", pad=10)
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Legit (0)", "Fraud (1)"], fontsize=9)
ax_cm.set_yticklabels(["Legit (0)", "Fraud (1)"], fontsize=9)
ax_cm.set_xlabel("Predicted", labelpad=6)
ax_cm.set_ylabel("Actual", labelpad=6)

# Annotate cells
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11, fontweight="bold",
                   color="white" if i == 1 else "black")

cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

# ---- Row 2: ROC (left) + Probability (right)
ax_roc = fig.add_subplot(gs[2, 0:6])
ax_prob = fig.add_subplot(gs[2, 7:12])

ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold", pad=10)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.grid(True, alpha=0.25)
ax_roc.legend(loc="lower right", fontsize=9)

ax_prob.hist(proba, bins=60)
ax_prob.axvline(THRESHOLD, linestyle="--", linewidth=1.6, label=f"Threshold = {THRESHOLD:.2f}")
ax_prob.set_title("Fraud Score Distribution (Predicted Probability)", fontsize=12, fontweight="bold", pad=10)
ax_prob.set_xlabel("Predicted Probability of Fraud")
ax_prob.set_ylabel("Frequency")
ax_prob.grid(True, alpha=0.25)
ax_prob.legend(loc="upper right", fontsize=9)

# ---- Row 3: Feature importance (left) + Model comparison (right)
ax_fi = fig.add_subplot(gs[3, 0:8])
ax_mc = fig.add_subplot(gs[3, 8:12])

# Feature importance
ax_fi.barh(list(reversed(feat_names)), list(reversed(feat_vals)))
ax_fi.set_title("Top 10 Important Features (Random Forest)", fontsize=12, fontweight="bold", pad=12)
ax_fi.set_xlabel("Importance Score")
ax_fi.grid(True, axis="x", alpha=0.25)

# Model comparison (F1 Fraud)
ax_mc.bar(comparison_df["Model"], comparison_df["F1_Fraud"])
ax_mc.set_title("Model Comparison (Fraud F1 Score)", fontsize=12, fontweight="bold", pad=12)
ax_mc.set_ylabel("F1 Score")
ax_mc.set_ylim(0, 1.0)
ax_mc.grid(True, axis="y", alpha=0.25)
ax_mc.tick_params(axis="x", rotation=10)

# Annotate bars
for i, (m, v) in enumerate(zip(comparison_df["Model"], comparison_df["F1_Fraud"])):
    ax_mc.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ---- Footer summary (never overlaps)
footer = (
    f"Evaluation Summary  |  Threshold: {THRESHOLD:.2f}  |  "
    f"TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}  |  "
    f"Precision(Fraud): {precision_f:.4f}  Recall(Fraud): {recall_f:.4f}  "
    f"F1(Fraud): {f1_f:.4f}  ROC-AUC: {roc_auc:.4f}"
)
fig.text(0.5, 0.015, footer, ha="center", va="center", fontsize=10, fontweight="bold")

# Make sure layout never collides with the title/footer
plt.subplots_adjust(top=0.91, bottom=0.085)

# -----------------------
# SAVE
# -----------------------
out_path = os.path.join(OUT_DIR, "premium_dashboard_rf.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved premium dashboard: {out_path}")