import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -------------------------
# Output folder
# -------------------------
os.makedirs("outputs/threshold_analysis", exist_ok=True)

# -------------------------
# Load data
# -------------------------
X_train = pd.read_csv("outputs/X_train_smote.csv")
y_train = pd.read_csv("outputs/y_train_smote.csv")["Class"].values

X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")["Class"].values

# -------------------------
# Train Random Forest
# -------------------------
print("Training Random Forest")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Probabilities for fraud class
y_prob = rf.predict_proba(X_test)[:, 1]

# -------------------------
# Threshold sweep
# -------------------------
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
rows = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    rows.append({
        "Threshold": t,
        "Precision_Fraud": round(prec, 4),
        "Recall_Fraud": round(rec, 4),
        "F1_Fraud": round(f1, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    })

results = pd.DataFrame(rows)

# Save table
results.to_csv("outputs/threshold_analysis/random_forest_threshold_results.csv", index=False)
print("Saved: outputs/threshold_analysis/random_forest_threshold_results.csv")
print(results)

# -------------------------
# Plot Precision/Recall/F1 vs Threshold
# -------------------------
plt.figure()
plt.plot(results["Threshold"], results["Precision_Fraud"], marker="o", label="Precision (Fraud)")
plt.plot(results["Threshold"], results["Recall_Fraud"], marker="o", label="Recall (Fraud)")
plt.plot(results["Threshold"], results["F1_Fraud"], marker="o", label="F1 (Fraud)")
plt.title("Random Forest: Threshold vs Precision/Recall/F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/threshold_analysis/threshold_tradeoff_plot.png")
plt.close()

print("Saved: outputs/threshold_analysis/threshold_tradeoff_plot.png")
