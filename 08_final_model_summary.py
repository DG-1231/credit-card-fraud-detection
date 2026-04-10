import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# -------------------------
# Output folder
# -------------------------
os.makedirs("outputs/final_results", exist_ok=True)

# -------------------------
# Load data
# -------------------------
X_train = pd.read_csv("outputs/X_train_smote.csv")
y_train = pd.read_csv("outputs/y_train_smote.csv")["Class"].values

X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")["Class"].values

# -------------------------
# Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

rows = []

for name, model in models.items():
    print(f"Training {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Focus on fraud class (Class = 1)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rows.append({
        "Model": name,
        "Precision_Fraud": round(prec, 4),
        "Recall_Fraud": round(rec, 4),
        "F1_Fraud": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })

# -------------------------
# Save comparison
# -------------------------
results_df = pd.DataFrame(rows).sort_values(by="F1_Fraud", ascending=False)
results_df.to_csv("outputs/final_results/final_model_comparison.csv", index=False)

print("Saved: outputs/final_results/final_model_comparison.csv")
print(results_df)
