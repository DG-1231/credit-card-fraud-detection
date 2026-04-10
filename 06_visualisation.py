import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# Create output folder
# -------------------------
os.makedirs("outputs/visuals", exist_ok=True)

# -------------------------
# Load data
# -------------------------
X_train = pd.read_csv("outputs/X_train_smote.csv")
y_train = pd.read_csv("outputs/y_train_smote.csv")["Class"].values

X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")["Class"].values

# -------------------------
# Define models
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

# -------------------------
# ROC curve plot
# -------------------------
plt.figure()

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"outputs/visuals/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    # Save summary
    results.append({"Model": name, "ROC_AUC": auc})

# ROC plot formatting
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("outputs/visuals/roc_comparison.png")
plt.close()

# Save AUC results
pd.DataFrame(results).to_csv("outputs/visuals/roc_auc_summary.csv", index=False)

print("✅ Visual results saved in outputs/visuals/")