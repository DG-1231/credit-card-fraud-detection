import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# Create output folder
os.makedirs("outputs/dashboard", exist_ok=True)

# Load data
X_train = pd.read_csv("outputs/X_train_smote.csv")
y_train = pd.read_csv("outputs/y_train_smote.csv")["Class"].values

X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")["Class"].values

# Train Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Get probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# Final prediction using threshold 0.8
threshold = 0.8
y_pred = (y_prob >= threshold).astype(int)

# 1) Fraud distribution
plt.figure()
pd.Series(y_test).value_counts().plot(kind="bar")
plt.title("Fraud vs Legit Transactions")
plt.xlabel("Class (0=Legit, 1=Fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/dashboard/fraud_distribution.png")
plt.close()

# 2) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Threshold 0.8)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("outputs/dashboard/confusion_matrix.png")
plt.close()

# 3) ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/dashboard/roc_curve.png")
plt.close()

# 4) Probability distribution
plt.figure()
plt.hist(y_prob, bins=50)
plt.title("Fraud Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/dashboard/probability_distribution.png")
plt.close()

print("Dashboard visuals saved in outputs/dashboard/")
