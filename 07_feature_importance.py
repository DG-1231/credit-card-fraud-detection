import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Create output folder
# -------------------------
os.makedirs("outputs/feature_analysis", exist_ok=True)

# -------------------------
# Load data
# -------------------------
X_train = pd.read_csv("outputs/X_train_smote.csv")
y_train = pd.read_csv("outputs/y_train_smote.csv")["Class"].values

# -------------------------
# Train Random Forest
# -------------------------
print("Training Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# -------------------------
# Extract feature importance
# -------------------------
importances = rf.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Save full importance table
importance_df.to_csv("outputs/feature_analysis/feature_importance_full.csv", index=False)

# -------------------------
# Plot Top 15 Features
# -------------------------
top15 = importance_df.head(15)

plt.figure(figsize=(8,6))
plt.barh(top15["Feature"], top15["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Most Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()

plt.savefig("outputs/feature_analysis/top15_feature_importance.png")
plt.close()

print("✅ Feature importance analysis completed.")
print("Saved in outputs/feature_analysis/")