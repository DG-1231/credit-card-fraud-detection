import pandas as pd
from imblearn.over_sampling import SMOTE

# Load training data
X_train = pd.read_csv("outputs/X_train.csv")
y_train = pd.read_csv("outputs/y_train.csv")

print("Original class distribution:")
print(y_train["Class"].value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train.values.ravel())

# Convert to DataFrame
X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
y_resampled = pd.DataFrame(y_resampled, columns=["Class"])

# Save
X_resampled.to_csv("outputs/X_train_smote.csv", index=False)
y_resampled.to_csv("outputs/y_train_smote.csv", index=False)

print("After SMOTE:")
print(y_resampled["Class"].value_counts())
