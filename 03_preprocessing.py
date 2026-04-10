import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (keep fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Amount column
scaler = StandardScaler()
X_train.loc[:, "Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test.loc[:, "Amount"] = scaler.transform(X_test[["Amount"]])

# Save files
X_train.to_csv("outputs/X_train.csv", index=False)
X_test.to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print("Preprocessing completed")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())
