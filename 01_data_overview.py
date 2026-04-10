import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nFraud Rate (%):")
print(df["Class"].mean() * 100)