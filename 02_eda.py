import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs("outputs/figures", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")
print(df.describe())

# -----------------------
# Fraud vs Non-Fraud
# -----------------------
plt.figure()
df["Class"].value_counts().plot(kind="bar")
plt.title("Fraud vs Non-Fraud Distribution")
plt.xlabel("Class (0 = Legit, 1 = Fraud)")
plt.ylabel("Count")
plt.savefig("outputs/figures/fraud_distribution.png")
plt.close()

# -----------------------
# Transaction Amount
# -----------------------
plt.figure()
plt.hist(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig("outputs/figures/amount_distribution.png")
plt.close()

# -----------------------
# Fraud Amount
# -----------------------
plt.figure()
plt.hist(df[df["Class"] == 1]["Amount"], bins=50)
plt.title("Fraud Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig("outputs/figures/fraud_amount_distribution.png")
plt.close()

# -----------------------
# Time distribution
# -----------------------
plt.figure()
plt.hist(df["Time"], bins=50)
plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.savefig("outputs/figures/time_distribution.png")
plt.close()

print("EDA figures saved to outputs/figures/")
