import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths
dashboard_dir = "outputs/dashboard"
output_path = "outputs/dashboard/combined_dashboard.png"

# Image files (must exist already)
images = {
    "Fraud Distribution": "fraud_distribution.png",
    "Confusion Matrix": "confusion_matrix.png",
    "ROC Curve": "roc_curve.png",
    "Probability Distribution": "probability_distribution.png",
}

# Load images
loaded = {}
for title, fname in images.items():
    path = os.path.join(dashboard_dir, fname)
    loaded[title] = mpimg.imread(path)

# Create combined dashboard (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Credit Card Fraud Detection — Python Dashboard Summary", fontsize=18)

titles = list(loaded.keys())
imgs = list(loaded.values())

for ax, title, img in zip(axes.flatten(), titles, imgs):
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_path, dpi=200)
plt.close()

print(f"✅ Combined dashboard saved to: {output_path}")