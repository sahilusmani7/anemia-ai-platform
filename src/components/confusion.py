import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# -----------------------------
# 1. Confusion Matrix
# -----------------------------
cm = np.array([[80, 15],
               [10, 113]])

labels = ["Anemia", "Non-Anemia"]

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 14}
)

plt.title("Confusion Matrix for Anemia Detection", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# 2. Classification Report Data
# -----------------------------
report_data = {
    "Class": ["Anemia", "Non-Anemia", "Accuracy", "Macro Avg", "Weighted Avg"],
    "Precision": [0.89, 0.88, "", 0.89, 0.89],
    "Recall": [0.84, 0.92, "", 0.88, 0.89],
    "F1-Score": [0.86, 0.90, 0.89, 0.88, 0.88],
    "Support": [95, 123, 218, 218, 218]
}

df = pd.DataFrame(report_data)

print("\nCLASSIFICATION REPORT:\n")
print(df)

# Save as CSV (optional)
df.to_csv("classification_report.csv", index=False)

# -----------------------------
# 3. Plot Precision / Recall / F1
# -----------------------------
metrics_df = pd.DataFrame({
    "Class": ["Anemia", "Non-Anemia"],
    "Precision": [0.89, 0.88],
    "Recall": [0.84, 0.92],
    "F1-Score": [0.86, 0.90]
})

metrics_df.set_index("Class").plot(kind="bar", figsize=(8, 6))

plt.title("Classification Metrics by Class", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("classification_metrics.png", dpi=300)
plt.show()