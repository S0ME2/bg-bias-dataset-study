import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = (
    "results/results_csv/results_aggregated.csv"  # Path to the CSV file with results
)
OUT_DIR = "results/graphs"  # Directory to save generated graphs
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print(df[["train_dataset", "test_dataset", "accuracy_mean", "accuracy_std"]])

heatmap_data = df.pivot(
    index="train_dataset", columns="test_dataset", values="accuracy_mean"
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "Accuracy"},
)

plt.xlabel("Test Dataset")
plt.ylabel("Train Dataset")
plt.title("Accuracy Heatmap Across Train–Test Dataset Pairs")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_heatmap.png", dpi=300)
plt.savefig(f"{OUT_DIR}/accuracy_heatmap.pdf")
plt.close()

key_pairs = [
    ("B", "B"),
    ("B", "C"),
    ("D", "C"),
    ("A", "C"),
]

key_df = df[
    df.apply(lambda r: (r.train_dataset, r.test_dataset) in key_pairs, axis=1)
].copy()

key_df["label"] = key_df.apply(
    lambda r: f"{r.train_dataset} → {r.test_dataset}", axis=1
)

plt.figure(figsize=(7, 4))

plt.bar(
    key_df["label"], key_df["accuracy_mean"], yerr=key_df["accuracy_std"], capsize=5
)

plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Generalization Performance on Key Dataset Transfers")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/key_comparisons_bar.png", dpi=300)
plt.savefig(f"{OUT_DIR}/key_comparisons_bar.pdf")
plt.close()

d_tests = df[df["test_dataset"] == "D"]

plt.figure(figsize=(6, 4))
plt.bar(
    d_tests["train_dataset"],
    d_tests["accuracy_mean"],
    yerr=d_tests["accuracy_std"],
    capsize=5,
)

plt.ylim(0, 1.05)
plt.xlabel("Train Dataset")
plt.ylabel("Accuracy on Dataset D")
plt.title("Robustness to Random Backgrounds")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/robustness_to_D.png", dpi=300)
plt.savefig(f"{OUT_DIR}/robustness_to_D.pdf")
plt.close()
