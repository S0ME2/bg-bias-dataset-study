import os
import json
import pandas as pd

RESULTS_DIR = "results/metrics"  # Directory containing JSON result files
OUTPUT_DIR = "models/results_csv"  # Directory to save the CSV results

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

for root, _, files in os.walk(RESULTS_DIR):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                data = json.load(f)

            row = {
                "train_dataset": data["train_dataset"],
                "test_dataset": data["test_dataset"],
                "seed": data["seed"],
                "accuracy": data["accuracy"],
                "precision": data["precision"],
                "recall": data["recall"],
                "f1": data["f1"],
                "model_path": data.get("model_path", ""),
            }

            rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(by=["train_dataset", "test_dataset", "seed"]).reset_index(drop=True)

per_run_csv = os.path.join(OUTPUT_DIR, "results_per_run.csv")
df.to_csv(per_run_csv, index=False)

print(f"Saved per-run results to {per_run_csv}")

agg = (
    df.groupby(["train_dataset", "test_dataset"])
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        runs=("seed", "count"),
    )
    .reset_index()
)

agg_csv = os.path.join(OUTPUT_DIR, "results_aggregated.csv")
agg.to_csv(agg_csv, index=False)

print(f"Saved aggregated results to {agg_csv}")
