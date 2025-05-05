import os
import glob
import pandas as pd

# Update this path to the directory containing your YOLOv8 experiment folders
base_dir = "/home/oem/Desktop/TrainingScripts/runs/detect"  # <<< UPDATE THIS PATH

# Recursively find all results.csv files under base_dir
csv_files = glob.glob(os.path.join(base_dir, "**", "results.csv"), recursive=True)
print(f"Found {len(csv_files)} results.csv files.\n")

if not csv_files:
    print("No CSV files found. Please check your base_dir path.")
    exit()

# Define the metric columns using the provided names
metric_columns = {
    "mAP50": ["metrics/mAP50(B)"],
    "mAP50-95": ["metrics/mAP50-95(B)"],
    "Precision": ["metrics/precision(B)"],
    "Recall": ["metrics/recall(B)"]  # Optional if available in your CSV
}

# List to store summary information from each experiment
results_summary = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue

    # Use the parent folder name as the experiment identifier
    experiment_name = os.path.basename(os.path.dirname(csv_file))
    exp_summary = {"Experiment": experiment_name}

    # For each metric, look for the specified column names and record the best (maximum) value
    for metric, possible_cols in metric_columns.items():
        best_val = None
        for col in possible_cols:
            if col in df.columns:
                val = df[col].max()  # get the best epoch value
                best_val = val if best_val is None or val > best_val else best_val
        if best_val is None:
            print(f"Warning: Metric '{metric}' not found in experiment '{experiment_name}'.")
            print("Available columns:", list(df.columns))
        exp_summary[metric] = best_val

    results_summary.append(exp_summary)

# Combine all experiment summaries into one DataFrame
summary_df = pd.DataFrame(results_summary)
print("Summary of best metrics from each experiment:")
print(summary_df)

# Detailed analysis: Find overall best experiment for each metric
overall_best = {}
print("\nDetailed Analysis of Best Models per Metric:")
for metric in metric_columns.keys():
    if summary_df[metric].notnull().any():
        best_idx = summary_df[metric].idxmax()
        best_row = summary_df.loc[best_idx]
        overall_best[metric] = best_row
        print(f"\nBest {metric}: {best_row[metric]}")
        print(f"  Experiment : {best_row['Experiment']}")
    else:
        overall_best[metric] = None
        print(f"\nNo data available for metric: {metric}")

# Save the overall summary to a CSV file for later review
output_file = "overall_experiment_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nOverall summary saved to {output_file}")

