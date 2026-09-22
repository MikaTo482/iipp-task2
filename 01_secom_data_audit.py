import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
import json
from datetime import datetime
from utils import save_result

# ----------------------------------------------------------------------
# sha256 hash for each file in the SECOM dataset
# ----------------------------------------------------------------------

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# ----------------------------------------------------------------------
# Save Figures
# ----------------------------------------------------------------------

def save_figure(fig, script_name, filename, base_dir="graph", dpi=150):
    target_dir = os.path.join(base_dir, script_name)
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        os.remove(target_path)

    fig.savefig(target_path, dpi=dpi, bbox_inches="tight")

# ----------------------------------------------------------------------
# Exact constant features
# ----------------------------------------------------------------------

def find_exact_constant_features(X):
    constant_features = []

    for col in X.columns:
        series = X[col].dropna()
        nunique = series.nunique()

        if nunique <= 1:
            constant_features.append(col)

    return constant_features

# ----------------------------------------------------------------------
# Near-zero variance features
# ----------------------------------------------------------------------

def find_near_zero_variance_features(X, freq_cut=95/5, unique_cut=10.0):
    n_samples = X.shape[0]
    results = []

    for col in X.columns:
        series = X[col].dropna()
        nunique = series.nunique()

        if nunique <= 1:
            continue

        value_counts = series.value_counts()
        freq_ratio = value_counts.iloc[0] / value_counts.iloc[1]
        percent_unique = (nunique / n_samples) * 100

        is_nzv = (freq_ratio > freq_cut) and (percent_unique < unique_cut)

        if is_nzv:
            results.append({
                "feature_id": col,
                "most_frequent_value": value_counts.index[0],
                "freq_ratio": round(freq_ratio, 2),
                "percent_unique": round(percent_unique, 3),
                "n_unique": nunique,
            })

    return pd.DataFrame(results)

# ----------------------------------------------------------------------
# Correlation Audits
# ----------------------------------------------------------------------

def audit_correlation_structure(X):
    corr_matrix = X.select_dtypes(include="number").corr().abs()

    upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(upper_mask)

    thresholds = [0.90, 0.95, 0.98, 0.995]
    result = {}
    pairs_by_threshold = {}

    all_pairs = upper.stack().dropna().reset_index()
    all_pairs.columns = ["feature_1", "feature_2", "correlation"]

    for t in thresholds:
        count = int((upper >= t).sum().sum())
        result[f">=|{t}|"] = count

        matched_pairs = all_pairs[all_pairs["correlation"] >= t].sort_values("correlation", ascending=False).reset_index(drop=True)
        pairs_by_threshold[f">=|{t}|"] = matched_pairs

    return result, upper, pairs_by_threshold


# ======================================================================
# Main execution (runs only when this file is executed directly,
# not when another script imports functions from it)
# ======================================================================
if __name__ == "__main__":

    files = ["secom.data", "secom_labels.data", "secom.names"]

    registry = {}
    for fname in files:
        file_hash = calculate_sha256(f"secom/{fname}")
        registry[fname] = file_hash
        print(f"{fname}: {file_hash}")

    registry_path = "data_registry.json"

    if os.path.exists(registry_path):
        os.remove(registry_path)
        print(f"deleted {registry_path} successfully")

    registry_record = {
        "dataset": "SECOM",
        "source": "UCI Machine Learning Repository (ID: 179)",
        "download_date": datetime.now().strftime("%Y-%m-%d"),
        "files": registry
    }

    with open(registry_path, "w") as f:
        json.dump(registry_record, f, indent=2)

    print(f"already created new {registry_path}")
    print(json.dumps(registry_record, indent=2))

    script_name = "01_secom_data_audit.py"
    graph1_name = "class_distribution.png"
    graph2_name = "missing_rate_histogram.png"
    graph3_name = "correlation_pairs_by_threshold.png"

    X = pd.read_csv("secom/secom.data", sep=r"\s+", header=None)

    y = pd.read_csv("secom/secom_labels.data", sep=r"\s+", header=None,
                               names=["label", "timestamp"])

    # ----------------------------------------------------------------------
    # Class Distribution and Basic Information
    # ----------------------------------------------------------------------

    print("X.shape:", X.shape)
    print("the number of labels:", y.shape[0])
    print("the number of predictor variables:", X.shape[1])


    class_distribution = pd.DataFrame(y["label"].value_counts())
    class_distribution["percentage"] = round((class_distribution["count"] / class_distribution["count"].sum()) * 100, 2)
    class_distribution.reset_index(inplace=True)
    class_distribution['label'] = class_distribution['label'].map({-1: 'Pass', 1: 'Fail'})
    print("Class distribution: ")
    print(class_distribution)

    # ----------------------------------------------------------------------
    # Missing Values Analysis
    # ----------------------------------------------------------------------

    total_missing = X.isnull().sum().sum()
    total_missing
    total_cell = X.shape[0] * X.shape[1]
    missing_percentage = (total_missing / total_cell) * 100

    missing_rate_per_feature = X.isnull().mean() * 100
    bins = [0, 5, 10, 20, 40, 50, 100]
    distribution_of_missing_rate = pd.DataFrame(pd.cut(missing_rate_per_feature, bins=bins).value_counts())
    distribution_of_missing_rate.reset_index(inplace=True)
    distribution_of_missing_rate.rename(columns={"index": "Missing Rate (%)","count": "Number of Features"}, inplace=True)
    distribution_of_missing_rate.sort_values(by='Missing Rate (%)', inplace=True)
    distribution_of_missing_rate["Missing Rate (%)"] = distribution_of_missing_rate["Missing Rate (%)"].astype(str).str.replace({"(": "", "]": "%", ", ": "-"})

    print(f"Total missing values: {total_missing}")
    print(f"Percentage of missing values: {missing_percentage:.2f}%")
    print(distribution_of_missing_rate)

    # ----------------------------------------------------------------------
    # Constant, near-zero variance, and duplicate features
    # ----------------------------------------------------------------------

    constant_features = find_exact_constant_features(X)

    nzv = find_near_zero_variance_features(X)

    duplicated_mask = X.T.duplicated(keep=False)
    duplicate_feature_ids = duplicated_mask[duplicated_mask].index.tolist()

    # ----------------------------------------------------------------------
    # Correlation Audits
    # ----------------------------------------------------------------------

    correlation_summary, upper_matrix, pairs_by_threshold = audit_correlation_structure(X)

    # ----------------------------------------------------------------------
    # Generate a report and save it to a text file
    # ----------------------------------------------------------------------

    content = f"""
X.shape: {X.shape}\n
the number of labels: {y.shape[0]}\n
the number of predictor variables: {X.shape[1]}\n
Class distribution:\n
{class_distribution.to_string(index=False)}\n
Total missing values: {total_missing}\n
Percentage of missing values: {missing_percentage:.2f}%\n
Distribution of missing rates:\n
{distribution_of_missing_rate.to_string(index=False)}\n
the number of constant features: {len(constant_features)}\n
Constant features:\n
{constant_features}\n
Near-zero variance features:\n
{nzv.to_string(index=False)}\n
the number of exact duplicate features: {len(duplicate_feature_ids)}\n
Exact duplicate features:\n
{duplicate_feature_ids}\n
Correlation Summary:\n
the number of correlation pairs by threshold:\n
{correlation_summary}\n
all pairs with correlation above each threshold:\n
Correlation >= 0.90:\n
{pairs_by_threshold[">=|0.9|"].to_string(index=False)}\n
Correlation >= 0.95:\n
{pairs_by_threshold[">=|0.95|"].to_string(index=False)}\n
Correlation >= 0.98:\n
{pairs_by_threshold[">=|0.98|"].to_string(index=False)}\n
Correlation >= 0.995:\n
{pairs_by_threshold[">=|0.995|"].to_string(index=False)}\n
"""

    save_result(
        content=content,
        script_name=script_name,
        filename="01_report.txt"
    )

    # ---------------------------------------------------------------
    # SECOM Class Distribution: Pass vs Fail
    # ---------------------------------------------------------------

    colors = {"Pass": "#4C72B0", "Fail": "#C44E52"}
    bar_colors = class_distribution["label"].map(colors)

    fig1, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        class_distribution["label"],
        class_distribution["count"],
        color=bar_colors,
        width=0.6,
    )

    ax.set_title("SECOM Class Distribution: Pass vs Fail", fontsize=13)
    ax.set_ylabel("Observation Count")
    ax.set_ylim(0, class_distribution["count"].max() * 1.15)

    for bar, count, pct in zip(bars, class_distribution["count"], class_distribution["percentage"]):
        ax.annotate(
            f"{count}\n({pct}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    fig1.tight_layout()
    save_figure(fig1, script_name, graph1_name, dpi=150)
    plt.close(fig1)

    print(f"Saved -> graph/{script_name}/{graph1_name}")

    # ----------------------------------------------------------------------
    # Plot: Histogram-style bar chart of missing rate distribution
    # ----------------------------------------------------------------------

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
    })

    fig2, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        distribution_of_missing_rate["Missing Rate (%)"],
        distribution_of_missing_rate["Number of Features"],
        width=1.0,
        color="#4C72B0",
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_title("Distribution of Missing Rate across Features", fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing Rate (%)")
    ax.set_ylabel("Number of Features")
    ax.grid(axis="y", alpha=0.25)

    for bar, count in zip(bars, distribution_of_missing_rate["Number of Features"]):
        if count > 0:
            ax.annotate(
                str(count),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.xticks(rotation=40, ha="right")
    fig2.tight_layout()

    save_figure(fig2, script_name=script_name, filename=graph2_name, dpi=150)
    plt.close(fig2)

    print(f"Saved -> graph/{script_name}/{graph2_name}")

    # ----------------------------------------------------------------------
    # Plot: Bar chart of number of feature pairs by correlation threshold
    # ----------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
    })

    labels = list(correlation_summary.keys())
    values = list(correlation_summary.values())

    fig3, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#4C72B0", width=0.6, edgecolor="white", linewidth=1.2)

    ax.set_title("Number of Feature Pairs by Correlation Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Correlation Threshold")
    ax.set_ylabel("Number of Feature Pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.15)

    for bar, count in zip(bars, values):
        ax.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig3.tight_layout()
    save_figure(fig3, script_name=script_name, filename=graph3_name, dpi=150)
    plt.close(fig3)

    print(f"Saved -> graph/{script_name}/{graph3_name}")
