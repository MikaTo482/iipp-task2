import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils import find_temporal_cutoff_candidates

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

RANDOM_SEED = 42
TARGET_COL = "label"
TEST_SIZE = 0.10

SPLIT_DIR = "splits"
SPLIT_DIR_RANDOM  = os.path.join(SPLIT_DIR, "random")
SPLIT_DIR_TEMPORAL = os.path.join(SPLIT_DIR, "temporal")
DEV_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "dev_indices_random.csv")
TEST_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "test_indices_random.csv")
SPLIT_CONFIG_PATH_RANDOM = os.path.join(SPLIT_DIR_RANDOM, "split_config.json")
DEV_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "dev_temporal.csv")
SPLIT_CONFIG_PATH_TEMPORAL = os.path.join(SPLIT_DIR_TEMPORAL, "split_config.json")
TEST_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "test_temporal.csv")


# =========================================================================
# Load & Merge Data
# =========================================================================
def load_secom_data(data_path="secom/secom.data", label_path="secom/secom_labels.data"):
    missing = [p for p in (data_path, label_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "SECOM file(s) not found:\n"
            + "\n".join(f"  - {os.path.abspath(p)}" for p in missing)
            + f"\nCurrent working directory: {os.getcwd()}"
            + "\nPlease check the path."
        )
    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    y = pd.read_csv(label_path, sep=r"\s+", header=None,
                    names=["label", "timestamp"])

    assert len(X) == len(y), "Mismatch in number of rows between X and y"
    assert X.index.equals(y.index), "Index of X and y do not match!"

    df = pd.concat([X, y], axis=1)
    print(f"Loaded SECOM: {df.shape[0]} rows, {df.shape[1]} columns "
          f"({X.shape[1]} features")
    return df

# =========================================================================
# create a directory to store the split indices and configuration
# =========================================================================
def setup_split_directory(split_dir: str = SPLIT_DIR):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        print(f"created '{split_dir}' successfully")
    else:
        print(f"folder '{split_dir}' already exists, continuing...")

# =========================================================================
# create a protected split of the dataset into development and test sets
# =========================================================================
def create_protected_split(df: pd.DataFrame, target_col: str):

    if not os.path.isdir(SPLIT_DIR):
        raise FileNotFoundError(
            f"Not found '{SPLIT_DIR}' Please run setup_split_directory() first to create the directory."
        )

    if not os.path.exists(SPLIT_DIR_RANDOM):
        os.makedirs(SPLIT_DIR_RANDOM, exist_ok=True)
        print(f"created '{SPLIT_DIR_RANDOM}' successfully")
    else:
        print(f"folder '{SPLIT_DIR_RANDOM}' already exists, continuing...")

    if os.path.exists(DEV_RANDOM_PATH) or os.path.exists(TEST_RANDOM_PATH):
        raise FileExistsError(
            f"⚠️ Protected Test Split already existed! "
            f"please delete the existing files in the {SPLIT_DIR_RANDOM} folder if you want to create a new split."
        )

    # ---------- Stratified split ----------
    all_idx = np.arange(len(df))
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        stratify=df[target_col],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # ---------- Save indices as .csv ----------
    pd.DataFrame({"index": dev_idx}).to_csv(DEV_RANDOM_PATH, index=False)
    pd.DataFrame({"index": test_idx}).to_csv(TEST_RANDOM_PATH, index=False)

    # ---------- Save metadata ----------
    config = {
        "split_type": "random",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "target_col": target_col,
        "n_total": len(df),
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "created_at": datetime.now().isoformat(),
    }

    with open(SPLIT_CONFIG_PATH_RANDOM, "w") as f:
        json.dump(config, f, indent=2)

    print("=== Stratified Protected Test Split Created ===")
    print(f"Dev size: {len(dev_idx)} | Test size: {len(test_idx)}")
    print("\nClass distribution (Dev):")
    print(df.iloc[dev_idx][target_col].value_counts(normalize=True))
    print("\nClass distribution (Test):")
    print(df.iloc[test_idx][target_col].value_counts(normalize=True))

def create_temporal_split(df, target, target_list, target_col=TARGET_COL):

    if not os.path.isdir(SPLIT_DIR):
        raise FileNotFoundError(
            f"Not found '{SPLIT_DIR}'. Please run setup_split_directory() first."
        )

    os.makedirs(SPLIT_DIR_TEMPORAL, exist_ok=True)

    existing = [p for p in (DEV_TEMPORAL_PATH, TEST_TEMPORAL_PATH, SPLIT_CONFIG_PATH_TEMPORAL)
                if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            f"⚠️ Protected temporal split already exists: {existing}. "
            f"Please delete the existing files in the '{SPLIT_DIR_TEMPORAL}' folder "
            f"if you want to create a new split."
        )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    target_fail_count = int(target)

    cutoff_input = df[[TARGET_COL, "timestamp"]].copy()

    candidates = find_temporal_cutoff_candidates(cutoff_input, target_fail_counts=target_list)
    print(candidates)

    cutoff = candidates[target_fail_count]["cutoff_timestamp"]
    print(cutoff)

    dev_temporal = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test_temporal = df[df["timestamp"] > cutoff].reset_index(drop=True)

    print(f"Train temporal shape: {dev_temporal.shape}")
    print(f"Test temporal shape: {test_temporal.shape}")
    print("Class Distribution (Train) \n")
    print(dev_temporal[target_col].value_counts())
    print("Class Distribution (Test) \n")
    print(test_temporal[target_col].value_counts())

    dev_temporal.to_csv(DEV_TEMPORAL_PATH, index=False)
    test_temporal.to_csv(TEST_TEMPORAL_PATH, index=False)

    config = {
        "split_type": "temporal",
        "target_col": target_col,
        "target_fail_count": target_fail_count,
        "candidate_fail_counts": [int(c) for c in target_list],
        "cutoff_timestamp": pd.Timestamp(cutoff).isoformat(),
        "n_total": len(df),
        "n_dev": len(dev_temporal),
        "n_test": len(test_temporal),
        "n_fail_dev": int((dev_temporal[target_col] == 1).sum()),
        "n_fail_test": int((test_temporal[target_col] == 1).sum()),
        "class_counts_dev": {str(k): int(v) for k, v in dev_temporal[target_col].value_counts().items()},
        "class_counts_test": {str(k): int(v) for k, v in test_temporal[target_col].value_counts().items()},
        "dev_time_range": [dev_temporal["timestamp"].min().isoformat(), dev_temporal["timestamp"].max().isoformat()],
        "test_time_range": [test_temporal["timestamp"].min().isoformat(), test_temporal["timestamp"].max().isoformat()],
        "created_at": datetime.now().isoformat(),
    }

    with open(SPLIT_CONFIG_PATH_TEMPORAL, "w") as f:
        json.dump(config, f, indent=2)

# =========================================================================
# Main execution
# =========================================================================
if __name__ == "__main__":

    df = load_secom_data()

    setup_split_directory()

    create_protected_split(df, target_col=TARGET_COL)

    create_temporal_split(df, target=15, target_list=[15, 20, 25], target_col=TARGET_COL)





