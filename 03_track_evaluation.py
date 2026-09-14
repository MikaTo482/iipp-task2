import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils import find_temporal_cutoff_candidates

RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
SPLIT_DIR = "splits"
TARGET_COL = "label"
DEV_INDICES_PATH = os.path.join(SPLIT_DIR, "dev_indices.csv")
TEST_INDICES_PATH = os.path.join(SPLIT_DIR, "test_indices.csv")
SPLIT_CONFIG_PATH = os.path.join(SPLIT_DIR, "split_config.json")
LOG_PATH = os.path.join(SPLIT_DIR, "test_set_access_log.csv")

# =========================================================================
# Load & Merge Data
# =========================================================================
def load_secom_data(data_path="secom/secom.data", label_path="secom/secom_labels.data"):
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

    if os.path.exists(DEV_INDICES_PATH) or os.path.exists(TEST_INDICES_PATH):
        raise FileExistsError(
            "⚠️ Protected Test Split have existed! "
            "please delete the existing files in the 'splits' folder if you want to create a new split."
        )

    # ---------- Stratified split ----------
    all_idx = np.arange(len(df)) # [0, 1, 2, ..., n-1]
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        stratify=df[target_col],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # ---------- Save indices as .csv ----------
    pd.DataFrame({"index": dev_idx}).to_csv(DEV_INDICES_PATH, index=False)
    pd.DataFrame({"index": test_idx}).to_csv(TEST_INDICES_PATH, index=False)

    # ---------- Save metadata ----------
    config = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "target_col": target_col,
        "n_total": len(df),
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "created_at": datetime.now().isoformat(),
    }

    with open(SPLIT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("=== Stratified Protected Test Split Created ===")
    print(f"Dev size: {len(dev_idx)} | Test size: {len(test_idx)}")
    print("\nClass distribution (Dev):")
    print(df.iloc[dev_idx][target_col].value_counts(normalize=True))
    print("\nClass distribution (Test):")
    print(df.iloc[test_idx][target_col].value_counts(normalize=True))

    return dev_idx, test_idx

def load_protected_split():
    if not os.path.exists(DEV_INDICES_PATH):
        raise FileNotFoundError(f"Not found: {DEV_INDICES_PATH} please run create_protected_split() first.")

    dev_idx = pd.read_csv(DEV_INDICES_PATH)["index"].to_numpy()

    with open(SPLIT_CONFIG_PATH) as f:
        config = json.load(f)

    print(f"Loaded split seed={config['random_seed']}, "
          f"created_at={config['created_at']}, n_dev={config['n_dev']}")
    return dev_idx


# =========================================================================
# Test Set Access Log + Leakage Guard
# =========================================================================

FORBIDDEN_KEYWORDS = [
    "feature selection", "model selection", "hyperparameter",
    "tuning", "select feature", "choose model", "compare model",
]

def log_test_access(purpose: str, accessed_by: str = "unknown", details: str = ""):
    purpose_lower = purpose.lower()
    violated = [kw for kw in FORBIDDEN_KEYWORDS if kw in purpose_lower]
    if violated:
        raise PermissionError(
            f"BLOCKED: this purpose have a forbidden keyword {violated}\n"
        )

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "accessed_by": accessed_by,
        "details": details,
    }

    if os.path.exists(LOG_PATH):
        log_df = pd.concat([pd.read_csv(LOG_PATH), pd.DataFrame([entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([entry])

    log_df.to_csv(LOG_PATH, index=False)
    print(f"Logged test access '{purpose}' (total entries: {len(log_df)})")


def _load_test_indices() -> np.ndarray:
    if not os.path.exists(TEST_INDICES_PATH):
        raise FileNotFoundError(f"Not found: {TEST_INDICES_PATH}")
    return pd.read_csv(TEST_INDICES_PATH)["index"].to_numpy()


def assert_no_test_leakage(indices, fold_name: str = "training fold"):
    test_idx = set(_load_test_indices())
    given_idx = set(np.asarray(indices).ravel().tolist())
    contaminated = given_idx & test_idx

    if contaminated:
        raise PermissionError(
            f"TEST SET LEAKAGE DETECTED in '{fold_name}'\n"
            f"found test index: {len(contaminated)} indices\n"
        )

def get_test_set(df: pd.DataFrame, purpose: str, accessed_by: str, details: str = ""):
    log_test_access(purpose=purpose, accessed_by=accessed_by, details=details)
    return df.iloc[_load_test_indices()].reset_index(drop=True)


def view_test_access_log():
    if not os.path.exists(LOG_PATH):
        print("No test set access log found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["timestamp", "purpose", "accessed_by", "details"])

    log_df = pd.read_csv(LOG_PATH)
    print(f"=== Test Set Access Log ({len(log_df)} entries) ===")
    print(log_df.to_string(index=False))

    if len(log_df) > 3:
        print(f"\nWARNING: Test set is touched {len(log_df)} times."
                " Please check the log carefully to avoid data leakage.")
    return log_df

# =========================================================================
# PART 4 - Main execution
# =========================================================================
if __name__ == "__main__":

    df = load_secom_data()

    setup_split_directory()

    if not os.path.exists(DEV_INDICES_PATH):
        dev_idx, _ = create_protected_split(df, target_col=TARGET_COL)
    else:
        dev_idx = load_protected_split()

    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    assert_no_test_leakage(dev_idx, "Development Set")

    print(f"Development set shape {dev_df.shape}")
    print(dev_df.columns.tolist())

    X = dev_df.drop([TARGET_COL, "timestamp"], axis = 1)
    y = dev_df[TARGET_COL]

    # =========================================================================
    # TRACK A
    # =========================================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        stratify=dev_df['label'],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    pipeline = Pipeline([
        ('remove_constant', VarianceThreshold(threshold=0.0)), # constant features removal
        ('median_imputation', SimpleImputer(missing_values=np.nan, strategy='median')), # imputation
        ('standard_scaling', StandardScaler()),
    ])

    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_val_processed = pipeline.transform(X_val)

    print("=== Stratified Protected Validation Split Created ===")
    print(f"Train size: {X_train.shape[0]} | Validation size: {X_val.shape[0]}")
    print("\nClass distribution (Test):")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution (Validation):")
    print(y_val.value_counts(normalize=True))

    print(f"Number of features in X_train: {X_train.shape[1]} | Number of features in X_train processed: {X_train_processed.shape[1]}")
    print(f"Number of features in X_val: {X_val.shape[1]} | Number of features in X_val processed: {X_val_processed.shape[1]}")


    # =========================================================================
    # TRACK B
    # =========================================================================

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    target_fail_count = int(15)

    cutoff_input = df[[TARGET_COL, "timestamp"]].copy()

    candidates = find_temporal_cutoff_candidates(cutoff_input, target_fail_counts=[15, 20, 25])
    print(candidates)

    cutoff = candidates[target_fail_count]["cutoff_timestamp"]
    print(cutoff)

    train_temporal = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test_temporal = df[df["timestamp"] > cutoff].reset_index(drop=True)

    print(f"Train temporal shape: {train_temporal.shape}")
    print(f"Test temporal shape: {test_temporal.shape}")
    print("Class Distribution (Train) \n")
    print(train_temporal[TARGET_COL].value_counts())
    print("Class Distribution (Test) \n")
    print(test_temporal[TARGET_COL].value_counts())

    view_test_access_log()




