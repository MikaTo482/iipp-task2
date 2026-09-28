
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils import (find_temporal_cutoff_candidates, 
                   missing_rate_filter, correlation_filter, 
                   duplicate_filter, save_figure,  VotingFeatureSelector)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn.metrics import (classification_report, recall_score, 
                             confusion_matrix, average_precision_score, 
                             precision_score, f1_score, matthews_corrcoef, 
                             roc_auc_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from boruta import BorutaPy
from itertools import combinations
import shap

SHAP_ENABLED = True
SHAP_MAX_BACKGROUND = 100

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoost is required by the locked protocol "
                      "(selector and model). Install with: pip install xgboost") from e

TARGET_COL = "label"
SPLIT_DIR = "splits"
VAL_SIZE = 0.1
RANDOM_SEED = 42

SPLIT_DIR_RANDOM  = os.path.join(SPLIT_DIR, "random")
SPLIT_DIR_TEMPORAL = os.path.join(SPLIT_DIR, "temporal")
DEV_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "dev_indices_random.csv")
TEST_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "test_indices_random.csv")
SPLIT_CONFIG_PATH_RANDOM = os.path.join(SPLIT_DIR_RANDOM, "split_config.json")
DEV_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "dev_temporal.csv")
SPLIT_CONFIG_PATH_TEMPORAL = os.path.join(SPLIT_DIR_TEMPORAL, "split_config.json")
TEST_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "test_temporal.csv")
RESULTS_DIR = "experiments"

N_SPLITS = 5
N_REPEATS = 10
N_JOBS = 1

MISSING_THRESHOLD = 50
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.95

SELECTORS = ["none", "mi", "l1", "xgb", "boruta"]   # "none" = all-feature reference for RQ1
IMBALANCE = ["none", "weight", "smote"]
MODELS = ["lr", "rf", "xgb"]

SELECTOR_PARAMS = {
    "mi": dict(top_k=20),
    "l1": dict(C=0.1),
    "xgb": dict(top_k=20, n_estimators=300, max_depth=3, learning_rate=0.05),
    "boruta": dict(max_iter=100, rf_max_depth=5),
}

MODEL_PARAMS = {
        "lr": dict(C=1.0, max_iter=5000),
        "rf": dict(n_estimators=500, min_samples_leaf=1),
        "xgb": dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist"),
}

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
          f"({X.shape[1]} features)")
    return df

def load_protected_split(dev_path, test_path, config_path):
    missing = [p for p in (dev_path, test_path, config_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Split file(s) not found:\n" + "\n".join(f"  - {p}" for p in missing)
            + "\nPlease run 03_protected_split.py first.")

    dev_idx = pd.read_csv(dev_path)["index"].to_numpy()
    test_idx = pd.read_csv(test_path)["index"].to_numpy()

    with open(config_path) as f:
        config = json.load(f)

    print(f"Loaded split seed={config['random_seed']}, "
          f"created_at={config['created_at']}, n_dev={config['n_dev']}")
    return dev_idx, test_idx

def apply_imbalance(X, y, condition, seed):
    if condition == "smote":
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
        return X_res, np.asarray(y_res), False
    return X, y, condition == "weight"

def setup_experiment_directory(split_dir: str = SPLIT_DIR):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        print(f"created '{split_dir}' successfully")
    else:
        print(f"folder '{split_dir}' already exists, continuing...")

def compute_scale_pos_weight(y):
    y = np.asarray(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0:
        return 1.0

    return n_neg / n_pos

def select_features(name, X, y, use_weights, seed):
    cols = X.columns
    cw = "balanced" if use_weights else None
    P = SELECTOR_PARAMS.get(name, {})
 
    if name == "none":                  # all-feature reference (RQ1)
        return list(cols)
 
    if name == "mi":                    # no weighting option -> 'weight' == 'none'
        scores = mutual_info_classif(X, y, random_state=seed)
        return list(cols[np.argsort(scores)[::-1][:P["top_k"]]])
 
    if name == "l1":
        m = LogisticRegression(penalty="l1", solver="liblinear", C=P["C"],
                               class_weight=cw, max_iter=5000, random_state=seed)
        m.fit(X, y)
        return list(cols[np.abs(m.coef_[0]) > 1e-8])
 
    if name == "xgb":
        xgb_params = dict(
            n_estimators=P["n_estimators"],
            max_depth=P["max_depth"],
            learning_rate=P["learning_rate"],
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=N_JOBS,
        )

        if use_weights:
            xgb_params["scale_pos_weight"] = compute_scale_pos_weight(y)

        m = XGBClassifier(**xgb_params)
        m.fit(X, y)
        return list(cols[np.argsort(m.feature_importances_)[::-1][:P["top_k"]]])
 
    if name == "boruta":
        rf = RandomForestClassifier(max_depth=P["rf_max_depth"], class_weight=cw,
                                    random_state=seed, n_jobs=N_JOBS)
        b = BorutaPy(rf, n_estimators="auto", max_iter=P["max_iter"], random_state=seed)
        b.fit(X.to_numpy(), y)
        return list(cols[b.support_])
 
    raise ValueError(f"Unknown selector: {name}")

def build_model(name, use_weights, y_train, seed):
    cw = "balanced" if use_weights else None
    P = MODEL_PARAMS[name]
    if name == "lr":
        return LogisticRegression(class_weight=cw, random_state=seed, **P)
    if name == "rf":
        return RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=N_JOBS, **P)
    if name == "xgb":
        xgb_params = P.copy()
        if use_weights:
            xgb_params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, **xgb_params)
    
    raise ValueError(f"Unknown model: {name}")

# =========================================================================
# Preprocessing - fitted on the TRAINING FOLD only, applied to the validation fold
# =========================================================================
def preprocess_fold(X_train, X_val):
    missing_list = missing_rate_filter(X_train, MISSING_THRESHOLD)
    X_train, X_val = X_train.drop(columns=missing_list), X_val.drop(columns=missing_list)

    if USE_CORRELATION_FILTER:
        correlation_list = correlation_filter(X_train, CORRELATION_THRESHOLD)
        X_train, X_val = X_train.drop(columns=correlation_list), X_val.drop(columns=correlation_list)

    duplicate_list = duplicate_filter(X_train)
    X_train, X_val = X_train.drop(columns=duplicate_list), X_val.drop(columns=duplicate_list)

    pipeline = Pipeline([
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("median_imputation", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("standard_scaling", StandardScaler()),
    ]).set_output(transform="pandas")              # keep column names for feature selection

    X_train_p = pipeline.fit_transform(X_train)    # fit on training fold only
    X_val_p = pipeline.transform(X_val)            # transform only
    return X_train_p, X_val_p


def to_matrix_from_names(subsets, feature_universe):
    feature_to_idx = {f: i for i, f in enumerate(feature_universe)}
    Z = np.zeros((len(subsets), len(feature_universe)), dtype=int)

    for i, s in enumerate(subsets):
        idx = [feature_to_idx[f] for f in s if f in feature_to_idx]
        Z[i, idx] = 1

    return Z

def selection_frequency(Z):
    return Z.mean(axis=0)

def nogueira_stability(Z):
    M, p = Z.shape
    p_hat = Z.mean(axis=0)
    s2 = M / (M - 1) * p_hat * (1 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / p) * (1 - k_bar / p)
    if denom == 0:
        return np.nan
    return 1 - s2.mean() / denom

def kuncheva_index(subsets, p):
    k = len(subsets[0])
    assert all(len(s) == k for s in subsets)
    assert 0 < k < p
    vals = [(len(set(a) & set(b)) * p - k**2) / (k * (p - k))
            for a, b in combinations(subsets, 2)]
    return np.mean(vals)

def mean_jaccard(subsets):
    vals = []
    for a, b in combinations(subsets, 2):
        a, b = set(a), set(b)
        union = a | b
        vals.append(len(a & b) / len(union) if union else 1.0)
    return np.mean(vals)

def summarize_feature_stability(selections_df, feature_universe):
    rows = []
    freq_rows = []
    p = len(feature_universe)

    for (imbalance, selector), g in selections_df.groupby(["imbalance", "selector"]):
        subsets = g["features"].tolist()
        Z = to_matrix_from_names(subsets, feature_universe)

        subset_sizes = np.array([len(s) for s in subsets])
        same_size = np.all(subset_sizes == subset_sizes[0])

        row = {
            "imbalance": imbalance,
            "selector": selector,
            "n_runs": len(subsets),
            "n_features_mean": subset_sizes.mean(),
            "n_features_std": subset_sizes.std(ddof=1) if len(subset_sizes) > 1 else 0,
            "n_features_min": subset_sizes.min(),
            "n_features_max": subset_sizes.max(),
            "nogueira": nogueira_stability(Z),
            "mean_jaccard": mean_jaccard(subsets),
            "kuncheva": kuncheva_index(subsets, p) if same_size and 0 < subset_sizes[0] < p else np.nan,
        }
        rows.append(row)

        freq = selection_frequency(Z)
        counts = Z.sum(axis=0)
        for feature, f, c in zip(feature_universe, freq, counts):
            freq_rows.append({
                "imbalance": imbalance,
                "selector": selector,
                "feature": feature,
                "selection_frequency": f,
                "selected_count": int(c),
                "n_runs": len(subsets),
            })

    stability_df = pd.DataFrame(rows).sort_values(
        ["imbalance", "selector"], ignore_index=True
    )

    feature_frequency_df = pd.DataFrame(freq_rows).sort_values(
        ["imbalance", "selector", "selection_frequency"],
        ascending=[True, True, False],
        ignore_index=True,
    )

    return stability_df, feature_frequency_df

# =========================================================================
# SHAP - post-hoc explanation only (never used to build the stable feature set)
# =========================================================================
def _positive_class(sv):
    if isinstance(sv, list):          
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                
        sv = sv[..., 1]
    return sv


def compute_shap(model, model_name, X_background, X_explain, seed):
    if model_name == "lr":
        bg = X_background.sample(min(SHAP_MAX_BACKGROUND, len(X_background)),
                                 random_state=seed)
        explainer = shap.LinearExplainer(
            model, shap.maskers.Independent(bg, max_samples=SHAP_MAX_BACKGROUND))
        sv = explainer.shap_values(X_explain)
        space = "log_odds"
    elif model_name in ("rf", "xgb"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain, check_additivity=False)
        space = "probability" if model_name == "rf" else "log_odds"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    sv = _positive_class(sv)
    assert sv.shape == X_explain.shape, f"SHAP shape {sv.shape} != {X_explain.shape}"
    return sv, space


def shap_to_records(sv, features, space, **keys):
    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)
    ranks = pd.Series(mean_abs).rank(ascending=False, method="average").to_numpy()
    return [{**keys, "feature": f, "mean_abs_shap": float(a), "mean_shap": float(s),
             "shap_rank": float(r), "n_features_in_model": len(features),
             "output_space": space}
            for f, a, s, r in zip(features, mean_abs, mean_signed, ranks)]


if __name__ == "__main__":
    setup_experiment_directory(RESULTS_DIR)

    df = load_secom_data()
    dev_idx, test_idx = load_protected_split(DEV_RANDOM_PATH, TEST_RANDOM_PATH, SPLIT_CONFIG_PATH_RANDOM)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Development set shape {dev_df.shape}")
    print(dev_df.columns.tolist())

    X = dev_df.drop([TARGET_COL, "timestamp"], axis = 1)
    X.columns = [f"f{c}" for c in X.columns] 
    feature_universe = X.columns.tolist()  
    y = (dev_df[TARGET_COL] == 1).astype(int) 

    # =========================================================================
    # TRACK A
    # =========================================================================

    results, selections, shap_rows = [], [], []
    n_folds = N_SPLITS * N_REPEATS
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    for fold_id, (tr, va) in enumerate(rskf.split(X, y), start=1):
        # seed = RANDOM_SEED + fold_id
        seed = RANDOM_SEED
        repeat = (fold_id - 1) // N_SPLITS + 1

        X_train_processed, X_val_processed = preprocess_fold(X.iloc[tr], X.iloc[va])
        y_train, y_val = y.iloc[tr], y.iloc[va]

        for imb in IMBALANCE:
            X_train_imb, y_train_imb, use_weight = apply_imbalance(X_train_processed, y_train, imb, seed)

            for selector in SELECTORS:
                features = select_features(selector, X_train_imb, y_train_imb, use_weight, seed)
                selections.append({
                    "fold": fold_id,
                    "repeat": repeat,
                    "imbalance": imb,
                    "selector": selector,
                    "n_features_after_preprocessing": X_train_processed.shape[1],
                    "n_features_selected": len(features),
                    "features": features,
                })

                if not features:
                    print(f"  [warning] fold {fold_id} {imb}|{selector}: no features selected")
                    continue

                for mdl in MODELS:
                    model = build_model(mdl, use_weight, y_train_imb, seed)
                    model.fit(X_train_imb[features], y_train_imb)
                    y_proba = model.predict_proba(X_val_processed[features])[:, 1]
                    y_pred = (y_proba >= 0.5).astype(int)

                    results.append({
                        "fold": fold_id, "repeat": repeat, "imbalance": imb,
                        "selector": selector, "model": mdl, "n_features": len(features),
                        "pr_auc": average_precision_score(y_val, y_proba),   # primary metric
                        "roc_auc": roc_auc_score(y_val, y_proba),
                        "recall": recall_score(y_val, y_pred, zero_division=0),
                        "precision": precision_score(y_val, y_pred, zero_division=0),
                        "f1": f1_score(y_val, y_pred, zero_division=0),
                        "mcc": matthews_corrcoef(y_val, y_pred),
                    })
                    if SHAP_ENABLED:
                        sv, space = compute_shap(
                            model, mdl,
                            X_background=X_train_processed[features],   # ก่อน SMOTE
                            X_explain=X_val_processed[features],
                            seed=seed)
                        shap_rows.extend(shap_to_records(
                            sv, features, space,
                            fold=fold_id, repeat=repeat, imbalance=imb,
                            selector=selector, model=mdl))

        print(f"fold {fold_id}/{n_folds} done")

    # =========================================================================
    # Save results
    # =========================================================================

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)

    stability_df, feature_frequency_df = summarize_feature_stability(
        selections_df,
        feature_universe
    )

    summary_df = (
        results_df.groupby(["imbalance", "selector", "model"])
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            mcc_mean=("mcc", "mean"),
            n_features_mean=("n_features", "mean"),
        )
        .reset_index()
        .sort_values("pr_auc_mean", ascending=False, ignore_index=True)
    )

    shap_df = pd.DataFrame(shap_rows)

    for name, frame in (
        ("results", results_df),
        ("selections", selections_df),
        ("summary", summary_df),
        ("stability", stability_df),
        ("feature_frequency", feature_frequency_df),
        ("shap", shap_df),  
    ):
        frame.to_json(
            os.path.join(RESULTS_DIR, f"{run_id}_{name}.json"),
            orient="records",
            indent=2,
            double_precision=15,
        )

    config = {
        "run_id": run_id, "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
        "missing_threshold": MISSING_THRESHOLD,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "selectors": SELECTORS, "imbalance": IMBALANCE, "models": MODELS,
        "selector_params": SELECTOR_PARAMS, "model_params": MODEL_PARAMS,
        "n_dev": len(dev_df), "n_features_original": len(feature_universe),
        "positive_rate_dev": float(y.mean()),
        "shap_enabled": SHAP_ENABLED,
        "shap_explain_data": "validation_fold",
        "shap_lr_background": "training_fold_before_smote",
        "shap_max_background": SHAP_MAX_BACKGROUND,
        "shap_version": shap.__version__,
    }
    with open(os.path.join(RESULTS_DIR, f"{run_id}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(summary_df.round(3).to_string(index=False))
    print(stability_df.round(3).to_string(index=False))
    print(feature_frequency_df.to_string(index=False))
    print(shap_df.to_string(index=False))




    