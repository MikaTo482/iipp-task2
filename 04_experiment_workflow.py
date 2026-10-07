import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif, RFE, SelectKBest, SelectFdr, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from utils import (missing_rate_filter, 
                   correlation_filter, 
                   duplicate_filter)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, recall_score, confusion_matrix, 
                             average_precision_score, precision_score)
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
from itertools import combinations
import warnings
import platform


try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoost is required by the locked protocol "
                      "(selector and model). Install with: pip install xgboost") from e

TARGET_COL = "label"
SPLIT_DIR = "splits"
DECISION_THRESHOLD = 0.5

SPLIT_DIR_RANDOM  = os.path.join(SPLIT_DIR, "random")
SPLIT_DIR_TEMPORAL = os.path.join(SPLIT_DIR, "temporal")
DEV_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "dev_indices_random.csv")
TEST_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "test_indices_random.csv")
SPLIT_CONFIG_PATH_RANDOM = os.path.join(SPLIT_DIR_RANDOM, "split_config.json")
DEV_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "dev_temporal.csv")
SPLIT_CONFIG_PATH_TEMPORAL = os.path.join(SPLIT_DIR_TEMPORAL, "split_config.json")
TEST_TEMPORAL_PATH = os.path.join(SPLIT_DIR_TEMPORAL, "test_temporal.csv")

N_SPLITS = 5
N_REPEATS = 10
N_JOBS = 1

RANDOM_SEED = 42
ALGORITHM_SEED = RANDOM_SEED
RESULTS_DIR = "experiments"

MISSING_THRESHOLD = 50
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.8

N_SPLITS = 5
N_REPEATS = 30
N_JOBS = 1

def resampling_seed(fold_id):
    return RANDOM_SEED + fold_id

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

def setup_experiment_directory(split_dir: str = SPLIT_DIR):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        print(f"created '{split_dir}' successfully")
    else:
        print(f"folder '{split_dir}' already exists, continuing...")

def apply_imbalance(X, y, condition, seed):
    if condition == "smote":
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
        return X_res, np.asarray(y_res), False
    return X, y, condition == "weight"

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

def build_model(name, seed):
    P = MODEL_PARAMS[name]
    if name == "lr":
        return LogisticRegression(random_state=seed, **P)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **P)
    if name == "xgb":
        xgb_params = P.copy()
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, **xgb_params)
    
    raise ValueError(f"Unknown model: {name}")

def preprocess_fold(X_train, X_val):
    n_original = X_train.shape[1]
    missing_list = missing_rate_filter(X_train, MISSING_THRESHOLD)
    X_train, X_val = X_train.drop(columns=missing_list), X_val.drop(columns=missing_list)
    n_after_missing = X_train.shape[1]

    if USE_CORRELATION_FILTER:
        correlation_list = correlation_filter(X_train, CORRELATION_THRESHOLD)
        X_train, X_val = X_train.drop(columns=correlation_list), X_val.drop(columns=correlation_list)
        n_after_correlation = X_train.shape[1]
    else:
        correlation_list = []
        n_after_correlation = n_after_missing

    duplicate_list = duplicate_filter(X_train)
    X_train, X_val = X_train.drop(columns=duplicate_list), X_val.drop(columns=duplicate_list)
    n_after_duplicate = X_train.shape[1]

    pipeline = Pipeline([
            ("remove_constant", VarianceThreshold(threshold=0.0)),
            ("median_imputation", SimpleImputer(missing_values=np.nan, strategy="median")),
            # ("missforest_imputation", missforest),
            ("robust_scaling", RobustScaler()),
        ]).set_output(transform="pandas")

    X_train_p = pipeline.fit_transform(X_train)
    X_val_p = pipeline.transform(X_val)
    
    n_after_constant = X_train_p.shape[1]

    keep_mask = pipeline.named_steps["remove_constant"].get_support()
    constant_list = list(X_train.columns[~keep_mask])

    report_removal = {
        "missing": missing_list,
        "correlation": correlation_list,
        "duplicate": duplicate_list,
        "constant": constant_list,
        "n_features_before": n_original,
        "n_features_after": X_train_p.shape[1],
        "n_after_missing": n_after_missing,
        "n_after_correlation": n_after_correlation,
        "n_after_duplicate": n_after_duplicate,
        "n_after_constant": n_after_constant
    }

    return X_train_p, X_val_p, report_removal

def to_matrix_from_names(subsets, feature_universe):
    if len(set(feature_universe)) != len(feature_universe):
        raise ValueError("feature_universe must contain unique feature names")
    feature_to_idx = {f: i for i, f in enumerate(feature_universe)}
    Z = np.zeros((len(subsets), len(feature_universe)), dtype=int)

    for i, s in enumerate(subsets):
        if len(set(s)) != len(s):
            raise ValueError(f"Subset {i} contains duplicate feature names")
        unknown = set(s) - feature_to_idx.keys()
        if unknown:
            raise ValueError(f"Subset {i} contains features outside feature_universe: {sorted(unknown)}")
        idx = [feature_to_idx[f] for f in s]
        Z[i, idx] = 1

    return Z

def selection_frequency(Z):
    return Z.mean(axis=0)

def nogueira_stability(Z):
    Z = np.asarray(Z)
    if Z.ndim != 2 or not np.isin(Z, [0, 1]).all():
        raise ValueError("Z must be a two-dimensional binary selection matrix")
    M, d = Z.shape
    if M < 2 or d == 0:
        return np.nan
    p_hat = Z.mean(axis=0)
    s2 = M / (M - 1) * p_hat * (1 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    if k_bar == 0 or k_bar == d:
        return np.nan
    denom = (k_bar / d) * (1 - k_bar / d)
    return 1 - s2.mean() / denom

def kuncheva_index(subsets, p):
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or p <= 0:
        raise ValueError("p must be a positive integer feature-universe size")
    if len(subsets) < 2:
        return np.nan
    k = len(subsets[0])
    if not all(len(s) == k for s in subsets) or not 0 < k < p:
        raise ValueError("Kuncheva requires equal subset sizes with 0 < k < p")
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
    """Summarize preprocessing + selection using one candidate-name union."""
    rows = []
    freq_rows = []
    d = len(feature_universe)

    for (imbalance, selector), g in selections_df.groupby(["imbalance", "selector"]):
        subsets = g["features"].tolist()
        Z = to_matrix_from_names(subsets, feature_universe)

        subset_sizes = np.array([len(s) for s in subsets])
        same_size = np.all(subset_sizes == subset_sizes[0])

        rows.append({
            "imbalance": imbalance,
            "selector": selector,
            "n_runs": len(subsets),
            "n_features_mean": subset_sizes.mean(),
            "n_features_std": subset_sizes.std(ddof=1) if len(subset_sizes) > 1 else 0,
            "n_features_min": subset_sizes.min(),
            "n_features_max": subset_sizes.max(),
            "nogueira": nogueira_stability(Z),
            "mean_jaccard": mean_jaccard(subsets),
            "kuncheva": (kuncheva_index(subsets, d)
                         if same_size and 0 < subset_sizes[0] < d else np.nan),
        })

        available = to_matrix_from_names(g["candidate_features"].tolist(),
                                         feature_universe).sum(axis=0)
        freq = selection_frequency(Z)
        counts = Z.sum(axis=0)
        for feature, f, c, a in zip(feature_universe, freq, counts, available):
            freq_rows.append({
                "imbalance": imbalance,
                "selector": selector,
                "feature": feature,
                "selection_frequency": f,
                "selected_count": int(c),
                "n_runs": len(subsets),
                "n_available": int(a),
                "frequency_when_available": c / a if a > 0 else np.nan,
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

SELECTORS = ["none", "mi", "l1", "xgb", "boruta"]
IMBALANCE = ["none", "weight", "smote"]
MODELS = ["lr", "rf", "xgb"]

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
    n_features_original = X.shape[1]
    y = (dev_df[TARGET_COL] == 1).astype(int) 

    results, selections, shap_rows = [], [], []
    removed_features = {}
    n_folds = N_SPLITS * N_REPEATS

    start_exp_time = datetime.now()
    print(f"Experiment started at {start_exp_time.isoformat()}")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    for fold_id, (tr, va) in enumerate(rskf.split(X, y), start=1):
        start_time = datetime.now()
        print(f"\nFold {fold_id}/{n_folds} started at {start_time.isoformat()}")

        seed = ALGORITHM_SEED
        data_seed = resampling_seed(fold_id)
        repeat = (fold_id - 1) // N_SPLITS + 1

        X_train_processed, X_val_processed, report_removal = preprocess_fold(X.iloc[tr], X.iloc[va])
        removed_features[fold_id] = report_removal
        y_train, y_val = y.iloc[tr], y.iloc[va]

        for imb in IMBALANCE:
            X_train_imb, y_train_imb, use_weight = apply_imbalance(X_train_processed, y_train, imb, data_seed)

            for selector in SELECTORS:
                t0 = time.perf_counter()
                features = select_features(selector, X_train_imb, y_train_imb, use_weight, seed)
                selection_time = time.perf_counter() - t0
                selections.append({
                    "fold": fold_id,
                    "repeat": repeat,
                    "imbalance": imb,
                    "selector": selector,
                    "n_features_after_preprocessing": X_train_processed.shape[1],
                    "candidate_features": list(X_train_processed.columns), 
                    "n_features_selected": len(features),
                    "features": features,
                    "selection_time(s)": selection_time
                })
                if not features:
                    print(f"  [warning] fold {fold_id} {imb}|{selector}: no features selected")
                    for mdl in MODELS:
                        results.append({"fold": fold_id, "repeat": repeat, "imbalance": imb,
                                        "selector": selector, "model": mdl, "n_features": 0,
                                        "fit_time(s)": np.nan,"fit_time(ms)": np.nan, "predict_time(s)": np.nan,
                                        "predict_time(ms)": np.nan,
                                        "predict_time_per_observation(ms)": np.nan,
                                        "ap": np.nan, "recall": np.nan, "precision": np.nan,
                                        "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan})
                    continue

                for mdl in MODELS:
                    model = build_model(mdl, seed)

                    t0 = time.perf_counter()
                    model.fit(X_train_imb[features], y_train_imb)
                    fit_time = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    y_proba = model.predict_proba(X_val_processed[features])[:, 1]
                    predict_time = time.perf_counter() - t0

                    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
                    conf_list = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel().tolist()  # [TN, FP, FN, TP]
                    tn, fp, fn, tp = conf_list
                    results.append({
                        "fold": fold_id, "repeat": repeat, "imbalance": imb,
                        "selector": selector, "model": mdl, "n_features": len(features),
                        "fit_time(s)": fit_time,
                        "fit_time(ms)": fit_time * 1000,
                        "predict_time(s)": predict_time,
                        "predict_time(ms)": predict_time * 1000,
                        "predict_time_per_observation(ms)": predict_time / len(y_val) * 1000,
                        "ap": average_precision_score(y_val, y_proba),   # primary metric
                        "recall": recall_score(y_val, y_pred, zero_division=0),
                        "precision": precision_score(y_val, y_pred, zero_division=0),
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    })

        print(f"fold {fold_id}/{n_folds} done at {datetime.now().isoformat()}")
        total_time = datetime.now() - start_time
        print(f"fold {fold_id}/{n_folds} took {total_time.total_seconds():.1f} seconds")

    end_exp_time = datetime.now()
    print(f"Experiment ended at {end_exp_time.isoformat()}")
    total_exp_time = end_exp_time - start_exp_time
    print(f"Experiment took {total_exp_time.total_seconds():.1f} seconds")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)
    removed_features_df = pd.DataFrame.from_dict(removed_features, orient="index").reset_index().rename(columns={"index": "fold"})

    feature_universe = sorted(set().union(*selections_df["candidate_features"]))

    stability_df, feature_frequency_df = summarize_feature_stability(
            selections_df,
            feature_universe,
    )

    summary_df = (
        results_df.groupby(["imbalance", "selector", "model"])
        .agg(
            average_precision_mean=("ap", "mean"),
            average_precision_std=("ap", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            tn_mean=("tn", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            tp_mean=("tp", "mean"),
            n_features_mean=("n_features", "mean"),
        )
        .reset_index()
        .sort_values("average_precision_mean", ascending=False, ignore_index=True)
    )
    
    # timing summary (median and IQR over the 50 folds)
    def q1(s): return s.quantile(0.25)
    def q3(s): return s.quantile(0.75)

    selection_time_df = (
        selections_df.groupby(["imbalance", "selector"])["selection_time(s)"]
        .agg(selection_time_median="median", selection_time_q1=q1, selection_time_q3=q3)
        .reset_index()
    )
    model_time_df = (
        results_df.groupby(["imbalance", "selector", "model"])
        .agg(fit_time_median=("fit_time(s)", "median"),
             fit_time_q1=("fit_time(s)", q1),
             fit_time_q3=("fit_time(s)", q3),
             predict_time_per_sample_ms_median=("predict_time_per_observation(ms)", "median"))
        .reset_index()
    )

    for name, frame in (
        ("results", results_df),
        ("selections", selections_df),
        ("summary", summary_df),
        ("stability", stability_df),
        ("feature_frequency", feature_frequency_df),
        ("removed_features", removed_features_df),
        ("selection_time", selection_time_df),
        ("model_time", model_time_df),
    ):
        frame.to_json(
            os.path.join(RESULTS_DIR, f"{run_id}_{name}.json"),
            orient="records",
            indent=2,
            double_precision=15,
        )

    config = {
            "run_id": run_id, "random_seed": RANDOM_SEED,
            "algorithm_seed": ALGORITHM_SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
            "n_features_original": n_features_original,
            "stability_feature_universe": feature_universe,
            "missing_threshold": MISSING_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "selectors": SELECTORS, "imbalance": IMBALANCE, "models": MODELS,
            "selector_params": SELECTOR_PARAMS, "model_params": MODEL_PARAMS,
            "scaler": "RobustScaler",
            "n_jobs": N_JOBS,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
    }
    with open(os.path.join(RESULTS_DIR, f"{run_id}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

        


        

