import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif, RFECV, SelectKBest, SelectFdr, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from utils import missing_rate_filter, correlation_filter, duplicate_filter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (matthews_corrcoef, recall_score, confusion_matrix, 
                             average_precision_score, precision_score)
from sklearn.svm import LinearSVC                 
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
N_REPEATS = 30
N_JOBS = 1

RANDOM_SEED = 42
ALGORITHM_SEED = RANDOM_SEED
RESULTS_DIR = "experiments"

MISSING_THRESHOLD = 50
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.8

def resampling_seed(fold_id):
    return RANDOM_SEED + fold_id

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


# --------------------------------------------------------------------------
# Selectors (single source of truth for the experiment AND the voters).
# Every selector returns exactly k column labels of X for each k in K_VALUES
# (fixed-k protocol), except "none", which keeps all features as the
# all-feature reference and is recorded with k = <NA>.
# --------------------------------------------------------------------------
K_VALUES = [10, 20, 30]

SELECTOR_PARAMS = {
    "mi": dict(rank_by="mutual_information"),
    "anova": dict(rank_by="f_statistic"),
    # Smallest C on this grid that gives >= k non-zero coefficients, then top-k by |coef|.
    "l1": dict(C_grid=np.round(np.logspace(-3, 2, 26), 6).tolist()),
    "xgb": dict(n_estimators=300, max_depth=3, learning_rate=0.05),
    # Rank by Boruta ranking_ (1 = confirmed); ties broken by RF importance.
    "boruta": dict(max_iter=100, rf_max_depth=5, tie_break_n_estimators=500),
    # Top-k by number of votes; ties broken by mean rank inside the voters' top-k lists.
    "vote": dict(methods=None, tie_break="mean_rank"),
}


def compute_scale_pos_weight(y):
    y = np.asarray(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0:
        return 1.0

    return n_neg / n_pos


def select_features_multi(name, X, y, use_weights, seed, k_values=None, n_jobs=None):
    """Return {k: top-k column labels} for every k in k_values.

    The selector is fitted once and the ranking is sliced per k, so expensive
    selectors (Boruta, vote) do not run once per k. Only L1 depends on k
    (C is searched until >= k coefficients are non-zero), so it refits per k.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    cols = X.columns
    k_values = K_VALUES if k_values is None else list(k_values)
    if not k_values or any(int(k) < 1 for k in k_values):
        raise ValueError(f"k_values must be positive integers, got {k_values!r}.")
    ks = {int(k): min(int(k), len(cols)) for k in k_values}  # requested k -> usable k
    cw = "balanced" if use_weights else None
    P = SELECTOR_PARAMS.get(name, {})

    def from_order(order):
        return {k: list(cols[order[:kk]]) for k, kk in ks.items()}

    def from_scores(scores):
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
        return from_order(np.argsort(-scores, kind="stable"))  # ties keep column order

    if name == "mi":
        return from_scores(mutual_info_classif(X, y, random_state=seed))

    if name == "anova":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # constant columns -> NaN F
            scores, _ = f_classif(X, y)
        return from_scores(scores)

    if name == "l1":
        out = {}
        for k, kk in ks.items():
            coef = None
            for C in P["C_grid"]:
                model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=C,
                    class_weight=cw,
                    max_iter=5000,
                    random_state=seed,
                )
                model.fit(X, y)
                coef = np.abs(model.coef_[0])
                if np.count_nonzero(coef > 1e-8) >= kk:
                    break
            else:
                warnings.warn(f"l1: only {np.count_nonzero(coef > 1e-8)} non-zero coefficients "
                              f"at the largest C; padding to k={kk} by column order.")
            out[k] = from_scores(coef)[k]
        return out

    if name == "xgb":
        xgb_params = dict(
            n_estimators=P["n_estimators"],
            max_depth=P["max_depth"],
            learning_rate=P["learning_rate"],
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=n_jobs,
        )

        if use_weights:
            xgb_params["scale_pos_weight"] = compute_scale_pos_weight(y)

        model = XGBClassifier(**xgb_params)
        model.fit(X, y)
        return from_scores(model.feature_importances_)

    if name == "boruta":
        rf = RandomForestClassifier(max_depth=P["rf_max_depth"], class_weight=cw,
                                    random_state=seed, n_jobs=n_jobs)
        b = BorutaPy(rf, n_estimators="auto", max_iter=P["max_iter"], random_state=seed)
        b.fit(X.to_numpy(), y)
        # BorutaPy re-sizes and refits rf on shadow features, so use a fresh forest
        # with the same settings to order features that share a Boruta rank.
        tie_rf = RandomForestClassifier(n_estimators=P["tie_break_n_estimators"],
                                        max_depth=P["rf_max_depth"], class_weight=cw,
                                        random_state=seed, n_jobs=n_jobs)
        importance = tie_rf.fit(X, y).feature_importances_
        return from_order(np.lexsort((np.arange(len(cols)), -importance, b.ranking_)))

    if name == "vote":
        results = vote_scores(X, y, use_weights, seed, methods=P["methods"],
                              k_values=list(ks), n_jobs=n_jobs)
        return {k: list(cols[r["support"].to_numpy()]) for k, r in results.items()}

    raise ValueError(f"Unknown selector: {name}")


def select_features(name, X, y, use_weights, seed, n_jobs=None, k=None):
    """Single-k convenience wrapper around select_features_multi."""
    k = K_VALUES[0] if k is None else int(k)
    return select_features_multi(name, X, y, use_weights, seed, k_values=[k], n_jobs=n_jobs)[k]


# Every real selector can vote. "none" keeps all features (it would give every
# feature a free vote) and "vote" is the ensemble itself, so neither is a voter.
VOTERS = tuple(m for m in SELECTOR_PARAMS if m not in ("none", "vote"))


def vote_scores(X, y, use_weights, seed, methods=None, k_values=None, n_jobs=None):
    """Fixed-k vote for every k: each voter submits its top-k, the ensemble keeps top-k by votes.

    Every voter is fitted once (select_features_multi) and reused for all k.
    Ties in vote count are broken by mean rank: a feature at position r in a
    voter's top-k list gets rank r, a feature the voter did not pick gets rank k.
    Remaining ties keep column order, so the result is deterministic.

    Returns {k: dict(scores, votes, mean_rank, selected_by, weights, support)},
    where support is a bool Series with exactly min(k, n_features) True values.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with unique feature names.")
    if X.empty or not X.columns.is_unique:
        raise ValueError("X must have rows and uniquely named feature columns.")
    if isinstance(methods, str):
        raise TypeError("methods must be a sequence of voter names, not a string.")
    methods = VOTERS if methods is None else tuple(methods)
    if not methods:
        raise ValueError("methods must contain at least one voter.")
    unknown = [m for m in methods if m not in VOTERS]
    if unknown:
        raise ValueError(f"Unknown voter(s) {unknown}. Choose from {list(VOTERS)}.")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate voter names in {methods}; each voter must vote once.")
    k_values = K_VALUES if k_values is None else list(k_values)

    ballots = {name: select_features_multi(name, X, y, use_weights, seed,
                                           k_values=k_values, n_jobs=n_jobs)
               for name in methods}

    p = X.shape[1]
    out = {}
    for k in k_values:
        k = int(k)
        kk = min(k, p)
        votes = np.zeros(p, dtype=int)
        rank_sum = np.zeros(p, dtype=float)
        selected_by = {}
        for name in methods:
            picked = list(ballots[name][k])
            positions = X.columns.get_indexer(picked)  # position of each picked name in X
            if (positions < 0).any() or len(set(picked)) != len(picked):
                raise ValueError(f"Voter {name!r} returned duplicate names or names outside X.columns.")
            selected_by[name] = picked
            rank = np.full(p, float(kk))
            rank[positions] = np.arange(len(picked))
            votes[positions] += 1
            rank_sum += rank

        mean_rank = rank_sum / len(methods)
        order = np.lexsort((np.arange(p), mean_rank, -votes))
        support = np.zeros(p, dtype=bool)
        support[order[:kk]] = True
        out[k] = {"scores": pd.Series(votes / len(methods), index=X.columns),
                  "votes": pd.Series(votes, index=X.columns),
                  "mean_rank": pd.Series(mean_rank, index=X.columns),
                  "selected_by": selected_by,
                  "weights": {name: 1.0 for name in methods},
                  "support": pd.Series(support, index=X.columns)}
    return out

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
        return np.nan
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

    # dropna=False keeps "none" (k = <NA>) as its own group.
    for (imbalance, selector, k), g in selections_df.groupby(["imbalance", "selector", "k"], dropna=False):
        subsets = g["features"].tolist()
        Z = to_matrix_from_names(subsets, feature_universe)

        subset_sizes = np.array([len(s) for s in subsets])
        same_size = np.all(subset_sizes == subset_sizes[0])

        rows.append({
            "imbalance": imbalance,
            "selector": selector,
            "k": k,
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
                "k": k,
                "feature": feature,
                "selection_frequency": f,
                "selected_count": int(c),
                "n_runs": len(subsets),
                "n_available": int(a),
                "frequency_when_available": c / a if a > 0 else np.nan,
            })

    stability_df = pd.DataFrame(rows).astype({"k": "Int64"}).sort_values(
        ["imbalance", "selector", "k"], ignore_index=True
    )

    feature_frequency_df = pd.DataFrame(freq_rows).astype({"k": "Int64"}).sort_values(
        ["imbalance", "selector", "k", "selection_frequency"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )

    return stability_df, feature_frequency_df

SELECTORS = ["none", "mi", "anova", "l1", "xgb", "boruta", "vote"]
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
                if selector == "none":
                    subsets = {None: list(X_train_imb.columns)}   # all features, no k
                else:
                    subsets = select_features_multi(selector, X_train_imb, y_train_imb,
                                                    use_weight, seed, k_values=K_VALUES)
                # One fit serves every k, so this time is shared by all k of this selector.
                selection_time = time.perf_counter() - t0

                for k, features in subsets.items():
                    selections.append({
                        "fold": fold_id,
                        "repeat": repeat,
                        "imbalance": imb,
                        "selector": selector,
                        "k": k,
                        "n_features_after_preprocessing": X_train_processed.shape[1],
                        "candidate_features": list(X_train_processed.columns),
                        "n_features_selected": len(features),
                        "features": features,
                        "selection_time(s)": selection_time
                    })
                    if not features:
                        print(f"  [warning] fold {fold_id} {imb}|{selector}|k={k}: no features selected")
                        for mdl in MODELS:
                            results.append({"fold": fold_id, "repeat": repeat, "imbalance": imb,
                                            "selector": selector, "k": k, "model": mdl, "n_features": 0,
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
                            "selector": selector, "k": k, "model": mdl, "n_features": len(features),
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
    # k is an integer, or <NA> for "none" (all features); Int64 keeps 10/20/30 as ints in JSON.
    results_df = pd.DataFrame(results).astype({"k": "Int64"})
    selections_df = pd.DataFrame(selections).astype({"k": "Int64"})
    removed_features_df = pd.DataFrame.from_dict(removed_features, orient="index").reset_index().rename(columns={"index": "fold"})

    feature_universe = sorted(set().union(*selections_df["candidate_features"]))

    stability_df, feature_frequency_df = summarize_feature_stability(
            selections_df,
            feature_universe,
    )
    def nanmean(s):
        return s.mean(skipna=True)

    summary_df = (
        results_df.groupby(["imbalance", "selector", "k", "model"], dropna=False)
        .agg(
            average_precision_mean=("ap", nanmean),
            average_precision_std=("ap", "std"),
            recall_mean=("recall", nanmean),
            precision_mean=("precision", nanmean),
            tn_mean=("tn", nanmean),
            fp_mean=("fp", nanmean),
            fn_mean=("fn", nanmean),
            tp_mean=("tp", nanmean),
            n_features_mean=("n_features", nanmean),
        )
        .reset_index()
        .sort_values("average_precision_mean", ascending=False, ignore_index=True)
    )
    
    # timing summary (median and IQR over the 50 folds)
    def q1(s): return s.quantile(0.25)
    def q3(s): return s.quantile(0.75)

    selection_time_df = (
        selections_df.groupby(["imbalance", "selector", "k"], dropna=False)["selection_time(s)"]
        .agg(selection_time_median="median", selection_time_q1=q1, selection_time_q3=q3)
        .reset_index()
    )
    model_time_df = (
        results_df.groupby(["imbalance", "selector", "k", "model"], dropna=False)
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
            "k_values": K_VALUES,
            "k_note": "k is <NA> for selector 'none' (all features). selection_time is "
                      "measured once per fold/imbalance/selector and shared by every k, "
                      "because one selector fit is sliced to all k values.",
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

        


        
