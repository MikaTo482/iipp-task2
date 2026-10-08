import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, mutual_info_classif, RFE, SelectKBest, SelectFdr, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils import missing_rate_filter, correlation_filter, duplicate_filter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier)
from sklearn.metrics import (recall_score, confusion_matrix, 
                             average_precision_score, 
                             precision_score, f1_score, matthews_corrcoef, 
                             roc_auc_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from boruta import BorutaPy
from itertools import combinations
import shap
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from imblearn.combine import SMOTEENN
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB



SHAP_ENABLED = True
SHAP_MAX_BACKGROUND = 100
SHAP_MODELS = ("lr", "rf", "xgb", "knn", "nb")
# Budget for this many forward/reverse orderings with all features varying.
# SHAP can do more orderings when some features are invariant for a row.
# This is not a guarantee of converged individual attributions/ranks.
SHAP_PERMUTATION_CYCLES = 20
SHAP_ADDITIVITY_ATOL = 1e-7
SHAP_ADDITIVITY_RTOL = 1e-6
SHAP_EXPLAINERS = {
    "lr": "linear", "rf": "tree", "xgb": "tree",
    "knn": "permutation", "nb": "permutation",
}
SHAP_OUTPUT_SPACES = {
    "lr": "log_odds", "rf": "probability", "xgb": "log_odds",
    "knn": "probability", "nb": "probability",
}

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoost is required by the locked protocol "
                      "(selector and model). Install with: pip install xgboost") from e

try:
    from ctgan import CTGAN
except ImportError as e:
    raise ImportError("CTGAN is required for the 'ctgan' imbalance condition. "
                      "Install with: pip install ctgan") from e

TARGET_COL = "label"
SPLIT_DIR = "splits"
RANDOM_SEED = 42

# Seeds. Feature selectors, classifiers and SHAP sampling use one fixed seed, so
# selection stability across splits reflects data resampling only, not algorithmic
# randomness. Data-level imbalance methods (SMOTE, SMOTE-ENN, CTGAN) draw a new
# seed per fold, because generating synthetic rows is part of the resampled data.
ALGORITHM_SEED = RANDOM_SEED


def resampling_seed(fold_id):
    return RANDOM_SEED + fold_id


# Probability cut-off for recall, precision, F1 and MCC (AP and ROC-AUC are
# threshold-free).
DECISION_THRESHOLD = 0.5

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
CORRELATION_THRESHOLD = 0.8

MISSFOREST_PARAMS = dict(n_estimators=100, max_features="sqrt", max_iter=10)

SELECTORS = ["none", "mi", "l1", "xgb", "boruta"]
IMBALANCE = ["none", "weight", "smote"]
MODELS = ["lr", "rf", "xgb"]

SELECTOR_PARAMS = {
    "mi": dict(top_k=20),
    "l1": dict(C=0.1),
    "xgb": dict(top_k=20, n_estimators=300, max_depth=3, learning_rate=0.05),
    "boruta": dict(max_iter=100, rf_max_depth=5),
    "svm_rfe": dict(top_k=20, C=0.1, step=0.1),
    "anova": dict(top_k=20),
    "fdr": dict(alpha=0.05),
}

# Fixed hyper-parameters, no tuning (same values in every fold and condition).
MODEL_PARAMS = {
        "lr": dict(C=1.0, max_iter=5000),
        "rf": dict(n_estimators=500, min_samples_leaf=1),
        "xgb": dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist"),
        "knn": dict(n_neighbors=5),
        "nb": dict(),
}

# =========================================================================
# CTGAN settings (light version: trained on minority rows only)
# =========================================================================
CTGAN_PARAMS = dict(
    epochs=300,              # ~1 training step per epoch on ~75 minority rows
    batch_size=50,           # must be even and divisible by pac
    pac=10,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    embedding_dim=64,
    verbose=False,
)
CTGAN_TARGET_RATIO = 1.0     # minority / majority after resampling (1.0 = balanced)
CTGAN_CLIP = True            # clip synthetic values to the observed minority range

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

# =========================================================================
# CTGAN oversampling - fitted on the minority rows of the TRAINING FOLD only
# =========================================================================
def ctgan_resample(X, y, seed):
    """Train CTGAN on the minority rows only and sample new minority rows."""
    y = np.asarray(y).astype(int)
    X_min = X[y == 1].reset_index(drop=True)
    n_new = int(round(CTGAN_TARGET_RATIO * (y == 0).sum())) - len(X_min)
    if n_new <= 0:
        return X, y

    model = CTGAN(**CTGAN_PARAMS)
    model.set_random_state(seed)
    model.fit(X_min)                                   # all columns continuous

    synth = model.sample(n_new)[list(X.columns)].astype(float)
    if CTGAN_CLIP:
        synth = synth.clip(lower=X_min.min(), upper=X_min.max(), axis=1)

    X_res = pd.concat([X.reset_index(drop=True), synth], ignore_index=True)
    y_res = np.concatenate([y, np.ones(n_new, dtype=int)])
    return X_res, y_res

def apply_imbalance(X, y, condition, seed):
    if condition == "smote":
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
        return X_res, np.asarray(y_res), False
    if condition == "smoteenn":
        X_res, y_res = SMOTEENN(random_state=seed).fit_resample(X, y)
        return X_res, np.asarray(y_res), False
    if condition == "ctgan":
        X_res, y_res = ctgan_resample(X, y, seed)
        return X_res, y_res, False
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
 
    if name == "fdr":
        s = SelectFdr(f_classif, alpha=P["alpha"]).fit(X, y)
        return list(cols[s.get_support()])
 
    if name == "l1":
        m = LogisticRegression(penalty="l1", solver="liblinear", C=P["C"],
                               class_weight=cw, max_iter=5000, random_state=seed)
        m.fit(X, y)
        return list(cols[np.abs(m.coef_[0]) > 1e-8])
    
    if name == "svm_rfe":
        svm = LinearSVC(C=P["C"], class_weight=cw, dual=False,
                        max_iter=5000, random_state=seed)
        rfe = RFE(svm, n_features_to_select=P["top_k"], step=P["step"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            rfe.fit(X, y)
        return list(cols[rfe.support_])

    if name == "anova":
        s = SelectKBest(f_classif, k=min(P["top_k"], X.shape[1])).fit(X, y)
        return list(cols[s.get_support()])
 
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
    if name == "knn":                       # no class weighting: 'weight' == 'none'
        return KNeighborsClassifier(n_jobs=N_JOBS, **P)
    if name == "nb":                        # equal priors stand in for class weighting
        return GaussianNB(priors=[0.5, 0.5] if use_weights else None, **P)
    
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

    # missforest = IterativeImputer(
    #     estimator=RandomForestRegressor(
    #         n_estimators=MISSFOREST_PARAMS["n_estimators"],
    #         max_features=MISSFOREST_PARAMS["max_features"],
    #         n_jobs=-1, random_state=RANDOM_SEED),
    #     initial_strategy="mean",          # Algorithm 1 step 1
    #     imputation_order="ascending",     # Algorithm 1 step 2
    #     max_iter=MISSFOREST_PARAMS["max_iter"],
    #     random_state=RANDOM_SEED,
    # )

    pipeline = Pipeline([
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("median_imputation", SimpleImputer(missing_values=np.nan, strategy="median")),
        # ("missforest_imputation", missforest),
        ("robust_scaling", RobustScaler()),
    ]).set_output(transform="pandas")              # keep column names for feature selection

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        X_train_p = pipeline.fit_transform(X_train)           # fit on training fold only
        X_val_p = pipeline.transform(X_val)                   # transform only
    return X_train_p, X_val_p


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
    """Definition 4 (Nogueira et al., 2018), on a shared feature universe.

    d is the number of columns of the binary selection matrix, never the
    average number of candidates remaining after per-fold preprocessing.
    """
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
    """Fixed-size subsets; p is the same integer universe size used by Z."""
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


def probability_shap(model, X_background, X_explain, seed):
    """Explain P(label=1) with interventional, permutation-sampled SHAP.

    The caller supplies preprocessed training rows BEFORE imbalance resampling
    as background, and all validation rows as the explanation data. No model is
    refitted here. Seeded background/permutation sampling is restored afterwards
    so SHAP cannot alter the experiment's global NumPy RNG state.
    """
    if not isinstance(X_background, pd.DataFrame) or not isinstance(X_explain, pd.DataFrame):
        raise ValueError("SHAP inputs must be DataFrames with ordered feature names")
    columns = X_explain.columns
    if not columns.is_unique or not columns.equals(X_background.columns):
        raise ValueError("SHAP background and validation feature names/order must match")
    if len(columns) == 0 or X_background.empty or X_explain.empty:
        raise ValueError("SHAP needs non-empty background and validation data")
    trained_columns = getattr(model, "feature_names_in_", None)
    if trained_columns is not None and list(trained_columns) != list(columns):
        raise ValueError("SHAP feature order differs from the fitted model")
    if not (np.isfinite(X_background.to_numpy(dtype=float)).all()
            and np.isfinite(X_explain.to_numpy(dtype=float)).all()):
        raise ValueError("SHAP inputs contain non-finite values")
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or len(classes) != 2 or len(positive) != 1:
        raise ValueError("SHAP requires a binary classifier with Fail labelled 1")
    positive_index = int(positive[0])
    if (isinstance(SHAP_PERMUTATION_CYCLES, bool)
            or not isinstance(SHAP_PERMUTATION_CYCLES, (int, np.integer))
            or SHAP_PERMUTATION_CYCLES < 1):
        raise ValueError("SHAP_PERMUTATION_CYCLES must be a positive integer")
    if (isinstance(SHAP_MAX_BACKGROUND, (bool, np.bool_))
            or not isinstance(SHAP_MAX_BACKGROUND, (int, np.integer))
            or SHAP_MAX_BACKGROUND < 1):
        raise ValueError("SHAP_MAX_BACKGROUND must be a positive integer")

    def predict_fail(values):
        # SHAP masks with arrays; restore names for sklearn's feature validation.
        data = pd.DataFrame(np.asarray(values), columns=columns)
        probabilities = np.asarray(model.predict_proba(data), dtype=float)
        if probabilities.shape != (len(data), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Classifier returned invalid SHAP probabilities")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Classifier returned probabilities outside [0, 1]")
        return probabilities[:, positive_index]

    bg = X_background.sample(min(SHAP_MAX_BACKGROUND, len(X_background)), random_state=seed)
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    # Unlike SHAP's fixed default of 500, this budget works with >249 features.
    max_evals = SHAP_PERMUTATION_CYCLES * (2 * len(columns) + 1)
    rng_state = np.random.get_state()
    try:
        explainer = shap.PermutationExplainer(
            predict_fail, masker, link=shap.links.identity, seed=seed)
        explanation = explainer(X_explain, max_evals=max_evals, silent=True)
    finally:
        np.random.set_state(rng_state)

    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float)
    if values.shape != X_explain.shape or base.shape != (len(X_explain),):
        raise ValueError("Unexpected SHAP shape for single-output P(Fail)")
    if not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError("SHAP returned non-finite values")
    expected_base = predict_fail(bg).mean()
    predictions = predict_fail(X_explain)
    if not np.allclose(base, expected_base, atol=SHAP_ADDITIVITY_ATOL, rtol=SHAP_ADDITIVITY_RTOL):
        raise ValueError("SHAP base values do not match the training background")
    if not np.allclose(base + values.sum(axis=1), predictions,
                       atol=SHAP_ADDITIVITY_ATOL, rtol=SHAP_ADDITIVITY_RTOL):
        raise ValueError("SHAP additivity check failed for P(Fail)")
    return values


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
    elif model_name in ("knn", "nb"):
        sv = probability_shap(model, X_background, X_explain, seed)
        space = "probability"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    sv = _positive_class(sv)
    if sv.shape != X_explain.shape:
        raise ValueError(f"SHAP shape {sv.shape} != {X_explain.shape}")
    if not np.isfinite(sv).all():
        raise ValueError(f"Non-finite SHAP values for {model_name}")
    return sv, space


def shap_to_records(sv, features, space, **keys):
    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)
    ranks = pd.Series(mean_abs).rank(ascending=False, method="average").to_numpy()
    return [{**keys, "feature": f, "mean_abs_shap": float(a), "mean_shap": float(s),
             "shap_rank": float(r), "n_features_in_model": len(features),
             "output_space": space}
            for f, a, s, r in zip(features, mean_abs, mean_signed, ranks)]


def compute_shap_for_fit(model, model_name, X_background, X_explain, seed, **fit):
    """Keep failed explanations separate from prediction results, never zero-fill."""
    approximate = model_name in ("knn", "nb")
    status = dict(
        **fit, model=model_name, status="failed",
        explainer=SHAP_EXPLAINERS.get(model_name, "unsupported"),
        output_space=SHAP_OUTPUT_SPACES.get(model_name), explained_class=1,
        n_features=X_explain.shape[1], n_explained=len(X_explain), seed=seed,
        approximate=approximate,
        background_rows=(min(SHAP_MAX_BACKGROUND, len(X_background))
                         if model_name in ("lr", "knn", "nb") else None),
        permutation_cycles_budget=SHAP_PERMUTATION_CYCLES if approximate else None,
        max_evals=(SHAP_PERMUTATION_CYCLES * (2 * X_explain.shape[1] + 1)
                   if approximate else None),
        additivity_checked=False,
    )
    try:
        sv, space = compute_shap(model, model_name, X_background, X_explain, seed)
        records = shap_to_records(sv, list(X_explain.columns), space, model=model_name, **fit)
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        warnings.warn(f"SHAP failed for {model_name} {fit}: {error}", RuntimeWarning)
        return [], status
    status.update(status="ok", output_space=space, additivity_checked=approximate,
                  error_type=None, error=None)
    return records, status


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

    # =========================================================================
    # TRACK A
    # =========================================================================

    results, selections, shap_rows, shap_status = [], [], [], []
    n_folds = N_SPLITS * N_REPEATS
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    for fold_id, (tr, va) in enumerate(rskf.split(X, y), start=1):
        seed = ALGORITHM_SEED                     # selectors, classifiers, SHAP
        data_seed = resampling_seed(fold_id)      # SMOTE / SMOTE-ENN / CTGAN only
        repeat = (fold_id - 1) // N_SPLITS + 1

        X_train_processed, X_val_processed = preprocess_fold(X.iloc[tr], X.iloc[va])
        y_train, y_val = y.iloc[tr], y.iloc[va]

        for imb in IMBALANCE:
            X_train_imb, y_train_imb, use_weight = apply_imbalance(
                X_train_processed, y_train, imb, data_seed)

            for selector in SELECTORS:
                features = select_features(selector, X_train_imb, y_train_imb, use_weight, seed)
                selections.append({
                    "fold": fold_id,
                    "repeat": repeat,
                    "imbalance": imb,
                    "selector": selector,
                    "n_features_after_preprocessing": X_train_processed.shape[1],
                    "candidate_features": list(X_train_processed.columns), 
                    "n_features_selected": len(features),
                    "features": features,
                })

                if not features:
                    print(f"  [warning] fold {fold_id} {imb}|{selector}: no features selected")
                    continue

                for mdl in MODELS:
                    model = build_model(mdl, use_weight, y_train_imb, seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        model.fit(X_train_imb[features], y_train_imb)
                    y_proba = model.predict_proba(X_val_processed[features])[:, 1]
                    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

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
                    if SHAP_ENABLED and mdl in SHAP_MODELS:
                        records, status = compute_shap_for_fit(
                            model, mdl,
                            X_background=X_train_processed[features],   # ก่อน resampling
                            X_explain=X_val_processed[features],
                            seed=seed,
                            fold=fold_id, repeat=repeat, imbalance=imb,
                            selector=selector)
                        shap_rows.extend(records)
                        shap_status.append(status)

        print(f"fold {fold_id}/{n_folds} done")

    # =========================================================================
    # Save results
    # =========================================================================

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(results)
    selections_df = pd.DataFrame(selections)

    # Shared identities across all folds/conditions, including features that
    # were eligible but never selected. Both stability indices use this space.
    feature_universe = sorted(set().union(*selections_df["candidate_features"]))

    stability_df, feature_frequency_df = summarize_feature_stability(
        selections_df,
        feature_universe,
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
    shap_status_df = pd.DataFrame(shap_status)

    for name, frame in (
        ("results", results_df),
        ("selections", selections_df),
        ("summary", summary_df),
        ("stability", stability_df),
        ("feature_frequency", feature_frequency_df),
        ("shap", shap_df),  
        ("shap_status", shap_status_df),
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
        "algorithm_seed_scope": "feature selectors, classifiers, SHAP background and permutation sampling (fixed across folds)",
        "resampling_seed": "RANDOM_SEED + fold_id",
        "resampling_seed_scope": "SMOTE, SMOTE-ENN, CTGAN (data-level imbalance methods)",
        "decision_threshold": DECISION_THRESHOLD,
        "decision_threshold_scope": "recall, precision, f1, mcc (y_proba >= threshold)",
        "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
        "n_features_original": n_features_original,
        "nogueira_d": len(feature_universe),
        "stability_universe": "union_of_candidate_features_across_all_folds",
        "stability_feature_universe": feature_universe,
        "stability_scope": "preprocessing_and_feature_selection",
        "missing_threshold": MISSING_THRESHOLD,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "selectors": SELECTORS, "imbalance": IMBALANCE, "models": MODELS,
        "selector_params": SELECTOR_PARAMS, "model_params": MODEL_PARAMS,
        "scaler": "RobustScaler",
        "hyperparameter_tuning": False,
        "ctgan_params": CTGAN_PARAMS,
        "ctgan_target_ratio": CTGAN_TARGET_RATIO,
        "ctgan_clip": CTGAN_CLIP,
        "ctgan_training_data": "minority_rows_of_preprocessed_training_fold",
        "n_dev": len(dev_df),
        "positive_rate_dev": float(y.mean()),
        "shap_enabled": SHAP_ENABLED,
        "shap_models": list(SHAP_MODELS),
        "shap_explain_data": "validation_fold",
        "shap_lr_background": "training_fold_before_resampling",
        "shap_model_agnostic_background": "training_fold_before_resampling",
        "shap_max_background": SHAP_MAX_BACKGROUND,
        "shap_explainers": SHAP_EXPLAINERS,
        "shap_output_spaces": SHAP_OUTPUT_SPACES,
        "shap_explained_class": 1,
        "shap_permutation_cycles": SHAP_PERMUTATION_CYCLES,
        "shap_permutation_cycles_scope": "budget with all features varying; actual orderings may be higher for invariant features",
        "shap_permutation_max_evals": "cycles * (2 * n_features + 1)",
        "shap_permutation_masker": "Independent (interventional, not conditional SHAP)",
        "shap_permutation_link": "identity",
        "shap_permutation_approximate": True,
        "shap_additivity_atol": SHAP_ADDITIVITY_ATOL,
        "shap_additivity_rtol": SHAP_ADDITIVITY_RTOL,
        "shap_failure_policy": "save failed fit in shap_status; no SHAP records or zero fill",
        "shap_version": shap.__version__,
    }
    with open(os.path.join(RESULTS_DIR, f"{run_id}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(summary_df.round(3).to_string(index=False))
    print(stability_df.round(3).to_string(index=False))
    print(feature_frequency_df.to_string(index=False))
    print(shap_df.to_string(index=False))
