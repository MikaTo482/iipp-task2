"""Fit one full-development configuration and save metrics and SHAP to experiments2.

No cross-validation is performed. Development scores are in-sample.
Protected-test evaluation requires the explicit --evaluate-test option.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import shap
import sklearn
from xgboost import XGBClassifier

from utils import missing_rate_filter, correlation_filter, duplicate_filter
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr, f_classif, mutual_info_classif, SelectFromModel
from boruta import BorutaPy
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
import joblib
from sklearn.utils import resample

import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import somersd 


run_id = "20261010_143609"
SELECTION_PATH = os.path.join("experiments", f"{run_id}_selections.json")

SPLIT_DIR = "splits"
RESULTS_DIR = "experiments2"
SPLIT_DIR_RANDOM  = os.path.join(SPLIT_DIR, "random")
SPLIT_CONFIG_PATH_RANDOM = os.path.join(SPLIT_DIR_RANDOM, "split_config.json")
DEV_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "dev_indices_random.csv")
TEST_RANDOM_PATH = os.path.join(SPLIT_DIR_RANDOM, "test_indices_random.csv")
TARGET_COL = "label"
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.8
MISSING_THRESHOLD = 50
RANDOM_SEED = 42
N_JOBS = -1
SHAP_BACKGROUND_SIZE = 1000
SHAP_MAX_DISPLAY = 20
N_RUNS = 150

mdl = "rf"
selector = "mi"
imb = "none"
DECISION_THRESHOLD = 0.17

MODEL_PARAMS = {
        "lr": dict(C=1.0, max_iter=5000),
        "rf": dict(n_estimators=500, min_samples_leaf=1),
        "xgb": dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist"),
}

SELECTOR_PARAMS = {
    "mi": dict(threshold="mean_score"),
    "fdr": dict(alpha=0.05),
    "l1": dict(C=0.1),
    "xgb": dict(n_estimators=300, max_depth=3, learning_rate=0.05),
    "boruta": dict(max_iter=100, rf_max_depth=5),
    "vote": dict(methods=None, weighting="equal"),
}

def compute_scale_pos_weight(y):
    y = np.asarray(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0:
        return 1.0

    return n_neg / n_pos

def load_selected_lists(path, imbalance, selector_name):
    sel = pd.read_json(path)
    groups = sel.groupby(["imbalance", "selector"])
    if (imbalance, selector_name) not in groups.groups:
        raise ValueError(f"No rows for imbalance={imbalance!r}, selector={selector_name!r} in {path}. "
                         f"Available: {sorted(groups.groups)}")
    g = groups.get_group((imbalance, selector_name)).sort_values("fold")
    if g["fold"].nunique() != len(g):
        raise ValueError(f"Duplicate folds for ({imbalance}, {selector_name}) in {path}")
    return [list(fs) for fs in g["features"]], [list(cs) for cs in g["candidate_features"]]

def _boyer_moore_support(ballots):
    ballots = np.asarray(ballots)
    if ballots.ndim != 2 or 0 in ballots.shape or not np.isin(ballots, [0, 1]).all():
        raise ValueError("ballots must be a nonempty two-dimensional binary matrix.")
    candidate = np.zeros(ballots.shape[1], dtype=bool)
    counter = np.zeros(ballots.shape[1], dtype=int)
    for ballot in ballots:
        reset = counter == 0
        candidate[reset] = ballot[reset]
        counter += np.where(ballot == candidate, 1, -1)
    matches = np.count_nonzero(ballots == candidate, axis=0)
    return candidate & (matches > ballots.shape[0] // 2)
 
 
def vote_final_stable_set(feature_lists, candidate_lists, quota=None):
    universe = sorted(set().union(*candidate_lists), key=lambda f: int(f[1:]))
    col = {f: i for i, f in enumerate(universe)}
    ballots = np.zeros((len(feature_lists), len(universe)), dtype=bool)
    for row, fs in enumerate(feature_lists):
        ballots[row, [col[f] for f in fs]] = True
 
    votes = ballots.sum(axis=0)
    available = np.zeros(len(universe), dtype=int)
    for cs in candidate_lists:
        available[[col[f] for f in cs]] += 1
    n_folds = ballots.shape[0]
 
    if quota is None:
        support = _boyer_moore_support(ballots)
    else:
        support = votes >= np.ceil(quota * n_folds)
 
    table = pd.DataFrame({"votes": votes, "n_folds": n_folds, "vote_share": votes / n_folds,
                          "n_available": available, "final_stable_set": support},
                         index=pd.Index(universe, name="feature"))
    table = table.sort_values(["votes", "n_available"], ascending=False)
    stable = list(table.index[table["final_stable_set"]])
    return stable, table


def setup_exp2_directory(split_dir: str = SPLIT_DIR):
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        print(f"created '{split_dir}' successfully")
    else:
        print(f"folder '{split_dir}' already exists, continuing...")

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

def preprocessing(X_train, X_test, missing_threshold=None, return_details=False):
    threshold = MISSING_THRESHOLD if missing_threshold is None else missing_threshold
    missing_list = missing_rate_filter(X_train, threshold)
    X_train, X_test = X_train.drop(columns=missing_list), X_test.drop(columns=missing_list)

    if USE_CORRELATION_FILTER:
        correlation_list = correlation_filter(X_train, CORRELATION_THRESHOLD)
        X_train, X_test = X_train.drop(columns=correlation_list), X_test.drop(columns=correlation_list)
    else:
        correlation_list = []

    duplicate_list = duplicate_filter(X_train)
    X_train, X_test = X_train.drop(columns=duplicate_list), X_test.drop(columns=duplicate_list)
    n_after_duplicate = X_train.shape[1]
    if not n_after_duplicate:
        raise ValueError("No features remain after preprocessing filters")
    filtered_columns = list(X_train.columns)

    pipeline = Pipeline([
            ("remove_constant", VarianceThreshold(threshold=0.0)),
            ("median_imputation", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("robust_scaling", RobustScaler()),
        ]).set_output(transform="pandas")

    X_train_p = pipeline.fit_transform(X_train)
    X_test_p = pipeline.transform(X_test)

    return X_train_p, X_test_p

def apply_imbalance(X, y, condition, seed):
    if condition not in ("none", "weight", "smote"):
        raise ValueError(f"Unknown imbalance condition: {condition}")
    if condition == "smote":
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
        return X_res, np.asarray(y_res), False
    return X, y, condition == "weight"

def build_model(name, seed, use_weights=False, y_train=None, n_jobs=None):
    if name not in ("lr", "rf", "xgb"):
        raise ValueError(f"Unknown model: {name}")
    P = MODEL_PARAMS[name].copy()
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    if name in ("lr", "rf"):
        P["class_weight"] = "balanced" if use_weights else None
    if name == "lr":
        return LogisticRegression(random_state=seed, **P)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **P)
    if name == "xgb":
        if use_weights:
            labels = np.asarray(y_train)
            if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
                raise ValueError("Weighted XGBoost requires y_train containing both 0 (Pass) and 1 (Fail)")
            P["scale_pos_weight"] = compute_scale_pos_weight(labels)
        else:
            P["scale_pos_weight"] = 1.0
        return XGBClassifier(random_state=seed, n_jobs=n_jobs, **P)

def compute_shap(model, model_name, X_background, X_explain):
    if model_name == "lr":
        masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
        explainer = shap.LinearExplainer(model, masker)
        explanation = explainer(X_explain)
    else:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_explain, check_additivity=False)

    if explanation.values.ndim == 3:  # RandomForest: (n, features, classes)
        explanation = explanation[:, :, list(model.classes_).index(1)]
    explanation.feature_names = list(X_explain.columns)
    return explanation

def save_shap_outputs(explanation, values_runs, out_dir, split_name, sample_index, max_display=SHAP_MAX_DISPLAY):
    out_dir = Path(out_dir)
    sample_index = np.asarray(sample_index)
    if len(sample_index) != explanation.values.shape[0]:
        raise ValueError(f"{split_name}: {len(sample_index)} indices for "
                         f"{explanation.values.shape[0]} explained rows")
 
    values = pd.DataFrame(explanation.values, columns=explanation.feature_names)
    values.insert(0, "index", sample_index)
    values.to_csv(out_dir / f"shap_values_{split_name}.csv", index=False)

    values_runs = np.asarray(values_runs)

    if (
        values_runs.ndim != 3
        or values_runs.shape[1:] != explanation.values.shape
    ):
        raise ValueError(
            "values_runs must have shape (runs, samples, features)"
        )

    mean_abs_shap = np.abs(values_runs).mean(axis=(0, 1))

    importance = (
        pd.DataFrame({
            "feature": explanation.feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": explanation.values.mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(out_dir / f"shap_importance_{split_name}.csv", index=False)
 
    n_display = min(max_display, len(explanation.feature_names))
    
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=n_display, show=False)
    plt.title(f"SHAP beeswarm ({split_name})")
    plt.savefig(out_dir / f"shap_beeswarm_{split_name}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    
    bar_explanation = shap.Explanation(
        values=mean_abs_shap,
        feature_names=list(explanation.feature_names),
    )

    plt.figure()
    shap.plots.bar(bar_explanation, max_display=n_display, show=False)
    plt.title(f"Mean |SHAP| ({split_name})")
    plt.savefig(out_dir / f"shap_bar_{split_name}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
 
    return importance

def save_waterfall(explanation, row, out_path, title, max_display=SHAP_MAX_DISPLAY):
    """Explain one wafer: how each feature pushes it away from the base value."""
    plt.figure()
    shap.plots.waterfall(explanation[row], max_display=max_display, show=False)
    plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")

def average_explanations(values_runs, base_runs, X_explain):
    values = np.stack(values_runs)            
    base = np.stack([np.broadcast_to(b, (values.shape[1],)) for b in base_runs])
    mean_exp = shap.Explanation(
        values=values.mean(axis=0),
        base_values=base.mean(axis=0),
        data=X_explain.to_numpy(),
        feature_names=list(X_explain.columns),
    )
    return mean_exp, values

if __name__ == "__main__":
    setup_exp2_directory(RESULTS_DIR)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(RESULTS_DIR) / f"{mdl}_{selector}_{imb}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving outputs to {run_dir}")

    df = load_secom_data()
    dev_idx, test_idx = load_protected_split(DEV_RANDOM_PATH, TEST_RANDOM_PATH, SPLIT_CONFIG_PATH_RANDOM)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Development set shape {dev_df.shape}, Test set shape {test_df.shape}")
    print(dev_df.columns.tolist())

    X_train, X_test = dev_df.drop([TARGET_COL, "timestamp"], axis = 1), test_df.drop([TARGET_COL, "timestamp"], axis = 1)
    X_train.columns = [f"f{c}" for c in X_train.columns]
    X_test.columns = [f"f{c}" for c in X_test.columns]
    n_features_original = X_train.shape[1]
    y_train = (dev_df[TARGET_COL] == 1).astype(int) 
    y_test = (test_df[TARGET_COL] == 1).astype(int)

    feature_lists, candidate_lists = load_selected_lists(SELECTION_PATH, imb, selector)
    features, vote_table = vote_final_stable_set(feature_lists, candidate_lists, quota=None)
    if not features:
        raise ValueError("No feature won the vote, choose another imb or selector.")

    X_train_p, X_test_p = preprocessing(X_train, X_test)
    voted = features
    features = [f for f in voted if f in X_train_p.columns]
    dropped = [f for f in voted if f not in X_train_p.columns]
    print(f"The number of voted features that didn't survived from preprocessing: {len(dropped)}")

    if not features:
        raise ValueError("All voted features were removed by preprocessing.")

    shap_dev_runs, shap_test_runs = [], []
    base_dev_runs, base_test_runs = [], []
    proba_runs = []
    X_dev_explain = X_train_p[features]
    X_test_explain = X_test_p[features]
    metrics = []

    for r in range(N_RUNS):
        seed_r = RANDOM_SEED + r
        Xb, yb = resample(X_train_p, y_train, replace=True, stratify=y_train, random_state=seed_r)
        X_train_imb, y_train_imb, use_weight = apply_imbalance(Xb, yb, imb, seed=seed_r)

        model = build_model(mdl, seed=seed_r, use_weights=use_weight, y_train=y_train_imb)
        print(f"[{r + 1}/{N_RUNS}] training {mdl} | {selector} | {imb} | seed={seed_r}")
        model.fit(X_train_imb[features], y_train_imb)
        y_proba = model.predict_proba(X_test_p[features])[:, 1]
        proba_runs.append(y_proba)

        metrics.append({
            "run": r, "seed": seed_r,
            "average_precision": average_precision_score(y_test, y_proba)
        })

        X_fit = X_train_imb[features]
        X_background = X_fit.sample(n=min(SHAP_BACKGROUND_SIZE, len(X_fit)), random_state=seed_r)

        shap_dev = compute_shap(model, mdl, X_background, X_dev_explain)
        shap_test = compute_shap(model, mdl, X_background, X_test_explain)
        shap_dev_runs.append(shap_dev.values)
        base_dev_runs.append(shap_dev.base_values)
        shap_test_runs.append(shap_test.values)
        base_test_runs.append(shap_test.base_values)

    pd.DataFrame(metrics).to_csv(run_dir / f"metrics_{mdl}_{selector}_{imb}.csv", index=False)

    shap_dev_mean, dev_values = average_explanations(shap_dev_runs, base_dev_runs, X_dev_explain)
    shap_test_mean, test_values = average_explanations(shap_test_runs, base_test_runs, X_test_explain)

    imp_dev = save_shap_outputs(shap_dev_mean, dev_values, run_dir, "dev", sample_index=dev_idx)
    imp_test = save_shap_outputs(shap_test_mean, test_values, run_dir, "test", sample_index=test_idx)

    freq = vote_table.loc[features, "vote_share"].rename("selection_freq")
    run_importance = pd.DataFrame(np.abs(test_values).mean(axis=1), columns=features)

    assert len(freq) == len(features) == run_importance.shape[1], f"size mismatch: freq={len(freq)}, features={len(features)}, shap={run_importance.shape[1]}"
    assert list(freq.index) == list(run_importance.columns), "feature order mismatch"

    if freq.nunique() < 2:
        print("All selection frequencies are tied; Kendall tau is undefined for this configuration.")
        taus = np.full(N_RUNS, np.nan)
    else:
        taus = []
        for _, imp in run_importance.iterrows():
            result = somersd(freq, imp)  
            taus.append(result.statistic)             
        taus = np.array(taus)   

    pd.DataFrame({"run": range(1, N_RUNS+1), "somer": taus}).to_csv(run_dir / "freq_shap_somer_runs.csv", index=False)
    print(f"\nKendall tau-b (selection frequency vs mean|SHAP|), n_features={len(freq)}: "
        f"mean {np.nanmean(taus):.3f}, sd {np.nanstd(taus, ddof=1):.3f}")

    mean_proba = np.mean(proba_runs, axis=0)
    top_row = int(np.argmax(mean_proba))
    save_waterfall(shap_test_mean, top_row, run_dir / "shap_waterfall_test_top_risk.png",
                   f"Test wafer index {test_idx[top_row]} "
                   f"(mean P(Fail)={mean_proba[top_row]:.2f}, true label={int(y_test.iloc[top_row])})")

    pd.set_option("display.width", 120)
    print(f"\nTop features by mean |SHAP| averaged over {N_RUNS} runs (dev):")
    print(imp_dev.head(SHAP_MAX_DISPLAY).to_string(index=False))
    print(f"\nTop features by mean |SHAP| averaged over {N_RUNS} runs (test):")
    print(imp_test.head(SHAP_MAX_DISPLAY).to_string(index=False))
    print(f"\nSaved SHAP outputs to {run_dir}")
