

"""
utils_save_result.py
=====================
Reusable helper to save text (.txt) result files into an organized
folder structure: result/<script_name>/<filename>.txt

Mirrors the same behavior as save_figure():
    - If the target folder does not exist -> create it (including parents).
    - If a file with the same name already exists -> delete it first,
      then save the new version (no stale leftovers from a previous run).
"""
from __future__ import annotations
import os
import warnings
from numbers import Integral
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectorMixin, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

def save_result(content: str, script_name: str, filename: str, base_dir: str = "result"):
    """
    Save a text string to: {base_dir}/{script_name}/{filename}

    Parameters
    ----------
    content : str
        The text content to write into the file.
    script_name : str
        Subfolder name, typically the script that generated the result
        (e.g. "01_secom_data_audit").
    filename : str
        Output file name, e.g. "class_distribution.txt".
    base_dir : str
        Root folder for all results (default "result").
    """
    target_dir = os.path.join(base_dir, script_name)

    # 1. Create folder (and parents) if it doesn't exist yet
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)

    # 2. If the file already exists, delete it before saving a fresh copy
    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"Existing file removed: {target_path}")

    # 3. Save the new content
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Result saved -> {target_path}")
    return target_path


def find_temporal_cutoff_candidates(cutoff_input, target_fail_counts=[15, 20, 25]):
    """
    หา Temporal Cutoff Candidates โดยใช้ Timestamp + Fail Count เท่านั้น
    (ห้ามดู Model Performance มาช่วยตัดสินใจ)

    สำหรับแต่ละ target_fail_count จะหาจุดตัดที่ทำให้ Test period
    (ข้อมูลหลัง cutoff) มี Fail observation เท่ากับเป้าหมายพอดี
    โดยให้ Dev period (ก่อน cutoff) มีขนาดใหญ่ที่สุดเท่าที่จะเป็นไปได้

    Parameters
    ----------
    y : DataFrame ที่มี column 'label' (-1/+1) และ 'timestamp' (datetime)
    target_fail_counts : list ของจำนวน Fail เป้าหมายที่ต้องการใน Test period

    Returns
    -------
    dict
        key = target_fail_count
        value = dict {
            "cutoff_timestamp", "dev_n", "dev_fail_count",
            "test_n", "test_fail_count", "test_fail_rate_pct"
        }
    """
    # ขั้นที่ 1-2: เรียงตามเวลา
    cutoff_input = cutoff_input.sort_values("timestamp").reset_index(drop=True)
    cutoff_input["is_fail"] = (cutoff_input["label"] == 1).astype(int)

    # ขั้นที่ 3: นับจำนวน Fail ที่เหลือ "หลัง" แต่ละจุด (ไม่รวมตัวเอง)
    cutoff_input["fails_strictly_after"] = cutoff_input["is_fail"][::-1].cumsum()[::-1] - cutoff_input["is_fail"]

    candidates = {}
    seen_cutoffs = set()

    for target in sorted(set(target_fail_counts)):
        # ขั้นที่ 4: หาจุดตัดที่ตรงเป้าหมาย
        eligible = cutoff_input[cutoff_input["fails_strictly_after"] >= target]
        if eligible.empty:
            continue

        cutoff_idx = eligible.index[-1]
        cutoff_time = cutoff_input.loc[cutoff_idx, "timestamp"]

        # ป้องกันคู่ซ้ำ (ถ้า target ต่างกันแต่ cutoff ตรงกันพอดี)
        if cutoff_time in seen_cutoffs:
            continue
        seen_cutoffs.add(cutoff_time)

        test_portion = cutoff_input[cutoff_input["timestamp"] > cutoff_time]
        dev_portion = cutoff_input[cutoff_input["timestamp"] <= cutoff_time]

        test_fail_rate_pct = round(
            test_portion["is_fail"].sum() / len(test_portion) * 100, 3
        ) if len(test_portion) > 0 else None

        candidates[target] = {
            "cutoff_timestamp": cutoff_time,
            "dev_n": int(dev_portion.shape[0]),
            "dev_fail_count": int(dev_portion["is_fail"].sum()),
            "test_n": int(test_portion.shape[0]),
            "test_fail_count": int(test_portion["is_fail"].sum()),
            "test_fail_rate_pct": test_fail_rate_pct,
        }

    return candidates

def correlation_filter(X, threshold=0.95, verbose=False):
    """Remove highly correlated predictors (Kuhn & Johnson, 2013,
    Applied Predictive Modeling, Section 3.5).
 
    1) Calculate the absolute correlation matrix of the predictors.
    2) Find the pair (A, B) with the largest absolute pairwise correlation.
    3) Compute the average absolute correlation of A, and of B,
       with the other predictors that are still retained.
    4) Remove A if its average is larger; otherwise remove B.
    Repeat until no absolute pairwise correlation is above the threshold.
 
    Returns the list of feature names to drop.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be a proportion in (0, 1]")
 
    num = X.select_dtypes(include="number")
    if num.shape[0] == 0 or num.shape[1] == 0:
        raise ValueError("X has no rows or no numeric columns")
 
    # Step 1: absolute correlation matrix
    corr = num.corr().abs()
    names = corr.columns.to_numpy()
    C = corr.to_numpy(copy=True)
    np.fill_diagonal(C, np.nan)                 # ignore self-correlation
 
    keep = np.ones(len(names), dtype=bool)      # True = predictor still retained
    to_drop = []
 
    while keep.sum() > 1:
        idx = np.flatnonzero(keep)
        sub = C[np.ix_(idx, idx)]
        if np.isnan(sub).all():
            break
 
        # Step 2: pair with the largest |r| (ties -> first pair in column order)
        i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
        if sub[i, j] < threshold:
            break
        a, b = idx[i], idx[j]
 
        # Step 3: average |r| with the predictors that are still retained
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_a = np.nanmean(C[a, idx])
            mean_b = np.nanmean(C[b, idx])
 
        # Step 4: remove A if its average is larger, otherwise remove B
        drop = a if mean_a > mean_b else b
        keep[drop] = False
        to_drop.append(names[drop])
 
    if verbose:
        print(f"{len(to_drop)} features removed (|r| > {threshold})")
    return to_drop


def missing_rate_filter(X, threshold):
    if not 0 <= threshold <= 100:
        raise ValueError("threshold have to be percentage format between 0 and 100")
    if len(X) == 0:
        raise ValueError("X data not found")

    miss_rate = pd.DataFrame(X.isna().sum() / len(X) * 100).reset_index()
    miss_rate.columns = ["feature", "missing_rate"]
    miss_by_thres = miss_rate[miss_rate["missing_rate"] > threshold]
    print(len(miss_by_thres), f"features have more than {threshold}% missing values")
    return miss_by_thres["feature"].tolist()

def duplicate_filter(X):
    to_drop = X.columns[X.T.duplicated(keep="first")].tolist()
    print(len(to_drop), "features are exact duplicates")
    return to_drop

def save_figure(fig, script_name, filename, base_dir="graph", dpi=150):
    target_dir = os.path.join(base_dir, script_name)
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        os.remove(target_path)

    fig.savefig(target_path, dpi=dpi, bbox_inches="tight")

"""
Voting (ensemble) feature selection.

Idea: run several independent feature-selection methods, each producing its own
top-k ranking / selected-feature set. Then "vote": a feature survives only if it
was picked by at least `min_votes` of the methods. This reduces the bias of any
single method (e.g. ANOVA F-test only catches linear relationships, RF importance
can be biased toward high-cardinality features, etc.).

IMPORTANT (leakage-free usage): wrap this in a sklearn Pipeline so that `fit`
(where the voting happens) is only ever called on the TRAINING fold inside your
cross-validation loop. Never call `.fit()` on the full dataset before splitting.

    Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("vote_select", VotingFeatureSelector(k=20, min_votes=3)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    cross_val_score(pipeline, X, y, cv=StratifiedKFold(5), scoring="f1")
    # -> imputer, scaler, AND the voting feature selector are all refit on each
    #    training fold only; the held-out fold is only ever `.transform()`-ed.
"""


# --------------------------------------------------------------------------
# Voters: each takes (X, y, k, random_state, n_estimators) and returns the
# indices of the k features it votes for (fewer is allowed, e.g. lasso).
# --------------------------------------------------------------------------
def _top_k(scores, k):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return np.argsort(scores)[::-1][:k]
 
 
def _vote_anova(X, y, k, **_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, _ = f_classif(X, y)
    return _top_k(scores, k)
 
 
def _vote_mutual_info(X, y, k, random_state=None, **_):
    return _top_k(mutual_info_classif(X, y, random_state=random_state), k)
 
 
def _vote_rf(X, y, k, random_state=None, n_estimators=300, **_):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X, y)
    return _top_k(rf.feature_importances_, k)
 
 
def _vote_lasso(X, y, k, random_state=None, **_):
    solver = "liblinear" if np.unique(y).size == 2 else "saga"
    model = LogisticRegression(
        solver=solver, l1_ratio=1, C=0.5, max_iter=5000,
        class_weight="balanced", random_state=random_state,
    ).fit(X, y)
    coef = np.abs(model.coef_).max(axis=0)
    nonzero = np.flatnonzero(coef > 1e-8)
    return nonzero[np.argsort(coef[nonzero])[::-1][:k]]
 
 
def _vote_rfe(X, y, k, random_state=None, **_):
    base = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    return np.flatnonzero(RFE(base, n_features_to_select=k, step=0.1).fit(X, y).support_)
 
 
def _vote_svm_rfe(X, y, k, random_state=None, **_):
    base = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConvergenceWarning on tiny/degenerate folds
        rfe = RFE(base, n_features_to_select=k, step=0.1).fit(X, y)
    return np.flatnonzero(rfe.support_)
 
 
VOTERS = {
    "anova": _vote_anova,
    "mutual_info": _vote_mutual_info,
    "rf": _vote_rf,
    "lasso": _vote_lasso,
    "rfe": _vote_rfe,
    "svm_rfe": _vote_svm_rfe,
}
 
 
# --------------------------------------------------------------------------
# The selector
# --------------------------------------------------------------------------
class VotingFeatureSelector(SelectorMixin, BaseEstimator):
 
    def __init__(self, methods=("rf", "anova", "lasso"), k=20, min_votes=None,
                 n_estimators=300, fallback=True, random_state=42):
        # sklearn rule: only store params here, no logic/validation.
        self.methods = methods
        self.k = k
        self.min_votes = min_votes
        self.n_estimators = n_estimators
        self.fallback = fallback
        self.random_state = random_state
 
    # ---- validation -------------------------------------------------------
    def _check_params(self, n_features):
        if isinstance(self.methods, str):
            raise TypeError(f"methods must be a list/tuple of names, e.g. ('{self.methods}',), not a string.")
        methods = tuple(self.methods)
        if not methods:
            raise ValueError("methods must contain at least one voter.")
        unknown = [m for m in methods if m not in VOTERS]
        if unknown:
            raise ValueError(f"Unknown method(s) {unknown}. Choose from {sorted(VOTERS)}.")
        if len(set(methods)) != len(methods):
            raise ValueError(f"Duplicate method names in {methods}; each voter must vote once.")
 
        if not isinstance(self.k, Integral) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k!r}.")
        k = min(int(self.k), n_features)
 
        min_votes = len(methods) // 2 + 1 if self.min_votes is None else self.min_votes
        if not isinstance(min_votes, Integral) or not 1 <= min_votes <= len(methods):
            raise ValueError(f"min_votes must be an integer in [1, {len(methods)}] "
                             f"for {len(methods)} voter(s), got {self.min_votes!r}.")
        return methods, k, int(min_votes)
 
    def _validate_input(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        if np.unique(y).size < 2:
            raise ValueError("y must contain at least 2 classes; got only one class.")
        return X, y
 
    # ---- sklearn API --------------------------------------------------------
    def fit(self, X, y):
        X, y = self._validate_input(X, y)  # also sets n_features_in_ / feature_names_in_
        n_features = X.shape[1]
        self.methods_, k, self.min_votes_ = self._check_params(n_features)
 
        self.selected_by_ = {}
        self.votes_ = np.zeros(n_features, dtype=int)
        for name in self.methods_:
            idx = np.unique(np.asarray(
                VOTERS[name](X, y, k, random_state=self.random_state,
                             n_estimators=self.n_estimators),
                dtype=int))
            self.selected_by_[name] = idx
            self.votes_[idx] += 1  # np.unique above -> one vote per voter per feature
 
        self.support_ = self.votes_ >= self.min_votes_
        if not self.support_.any():
            msg = (f"No feature reached min_votes={self.min_votes_} "
                   f"(max votes = {self.votes_.max()}).")
            if not self.fallback:
                raise ValueError(msg)
            warnings.warn(msg + f" Falling back to top-{k} features by vote count.", UserWarning)
            order = np.lexsort((np.arange(n_features), -self.votes_))  # stable tie-break by index
            self.support_ = np.zeros(n_features, dtype=bool)
            self.support_[order[:k]] = True
        return self
 
    def _get_support_mask(self):
        # SelectorMixin builds transform / get_support / get_feature_names_out on top of this.
        check_is_fitted(self, "support_")
        return self.support_
 
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True  # fit needs y
        return tags
 
    # ---- reporting ----------------------------------------------------------
    def vote_report(self, feature_names=None):
        """List of dicts per feature: votes, which voters picked it, selected or not."""
        check_is_fitted(self, "support_")
        if feature_names is None:
            feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        sets = {m: set(idx.tolist()) for m, idx in self.selected_by_.items()}
        rows = [{
            "feature": str(name),
            "votes": int(self.votes_[i]),
            "picked_by": ", ".join(m for m in self.methods_ if i in sets[m]) or "-",
            "selected": bool(self.support_[i]),
        } for i, name in enumerate(feature_names)]
        return sorted(rows, key=lambda r: -r["votes"])