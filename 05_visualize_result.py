"""
05_visualize_result.py
======================
Turn the JSON artefacts written by 04_track_evaluation.py into the figures that
answer the five Phase 2 Research Questions.

    python 05_visualize_result.py                          # newest run in experiments/
    python 05_visualize_result.py --run-id 20260925_022325
    python 05_visualize_result.py --metric roc_auc

Reproducibility contract
    The script is a pure function of the input JSON files. Nothing is retrained
    and nothing is resampled from the raw data. Category orders, sort orders and
    colour assignments are fixed in code and no timestamp is written into an
    output name, so the same run_id always produces byte identical figures. The
    only stochastic step is the paired bootstrap confidence interval, which is
    seeded with BOOTSTRAP_SEED.

Robustness contract
    The experiment grid is fixed by 04_track_evaluation.py, but this script never
    assumes it. Selector, imbalance and model levels are read from the data and
    only ordered by the canonical lists below, unknown levels are appended, absent
    levels are dropped, and every view is registered on its own so a missing or
    empty input costs one figure and never the rest of the run. Skips are printed
    and recorded in manifest.json. Reruns overwrite in place.

Layout rules
    Long category names live on the vertical axis, so bar charts with wordy
    categories are drawn horizontally instead of rotating the tick labels.
    Legends sit outside the plotting area and no explanatory text is drawn inside
    it, so nothing can land on top of a bar, a marker or a line. The written
    explanation of each figure is saved next to the figures as captions.md rather
    than printed underneath them.

Output
    experiments/graph/<run_id>/
        *.png            one figure per view
        captions.md      what each figure shows and how to read it
        tables/*.csv     the numbers behind every figure
        manifest.json    run_id, input checksums, parameters, figures, skips
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import traceback
from itertools import combinations

import matplotlib
matplotlib.use("Agg")                     # headless and deterministic rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# =========================================================================
# Configuration
# =========================================================================
RESULTS_DIR = "experiments"
GRAPH_ROOT = os.path.join("experiments", "graph")
PRIMARY_METRIC = "pr_auc"
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 5000
TOP_K_FEATURES = 15          # rows in the feature frequency views
CONSENSUS_K = 10             # size of the per selector explanation set

CANONICAL_SELECTORS = ["none", "mi", "l1", "xgb", "boruta"]
CANONICAL_IMBALANCE = ["none", "weight", "smote"]
CANONICAL_MODELS = ["lr", "rf", "xgb"]

SELECTOR_LABEL = {
    "none": "all features",
    "mi": "mutual information top 20",
    "l1": "L1 logistic",
    "xgb": "XGBoost importance top 20",
    "boruta": "Boruta",
}
IMBALANCE_LABEL = {
    "none": "no imbalance handling",
    "weight": "class weighting",
    "smote": "SMOTE oversampling",
}
IMBALANCE_SHORT = {"none": "none", "weight": "class weight", "smote": "SMOTE"}
MODEL_LABEL = {"lr": "logistic regression", "rf": "random forest", "xgb": "XGBoost"}
METRIC_LABEL = {
    "pr_auc": "precision recall AUC",
    "roc_auc": "ROC AUC",
    "recall": "recall at threshold 0.5",
    "precision": "precision at threshold 0.5",
    "f1": "F1 at threshold 0.5",
    "mcc": "Matthews correlation",
}
THRESHOLD_METRICS = ["recall", "precision", "f1", "mcc"]
STABILITY_AXIS = "stability, one means identical subsets"

# Categorical hues in fixed order, validated for colour vision deficiency
# separation on the light surface below. Hues are never recycled: a level beyond
# the palette is drawn in neutral grey and stays identifiable by its legend entry.
PALETTE5 = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#7B3294"]
PALETTE3 = ["#0072B2", "#D55E00", "#009E73"]
OVERFLOW_COLOR = "#8a8a86"
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]

SURFACE = "#fcfcfb"
INK = "#1b1b1a"
INK_MUTED = "#6b6b68"
GRID = "#dcdcd8"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f2f7fb", "#0072B2", "#003a5c"])

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "font.size": 9.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "svg.hashsalt": "iipp-task2",
}


# =========================================================================
# Small utilities that keep the script alive when the input changes
# =========================================================================
class SkipFigure(Exception):
    """Raised when a view has nothing to draw. Caught and reported, never fatal."""


def present(values, canonical) -> list[str]:
    """Levels actually in the data, canonical ones first, unknown ones appended."""
    seen = list(dict.fromkeys(str(v) for v in pd.Series(list(values)).dropna()))
    ordered = [c for c in canonical if c in seen]
    return ordered + sorted(v for v in seen if v not in canonical)


def label_of(level, mapping) -> str:
    return mapping.get(str(level), str(level))


def colour_map(levels, palette) -> dict[str, str]:
    return {lv: (palette[i] if i < len(palette) else OVERFLOW_COLOR)
            for i, lv in enumerate(levels)}


def cycle_map(levels, options) -> dict[str, object]:
    return {lv: options[i % len(options)] for i, lv in enumerate(levels)}


def safe_max(values, default=1.0) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float(default)


def require(condition, reason: str):
    if not condition:
        raise SkipFigure(reason)


# =========================================================================
# Loading
# =========================================================================
def discover_run_id(results_dir: str) -> str:
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    pat = re.compile(r"^(\d{8}_\d{6})_results\.json$")
    ids = sorted(m.group(1) for m in (pat.match(f) for f in os.listdir(results_dir)) if m)
    if not ids:
        raise FileNotFoundError(
            f"No '*_results.json' found in {results_dir}. Run 04_track_evaluation.py first."
        )
    return ids[-1]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(results_dir: str, run_id: str):
    """Required artefacts raise, optional ones come back as empty frames."""
    required = ["results", "selections"]
    optional = ["summary", "stability", "feature_frequency"]
    paths, frames = {}, {}
    for name in required + optional:
        path = os.path.join(results_dir, f"{run_id}_{name}.json")
        if not os.path.exists(path):
            if name in required:
                raise FileNotFoundError(f"Required artefact missing: {path}")
            print(f"  note: optional artefact missing, continuing without it: {path}")
            frames[name] = pd.DataFrame()
            continue
        try:
            frames[name] = pd.read_json(path)
            paths[name] = path
        except ValueError as exc:
            if name in required:
                raise
            print(f"  note: could not read {path} ({exc}), continuing without it")
            frames[name] = pd.DataFrame()
    return frames, paths


class Grid:
    """The experiment grid as it actually appears in this run."""

    def __init__(self, results: pd.DataFrame, selections: pd.DataFrame):
        source = results if not results.empty else selections
        self.selectors = present(source["selector"] if "selector" in source else [],
                                 CANONICAL_SELECTORS)
        self.imbalance = present(source["imbalance"] if "imbalance" in source else [],
                                 CANONICAL_IMBALANCE)
        self.models = present(results["model"] if "model" in results else [],
                              CANONICAL_MODELS)
        self.selector_color = colour_map(self.selectors, PALETTE5)
        self.imbalance_color = colour_map(self.imbalance, PALETTE3)
        self.model_color = colour_map(self.models, PALETTE3)
        self.imbalance_marker = cycle_map(self.imbalance, MARKERS)
        self.imbalance_dash = cycle_map(self.imbalance, DASHES)

    def order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Freeze row order so reruns are identical and panels stay comparable."""
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()
        df = df.copy()
        for col, levels in (("selector", self.selectors), ("imbalance", self.imbalance),
                            ("model", self.models)):
            if col in df.columns and levels:
                df[col] = pd.Categorical(df[col].astype(str), levels, ordered=True)
        sort_cols = [c for c in ["imbalance", "selector", "model", "fold"] if c in df.columns]
        return df.sort_values(sort_cols, ignore_index=True) if sort_cols else df

    def selector_labels(self):
        return [label_of(s, SELECTOR_LABEL) for s in self.selectors]

    def imbalance_labels(self):
        return [label_of(i, IMBALANCE_LABEL) for i in self.imbalance]

    def imbalance_short(self):
        return [label_of(i, IMBALANCE_SHORT) for i in self.imbalance]


def positive_prevalence(dev_index_path="splits/random/dev_indices_random.csv",
                        label_path="secom/secom_labels.data"):
    """Baseline for precision recall AUC. None when the raw files are unavailable."""
    try:
        if not (os.path.exists(dev_index_path) and os.path.exists(label_path)):
            return None
        y = pd.read_csv(label_path, sep=r"\s+", header=None, names=["label", "timestamp"])
        idx = pd.read_csv(dev_index_path)["index"].to_numpy()
        idx = idx[(idx >= 0) & (idx < len(y))]
        if idx.size == 0:
            return None
        return float((y.iloc[idx]["label"] == 1).mean())
    except Exception as exc:                                  # never block the figures
        print(f"  note: could not compute the positive rate ({exc})")
        return None


# =========================================================================
# Statistics
# =========================================================================
def paired_bootstrap_ci(deltas, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Percentile confidence interval for the mean of paired per fold differences."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, np.nan, 0
    if d.size == 1:
        return float(d[0]), float(d[0]), float(d[0]), 1
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi), int(d.size)


def paired_delta_table(grid: Grid, results: pd.DataFrame, metric: str, factor: str,
                       reference: str, within: list[str]) -> pd.DataFrame:
    """
    Per fold paired differences of one metric for every level of `factor` against
    `reference`, holding `within` fixed, with a bootstrap confidence interval.
    """
    require(metric in results.columns, f"the metric '{metric}' is not in the results file")
    require("fold" in results.columns, "the results file has no fold column")
    missing = [c for c in within + [factor] if c not in results.columns]
    require(not missing, f"the results file is missing the columns {missing}")

    wide = results.pivot_table(index=within + ["fold"], columns=factor, values=metric,
                               observed=True)
    require(reference in [str(c) for c in wide.columns],
            f"there is no '{reference}' level of '{factor}' to compare against")
    reference_col = next(c for c in wide.columns if str(c) == reference)

    rows = []
    for level in wide.columns:
        if str(level) == reference:
            continue
        diff = (wide[level] - wide[reference_col]).dropna()
        if diff.empty:
            continue
        frame = diff.reset_index(name="delta")
        for combo, g in frame.groupby(within, observed=True):
            combo = combo if isinstance(combo, tuple) else (combo,)
            mean, lo, hi, n = paired_bootstrap_ci(g["delta"].to_numpy())
            row = dict(zip(within, [str(c) for c in combo]))
            row.update({factor: str(level), "delta_mean": mean, "ci_low": lo,
                        "ci_high": hi, "n_folds": n,
                        "folds_better": int((g["delta"] > 0).sum()),
                        "folds_worse": int((g["delta"] < 0).sum())})
            rows.append(row)
    require(rows, f"no paired differences could be formed for '{factor}'")
    return grid.order(pd.DataFrame(rows))


# ---- stability recomputed on the pool a selector can actually reach ------
def _to_matrix(subsets, universe):
    pos = {f: i for i, f in enumerate(universe)}
    Z = np.zeros((len(subsets), len(universe)), dtype=int)
    for i, s in enumerate(subsets):
        idx = [pos[str(f)] for f in s if str(f) in pos]
        Z[i, idx] = 1
    return Z


def _nogueira(Z):
    M, p = Z.shape
    if M < 2 or p < 2:
        return np.nan
    p_hat = Z.mean(axis=0)
    s2 = M / (M - 1) * p_hat * (1 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / p) * (1 - k_bar / p)
    return np.nan if denom == 0 else float(1 - s2.mean() / denom)


def _mean_jaccard(subsets):
    vals = []
    for a, b in combinations(subsets, 2):
        a, b = set(a), set(b)
        union = a | b
        vals.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(vals)) if vals else np.nan


def _feature_sort_key(name):
    digits = re.findall(r"\d+", str(name))
    return (int(digits[0]) if digits else 10**9, str(name))


def as_feature_list(value) -> list[str]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(v) for v in value]
    return []


def candidate_universe(selections: pd.DataFrame) -> list[str]:
    """
    The variables a selector can actually choose from, which is the union of the
    per fold sets that survived preprocessing. Script 04 uses the full raw feature
    list instead, and that inflates every stability index because variables that
    can never be selected contribute no variation.
    """
    if selections.empty or "features" not in selections.columns:
        return []
    ref = selections[selections["selector"].astype(str) == "none"]
    ref = ref if not ref.empty else selections
    pool = set()
    for subset in ref["features"]:
        pool.update(as_feature_list(subset))
    return sorted(pool, key=_feature_sort_key)


def full_universe(selections: pd.DataFrame, feature_frequency: pd.DataFrame) -> list[str]:
    pool = set()
    if not feature_frequency.empty and "feature" in feature_frequency.columns:
        pool.update(feature_frequency["feature"].astype(str))
    if not pool and not selections.empty and "features" in selections.columns:
        for subset in selections["features"]:
            pool.update(as_feature_list(subset))
    return sorted(pool, key=_feature_sort_key)


def recompute_stability(grid: Grid, selections: pd.DataFrame,
                        universes: dict[str, list[str]]) -> pd.DataFrame:
    if selections.empty or "features" not in selections.columns:
        return pd.DataFrame()
    rows = []
    for (imb, sel), g in selections.groupby(["imbalance", "selector"], observed=True):
        g = g.sort_values("fold") if "fold" in g.columns else g
        subsets = [as_feature_list(s) for s in g["features"]]
        subsets = [s for s in subsets if s]
        if not subsets:
            continue
        sizes = np.array([len(s) for s in subsets], dtype=float)
        row = {"imbalance": str(imb), "selector": str(sel), "n_runs": len(subsets),
               "n_features_mean": float(sizes.mean()),
               "n_features_std": float(sizes.std(ddof=1)) if sizes.size > 1 else 0.0,
               "n_features_min": float(sizes.min()), "n_features_max": float(sizes.max()),
               "mean_jaccard": _mean_jaccard(subsets)}
        for tag, universe in universes.items():
            row[f"nogueira_{tag}"] = (_nogueira(_to_matrix(subsets, universe))
                                      if universe else np.nan)
            row[f"p_{tag}"] = len(universe)
        rows.append(row)
    return grid.order(pd.DataFrame(rows))


def frequency_from_selections(grid: Grid, selections: pd.DataFrame,
                              universe: list[str]) -> pd.DataFrame:
    """Rebuilt locally so the figures still work if the frequency artefact is absent."""
    if selections.empty or not universe or "features" not in selections.columns:
        return pd.DataFrame()
    rows = []
    for (imb, sel), g in selections.groupby(["imbalance", "selector"], observed=True):
        subsets = [as_feature_list(s) for s in g["features"]]
        subsets = [s for s in subsets if s]
        if not subsets:
            continue
        freq = _to_matrix(subsets, universe).mean(axis=0)
        for feature, f in zip(universe, freq):
            rows.append({"imbalance": str(imb), "selector": str(sel), "feature": feature,
                         "selection_frequency": float(f),
                         "selected_count": int(round(f * len(subsets))),
                         "n_runs": len(subsets)})
    return grid.order(pd.DataFrame(rows))


def top_sets(feature_frequency: pd.DataFrame, k: int) -> dict[tuple[str, str], list[str]]:
    out = {}
    for (imb, sel), g in feature_frequency.groupby(["imbalance", "selector"], observed=True):
        g = g[g["selection_frequency"] > 0].sort_values(
            ["selection_frequency", "feature"], ascending=[False, True])
        if not g.empty:
            out[(str(imb), str(sel))] = g["feature"].astype(str).head(k).tolist()
    return out


# =========================================================================
# Plot plumbing
# =========================================================================
class Figures:
    """Collects the figures, their tables and their captions, and writes them."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.table_dir = os.path.join(out_dir, "tables")
        os.makedirs(self.table_dir, exist_ok=True)
        self.figures: list[str] = []
        self.tables: list[str] = []
        self.captions: list[tuple[str, str]] = []
        self.skipped: dict[str, str] = {}

    def save(self, fig, name: str, caption: str = "", table: pd.DataFrame | None = None):
        path = os.path.join(self.out_dir, f"{name}.png")
        fig.savefig(path)
        plt.close(fig)
        self.figures.append(os.path.basename(path))
        if caption:
            self.captions.append((name, " ".join(caption.split())))
        print(f"  figure written: {path}")
        self.table(table, name)

    def table(self, df: pd.DataFrame | None, name: str):
        if df is None or df.empty:
            return
        path = os.path.join(self.table_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        self.tables.append(os.path.basename(path))

    def write_captions(self, run_id: str):
        if not self.captions:
            return
        lines = [f"# How to read the figures of run {run_id}", ""]
        for name, caption in self.captions:
            lines += [f"## {name}", "", caption, ""]
        with open(os.path.join(self.out_dir, "captions.md"), "w") as f:
            f.write("\n".join(lines))

    def skip(self, name: str, reason: str):
        self.skipped[name] = reason
        print(f"  view skipped: {name} because {reason}")

    def run(self, name: str, fn, *args, **kwargs):
        """Never let one broken view take the whole run down."""
        try:
            return fn(*args, **kwargs)
        except SkipFigure as exc:
            self.skip(name, str(exc))
        except Exception as exc:
            self.skip(name, f"an unexpected error occurred, {exc}")
            traceback.print_exc()
        finally:
            plt.close("all")
        return None


def title(fig, text: str):
    fig.suptitle(text, fontsize=12.5, fontweight="semibold", x=0.005, ha="left")


def style(ax, xgrid=False, ygrid=True):
    ax.set_axisbelow(True)
    ax.grid(visible=False)
    if ygrid:
        ax.grid(axis="y", alpha=0.9)
    if xgrid:
        ax.grid(axis="x", alpha=0.9)
    return ax


def panels(n, width_each=4.5, height=4.4, sharey=False, sharex=False):
    n = max(int(n), 1)
    fig, axes = plt.subplots(1, n, figsize=(width_each * n, height), sharey=sharey,
                             sharex=sharex, squeeze=False)
    return fig, list(axes[0])


def swatches(mapping, labels):
    return [Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=c,
                   markeredgecolor=SURFACE, label=label_of(k, labels))
            for k, c in mapping.items()]


def shapes(mapping, labels):
    return [Line2D([0], [0], marker=m, linestyle="", markersize=8, color=INK_MUTED,
                   label=label_of(k, labels)) for k, m in mapping.items()]


def side_legend(fig, handles, title_text=None):
    """One legend, outside the plotting area, never on top of the data."""
    if not handles:
        return
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.004, 0.5),
               frameon=False, title=title_text, alignment="left")


def series_for(frame: pd.DataFrame, level_col: str, level: str, index_col: str,
               value_col: str, index_order: list[str]) -> np.ndarray:
    """One bar group, aligned to a fixed category order, missing cells become NaN."""
    sub = frame[frame[level_col].astype(str) == str(level)]
    if sub.empty:
        return np.full(len(index_order), np.nan)
    s = sub.set_index(sub[index_col].astype(str))[value_col]
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(index_order).to_numpy(dtype=float)


def grouped_hbars(ax, category_labels, groups, values, errs, colors, labels,
                  value_fmt="{:.3f}", annotate=True):
    """
    Horizontal grouped bars. Category names go on the vertical axis, where a long
    name has room, and each value is printed just past the end of its own bar, so
    no text can land on a bar or on a neighbouring label.
    """
    n_g = max(len(groups), 1)
    slot = 0.84 / n_g
    y = np.arange(len(category_labels), dtype=float)[::-1]
    reach = 0.0
    for j, g in enumerate(groups):
        off = -0.42 + slot * (j + 0.5)
        v = np.asarray(values[j], dtype=float)
        e = None if errs is None else np.nan_to_num(np.asarray(errs[j], dtype=float))
        ax.barh(y + off, np.nan_to_num(v, nan=0.0), height=slot * 0.9,
                color=colors.get(g, OVERFLOW_COLOR), linewidth=0.8, edgecolor=SURFACE,
                label=label_of(g, labels), xerr=e,
                error_kw=dict(ecolor=INK_MUTED, elinewidth=1.0, capsize=0))
        spans = np.nan_to_num(v, nan=0.0) + (e if e is not None else 0.0)
        reach = max(reach, float(np.max(spans)) if spans.size else 0.0)
        if annotate:
            for yi, vi, si in zip(y + off, v, spans):
                if np.isfinite(vi):
                    ax.annotate(value_fmt.format(vi), (si, yi), xytext=(4, 0),
                                textcoords="offset points", va="center", ha="left",
                                fontsize=7, color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(category_labels)
    ax.set_ylim(-0.6, len(category_labels) - 0.4)
    ax.set_xlim(0, reach * 1.24 if reach > 0 else 1.0)
    style(ax, xgrid=True, ygrid=False)
    return ax


def dot_whisker(ax, frame, ylabels, color_key, colors, xlabel):
    y = np.arange(len(frame))[::-1]
    for yi, (_, r) in zip(y, frame.iterrows()):
        c = colors.get(str(r[color_key]), OVERFLOW_COLOR)
        lo, hi, mid = r["ci_low"], r["ci_high"], r["delta_mean"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.0, solid_capstyle="round",
                    alpha=0.85)
        if np.isfinite(mid):
            ax.plot([mid], [yi], marker="o", markersize=7, color=c,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_xlabel(xlabel)
    style(ax, xgrid=True, ygrid=False)
    return ax


def pareto_front(points):
    """Row indices that nothing else beats on both maximised columns at once."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return []
    ok = np.flatnonzero(np.all(np.isfinite(pts), axis=1))
    keep = [int(i) for i in ok
            if not np.any(np.all(pts[ok] >= pts[i], axis=1) &
                          np.any(pts[ok] > pts[i], axis=1))]
    return sorted(keep, key=lambda i: pts[i, 0])


def heatmap(ax, matrix, xlabels, ylabels, value_fmt="{:.2f}"):
    data = np.asarray(matrix, dtype=float)
    im = ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    many = len(xlabels) > 8
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=90 if many else 30,
                  ha="center" if many else "right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, value_fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=7, color=SURFACE if data[i, j] > 0.55 else INK)
    ax.grid(False)
    return im


# =========================================================================
# RQ1  how far the variable set can shrink before prediction suffers
# =========================================================================
def rq1_performance(fx: Figures, grid: Grid, results: pd.DataFrame, metric: str,
                    prevalence):
    require(not results.empty, "the results file contains no rows")
    require(metric in results.columns, f"the metric '{metric}' is not in the results file")
    agg = (results.groupby(["imbalance", "selector", "model"], observed=True)[metric]
           .agg(["mean", "std", "count"]).reset_index())
    require(not agg.empty, "no performance rows could be aggregated")
    agg["standard_error"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg = grid.order(agg)

    bars = max(len(grid.selectors) * max(len(grid.models), 1), 1)
    fig, axes = panels(len(grid.imbalance), width_each=4.4, height=1.6 + 0.30 * bars,
                       sharey=True, sharex=True)
    upper = safe_max(agg["mean"] + agg["standard_error"].fillna(0)) * 1.26
    for ax, imb in zip(axes, grid.imbalance):
        sub = agg[agg["imbalance"].astype(str) == imb]
        values = [series_for(sub, "model", m, "selector", "mean", grid.selectors)
                  for m in grid.models]
        errs = [series_for(sub, "model", m, "selector", "standard_error", grid.selectors)
                for m in grid.models]
        grouped_hbars(ax, grid.selector_labels(), grid.models, values, errs,
                      grid.model_color, MODEL_LABEL)
        ax.set_xlim(0, upper)
        if prevalence is not None:
            ax.axvline(prevalence, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_title(label_of(imb, IMBALANCE_LABEL))
        ax.set_xlabel(METRIC_LABEL.get(metric, metric))
    handles = swatches(grid.model_color, MODEL_LABEL)
    if prevalence is not None:
        handles.append(Line2D([0], [0], color=INK_MUTED, linestyle=(0, (4, 3)),
                              label=f"random baseline {prevalence:.3f}"))
    side_legend(fig, handles, "classifier")
    title(fig, "RQ1  how well each feature selection method predicts the quality anomaly")
    fx.save(fig, "rq1_01_performance_by_selector", caption=(
        "Each bar is one classifier trained on the variables that one selection method kept, averaged over "
        "every validation fold, and the thin line at its end is one standard error. The top block keeps all "
        "variables and is the reference the other blocks have to match. The dashed vertical line is what a "
        "model that guesses at random would score, which for this metric equals the share of failed wafers in "
        "the development set. A method preserves prediction quality when its bars stay level with the top "
        "block, not merely when they stay to the right of the dashed line."), table=agg)
    return agg


def rq1_delta(fx: Figures, grid: Grid, results: pd.DataFrame, metric: str):
    delta = paired_delta_table(grid, results, metric, factor="selector", reference="none",
                               within=["imbalance", "model"])
    rows = max(int(delta.groupby("imbalance", observed=True).size().max()), 1)
    fig, axes = panels(len(grid.imbalance), width_each=4.6, height=1.9 + 0.32 * rows,
                       sharex=True)
    for ax, imb in zip(axes, grid.imbalance):
        sub = delta[delta["imbalance"].astype(str) == imb].sort_values(
            ["model", "selector"], ignore_index=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        labels = [f"{label_of(r['model'], MODEL_LABEL)}, "
                  f"{label_of(r['selector'], SELECTOR_LABEL)}" for _, r in sub.iterrows()]
        dot_whisker(ax, sub, labels, "selector", grid.selector_color,
                    f"change in {METRIC_LABEL.get(metric, metric)}")
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title(label_of(imb, IMBALANCE_LABEL))
    shown = set(delta["selector"].astype(str))
    side_legend(fig, swatches({k: v for k, v in grid.selector_color.items() if k in shown},
                              SELECTOR_LABEL), "selection method")
    title(fig, "RQ1  what each selection method costs compared with keeping every variable")
    fx.save(fig, "rq1_02_change_against_all_features", caption=(
        "The same fold is scored twice, once with all variables and once with the reduced set, and the two "
        "scores are subtracted. The dot is the average of those paired differences and the bar around it is "
        "the ninety five percent bootstrap confidence interval. A dot on the dashed zero line means the "
        "reduced set matched the full set on the same data. A bar that crosses the line means the difference "
        "stays inside resampling noise, so the reduction comes for free. A bar lying entirely to the left of "
        "the line means the reduction really did cost prediction quality."), table=delta)


# =========================================================================
# RQ2  how stable the selected variable set is
# =========================================================================
def rq2_stability(fx: Figures, grid: Grid, stab: pd.DataFrame):
    require(not stab.empty, "no stability rows could be computed from the selections file")
    columns = [(c, t) for c, t in [
        ("nogueira_corrected", "counted on the selectable pool"),
        ("nogueira_raw", "counted on the raw list, as in script 04"),
        ("mean_jaccard", "plain overlap of two subsets")]
        if c in stab.columns]
    require(columns, "the stability table has none of the expected stability columns")

    bars = max(len(grid.selectors) * max(len(grid.imbalance), 1), 1)
    fig, axes = panels(len(columns), width_each=4.4, height=1.6 + 0.30 * bars,
                       sharey=True, sharex=True)
    for ax, (col, subtitle) in zip(axes, columns):
        values = [series_for(stab, "imbalance", imb, "selector", col, grid.selectors)
                  for imb in grid.imbalance]
        grouped_hbars(ax, grid.selector_labels(), grid.imbalance, values, None,
                      grid.imbalance_color, IMBALANCE_LABEL)
        ax.set_xlim(0, 1.18)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_title(subtitle)
        ax.set_xlabel(STABILITY_AXIS)
    side_legend(fig, swatches(grid.imbalance_color, IMBALANCE_LABEL), "imbalance strategy")
    runs = int(stab["n_runs"].max()) if "n_runs" in stab.columns else 0
    title(fig, "RQ2  how often the same process variables come back when the data is resampled")
    fx.save(fig, "rq2_01_stability_by_selector", caption=(
        f"Stability is measured over the {runs} resampled training folds. One means every fold selected exactly "
        "the same variables and zero means the picks look random. The first two panels use the same Nogueira "
        "index and differ only in how many variables count as selectable. The left panel counts the pool that "
        "actually survives the per fold missing value and correlation filters, which is the pool a method can "
        "really choose from. The middle panel counts the whole raw variable list the way script 04 does, and "
        "comes out higher because variables that can never be selected add no variation and quietly pull the "
        "index up. The right panel is the plain overlap between two subsets, which is easier to read but "
        "unfair to methods that keep many variables, because large subsets overlap more by chance. Notice that "
        "even keeping all variables does not score one, because the preprocessing filters themselves change "
        "from fold to fold, and that value is the floor every method inherits."), table=stab)


def rq2_profiles(fx: Figures, grid: Grid, feature_frequency: pd.DataFrame):
    require(not feature_frequency.empty, "no feature frequency information is available")
    fig, axes = panels(len(grid.imbalance), width_each=4.4, height=4.2, sharey=True)
    drawn = []
    for ax, imb in zip(axes, grid.imbalance):
        for sel in grid.selectors:
            g = feature_frequency[(feature_frequency["imbalance"].astype(str) == imb) &
                                  (feature_frequency["selector"].astype(str) == sel)]
            f = np.sort(g["selection_frequency"].to_numpy(dtype=float))[::-1]
            f = f[np.isfinite(f) & (f > 0)]
            if f.size:
                ax.plot(np.arange(1, f.size + 1), f,
                        color=grid.selector_color.get(sel, OVERFLOW_COLOR))
                drawn.append(sel)
        ax.set_xscale("log")
        ax.set_xlabel("variables ranked from most to least selected")
        ax.set_title(label_of(imb, IMBALANCE_LABEL))
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("share of folds in which the variable was selected")
    side_legend(fig, swatches({k: v for k, v in grid.selector_color.items() if k in drawn},
                              SELECTOR_LABEL), "selection method")
    title(fig, "RQ2  whether a method keeps returning to the same few variables")
    fx.save(fig, "rq2_02_selection_frequency_profiles", caption=(
        "Every curve belongs to one selection method and sorts its variables from the most often chosen to the "
        "least. A curve that starts high and then falls off a cliff means a small core of variables is chosen "
        "in almost every fold, which is the shape of a stable method. A curve that slides down gradually and "
        "runs on into a long flat tail means the method keeps swapping variables in and out as the training "
        "data changes. The horizontal axis is logarithmic so that short and long variable lists both fit."))


def rq2_sizes(fx: Figures, grid: Grid, selections: pd.DataFrame):
    require("n_features_selected" in selections.columns,
            "the selections file has no n_features_selected column")
    step = len(grid.imbalance) + 1
    rows, ticks = [], []
    for i, sel in enumerate(grid.selectors):
        base = (len(grid.selectors) - 1 - i) * step
        for j, imb in enumerate(grid.imbalance):
            g = selections[(selections["selector"].astype(str) == sel) &
                           (selections["imbalance"].astype(str) == imb)]
            values = g["n_features_selected"].to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            if values.size == 0:
                continue
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            rows.append({"y": base + (len(grid.imbalance) - 1 - j),
                         "color": grid.imbalance_color.get(imb, OVERFLOW_COLOR),
                         "low": float(values.min()), "high": float(values.max()),
                         "q1": float(q1), "median": float(med), "q3": float(q3)})
        ticks.append((base + (len(grid.imbalance) - 1) / 2, label_of(sel, SELECTOR_LABEL)))
    require(rows, "no positive subset sizes were recorded")

    # A range row rather than a box plot: a selector with a fixed size collapses a
    # box to an invisible sliver, while its median dot is always visible here.
    fig, ax = plt.subplots(figsize=(7.8, 1.8 + 0.30 * len(rows)))
    for r in rows:
        ax.plot([r["low"], r["high"]], [r["y"]] * 2, color=r["color"], linewidth=1.4,
                alpha=0.55, solid_capstyle="butt", zorder=1)
        ax.plot([r["q1"], r["q3"]], [r["y"]] * 2, color=r["color"], linewidth=7,
                alpha=0.9, solid_capstyle="butt", zorder=2)
        ax.plot([r["median"]], [r["y"]], marker="o", markersize=7, color=r["color"],
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks])
    ax.set_ylim(min(r["y"] for r in rows) - 1, max(r["y"] for r in rows) + 1)
    ax.set_xlabel("number of variables kept, logarithmic")
    side_legend(fig, swatches(grid.imbalance_color, IMBALANCE_LABEL) +
                [Line2D([0], [0], color=INK_MUTED, linewidth=7, alpha=0.9,
                        label="middle half of the folds"),
                 Line2D([0], [0], color=INK_MUTED, linewidth=1.4, alpha=0.55,
                        label="smallest to largest fold"),
                 Line2D([0], [0], marker="o", linestyle="", markersize=7,
                        markerfacecolor=INK_MUTED, markeredgecolor=SURFACE,
                        label="median fold")], "imbalance strategy")
    ax.set_title("RQ2  how much the size of the selected set moves between resamples")
    style(ax, xgrid=True, ygrid=False)
    fx.save(fig, "rq2_03_subset_size_spread", caption=(
        "Each row is one method under one imbalance strategy. The thin line runs from the smallest to the "
        "largest number of variables that method kept across the resampled folds, the thick segment covers the "
        "middle half of the folds and the dot marks the median fold. A method shown as a bare dot was told in "
        "advance how many variables to keep, so its size cannot vary. A wide row means the method cannot agree "
        "with itself even on how many variables the problem needs, and that is instability the overlap "
        "measures on their own would hide."),
        table=selections.groupby(["imbalance", "selector"], observed=True)
        ["n_features_selected"].describe().reset_index())


def rq2_recurrent(fx: Figures, grid: Grid, feature_frequency: pd.DataFrame):
    require(not feature_frequency.empty, "no feature frequency information is available")
    reference_imbalance = grid.imbalance[0] if grid.imbalance else None
    require(reference_imbalance is not None, "no imbalance condition is present")
    ref = feature_frequency[feature_frequency["imbalance"].astype(str) == reference_imbalance]
    require(not ref.empty, "no feature frequencies for the first imbalance condition")
    order = (ref.groupby("feature", observed=True)["selection_frequency"].mean()
             .sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist())
    require(order, "no variable was ever selected")
    mat = (ref[ref["feature"].isin(order)]
           .pivot_table(index="selector", columns="feature", values="selection_frequency",
                        observed=True)
           .reindex(index=grid.selectors, columns=order))
    fig, ax = plt.subplots(figsize=(3.6 + 0.56 * len(order),
                                    0.55 * max(len(grid.selectors), 1) + 2.0))
    im = heatmap(ax, mat.to_numpy(), [str(o) for o in order], grid.selector_labels())
    ax.set_xlabel("process variable")
    ax.set_title("RQ2  the process variables that keep coming back, and which method finds them")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="share of folds selected")
    fx.save(fig, "rq2_04_recurrent_variables", caption=(
        f"The columns are the {len(order)} variables with the highest average selection frequency under "
        f"{label_of(reference_imbalance, IMBALANCE_LABEL)} and the rows are the selection methods. A dark cell "
        "means that method chose that variable in nearly every fold. A column that stays dark all the way down "
        "means different methods independently agree the variable matters, and that is the strongest kind of "
        "evidence this run can offer for taking a variable to the process engineers."),
        table=mat.reset_index())


# =========================================================================
# RQ3  the trade off between performance, reduction and stability
# =========================================================================
def build_tradeoff(grid: Grid, agg: pd.DataFrame, stab: pd.DataFrame,
                   selections: pd.DataFrame, pool_size: int = 0) -> pd.DataFrame:
    """
    Reduction is the per fold ratio of variables kept to variables that survived
    preprocessing, averaged over folds, so it is never a ratio of two averages and
    never measured against the raw variable count.
    """
    require(not stab.empty, "no stability rows are available")
    require(not agg.empty, "no performance summary is available")
    best = (agg.sort_values("mean", ascending=False)
            .groupby(["imbalance", "selector"], observed=True)
            .head(1)[["imbalance", "selector", "model", "mean"]]
            .rename(columns={"mean": "best_performance", "model": "best_model"}))
    for c in ["imbalance", "selector", "best_model"]:
        best[c] = best[c].astype(str)

    red = selections.copy()
    if "n_features_after_preprocessing" in red.columns:
        denom = red["n_features_after_preprocessing"].replace(0, np.nan)
    elif pool_size > 0:
        # the per fold pool was not recorded, so fall back to the pool as a whole
        denom = pd.Series(float(pool_size), index=red.index)
    else:
        denom = pd.Series(np.nan, index=red.index)
    red["ratio"] = (red["n_features_selected"] / denom
                    if "n_features_selected" in red.columns else np.nan)
    red = (red.groupby(["imbalance", "selector"], observed=True)["ratio"].mean()
           .reset_index())
    red["imbalance"] = red["imbalance"].astype(str)
    red["selector"] = red["selector"].astype(str)
    red["reduction"] = 1 - red["ratio"]

    t = stab.copy()
    t["imbalance"] = t["imbalance"].astype(str)
    t["selector"] = t["selector"].astype(str)
    t = t.merge(best, on=["imbalance", "selector"], how="left")
    t = t.merge(red[["imbalance", "selector", "reduction"]], on=["imbalance", "selector"],
                how="left")
    t["stability"] = t["nogueira_corrected"] if "nogueira_corrected" in t.columns else np.nan
    require(t["reduction"].notna().any() and t["best_performance"].notna().any(),
            "reduction or performance could not be computed for any configuration")
    return grid.order(t)


def rq3_tradeoff(fx: Figures, grid: Grid, tradeoff: pd.DataFrame, metric: str):
    fig, axes = panels(2, width_each=5.6, height=4.8)
    for ax, ycol, ylab, subtitle in [
            (axes[0], "best_performance",
             f"best {METRIC_LABEL.get(metric, metric)} across classifiers",
             "prediction quality against reduction"),
            (axes[1], "stability", STABILITY_AXIS, "stability against reduction")]:
        for _, r in tradeoff.iterrows():
            ax.plot(r["reduction"], r[ycol],
                    marker=grid.imbalance_marker.get(str(r["imbalance"]), "o"),
                    markersize=11,
                    color=grid.selector_color.get(str(r["selector"]), OVERFLOW_COLOR),
                    markeredgecolor=SURFACE, markeredgewidth=1.6, linestyle="")
        ax.set_xlim(-0.06, 1.06)
        ax.margins(y=0.16)
        ax.set_xlabel("share of the surviving variables removed")
        ax.set_ylabel(ylab)
        ax.set_title(subtitle)
        style(ax, xgrid=True)
    front = pareto_front(tradeoff[["reduction", "best_performance"]].to_numpy())
    if len(front) > 1:
        axes[0].plot(tradeoff.iloc[front]["reduction"],
                     tradeoff.iloc[front]["best_performance"], color=INK_MUTED,
                     linewidth=1.2, linestyle=(0, (4, 3)), zorder=0)
    handles = (swatches(grid.selector_color, SELECTOR_LABEL) +
               shapes(grid.imbalance_marker, IMBALANCE_LABEL) +
               [Line2D([0], [0], color=INK_MUTED, linestyle=(0, (4, 3)),
                       label="best achievable frontier")])
    side_legend(fig, handles, "colour is the method, shape is the strategy")
    title(fig, "RQ3  reading the three way trade off as two side by side projections")
    fx.save(fig, "rq3_01_tradeoff_projections", caption=(
        "Colour is the selection method and marker shape is the imbalance strategy, so one configuration "
        "appears once in each panel at the same horizontal position and can be traced between them. In the "
        "left panel the dashed line joins the configurations that no other configuration beats on both "
        "prediction quality and reduction at the same time, and that is the set worth arguing about. The right "
        "panel swaps prediction quality for stability, so reading both panels at one horizontal position shows "
        "what a given amount of reduction costs on the other two criteria. The exact value behind every marker "
        "is in the table of the same name."), table=tradeoff)


def rq3_profile(fx: Figures, grid: Grid, tradeoff: pd.DataFrame, metric: str):
    cols = [("best_performance", f"best {METRIC_LABEL.get(metric, metric)}"),
            ("reduction", "variable reduction"), ("stability", "selection stability")]
    norm = tradeoff.copy()
    for c, _ in cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c + "_scaled"] = (0.5 if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo
                               else (norm[c] - lo) / (hi - lo))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    xs = np.arange(len(cols))
    for _, r in norm.iterrows():
        ax.plot(xs, [r[c + "_scaled"] for c, _ in cols],
                color=grid.selector_color.get(str(r["selector"]), OVERFLOW_COLOR),
                linewidth=2.0, alpha=0.85,
                linestyle=grid.imbalance_dash.get(str(r["imbalance"]), "-"),
                marker=grid.imbalance_marker.get(str(r["imbalance"]), "o"), markersize=7,
                markeredgecolor=SURFACE)
    ax.set_xticks(xs, [lab for _, lab in cols])
    ax.set_xlim(-0.15, len(cols) - 0.85)
    ax.set_ylim(-0.06, 1.06)
    ax.set_ylabel("rescaled within this run, one is the best value seen")
    ax.set_title("RQ3  the profile of every configuration across all three criteria")
    side_legend(fig, swatches(grid.selector_color, SELECTOR_LABEL) +
                shapes(grid.imbalance_marker, IMBALANCE_LABEL),
                "colour is the method, shape is the strategy")
    style(ax)
    fx.save(fig, "rq3_02_profile_across_criteria", caption=(
        "Every line is one combination of selection method and imbalance strategy. The three criteria live on "
        "different natural scales, so each one is rescaled to put the worst configuration in this run at zero "
        "and the best at one, which makes the lines comparable as rankings rather than as absolute values. A "
        "line that stays high across all three points is a genuine compromise, while a line that spikes on one "
        "criterion and collapses on another is buying one property at the expense of the others."), table=norm)


# =========================================================================
# RQ4  imbalance handling, minority detection and stability
# =========================================================================
def rq4_effect(fx: Figures, grid: Grid, results: pd.DataFrame, metric: str):
    require(not results.empty, "the results file contains no rows")
    metrics = [m for m in [metric] + THRESHOLD_METRICS if m in results.columns]
    metrics = list(dict.fromkeys(metrics))
    require(metrics, "none of the expected metric columns are in the results file")

    long = (results.groupby(["imbalance", "selector", "model"], observed=True)[metrics]
            .mean().reset_index())
    bars = max(len(grid.imbalance) * max(len(grid.models), 1), 1)
    fig, axes = panels(len(metrics), width_each=3.4, height=1.8 + 0.34 * bars, sharey=True)
    for ax, m in zip(axes, metrics):
        per_model = (long.groupby([long["model"].astype(str), long["imbalance"].astype(str)],
                                  observed=True)[m].mean().reset_index())
        per_model.columns = ["model", "imbalance", m]
        values = [series_for(per_model, "model", mdl, "imbalance", m, grid.imbalance)
                  for mdl in grid.models]
        grouped_hbars(ax, grid.imbalance_labels(), grid.models, values, None,
                      grid.model_color, MODEL_LABEL, value_fmt="{:.2f}")
        ax.set_title(METRIC_LABEL.get(m, m))
        ax.set_xlabel("averaged over folds and methods")
    side_legend(fig, swatches(grid.model_color, MODEL_LABEL), "classifier")
    title(fig, "RQ4  what imbalance handling does to minority class detection")
    fx.save(fig, "rq4_01_imbalance_effect", caption=(
        "Bars average over every fold and every selection method, so they describe the imbalance strategy "
        "itself rather than any single pipeline. Only the first panel is free of a decision threshold. The "
        "remaining panels are computed at the fixed threshold of one half, and on a data set where roughly "
        "seven wafers in a hundred fail, that threshold is almost never crossed by the tree models. Their near "
        "zero recall therefore says the cut off is wrong for this class balance, not that the models learned "
        "nothing, and the first panel confirms it."), table=long)


def rq4_delta(fx: Figures, grid: Grid, results: pd.DataFrame, metric: str):
    delta = paired_delta_table(grid, results, metric, factor="imbalance", reference="none",
                               within=["selector", "model"])
    others = [i for i in grid.imbalance if i != "none"]
    rows = max(int(delta.groupby("imbalance", observed=True).size().max()), 1)
    fig, axes = panels(len(others), width_each=5.6, height=1.9 + 0.30 * rows, sharex=True)
    for ax, imb in zip(axes, others):
        sub = delta[delta["imbalance"].astype(str) == imb].sort_values(
            ["selector", "model"], ignore_index=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        labels = [f"{label_of(r['selector'], SELECTOR_LABEL)}, "
                  f"{label_of(r['model'], MODEL_LABEL)}" for _, r in sub.iterrows()]
        dot_whisker(ax, sub, labels, "selector", grid.selector_color,
                    f"change in {METRIC_LABEL.get(metric, metric)}")
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title(label_of(imb, IMBALANCE_LABEL))
    shown = set(delta["selector"].astype(str))
    side_legend(fig, swatches({k: v for k, v in grid.selector_color.items() if k in shown},
                              SELECTOR_LABEL), "selection method")
    title(fig, "RQ4  what imbalance handling changes when the same fold is scored both ways")
    fx.save(fig, "rq4_02_change_against_no_handling", caption=(
        "Every pipeline is scored twice on the same fold, once without any imbalance handling and once with "
        "it, and the two scores are subtracted. The dot is the average of those paired differences and the bar "
        "is the ninety five percent bootstrap confidence interval. A bar that crosses the dashed zero line "
        "means the strategy did not move this metric beyond resampling noise. Reading the rows by colour shows "
        "whether a strategy helps every selection method or only rescues the weak ones."), table=delta)


def rq4_stability_shift(fx: Figures, grid: Grid, stab: pd.DataFrame):
    require(not stab.empty, "no stability rows are available")
    require("nogueira_corrected" in stab.columns,
            "the recomputed stability table has no corrected Nogueira column")
    stability_by = stab.pivot_table(index="selector", columns="imbalance",
                                    values="nogueira_corrected", observed=True)
    size_by = stab.pivot_table(index="selector", columns="imbalance",
                               values="n_features_mean", observed=True)
    for frame in (stability_by, size_by):
        frame.index = frame.index.astype(str)
        frame.columns = [str(c) for c in frame.columns]
    stability_by = stability_by.reindex(index=grid.selectors, columns=grid.imbalance)
    size_by = size_by.reindex(index=grid.selectors, columns=grid.imbalance)

    fig, axes = panels(2, width_each=5.2, height=4.4)
    for ax, frame, ylab, subtitle, log in [
            (axes[0], stability_by, STABILITY_AXIS,
             "how repeatable the selection becomes", False),
            (axes[1], size_by, "average number of variables kept",
             "how many variables survive", True)]:
        for sel in grid.selectors:
            ys = [float(frame.loc[sel, imb]) if (sel in frame.index and imb in frame.columns)
                  else np.nan for imb in grid.imbalance]
            ax.plot(range(len(grid.imbalance)), ys,
                    color=grid.selector_color.get(sel, OVERFLOW_COLOR), marker="o",
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(range(len(grid.imbalance)), grid.imbalance_short())
        ax.set_xlim(-0.25, len(grid.imbalance) - 0.75)
        ax.set_xlabel("imbalance strategy")
        ax.set_ylabel(ylab)
        ax.set_title(subtitle)
        if log:
            ax.set_yscale("log")
        style(ax)
    side_legend(fig, swatches(grid.selector_color, SELECTOR_LABEL), "selection method")
    title(fig, "RQ4  how data level resampling changes the variables that get selected")
    fx.save(fig, "rq4_03_stability_under_resampling", caption=(
        "Each line follows one selection method as the imbalance strategy changes from left to right, and the "
        "two panels have to be read together. A method whose stability climbs while its variable count also "
        "explodes is not becoming more decisive, it is accepting almost everything, and near total agreement "
        "about keeping everything is easy to reach and says little. A method whose stability climbs while its "
        "variable count stays fixed is agreeing more firmly about the same short list, but that agreement is "
        "measured on partly synthetic training samples, so it describes the resampled data rather than proving "
        "the signal in the real wafers got stronger."),
        table=stab[[c for c in ["imbalance", "selector", "nogueira_corrected", "mean_jaccard",
                                "n_features_mean"] if c in stab.columns]])


# =========================================================================
# RQ5  stability and explainability
# =========================================================================
def rq5_explainability(fx: Figures, grid: Grid, feature_frequency: pd.DataFrame,
                       stab: pd.DataFrame):
    require(not feature_frequency.empty, "no feature frequency information is available")
    require(not stab.empty, "no stability rows are available")
    sets = top_sets(feature_frequency, CONSENSUS_K)
    require(sets, "no method produced a non empty explanation set")
    reference_imbalance = grid.imbalance[0] if grid.imbalance else None
    require(reference_imbalance is not None, "no imbalance condition is present")
    sels = [s for s in grid.selectors if (reference_imbalance, s) in sets]
    require(len(sels) >= 2, "at least two methods are needed to compare explanation sets")

    overlap = np.full((len(sels), len(sels)), np.nan)
    for i, a in enumerate(sels):
        for j, b in enumerate(sels):
            A, B = set(sets[(reference_imbalance, a)]), set(sets[(reference_imbalance, b)])
            overlap[i, j] = len(A & B) / len(A | B) if (A | B) else np.nan

    fig, axes = panels(2, width_each=6.0, height=5.0)
    im = heatmap(axes[0], overlap, [label_of(s, SELECTOR_LABEL) for s in sels],
                 [label_of(s, SELECTOR_LABEL) for s in sels])
    axes[0].set_title(f"agreement on the top {CONSENSUS_K} variables")
    fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.02, label="share of the lists shared")

    rows = []
    for _, r in stab.iterrows():
        key = (str(r["imbalance"]), str(r["selector"]))
        if key not in sets:
            continue
        freq = feature_frequency[
            (feature_frequency["imbalance"].astype(str) == key[0]) &
            (feature_frequency["selector"].astype(str) == key[1])]
        f = freq.set_index(freq["feature"].astype(str))["selection_frequency"]
        f = f[~f.index.duplicated(keep="first")]
        own = sets[key]
        firmness = float(f.reindex(own).mean()) if own else np.nan
        peers = [k for k in sets if k[0] == key[0] and k[1] != key[1]]
        agreement = float(np.mean([
            len(set(own) & set(sets[p])) / len(set(own) | set(sets[p])) for p in peers
        ])) if peers else np.nan
        rows.append({"imbalance": key[0], "selector": key[1],
                     "stability": r.get("nogueira_corrected", np.nan),
                     "explanation_set_firmness": firmness,
                     "agreement_with_other_methods": agreement,
                     "n_features_mean": r.get("n_features_mean", np.nan)})
    expl = grid.order(pd.DataFrame(rows))
    require(not expl.empty, "no configuration had both a stability value and a top set")

    ax = axes[1]
    for _, r in expl.iterrows():
        ax.plot(r["stability"], r["explanation_set_firmness"],
                marker=grid.imbalance_marker.get(str(r["imbalance"]), "o"), markersize=11,
                color=grid.selector_color.get(str(r["selector"]), OVERFLOW_COLOR),
                markeredgecolor=SURFACE, markeredgewidth=1.6, linestyle="")
    ax.margins(x=0.12, y=0.16)
    ax.set_xlabel(STABILITY_AXIS)
    ax.set_ylabel(f"how often the same top {CONSENSUS_K} variables are chosen")
    ax.set_title("a repeatable selection gives a repeatable explanation")
    style(ax, xgrid=True)
    side_legend(fig, swatches(grid.selector_color, SELECTOR_LABEL) +
                shapes(grid.imbalance_marker, IMBALANCE_LABEL),
                "colour is the method, shape is the strategy")
    title(fig, "RQ5  whether a stable selection also gives an explanation worth trusting")
    fx.save(fig, "rq5_01_stability_and_explainability", caption=(
        "An explanation set here means the ten variables a method chooses most often, which is what an "
        "engineer would be handed as the answer to which process variables drive the failures. The left panel "
        "asks how far two methods hand over the same ten variables under "
        f"{label_of(reference_imbalance, IMBALANCE_LABEL)}. The right panel asks how firm each list is, "
        "measured as how often its own ten variables were actually selected. Points towards the upper right "
        "are methods whose explanation would survive a rerun on fresh data. This run stores no SHAP values and "
        "no model coefficients, so selection frequency stands in for explanation strength, and a full answer "
        "to this question needs script 04 to save the per fold importances as well."), table=expl)


def rq5_consensus(fx: Figures, grid: Grid, feature_frequency: pd.DataFrame):
    require(not feature_frequency.empty, "no feature frequency information is available")
    reference_imbalance = grid.imbalance[0] if grid.imbalance else None
    require(reference_imbalance is not None, "no imbalance condition is present")
    ref = feature_frequency[
        (feature_frequency["imbalance"].astype(str) == reference_imbalance) &
        (feature_frequency["selector"].astype(str) != "none")]
    require(not ref.empty, "no method other than the all feature reference is available")
    consensus = (ref.groupby("feature", observed=True)
                 .agg(mean_frequency=("selection_frequency", "mean"),
                      methods_that_ever_chose_it=("selection_frequency",
                                                  lambda s: int((s > 0).sum())))
                 .sort_values(["mean_frequency", "methods_that_ever_chose_it"],
                              ascending=False).head(TOP_K_FEATURES).reset_index())
    consensus = consensus[consensus["mean_frequency"] > 0]
    require(not consensus.empty, "no variable was ever selected")

    used = [s for s in grid.selectors if s != "none"]
    fig, ax = plt.subplots(figsize=(8.0, 0.40 * len(consensus) + 2.0))
    y = np.arange(len(consensus))[::-1]
    for yi, (_, r) in zip(y, consensus.iterrows()):
        ax.plot([0, r["mean_frequency"]], [yi, yi], color=GRID, linewidth=2.5,
                solid_capstyle="round", zorder=0)
        for sel in used:
            v = ref[(ref["selector"].astype(str) == sel) &
                    (ref["feature"].astype(str) == str(r["feature"]))]
            if not v.empty and float(v["selection_frequency"].iloc[0]) > 0:
                ax.plot([float(v["selection_frequency"].iloc[0])], [yi], marker="o",
                        markersize=8, color=grid.selector_color.get(sel, OVERFLOW_COLOR),
                        markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.set_yticks(y, consensus["feature"].astype(str).tolist())
    ax.set_ylim(-0.7, len(consensus) - 0.3)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("share of folds in which the variable was selected")
    ax.set_ylabel("process variable")
    ax.set_title("RQ5  the variables that several methods keep pointing at")
    side_legend(fig, swatches({k: v for k, v in grid.selector_color.items() if k != "none"},
                              SELECTOR_LABEL) +
                [Line2D([0], [0], color=GRID, linewidth=3,
                        label="average across the methods")], "selection method")
    style(ax, xgrid=True, ygrid=False)
    fx.save(fig, "rq5_02_consensus_variables", caption=(
        "One row is one process variable. The grey bar ends at its average selection frequency across the "
        "methods and each coloured dot is one method. Dots clustered together and far to the right mean the "
        "methods independently and repeatedly chose the same variable, and that is the strongest evidence this "
        "run can offer for an explanation. Dots scattered across the row mean the high average comes from a "
        "single enthusiastic method and should not be reported as a finding."), table=consensus)


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Build the figures for RQ1 to RQ5 from a 04_track_evaluation.py run.")
    ap.add_argument("--run-id", default=None,
                    help="for example 20260925_022325, the default is the newest run")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out-dir", default=None, help=f"the default is {GRAPH_ROOT}/<run_id>")
    ap.add_argument("--metric", default=PRIMARY_METRIC,
                    help="primary metric for RQ1, RQ3 and RQ4, the default is pr_auc")
    args = ap.parse_args()

    run_id = args.run_id or discover_run_id(args.results_dir)
    out_dir = args.out_dir or os.path.join(GRAPH_ROOT, run_id)
    frames, paths = load_run(args.results_dir, run_id)
    print(f"run id {run_id}\noutput directory {out_dir}\nrequested metric {args.metric}")

    grid = Grid(frames["results"], frames["selections"])
    results = grid.order(frames["results"])
    selections = grid.order(frames["selections"])
    if "features" in selections.columns:
        selections["features"] = selections["features"].apply(as_feature_list)
    feature_frequency = grid.order(frames["feature_frequency"])

    metric = args.metric
    if metric not in results.columns:
        fallback = next((m for m in [PRIMARY_METRIC, "roc_auc"] + THRESHOLD_METRICS
                         if m in results.columns), None)
        if fallback is None:
            numeric = [c for c in results.select_dtypes("number").columns
                       if c not in {"fold", "repeat", "n_features"}]
            fallback = numeric[0] if numeric else None
        print(f"  note: the metric '{metric}' is not in this run, using '{fallback}' instead")
        metric = fallback

    universes = {"corrected": candidate_universe(selections),
                 "raw": full_universe(selections, feature_frequency)}
    print(f"selectable pool after preprocessing {len(universes['corrected'])} variables, "
          f"raw variable list {len(universes['raw'])} variables")

    stab = recompute_stability(grid, selections, universes)
    if feature_frequency.empty:
        print("  note: rebuilding the selection frequencies from the selections file")
        feature_frequency = frequency_from_selections(grid, selections,
                                                      universes["corrected"])

    fx = Figures(out_dir)
    with plt.rc_context(RC):
        prevalence = positive_prevalence()
        # Every view is registered on its own, so one missing input costs one
        # figure and never the rest of the run.
        agg = fx.run("rq1_performance", rq1_performance, fx, grid, results, metric,
                     prevalence)
        fx.run("rq1_delta", rq1_delta, fx, grid, results, metric)
        fx.run("rq2_stability", rq2_stability, fx, grid, stab)
        fx.run("rq2_profiles", rq2_profiles, fx, grid, feature_frequency)
        fx.run("rq2_sizes", rq2_sizes, fx, grid, selections)
        fx.run("rq2_recurrent", rq2_recurrent, fx, grid, feature_frequency)
        if agg is None or agg.empty:
            fx.skip("rq3", "there is no performance summary to trade off against")
        else:
            tradeoff = fx.run("rq3_inputs", build_tradeoff, grid, agg, stab, selections,
                              len(universes["corrected"]))
            if tradeoff is not None and not tradeoff.empty:
                fx.run("rq3_tradeoff", rq3_tradeoff, fx, grid, tradeoff, metric)
                fx.run("rq3_profile", rq3_profile, fx, grid, tradeoff, metric)
        fx.run("rq4_effect", rq4_effect, fx, grid, results, metric)
        fx.run("rq4_delta", rq4_delta, fx, grid, results, metric)
        fx.run("rq4_stability_shift", rq4_stability_shift, fx, grid, stab)
        fx.run("rq5_explainability", rq5_explainability, fx, grid, feature_frequency, stab)
        fx.run("rq5_consensus", rq5_consensus, fx, grid, feature_frequency)

    fx.table(stab, "stability_recomputed")
    fx.table(frames["summary"], "summary_from_04")
    fx.write_captions(run_id)

    manifest = {
        "script": os.path.basename(__file__),
        "run_id": run_id,
        "primary_metric": metric,
        "n_folds": int(results["fold"].nunique()) if "fold" in results.columns else None,
        "selectors": grid.selectors,
        "imbalance_strategies": grid.imbalance,
        "models": grid.models,
        "selectable_pool_size": len(universes["corrected"]),
        "raw_variable_count": len(universes["raw"]),
        "positive_rate_in_development_set": prevalence,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": N_BOOTSTRAP, "alpha": 0.05,
                      "kind": "paired percentile interval over folds"},
        "inputs": {k: {"path": v, "sha256": sha256(v)} for k, v in sorted(paths.items())},
        "figures": sorted(fx.figures),
        "tables": sorted(fx.tables),
        "skipped": fx.skipped,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(fx.figures)} figures and {len(fx.tables)} tables written to {out_dir}")
    if fx.skipped:
        print(f"{len(fx.skipped)} views were skipped, see manifest.json for the reasons")


if __name__ == "__main__":
    main()
