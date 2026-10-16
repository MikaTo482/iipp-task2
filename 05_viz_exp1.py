"""Reproduce experiment figures from saved JSON without running model fits.

Run with --run-id to select an experiment and --out-dir to change the destination.
Threshold comparison curves use the full saved grid.
Use --list-options to see choices, --plots to choose figures, and --selectors,
--exclude-selectors, --models, or --imbalances to choose displayed results.
Customized views are saved in separate view folders to avoid mixing outputs.
RQ4 plots imbalance_ap and imbalance_stability retain the no-handling reference
even when --imbalances displays only weight or smote.
Importing this module does not load experiments or create output files.
"""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, FixedLocator, FormatStrFormatter

BASE_DIR = Path(__file__).resolve().parent / "experiments"
RUN_ID = None
# Set lists here for normal script runs, or override them with command-line options.
PLOTS_TO_SHOW = None
SELECTORS_TO_SHOW = None
SELECTORS_TO_HIDE = ["vote"]
MODELS_TO_SHOW = None
IMBALANCES_TO_SHOW = None
PLOT_LABELS = {
    "tradeoff": "Average precision versus feature count",
    "ap_difference": "AP change from all features",
    "stability": "Feature selection stability",
    "feature_frequency": "Most frequently selected features",
    "heatmap": "Performance, reduction and stability heatmap",
    "imbalance_ap": "AP change from imbalance handling",
    "imbalance_stability": "Stability and feature counts under imbalance handling",
    "thresholds": "Recall and precision versus decision threshold",
    "timing": "Training and prediction times",
}
REQUIRED_TABLES = ("config", "results", "summary", "stability", "feature_frequency")
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}

SELECTOR_STYLES = {
    "none": ("All features", "#333333", "D"),
    "mi": ("Mutual information", "#0072B2", "o"),
    "fdr": ("FDR ANOVA", "#56B4E9", "X"),
    "l1": ("L1 logistic regression", "#E69F00", "s"),
    "xgb": ("XGBoost selector", "#009E73", "^"),
    "boruta": ("Boruta", "#CC79A7", "P"),
    "vote": ("Voting", "#D55E00", "*"),
}

MODEL_LABELS = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "xgb": "XGBoost",
}

IMBALANCE_LABELS = {
    "none": "No resampling",
    "weight": "Class weighting",
    "smote": "SMOTE",
}

SELECTOR_ORDER = list(SELECTOR_STYLES)
SELECTOR_LABELS = {key: style[0] for key, style in SELECTOR_STYLES.items()}
SELECTOR_COLORS = {key: style[1] for key, style in SELECTOR_STYLES.items()}
IMBALANCE_ORDER = list(IMBALANCE_LABELS)
MODEL_ORDER = list(MODEL_LABELS)
IMBALANCE_MARKERS = {"none": "o", "weight": "s", "smote": "^"}
STABILITY_METRICS = {
    "nogueira": "Nogueira stability",
    "mean_jaccard": "Mean Jaccard similarity",
}
FIGURE_TITLE = "Average precision, feature count and selection stability"
TIME_METRICS = {
    "selection_s": "Feature selection time (s)",
    "fit_s": "Model training time (s)",
    "train_total_s": "Selection and training time (s)",
    "predict_ms_per_obs": "Prediction time per observation (ms)",
}
TIME_KEYS = ["imbalance", "selector", "model"]


class PlotUnavailable(ValueError):
    """A valid experiment has insufficient observations for this figure."""


def resolve_run_id(base_dir, run_id=None):
    """Choose the latest complete core bundle by its timestamped filename."""
    base_dir = Path(base_dir)
    if run_id is not None:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a filename prefix, not a path")
        return run_id
    for path in sorted(base_dir.glob("*_config.json"), reverse=True):
        candidate = path.name.removesuffix("_config.json")
        if all((base_dir / f"{candidate}_{name}.json").is_file() for name in REQUIRED_TABLES):
            return candidate
    raise ValueError(f"No complete experiment bundle found in {base_dir}")


def output_directory(base_dir, run_id, out_dir=None):
    path = Path(out_dir) if out_dir is not None else Path(base_dir) / "graph" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_config(base_dir, run_id):
    path = Path(base_dir) / f"{run_id}_config.json"
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return config


def save_figure(fig, out_dir, stem):
    """Use stable filenames and metadata so identical inputs reproduce outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight",
                    facecolor="white", metadata=metadata)
    print(f"Saved: {out_dir / stem} (png, pdf)")


def add_figure_legends(fig, selectors, secondary=(), note=None):
    """Reserve physical space for legends even for a single panel."""
    width, height = fig.get_size_inches()
    columns = min(len(selectors), max(1, int(width / 2.5)))
    bottom = 0.15
    if note:
        fig.text(0.5, bottom / height, note, ha="center", fontsize=9)
        bottom += 0.18 + 0.16 * (note.count("\n") + 1)
    if secondary:
        fig.legend(handles=secondary, loc="lower center", frameon=False,
                   bbox_to_anchor=(0.5, bottom / height), ncol=2, fontsize=9)
        bottom += 0.35
    fig.legend(handles=selectors, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, bottom / height), ncol=columns,
               fontsize=9, title="Feature selection method")
    bottom += 0.3 + 0.25 * int(np.ceil(len(selectors) / columns))
    fig.tight_layout(rect=(0, (bottom + 0.18) / height, 1, 1 - 0.65 / height))


def ordered_present(values, preferred):
    values = [str(v) for v in pd.Series(list(values), dtype=object).dropna().unique()]
    return [v for v in preferred if v in values] + sorted(v for v in values if v not in preferred)


def selector_style(name):
    return SELECTOR_STYLES.get(name, (name, "#666666", "o"))


def filter_records(data, filters=None, keep_baseline=False):
    """Filter rows without recalculating scores; paired AP retains its reference."""
    if not filters:
        return data
    mask = pd.Series(True, index=data.index)
    for column in ("selector", "model", "imbalance"):
        allowed = filters.get(column)
        if column in data and allowed is not None:
            allowed = set(allowed)
            if column == "selector" and keep_baseline:
                allowed.add("none")
            mask &= data[column].isin(allowed)
    if "selector" in data:
        excluded = set(filters.get("exclude_selectors", ()))
        if keep_baseline:
            excluded.discard("none")
        mask &= ~data.selector.isin(excluded)
    return data.loc[mask].copy()


def check_complete_grid(data, imbalances, selectors, models, message):
    expected = {(i, s, m) for i in imbalances for s in selectors for m in models}
    actual = set(data[["imbalance", "selector", "model"]].itertuples(index=False, name=None))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{message}. Missing={missing[:10]}, extra={extra[:10]}")


def validate_splits(data, config):
    keys = ["repeat", "fold", "imbalance", "selector", "model"]
    if data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Missing identifiers or duplicate split/configuration records")
    splits = data[["repeat", "fold"]].drop_duplicates()
    counts = splits.groupby("repeat").size()
    if len(counts) != config["n_repeats"] or not counts.eq(config["n_splits"]).all():
        raise ValueError("Split counts do not match config")
    reference = set(splits.itertuples(index=False, name=None))
    for _, group in data.groupby(["imbalance", "selector", "model"]):
        if set(group[["repeat", "fold"]].itertuples(index=False, name=None)) != reference:
            raise ValueError("Configurations must contain identical splits")


@plt.rc_context(PLOT_STYLE)
def plot_tradeoff(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    summary = read_json_records(base_dir / f"{run_id}_summary.json", filters)
    stability = read_json_records(base_dir / f"{run_id}_stability.json", filters)

    merge_keys = ["imbalance", "selector"]
    if "model" in stability.columns:
        merge_keys.append("model")

    data = summary.merge(
        stability[merge_keys + ["n_features_std"]],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    metrics = [
        "n_features_mean",
        "n_features_std",
        "average_precision_mean",
        "average_precision_std",
    ]
    if data[metrics].isna().any().any():
        raise PlotUnavailable("Tradeoff figure needs finite summary values and matching stability records")
    if data.duplicated(["imbalance", "selector", "model"]).any():
        raise ValueError("Duplicate summary configurations")

    selectors = ordered_present(data["selector"], SELECTOR_STYLES)
    imbalances = ordered_present(data["imbalance"], IMBALANCE_LABELS)
    models = ordered_present(data["model"], MODEL_LABELS)
    check_complete_grid(data, imbalances, selectors, models, "Expected one row per imbalance/selector/model configuration")

    fig, axes = plt.subplots(
        len(imbalances),
        len(models),
        figsize=(max(10, 4.2 * len(models)), max(6, 3.2 * len(imbalances))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, imbalance in enumerate(imbalances):
        for j, model in enumerate(models):
            ax = axes[i, j]
            subset = data[(data["imbalance"].eq(imbalance)) & (data["model"].eq(model))]

            for selector in selectors:
                label, color, marker = SELECTOR_STYLES.get(selector, (selector, "#666666", "o"))
                row = subset[subset["selector"].eq(selector)].iloc[0]
                ax.errorbar(
                    row["n_features_mean"],
                    row["average_precision_mean"],
                    xerr=row["n_features_std"],
                    yerr=row["average_precision_std"],
                    fmt=marker,
                    color=color,
                    markersize=7,
                    capsize=3,
                    elinewidth=1.1,
                    alpha=0.85,
                )

            if i == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{IMBALANCE_LABELS.get(imbalance, imbalance)}\nAverage precision (AP)")
            if i == len(imbalances) - 1:
                ax.set_xlabel("Mean selected feature count")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    xmax = (data["n_features_mean"] + data["n_features_std"]).max()
    ymax = (data["average_precision_mean"] + data["average_precision_std"]).max()
    axes[0, 0].set_xlim(0, xmax * 1.08)
    axes[0, 0].set_ylim(0, ymax * 1.1)

    handles = [
        Line2D([], [], color=selector_style(s)[1], marker=selector_style(s)[2],
               linestyle="none", markersize=7, label=selector_style(s)[0])
        for s in selectors
    ]

    fig.suptitle("Feature reduction and quality anomaly prediction", fontsize=15, y=0.99)
    add_figure_legends(fig, handles)

    out = output_directory(base_dir, run_id, out_dir)
    save_figure(fig, out, f"{run_id}_rq1_tradeoff")

    return fig, axes, data


@plt.rc_context(PLOT_STYLE)
def plot_ap_difference(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    data = read_json_records(base_dir / f"{run_id}_results.json", filters, keep_baseline=True)
    config = read_config(base_dir, run_id)

    keys = ["repeat", "fold", "imbalance", "model"]
    selectors = [s for s in ordered_present(data["selector"], SELECTOR_STYLES) if s != "none"]
    all_selectors = ["none"] + selectors
    imbalances = ordered_present(data["imbalance"], IMBALANCE_LABELS)
    models = ordered_present(data["model"], MODEL_LABELS)

    validate_splits(data, config)
    if not selectors or "none" not in set(data["selector"]):
        raise PlotUnavailable("AP comparison needs all features and at least one selector")
    if data["ap"].isna().any():
        raise PlotUnavailable("AP comparison needs recorded scores for every paired split")
    if not np.isfinite(data["ap"]).all() or not data["ap"].between(0, 1).all():
        raise ValueError("AP must be finite and between zero and one")
    check_complete_grid(data, imbalances, all_selectors, models, "Unexpected or missing configurations")

    baseline = data.loc[data["selector"].eq("none"), keys + ["ap"]].rename(columns={"ap": "baseline_ap"})
    paired = data.loc[data["selector"].isin(selectors)].merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Unmatched baseline splits")

    paired = paired.drop(columns="_merge")
    paired["delta_ap"] = paired["ap"] - paired["baseline_ap"]

    repeat_means = paired.groupby(
        ["repeat", "imbalance", "model", "selector"],
        as_index=False,
    ).agg(delta_ap=("delta_ap", "mean"), n_folds=("fold", "nunique"))

    assert repeat_means["n_folds"].eq(config["n_splits"]).all()

    direct = data.groupby(["imbalance", "model", "selector"])["ap"].mean()
    for (imbalance, model, selector), value in repeat_means.groupby(
        ["imbalance", "model", "selector"]
    )["delta_ap"].mean().items():
        np.testing.assert_allclose(
            value,
            direct[imbalance, model, selector] - direct[imbalance, model, "none"],
            atol=1e-12,
        )

    fig, axes = plt.subplots(
        len(imbalances),
        len(models),
        figsize=(max(10, 4.2 * len(models)), max(6, 3.2 * len(imbalances))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, imbalance in enumerate(imbalances):
        for j, model in enumerate(models):
            ax = axes[i, j]
            ax.axhline(0, color="#444444", linestyle="--", linewidth=1)

            for x, selector in enumerate(selectors):
                _, color, _ = selector_style(selector)
                values = repeat_means.loc[
                    repeat_means["imbalance"].eq(imbalance)
                    & repeat_means["model"].eq(model)
                    & repeat_means["selector"].eq(selector)
                ].sort_values("repeat")["delta_ap"].to_numpy()

                ax.scatter(x + np.linspace(-0.15, 0.15, len(values)), values,
                           s=25, color=color, alpha=0.65, zorder=3)
                ax.scatter(x, values.mean(), marker="D", s=70, color=color,
                           edgecolor="black", linewidth=0.8, zorder=4)

            ax.set_xticks([])
            ax.set_xlim(-0.5, len(selectors) - 0.5)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

            if i == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            if j == 0:
                ax.set_ylabel(f"{IMBALANCE_LABELS.get(imbalance, imbalance)}\nAP change from all features")

    limit = max(repeat_means["delta_ap"].abs().max() * 1.15, 0.01)
    axes[0, 0].set_ylim(-limit, limit)

    selector_handles = [
        Line2D([], [], color=selector_style(s)[1], marker="o",
               linestyle="none", markersize=7, label=selector_style(s)[0])
        for s in selectors
    ]
    shape_handles = [
        Line2D([], [], color="#555555", marker="o", linestyle="none",
               label="Mean paired difference within a repeat"),
        Line2D([], [], color="#555555", marker="D", linestyle="none",
               markeredgecolor="black", label="Mean across repeats"),
    ]

    fig.suptitle("Change in average precision after feature selection", fontsize=15)
    add_figure_legends(fig, selector_handles, shape_handles)

    out = output_directory(base_dir, run_id, out_dir)
    save_figure(fig, out, f"{run_id}_rq1_ap_difference")

    repeat_means.to_csv(out / f"{run_id}_rq1_ap_difference_repeat_means.csv", index=False)
    return fig, axes, repeat_means, paired


def load_imbalance_comparison(base_dir, run_id, table, filters=None):
    """Keep the no-handling reference even when only SMOTE or weighting is shown."""
    reference_filters = dict(filters or {})
    if reference_filters.get("imbalance") is not None:
        reference_filters["imbalance"] = list(set(reference_filters["imbalance"]) | {"none"})
    data = read_json_records(Path(base_dir) / f"{run_id}_{table}.json", reference_filters)
    displayed = filter_records(data, filters)
    conditions = [i for i in ordered_present(displayed.imbalance, IMBALANCE_ORDER) if i != "none"]
    if not conditions or not data.imbalance.eq("none").any():
        raise PlotUnavailable("Imbalance comparison needs no-handling baseline and a displayed treatment")
    return data, conditions


@plt.rc_context(PLOT_STYLE)
def plot_imbalance_ap(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    """Compare the same selector, learner and validation split across conditions."""
    run_id = resolve_run_id(base_dir, run_id)
    data, conditions = load_imbalance_comparison(base_dir, run_id, "results", filters)
    config = read_config(base_dir, run_id)
    validate_splits(data, config)
    data["ap"] = pd.to_numeric(data.ap, errors="raise")
    recorded = data.ap.dropna()
    if not np.isfinite(recorded).all() or not recorded.between(0, 1).all():
        raise ValueError("AP must be between zero and one or missing")
    keys = ["repeat", "fold", "selector", "model"]
    baseline = data.loc[data.imbalance.eq("none"), keys + ["ap"]].rename(columns={"ap": "baseline_ap"})
    paired = data.loc[data.imbalance.isin(conditions)].merge(
        baseline, on=keys, how="left", validate="many_to_one", indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise ValueError("Imbalance AP comparison has unmatched baseline splits")
    paired = paired.drop(columns="_merge").sort_values(["imbalance", "model", "selector", "repeat", "fold"])
    paired["delta_ap"] = paired.ap - paired.baseline_ap
    group_keys = ["imbalance", "model", "selector"]
    repeats = paired.groupby(group_keys + ["repeat"], as_index=False).agg(
        ap=("ap", "mean"), baseline_ap=("baseline_ap", "mean"),
        delta_ap=("delta_ap", "mean"), n_paired_folds=("delta_ap", "count"),
    )
    # Missing scores must not silently turn a full-repeat estimate into a partial mean.
    complete = repeats.n_paired_folds.eq(config["n_splits"])
    repeats.loc[~complete, ["ap", "baseline_ap", "delta_ap"]] = np.nan
    summary = repeats.groupby(group_keys, as_index=False).agg(
        ap_mean=("ap", "mean"), baseline_ap_mean=("baseline_ap", "mean"),
        delta_ap_mean=("delta_ap", "mean"), delta_ap_sd=("delta_ap", "std"),
        n_complete_repeats=("delta_ap", "count"), n_expected_repeats=("repeat", "size"),
    )
    selectors = ordered_present(repeats.selector, SELECTOR_ORDER)
    models = ordered_present(repeats.model, MODEL_ORDER)
    fig, axes = plt.subplots(
        len(conditions), len(models), squeeze=False, sharex=True, sharey=True,
        figsize=(max(9, 4.3 * len(models)), max(5.5, (0.46 * len(selectors) + 1.4) * len(conditions) + 1)),
    )
    for row, condition in enumerate(conditions):
        for col, model in enumerate(models):
            ax = axes[row, col]
            ax.axvline(0, color="#777777", linewidth=1)
            for y, selector in enumerate(selectors):
                values = repeats.loc[
                    repeats.imbalance.eq(condition) & repeats.model.eq(model)
                    & repeats.selector.eq(selector), "delta_ap"
                ].dropna().to_numpy()
                color = selector_style(selector)[1]
                if len(values):
                    ax.scatter(values, y + np.linspace(-0.14, 0.14, len(values)),
                               s=18, alpha=0.4, color=color)
                    ax.scatter(values.mean(), y, marker="D", s=58, color=color,
                               edgecolor="black", linewidth=0.7, zorder=3)
                else:
                    ax.text(0.97, y, "Unavailable", transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=8)
            ax.set_yticks(range(len(selectors)), [textwrap.fill(selector_style(s)[0], 19) for s in selectors])
            ax.set_ylim(len(selectors) - 0.5, -0.5)
            ax.set_title(f"{IMBALANCE_LABELS.get(condition, condition)}\n{MODEL_LABELS.get(model, model)}")
            ax.set_xlabel("Change in average precision")
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    limit = max(float(repeats.delta_ap.abs().max()), 0.01) if repeats.delta_ap.notna().any() else 0.01
    axes[0, 0].set_xlim(-limit * 1.15, limit * 1.15)
    fig.suptitle("Effect of imbalance handling on quality anomaly prediction", fontsize=14)
    handles = [Line2D([], [], marker=m, linestyle="none", color="#555555", label=label)
               for m, label in [("o", "Mean paired change within a repeat"), ("D", "Mean across complete repeats")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    out = output_directory(base_dir, run_id, out_dir)
    stem = f"{run_id}_rq4_imbalance_ap"
    save_figure(fig, out, stem)
    paired.to_csv(out / f"{stem}_paired_folds.csv", index=False)
    repeats.to_csv(out / f"{stem}_repeat_means.csv", index=False)
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    return fig, axes, repeats, paired


@plt.rc_context(PLOT_STYLE)
def plot_imbalance_stability(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    """Show saved stability and subset sizes, without inventing sampling intervals."""
    run_id = resolve_run_id(base_dir, run_id)
    data, conditions = load_imbalance_comparison(base_dir, run_id, "stability", filters)
    data = data.loc[data.selector.ne("none")].copy()
    if data.empty:
        raise PlotUnavailable("Imbalance stability comparison needs a feature selector")
    keys = ["imbalance", "selector"] + (["model"] if "model" in data else [])
    if "n_features_mean" not in data:
        reference_filters = {**(filters or {}), "imbalance": None}
        counts = read_json_records(Path(base_dir) / f"{run_id}_summary.json", reference_filters)
        counts = counts[keys + ["n_features_mean"]].drop_duplicates()
        if counts.duplicated(keys).any():
            raise ValueError("Model-independent stability has inconsistent feature counts across models")
        data = data.merge(counts, on=keys, how="left", validate="one_to_one")
    metrics = ["nogueira", "n_features_mean"]
    data = validate_frame(data, keys, metrics, "stability.json")
    if np.isinf(data[metrics].to_numpy()).any() or data.n_features_mean.dropna().lt(0).any():
        raise ValueError("Stability comparison needs finite or missing scores and nonnegative feature counts")
    pair_keys = [key for key in keys if key != "imbalance"]
    baseline = data.loc[data.imbalance.eq("none"), pair_keys + metrics].rename(
        columns={metric: f"baseline_{metric}" for metric in metrics})
    paired = data.loc[data.imbalance.isin(conditions)].merge(
        baseline, on=pair_keys, how="left", validate="many_to_one",
    ).sort_values(keys)
    if paired.empty:
        raise PlotUnavailable("No selected stability comparisons are available")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"baseline_{metric}"]
    if "model" not in paired:
        paired["model"] = "all_models"
    selectors = ordered_present(paired.selector, SELECTOR_ORDER)
    models = ordered_present(paired.model, MODEL_ORDER)
    panels = [(condition, model) for condition in conditions for model in models]
    fig, axes = plt.subplots(
        len(panels), 2, squeeze=False, sharey=True, sharex="col",
        figsize=(12, max(5.5, (0.5 * len(selectors) + 1.3) * len(panels) + 1)),
    )
    for row, (condition, model) in enumerate(panels):
        subset = paired.loc[paired.imbalance.eq(condition) & paired.model.eq(model)].set_index("selector")
        for col, (metric, label) in enumerate(zip(metrics, ["Nogueira stability", "Mean selected feature count"])):
            ax = axes[row, col]
            for y, selector in enumerate(selectors):
                if selector not in subset.index:
                    ax.text(0.97, y, "Unavailable", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
                    continue
                before = subset.loc[selector, f"baseline_{metric}"]
                after = subset.loc[selector, metric]
                color = selector_style(selector)[1]
                if np.isfinite(before) and np.isfinite(after):
                    ax.plot([before, after], [y, y], color=color, alpha=0.6, linewidth=2)
                if np.isfinite(before):
                    ax.scatter(before, y, marker="o", s=70, facecolor="white", edgecolor=color, linewidth=1.5, zorder=3)
                if np.isfinite(after):
                    ax.scatter(after, y, marker="D", s=35, color=color, zorder=4)
                if not (np.isfinite(before) and np.isfinite(after)):
                    ax.text(0.97, y, "Incomplete", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
            title = IMBALANCE_LABELS.get(condition, condition)
            if model != "all_models":
                title += f"\n{MODEL_LABELS.get(model, model)}"
            ax.set_title(title)
            ax.set_xlabel(label)
            ax.set_yticks(range(len(selectors)), [textwrap.fill(selector_style(s)[0], 19) for s in selectors])
            ax.set_ylim(len(selectors) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    finite = paired[["nogueira", "baseline_nogueira"]].to_numpy()
    finite = finite[np.isfinite(finite)]
    axes[0, 0].set_xlim(min(0, finite.min()) - 0.04 if finite.size else -0.04,
                        max(1, finite.max()) + 0.04 if finite.size else 1.04)
    axes[0, 1].set_xlim(left=-0.5)
    fig.suptitle("Effect of imbalance handling on selection stability and feature count", fontsize=14)
    handles = [
        Line2D([], [], marker="o", markerfacecolor="white", color="#555555", linestyle="none",
               label="No resampling and no class weighting"),
        Line2D([], [], marker="D", color="#555555", linestyle="none", label="Condition shown in the panel"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    out = output_directory(base_dir, run_id, out_dir)
    stem = f"{run_id}_rq4_imbalance_stability"
    save_figure(fig, out, stem)
    paired.to_csv(out / f"{stem}.csv", index=False)
    return fig, axes, paired


def load_threshold_results(base_dir, run_id, filters=None):
    """Normalize new threshold tables and legacy single-threshold results."""
    config = read_config(base_dir, run_id)
    path = Path(base_dir) / f"{run_id}_thresholds.json"
    if path.exists():
        data = read_json_records(path, filters).copy()
    else:
        data = read_json_records(Path(base_dir) / f"{run_id}_results.json", filters).copy()
        if "recall" not in data or "decision_threshold" not in config:
            raise PlotUnavailable("No saved threshold scores; generate the thresholds JSON with 04_exp_1.py")
        data["threshold"] = config["decision_threshold"]
    metrics = ["recall", "precision", "tp", "fp", "fn", "tn"]
    required = {"repeat", "fold", "imbalance", "selector", "model", "threshold", *metrics}
    if required - set(data):
        raise ValueError(f"Threshold table is missing columns: {sorted(required - set(data))}")
    if data.empty:
        raise PlotUnavailable("The threshold table is empty")
    for column in ["threshold", *metrics]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data.threshold).all() or not data.threshold.between(0, 1).all():
        raise ValueError("Thresholds must be finite values between zero and one")
    available = sorted(data.threshold.unique())
    if "threshold_grid" in config:
        expected = sorted(config["threshold_grid"])
        if len(expected) != len(available) or not np.allclose(expected, available, rtol=0, atol=1e-12):
            raise ValueError("Saved thresholds do not match the configured threshold grid")
    reference = None
    keys = ["repeat", "fold", "imbalance", "selector", "model"]
    for _, group in data.groupby("threshold"):
        validate_splits(group, config)
        identifiers = set(group[keys].itertuples(index=False, name=None))
        if reference is not None and identifiers != reference:
            raise ValueError("Thresholds must contain identical configurations and splits")
        reference = identifiers
    missing = data[metrics].isna()
    if (missing.any(axis=1) & ~missing.all(axis=1)).any():
        raise ValueError("Threshold rows must have all scores present or all scores missing")
    valid = ~missing.any(axis=1)
    recorded = data.loc[valid]
    if not np.isfinite(recorded[metrics]).all().all():
        raise ValueError("Threshold scores must be finite")
    for metric in ["recall", "precision"]:
        if not recorded[metric].between(0, 1).all():
            raise ValueError(f"Invalid {metric}")
    counts = recorded[["tp", "fp", "fn", "tn"]]
    if (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("Confusion counts must be nonnegative integers")
    for metric, denominator in [("recall", recorded.tp + recorded.fn),
                                ("precision", recorded.tp + recorded.fp)]:
        calculated = np.divide(recorded.tp.to_numpy(dtype=float), denominator.to_numpy(dtype=float),
                               out=np.zeros(len(recorded)), where=denominator.to_numpy() != 0)
        if not np.allclose(recorded[metric], calculated, rtol=0, atol=1e-10):
            raise ValueError(f"Recorded {metric} does not match confusion counts")
    data["valid_score"] = valid
    data["no_predicted_failures"] = valid & (data.tp + data.fp).eq(0)
    return data, config


def threshold_repeat_means(data):
    """Exclude incomplete repeats, preserving their coverage in the export."""
    means = data.groupby(["threshold", "repeat", "imbalance", "selector", "model"],
                         as_index=False).agg(
        recall=("recall", "mean"), precision=("precision", "mean"),
        n_folds=("fold", "nunique"), n_valid_folds=("valid_score", "sum"),
        folds_without_predicted_failures=("no_predicted_failures", "sum"))
    means["complete_repeat"] = means.n_valid_folds.eq(means.n_folds)
    means.loc[~means.complete_repeat, ["recall", "precision"]] = np.nan
    return means


def summarize_threshold_scores(data, config):
    """Use identical complete evaluation coverage for threshold comparisons."""
    repeats = threshold_repeat_means(data)
    summary = repeats.groupby(["imbalance", "selector", "model", "threshold"], as_index=False).agg(
        recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
        n_complete_repeats=("complete_repeat", "sum"),
        n_valid_folds=("n_valid_folds", "sum"), n_expected_folds=("n_folds", "sum"),
        folds_without_predicted_failures=("folds_without_predicted_failures", "sum"))
    # Do not compare a threshold's easy/successful repeats against a different set.
    incomplete = summary.n_complete_repeats.ne(config["n_repeats"])
    summary.loc[incomplete, ["recall_mean", "precision_mean"]] = np.nan
    return repeats, summary




@plt.rc_context(PLOT_STYLE)
def plot_threshold_sweep(data, config, out_dir, run_id, show=False):
    """Compare both metrics at saved operating points, not an interpolated PR curve."""
    repeats, summary = summarize_threshold_scores(data, config)
    repeats.to_csv(Path(out_dir) / f"{run_id}_threshold_repeat_means.csv", index=False)
    summary.to_csv(Path(out_dir) / f"{run_id}_threshold_comparison.csv", index=False)
    for (imbalance, model), group in summary.groupby(["imbalance", "model"], sort=True):
        selectors = ordered_present(group.selector, SELECTOR_STYLES)
        columns = min(3, len(selectors))
        rows = int(np.ceil(len(selectors) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows + 1.4),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, selector in zip(axes.flat, selectors):
            values = group.loc[group.selector.eq(selector)].sort_values("threshold")
            for metric, label, color, marker in [("recall", "Recall", "#0072B2", "o"),
                                                  ("precision", "Precision", "#D55E00", "s")]:
                ax.plot(values.threshold, values[f"{metric}_mean"], color=color,
                        marker=marker, markersize=4, linewidth=1.5, label=label)
            if values.recall_mean.isna().any():
                ax.text(0.5, 0.5, "Incomplete evaluation\nSee coverage in the CSV table",
                        transform=ax.transAxes, ha="center", fontsize=9)
            ax.set_title(SELECTOR_LABELS.get(selector, selector))
            ax.set_xlabel("Decision threshold")
            ax.set_ylabel("Mean score across repeats")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.025, 1.025)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(selectors):]:
            ax.set_visible(False)
        fig.suptitle("Recall and precision at different decision thresholds\n"
                     f"{IMBALANCE_LABELS.get(imbalance, imbalance)} | {MODEL_LABELS.get(model, model)}",
                     fontsize=14)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045),
                   ncol=2, frameon=False)
        fig.text(0.5, 0.02, "All repeats required for each point. Precision is zero when no failures are predicted.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.85 / fig.get_size_inches()[1], 1, 1))
        save_figure(fig, out_dir, f"{run_id}_threshold_comparison_{imbalance}_{model}")
        finish_figure(fig, show)
    return summary


@plt.rc_context(PLOT_STYLE)
def plot_rq2a_nogueira_stability(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    """Plot saved stability metrics. Supports MI, FDR, L1, XGB, Boruta, and Voting."""
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)

    df = read_json_records(base_dir / f'{run_id}_stability.json', filters)

    required = {'imbalance', 'selector', *STABILITY_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in stability.json: {sorted(missing)}')

    df = df.copy()
    if 'model' not in df.columns:
        df['model'] = 'all_models'

    keys = ['model', 'imbalance', 'selector']
    if df[keys].isna().any().any():
        raise ValueError('Missing model/imbalance/selector identifiers')

    df[keys] = df[keys].astype(str)
    df = df[df['selector'].ne('none')].copy()

    if df.empty:
        raise PlotUnavailable('No feature selector rows to plot')
    if df[keys].duplicated().any():
        raise ValueError('Duplicate model/imbalance/selector rows in stability.json')

    for metric in STABILITY_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='raise')

    values = df[list(STABILITY_METRICS)].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise PlotUnavailable('No finite stability values to plot')

    selectors = ordered_present(df['selector'], SELECTOR_ORDER)
    imbalances = ordered_present(df['imbalance'], IMBALANCE_ORDER)
    models = ordered_present(df['model'], MODEL_ORDER)

    fig, axes = plt.subplots(
        len(models),
        len(STABILITY_METRICS),
        figsize=(13.5, max(3.8, 0.5 * len(selectors) + 1.8) * len(models)),
        squeeze=False,
        layout='constrained',
    )

    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad('#e5e7eb')

    norm = plt.Normalize(
        vmin=min(0.0, float(finite.min())),
        vmax=max(1.0, float(finite.max())),
    )

    for col, (metric, title) in enumerate(STABILITY_METRICS.items()):
        for row, model in enumerate(models):
            ax = axes[row, col]

            matrix = (
                df[df['model'].eq(model)]
                .pivot(index='selector', columns='imbalance', values=metric)
                .reindex(index=selectors, columns=imbalances)
            )
            arr = matrix.to_numpy(dtype=float)

            im = ax.imshow(
                np.ma.masked_invalid(arr),
                cmap=cmap,
                norm=norm,
                aspect='auto',
            )

            ax.set_title(title if model == 'all_models' else f'{title}\n{MODEL_LABELS.get(model, model)}')
            ax.set_xticks(np.arange(len(imbalances)),
                          [textwrap.fill(IMBALANCE_LABELS.get(i, i), 12) for i in imbalances])
            ax.set_yticks(
                np.arange(len(selectors)),
                [textwrap.fill(SELECTOR_LABELS.get(s, s), 18) for s in selectors],
            )
            ax.set_xlabel('Imbalance method')

            if col == 0:
                ax.set_ylabel('Feature selector')

            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    value = arr[r, c]
                    valid = np.isfinite(value)
                    text = f'{value:.2f}' if valid else 'NA'
                    color = 'white' if valid and im.norm(value) > 0.55 else 'black'
                    ax.text(c, r, text, ha='center', va='center', fontsize=11, color=color)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label('Stability score (shared scale)')

    fig.suptitle('Feature selection stability across repeated resampling', fontsize=13)
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))
    fig.text(
        0.985,
        0.015,
        'NA indicates an unavailable score\nEach stability measure has its own definition',
        ha='right',
        va='bottom',
        fontsize=8,
        color='#555555',
    )

    save_figure(fig, out_dir, f'{run_id}_rq2a_stability_metrics')

    df[keys + list(STABILITY_METRICS)].to_csv(
        out_dir / f'{run_id}_rq2a_stability_metrics.csv',
        index=False,
    )

    return fig, axes, df


@plt.rc_context(PLOT_STYLE)
def plot_rq2b_top_feature_frequency(base_dir=BASE_DIR, run_id=RUN_ID, top_n=10, out_dir=None,
                                    filters=None):
    """Plot top selected features per imbalance/selector configuration. Supports FDR."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError('top_n must be a positive integer')

    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)

    df = read_json_records(base_dir / f'{run_id}_feature_frequency.json', filters)

    keys = ['imbalance', 'selector', 'feature']
    required = keys + ['selection_frequency', 'selected_count', 'n_runs']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    if 'model' in df.columns:
        raise ValueError('This plot expects model-independent selection records; do not pool model-specific rows')

    df = df.loc[df.selector.ne('none')].copy()
    if df.empty:
        raise PlotUnavailable('No feature selector frequency records to plot')

    if df[required].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Missing values or duplicate feature/configuration records')
    if not np.isfinite(df[['selection_frequency', 'selected_count', 'n_runs']]).all().all():
        raise ValueError('Nonfinite values')
    if not df.selection_frequency.between(0, 1).all() or not df.n_runs.gt(0).all():
        raise ValueError('Invalid frequency or run count')

    np.testing.assert_allclose(
        df.selection_frequency,
        df.selected_count / df.n_runs,
        atol=1e-12,
    )

    selectors = ordered_present(df['selector'], SELECTOR_ORDER)
    settings = ordered_present(df['imbalance'], IMBALANCE_ORDER)

    expected = {(i, s) for i in settings for s in selectors}
    actual = set(df[['imbalance', 'selector']].itertuples(index=False, name=None))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Expected all configured imbalance/selector combinations. Missing={missing}, extra={extra}')

    if not df.feature.astype(str).str.fullmatch(r'f\d+').all():
        raise ValueError('Expected feature identifiers of form f<number>')

    df['_feature_number'] = df.feature.str[1:].astype(int)

    ranked = df[df.selection_frequency.gt(0)].sort_values(
        ['imbalance', 'selector', 'selection_frequency', '_feature_number'],
        ascending=[True, True, False, True],
    )

    top = ranked.groupby(['imbalance', 'selector'], sort=False).head(top_n).copy()
    top['rank'] = top.groupby(['imbalance', 'selector']).cumcount() + 1

    top['cutoff_tie_total'] = 0
    for (imbalance, selector), part in top.groupby(['imbalance', 'selector']):
        source = ranked[ranked.imbalance.eq(imbalance) & ranked.selector.eq(selector)]
        assert len(part) == min(top_n, len(source))
        cutoff = part.selection_frequency.min()
        top.loc[part.index, 'cutoff_tie_total'] = int(source.selection_frequency.eq(cutoff).sum())

    fig, axes = plt.subplots(
        len(settings),
        len(selectors),
        figsize=(3.4 * len(selectors), 4.0 * len(settings)),
        sharex=True,
        squeeze=False,
        layout='constrained',
    )

    for i, setting in enumerate(settings):
        for j, selector in enumerate(selectors):
            ax = axes[i, j]

            part = (
                top[top.imbalance.eq(setting) & top.selector.eq(selector)]
                .sort_values('rank')
            )

            color = SELECTOR_COLORS.get(selector, '#666666')
            bars = ax.barh(part.feature, part.selection_frequency, color=color, height=0.7)
            ax.invert_yaxis()

            if part.empty:
                ax.text(
                    0.5,
                    0.5,
                    'No selected features',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=9,
                )

            for bar, value in zip(bars, part.selection_frequency):
                inside = value >= 0.86
                ax.text(
                    value - 0.02 if inside else value + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:.0%}',
                    ha='right' if inside else 'left',
                    va='center',
                    fontsize=9,
                    color='white' if inside else '#222222',
                )

            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.grid(axis='x', alpha=0.18)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)

            ax.set_title(f'{SELECTOR_LABELS.get(selector, selector)}\n{IMBALANCE_LABELS.get(setting, setting)}', fontsize=11)

            if j == 0:
                ax.set_ylabel('Feature')
            if i == len(settings) - 1:
                ax.set_xlabel('Selection frequency')

    fig.suptitle(f'Top {top_n} most frequently selected features per configuration', fontsize=16)
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))

    save_figure(fig, out_dir, f'{run_id}_rq2b_top{top_n}_feature_bars')

    top = top.drop(columns='_feature_number')
    top.to_csv(out_dir / f'{run_id}_rq2b_top{top_n}_feature_bars.csv', index=False)

    return fig, axes, top

def read_json_records(path, filters=None, keep_baseline=False):
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{Path(path).name}: expected a nonempty list of records"
        )

    data = filter_records(pd.DataFrame(records), filters, keep_baseline)
    if data.empty:
        raise PlotUnavailable(f"{Path(path).name}: no records match the display selection")
    return data


def validate_frame(df, keys, metrics, filename):
    missing = set(keys + metrics) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename}: missing columns {sorted(missing)}"
        )

    if df[keys].isna().any().any():
        raise ValueError(
            f"{filename}: missing configuration identifiers"
        )

    df = df.copy()
    df[keys] = df[keys].astype(str)

    if df.duplicated(keys).any():
        raise ValueError(
            f"{filename}: duplicate configuration rows"
        )

    for metric in metrics:
        df[metric] = pd.to_numeric(
            df[metric], errors="raise"
        ).astype(float)

    return df


@plt.rc_context(PLOT_STYLE)
def plot_rq3_main_tradeoff(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None, filters=None):
    """
    Plot aligned heatmaps using saved JSON metrics.

    Panels:
        1. Mean average precision for each model.
        2. Mean selected feature count.
        3. Nogueira stability.

    Missing/nonfinite values appear as NA.
    No metrics are recalculated or averaged across models.

    Returns:
        fig, axes, data
    """
    base_dir = Path(base_dir)

    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)

    score_col = "average_precision_mean"
    count_col = "n_features_mean"

    keys = ["selector", "imbalance"]
    model_keys = keys + ["model"]

    # ---------------------------------------------------------
    # Read and validate saved results
    # ---------------------------------------------------------
    summary = read_json_records(
        base_dir / f"{run_id}_summary.json", filters
    )
    stability = read_json_records(
        base_dir / f"{run_id}_stability.json", filters
    )

    stability_keys = (
        model_keys if "model" in stability.columns else keys
    )

    summary = validate_frame(
        summary,
        model_keys,
        [score_col, count_col],
        "summary.json",
    )

    stability = validate_frame(
        stability,
        stability_keys,
        ["nogueira"],
        "stability.json",
    )

    data = summary.merge(
        stability[stability_keys + ["nogueira"]],
        on=stability_keys,
        how="left",
        validate=(
            "one_to_one"
            if "model" in stability_keys
            else "many_to_one"
        ),
    )

    # ---------------------------------------------------------
    # Read category levels dynamically
    # ---------------------------------------------------------
    selectors = ordered_present(
        set(summary["selector"])
        | set(stability["selector"]),
        SELECTOR_ORDER,
    )

    imbalances = ordered_present(
        set(summary["imbalance"])
        | set(stability["imbalance"]),
        IMBALANCE_ORDER,
    )

    model_values = set(summary["model"])

    if "model" in stability.columns:
        model_values |= set(stability["model"])

    models = ordered_present(model_values, MODEL_ORDER)

    rows = pd.MultiIndex.from_product(
        [selectors, imbalances],
        names=keys,
    )

    # ---------------------------------------------------------
    # Build matrices without aggregating saved values
    # ---------------------------------------------------------
    ap_matrix = (
        summary.pivot(
            index=keys,
            columns="model",
            values=score_col,
        )
        .reindex(index=rows, columns=models)
    )

    count_matrix = (
        summary.pivot(
            index=keys,
            columns="model",
            values=count_col,
        )
        .reindex(index=rows, columns=models)
    )

    # Show one shared count column only when all model columns
    # contain exactly the same saved values, including missingness.
    shared_count = all(
        count_matrix[model].equals(count_matrix.iloc[:, 0])
        for model in models
    )

    if shared_count:
        count_matrix = count_matrix.iloc[:, [0]].copy()
        count_matrix.columns = ["Shared"]

    if "model" in stability.columns:
        stability_matrix = (
            stability.pivot(
                index=keys,
                columns="model",
                values="nogueira",
            )
            .reindex(index=rows, columns=models)
        )
    else:
        stability_matrix = (
            stability.set_index(keys)[["nogueira"]]
            .reindex(rows)
        )
        stability_matrix.columns = ["Shared"]

    matrices = [
        ap_matrix,
        count_matrix,
        stability_matrix,
    ]

    arrays = [
        matrix.to_numpy(dtype=float)
        for matrix in matrices
    ]

    finite = [
        values[np.isfinite(values)]
        for values in arrays
    ]

    # ---------------------------------------------------------
    # Colour scales: display limits only, not new metrics
    # ---------------------------------------------------------
    ap_min = (
        np.floor(finite[0].min() / 0.01) * 0.01
        if finite[0].size else 0.0
    )
    ap_max = (
        np.ceil(finite[0].max() / 0.01) * 0.01
        if finite[0].size else 1.0
    )

    if ap_min == ap_max:
        ap_min = max(0.0, ap_min - 0.01)
        ap_max = min(1.0, ap_max + 0.01)

    count_max = (
        max(1.0, float(finite[1].max()))
        if finite[1].size else 1.0
    )

    stability_min = (
        min(0.0, float(finite[2].min()))
        if finite[2].size else 0.0
    )
    stability_max = (
        max(1.0, float(finite[2].max()))
        if finite[2].size else 1.0
    )

    norms = [
        plt.Normalize(ap_min, ap_max),
        plt.Normalize(0.0, count_max),
        plt.Normalize(stability_min, stability_max),
    ]

    cmaps = [
        "YlGnBu",
        "YlOrBr_r",
        "RdPu",
    ]

    title_colors = [
        "#135d88",
        "#a64b08",
        "#a80f68",
    ]

    titles = [
        "Average precision",
        "Mean selected\nfeatures",
        "Nogueira\nstability",
    ]

    formats = [".3f", ".1f", ".3f"]

    labels = [
        f"{SELECTOR_LABELS.get(selector, selector)}\n"
        f"{IMBALANCE_LABELS.get(imbalance, imbalance)}"
        for selector, imbalance in rows
    ]

    # ---------------------------------------------------------
    # Figure dimensions grow with the number of rows/models
    # ---------------------------------------------------------
    widths = [
        len(matrix.columns)
        for matrix in matrices
    ]

    fig_width = max(8.0, 3.0 + sum(widths))

    label_lines = max(
        label.count("\n") + 1
        for label in labels
    )

    fig_height = max(
        4.0,
        1.3 + len(rows) * (
            0.245 + 0.095 * (label_lines - 1)
        ),
    )

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "text.color": "#243247",
        "xtick.color": "#243247",
        "ytick.color": "#243247",
    }

    with plt.rc_context(style):
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            facecolor="white",
        )

        fig.suptitle(
            textwrap.fill(FIGURE_TITLE, width=60),
            x=0.02,
            y=0.985,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

        # Dedicated bottom row for colour bars.
        grid = fig.add_gridspec(
            2,
            3,
            left=2.0 / fig_width,
            right=1.0 - 0.24 / fig_width,
            top=1.0 - 1.25 / fig_height,
            bottom=0.46 / fig_height,
            height_ratios=[20, 0.65],
            width_ratios=widths,
            wspace=0.25,
            hspace=0.18,
        )

        axes = np.empty(3, dtype=object)

        for j, (matrix, arr) in enumerate(
            zip(matrices, arrays)
        ):
            ax = axes[j] = fig.add_subplot(grid[0, j])

            cmap = plt.get_cmap(cmaps[j]).copy()
            cmap.set_bad("#e5e7eb")

            im = ax.imshow(
                np.ma.masked_invalid(arr),
                cmap=cmap,
                norm=norms[j],
                aspect="auto",
            )

            ax.set_title(
                titles[j],
                fontsize=10,
                fontweight="bold",
                pad=25,
                color=title_colors[j],
            )

            column_labels = [
                textwrap.fill(
                    MODEL_LABELS.get(model, model),
                    width=15,
                )
                for model in matrix.columns
            ]

            ax.set_xticks(
                np.arange(arr.shape[1]),
                column_labels,
            )
            ax.tick_params(
                axis="x",
                top=True,
                labeltop=True,
                bottom=False,
                labelbottom=False,
                length=0,
                pad=6,
            )

            ax.set_yticks(
                np.arange(len(rows)),
                labels,
            )
            ax.tick_params(
                axis="y",
                labelleft=(j == 0),
                length=0,
                pad=7,
            )
            ax.set_ylim(len(rows) - 0.5, -0.5)

            # Separate feature-selector groups.
            for r in range(1, len(rows)):
                if rows[r][0] != rows[r - 1][0]:
                    ax.axhline(
                        r - 0.5,
                        color="#c7ced6",
                        linewidth=0.7,
                        zorder=2,
                    )

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xticks(
                np.arange(-0.5, arr.shape[1], 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, arr.shape[0], 1),
                minor=True,
            )
            ax.grid(
                which="minor",
                color="white",
                linewidth=1.3,
            )
            ax.tick_params(
                which="minor",
                length=0,
            )

            # Annotate each cell; choose readable black/white text.
            for (r, c), value in np.ndenumerate(arr):
                valid = np.isfinite(value)

                rgba = (
                    cmap(norms[j](value))
                    if valid
                    else (
                        229 / 255,
                        231 / 255,
                        235 / 255,
                        1,
                    )
                )

                rgb = np.asarray(rgba[:3])

                linear = np.where(
                    rgb <= 0.04045,
                    rgb / 12.92,
                    ((rgb + 0.055) / 1.055) ** 2.4,
                )

                luminance = np.dot(
                    linear,
                    [0.2126, 0.7152, 0.0722],
                )

                black_contrast = (
                    luminance + 0.05
                ) / 0.05

                white_contrast = 1.05 / (
                    luminance + 0.05
                )

                text_color = (
                    "black"
                    if black_contrast >= white_contrast
                    else "white"
                )

                ax.text(
                    c,
                    r,
                    (
                        format(value, formats[j])
                        if valid else "NA"
                    ),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                )

            # Colour bar outside the heatmap.
            cax = fig.add_subplot(grid[1, j])

            cbar = fig.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
            )

            cbar.locator = FixedLocator(
                np.linspace(
                    norms[j].vmin,
                    norms[j].vmax,
                    3,
                )
            )

            cbar.formatter = FormatStrFormatter(
                "%.3f" if j == 0 else "%.1f"
            )

            cbar.update_ticks()
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(
                labelsize=7,
                length=2,
            )

        # No RQ3 subtitle or small explanatory footer.
        stem = f"{run_id}_rq3_main_heatmap"

        save_figure(fig, out_dir, stem)

    # Export saved values behind the figure.
    data.to_csv(
        out_dir / f"{stem}.csv",
        index=False,
    )

    for name, matrix in zip(
        ["ap", "feature_count", "nogueira"],
        matrices,
    ):
        matrix.to_csv(
            out_dir / f"{stem}_{name}.csv"
        )

    return fig, axes, data


def _q1(s):
    return s.quantile(0.25)


def _q3(s):
    return s.quantile(0.75)


def load_timing(base_dir, run_id, filters=None):
    """One row per fold x imbalance x selector x model, with selection, fit and predict time."""
    base_dir = Path(base_dir)
    results = read_json_records(base_dir / f"{run_id}_results.json", filters)
    selections = read_json_records(base_dir / f"{run_id}_selections.json", filters)

    sel_keys = ["fold", "repeat", "imbalance", "selector"]
    selections = selections[sel_keys + ["n_features_after_preprocessing", "selection_time(s)"]]
    if selections.duplicated(sel_keys).any():
        raise ValueError("Duplicate fold/imbalance/selector rows in selections.json")

    df = results.merge(selections, on=sel_keys, how="left", validate="many_to_one")
    if df["selection_time(s)"].isna().any():
        raise ValueError("Some results rows have no matching selection_time in selections.json")

    df = df.rename(columns={
        "selection_time(s)": "selection_s",
        "fit_time(s)": "fit_s",
        "predict_time_per_observation(ms)": "predict_ms_per_obs",
    })
    df["train_total_s"] = df["selection_s"] + df["fit_s"]

    # Unavailable timings must not be treated as measured zeros.
    n_empty = int(df["fit_s"].isna().sum())
    if n_empty:
        print(f"Timing data: {n_empty} rows have no recorded model training time")
    return df.dropna(subset=["fit_s", "predict_ms_per_obs"]).reset_index(drop=True)


def timing_summary(df, out_dir, run_id):
    """Median and IQR of every time per configuration, slowest (selection + training) first."""
    out_dir = output_directory(BASE_DIR, run_id, out_dir)
    summary = (
        df.groupby(TIME_KEYS)
        .agg(
            n_folds=("fit_s", "size"),
            n_features_median=("n_features", "median"),
            selection_s_median=("selection_s", "median"),
            fit_s_median=("fit_s", "median"),
            fit_s_q1=("fit_s", _q1),
            fit_s_q3=("fit_s", _q3),
            train_total_s_median=("train_total_s", "median"),
            predict_ms_per_obs_median=("predict_ms_per_obs", "median"),
            predict_ms_per_obs_q1=("predict_ms_per_obs", _q1),
            predict_ms_per_obs_q3=("predict_ms_per_obs", _q3),
        )
        .reset_index()
        .sort_values("train_total_s_median", ascending=False, ignore_index=True)
    )
    path = out_dir / f"{run_id}_timing_summary.csv"
    summary.to_csv(path, index=False)
    print(f"[timing_summary] saved {path}")
    return summary


def time_feature_correlation(df, out_dir, run_id):
    """Spearman rho between number of features and time, per model x imbalance.

    Descriptive only: repeated folds share training data, so they are not
    independent observations and no p-value is reported.
    """
    out_dir = output_directory(BASE_DIR, run_id, out_dir)
    rows = []
    for (model, imbalance), g in df.groupby(["model", "imbalance"]):
        rows.append({
            "model": model,
            "imbalance": imbalance,
            "n_rows": len(g),
            "n_features_min": int(g["n_features"].min()),
            "n_features_max": int(g["n_features"].max()),
            "rho_fit_time": g["n_features"].corr(g["fit_s"], method="spearman"),
            "rho_predict_time": g["n_features"].corr(g["predict_ms_per_obs"], method="spearman"),
        })
    corr = pd.DataFrame(rows)
    corr["model"] = pd.Categorical(corr["model"], ordered_present(corr["model"], MODEL_ORDER), ordered=True)
    corr["imbalance"] = pd.Categorical(corr["imbalance"], ordered_present(corr["imbalance"], IMBALANCE_ORDER), ordered=True)
    corr = corr.sort_values(["model", "imbalance"], ignore_index=True)

    path = out_dir / f"{run_id}_time_feature_spearman.csv"
    corr.to_csv(path, index=False)
    print(f"[time_feature_correlation] saved {path}")
    return corr


@plt.rc_context(PLOT_STYLE)
def plot_time_vs_features(df, out_dir, run_id, metric="fit_s", log_y=True):
    """Number of features (x) vs time (y), one panel per model.

    Small faint points = individual folds; large points = configuration median.
    Colour = selector, marker = imbalance handling.
    """
    if metric not in ("fit_s", "predict_ms_per_obs", "train_total_s"):
        raise ValueError(f"metric must be fit_s, predict_ms_per_obs or train_total_s, got {metric!r}")
    out_dir = output_directory(BASE_DIR, run_id, out_dir)

    models = ordered_present(df["model"], MODEL_ORDER)
    selectors = ordered_present(df["selector"], SELECTOR_ORDER)
    imbalances = ordered_present(df["imbalance"], IMBALANCE_ORDER)
    medians = df.groupby(TIME_KEYS)[["n_features", metric]].median().reset_index()

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6),
                             squeeze=False, layout="constrained")
    for ax, model in zip(axes[0], models):
        raw = df[df["model"] == model]
        med = medians[medians["model"] == model]
        for selector in selectors:
            color = SELECTOR_COLORS.get(selector, "#777777")
            for imbalance in imbalances:
                marker = IMBALANCE_MARKERS.get(imbalance, "o")
                r = raw[(raw["selector"] == selector) & (raw["imbalance"] == imbalance)]
                m = med[(med["selector"] == selector) & (med["imbalance"] == imbalance)]
                ax.scatter(r["n_features"], r[metric], s=6, alpha=0.12, color=color,
                           marker=marker, linewidths=0, rasterized=True)
                ax.scatter(m["n_features"], m[metric], s=70, color=color, marker=marker,
                           edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(MODEL_LABELS.get(model, model), loc="left", fontweight="bold")
        ax.set_xlabel("Number of selected features")
        if log_y and (raw[metric] > 0).any():
            ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel(TIME_METRICS[metric] + (" (log scale)" if log_y else ""))

    selector_handles = [Line2D([], [], color=SELECTOR_COLORS.get(s, "#777777"), marker="o",
                               linestyle="none", markersize=7, label=SELECTOR_LABELS.get(s, s))
                        for s in selectors]
    imbalance_handles = [Line2D([], [], color="#555555", marker=IMBALANCE_MARKERS.get(i, "o"),
                                linestyle="none", markersize=7, label=IMBALANCE_LABELS.get(i, i))
                         for i in imbalances]
    fig.legend(handles=selector_handles + imbalance_handles, loc="outside right center", frameon=False)
    fig.suptitle(f"{TIME_METRICS[metric]} by selected feature count",
                 x=0.01, ha="left", fontweight="bold")

    save_figure(fig, out_dir, f"{run_id}_time_vs_features_{metric}")
    return fig, axes


@plt.rc_context(PLOT_STYLE)
def plot_slowest_configs(df, out_dir, run_id, top_n=15):
    """Rank configurations: (a) selection + training time, (b) prediction time per observation."""
    out_dir = output_directory(BASE_DIR, run_id, out_dir)
    med = df.groupby(TIME_KEYS)[["selection_s", "fit_s", "train_total_s", "predict_ms_per_obs",
                                 "n_features"]].median().reset_index()
    med["label"] = (
        med["selector"].map(lambda s: SELECTOR_LABELS.get(s, s)) + "\n"
        + med["imbalance"].map(lambda i: IMBALANCE_LABELS.get(i, i)) + ", "
        + med["model"].map(lambda m: MODEL_LABELS.get(m, m))
        + "\n" + med["n_features"].round().astype(int).astype(str) + " features"
    )
    top_n = min(top_n, len(med))

    train = med.nlargest(top_n, "train_total_s").iloc[::-1]
    pred = med.nlargest(top_n, "predict_ms_per_obs").iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 0.7 * top_n + 1.8), layout="constrained")

    y = np.arange(len(train))
    ax1.barh(y, train["selection_s"], color="#0072B2", label="Feature selection")
    ax1.barh(y, train["fit_s"], left=train["selection_s"], color="#E69F00", label="Model training")
    for yi, total in zip(y, train["train_total_s"]):
        ax1.text(total, yi, f" {total:.1f}s", va="center", fontsize=8)
    ax1.set_yticks(y, train["label"], fontsize=8)
    ax1.set_xlabel("Median time per fold (s)")
    ax1.set_title(f"Slowest {top_n} configurations for selection and training", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower right")

    y = np.arange(len(pred))
    ax2.barh(y, pred["predict_ms_per_obs"], color="#009E73")
    for yi, v in zip(y, pred["predict_ms_per_obs"]):
        ax2.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    ax2.set_yticks(y, pred["label"], fontsize=8)
    ax2.set_xlabel("Median prediction time per observation (ms)")
    ax2.set_title(f"Slowest {top_n} configurations for prediction", loc="left", fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    save_figure(fig, out_dir, f"{run_id}_slowest_configs")
    return fig, (ax1, ax2)


def run_timing_analysis(base_dir, run_id, out_dir, top_n=15, show=False, filters=None):
    """Run every timing table and figure; everything is written to out_dir."""
    df = load_timing(base_dir, run_id, filters)
    if df.empty:
        print("Skipped timing analysis: no complete timing records")
        return df, pd.DataFrame(), pd.DataFrame()
    summary = timing_summary(df, out_dir, run_id)
    corr = time_feature_correlation(df, out_dir, run_id)
    for metric in ("fit_s", "predict_ms_per_obs"):
        fig, _ = plot_time_vs_features(df, out_dir, run_id, metric=metric)
        finish_figure(fig, show)
    fig, _ = plot_slowest_configs(df, out_dir, run_id, top_n=top_n)
    finish_figure(fig, show)
    return df, summary, corr



def finish_figure(fig, show=False):
    if show:
        plt.show()
    plt.close(fig)


def remove_obsolete_threshold_outputs(out_dir, run_id):
    """Remove retired threshold figures and their score tables."""
    patterns = [f"{run_id}_rq1_recall_precision_threshold_*_repeat_means.csv",
                f"{run_id}_rq1_recall_precision_repeat_means.csv",
                f"{run_id}_rq1_precision_recall_curve.png",
                f"{run_id}_rq1_precision_recall_curve.pdf",
                f"{run_id}_rq1_precision_recall_curve.csv",
                f"{run_id}_rq1_precision_recall_curve_repeat_means.csv"]
    for metric in ("recall", "precision"):
        for extension in ("png", "pdf"):
            patterns.extend([f"{run_id}_rq1_{metric}_threshold_*.{extension}",
                             f"{run_id}_rq1_{metric}.{extension}"])
    for pattern in patterns:
        for path in Path(out_dir).glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)


def display_options(base_dir, run_id):
    data = read_json_records(Path(base_dir) / f"{run_id}_results.json")
    return {
        "selector": ordered_present(data.selector, SELECTOR_STYLES),
        "model": ordered_present(data.model, MODEL_LABELS),
        "imbalance": ordered_present(data.imbalance, IMBALANCE_LABELS),
    }


def normalize_choices(values, label):
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{label} must be a list of names")
    return sorted(set(values))


def run_all(base_dir=BASE_DIR, run_id=None, out_dir=None, top_n=10,
            timing_top_n=15, show=False, plots=None, selectors=None,
            exclude_selectors=None, models=None, imbalances=None):
    """Render one run using the same labels, style and destination throughout."""
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    selected_plots = normalize_choices(plots, "plots")
    selected_plots = list(PLOT_LABELS) if selected_plots is None else selected_plots
    if not selected_plots or set(selected_plots) - PLOT_LABELS.keys():
        raise ValueError(f"Choose plots from: {', '.join(PLOT_LABELS)}")
    filters = {
        "selector": normalize_choices(selectors, "selectors"),
        "model": normalize_choices(models, "models"),
        "imbalance": normalize_choices(imbalances, "imbalances"),
        "exclude_selectors": normalize_choices(exclude_selectors, "exclude_selectors") or [],
    }
    available_options = display_options(base_dir, run_id)
    for column, available in available_options.items():
        requested = filters[column]
        if requested is not None and (not requested or set(requested) - set(available)):
            raise ValueError(f"Unavailable {column} selection {requested}. Available: {', '.join(available)}")
    results = read_json_records(base_dir / f"{run_id}_results.json", filters)
    out_dir = output_directory(base_dir, run_id, out_dir)
    view = {"plots": sorted(selected_plots), **filters,
            "top_n": top_n, "timing_top_n": timing_top_n}
    if plots is not None or any(filters.values()):
        fingerprint = hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest()[:12]
        out_dir = output_directory(base_dir, run_id, out_dir / "views" / fingerprint)
    with (out_dir / "view_config.json").open("w", encoding="utf-8") as file:
        json.dump({"run_id": run_id, **view}, file, indent=2, sort_keys=True)
    remove_obsolete_threshold_outputs(out_dir, run_id)
    print(f"Experiment: {run_id}\nOutput: {out_dir}")
    plot_functions = {
        "tradeoff": plot_tradeoff,
        "ap_difference": plot_ap_difference,
        "stability": plot_rq2a_nogueira_stability,
        "feature_frequency": plot_rq2b_top_feature_frequency,
        "heatmap": plot_rq3_main_tradeoff,
        "imbalance_ap": plot_imbalance_ap,
        "imbalance_stability": plot_imbalance_stability,
    }
    for name, plot in plot_functions.items():
        if name not in selected_plots:
            continue
        options = {"filters": filters}
        if name == "feature_frequency":
            options["top_n"] = top_n
        try:
            result = plot(base_dir, run_id, out_dir=out_dir, **options)
        except PlotUnavailable as exc:
            print(f"Skipped {plot.__name__}: {exc}")
            continue
        figures = result[0]
        if isinstance(figures, dict):
            for fig, _ in figures.values():
                finish_figure(fig, show)
        else:
            finish_figure(figures, show)

    if "thresholds" in selected_plots:
        try:
            data, config = load_threshold_results(base_dir, run_id, filters)
            if "threshold_grid" in config or data.threshold.nunique() > 1:
                plot_threshold_sweep(data, config, out_dir, run_id, show)
        except PlotUnavailable as exc:
            print(f"Skipped threshold analysis: {exc}")

    if "timing" in selected_plots:
        try:
            selection_path = base_dir / f"{run_id}_selections.json"
            selections = read_json_records(selection_path, filters) if selection_path.exists() else pd.DataFrame()
            has_timing = (
                {"fit_time(s)", "predict_time_per_observation(ms)"}.issubset(results.columns)
                and {"selection_time(s)", "n_features_after_preprocessing"}.issubset(selections.columns)
            )
            if has_timing:
                run_timing_analysis(base_dir, run_id, out_dir / "timing", timing_top_n, show, filters)
            else:
                print("Skipped timing analysis: this experiment has no complete timing fields")
        except PlotUnavailable as exc:
            print(f"Skipped timing analysis: {exc}")
    return out_dir


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--run-id", default=RUN_ID, help="Defaults to the latest complete bundle")
    parser.add_argument("--out-dir", type=Path, help="Defaults to experiments/graph/RUN_ID")
    parser.add_argument("--top-n", type=positive_int, default=10)
    parser.add_argument("--timing-top-n", type=positive_int, default=15)
    parser.add_argument("--plots", nargs="+", choices=list(PLOT_LABELS), default=PLOTS_TO_SHOW,
                        help="Only create these plot groups; defaults to all")
    parser.add_argument("--selectors", nargs="+", default=SELECTORS_TO_SHOW,
                        help="Only display these selectors, e.g. mi l1 xgb boruta")
    parser.add_argument("--exclude-selectors", nargs="*", default=SELECTORS_TO_HIDE,
                        help="Hide these selectors, e.g. vote; no names clears the default exclusions")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_SHOW,
                        help="Only display these models, e.g. lr rf")
    parser.add_argument("--imbalances", nargs="+", default=IMBALANCES_TO_SHOW,
                        help="Only display these conditions, e.g. none weight smote")
    parser.add_argument("--list-options", action="store_true",
                        help="List plot groups and available experiment categories without creating figures")
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    args = parser.parse_args(argv)
    if not args.show:
        plt.switch_backend("Agg")
    try:
        if args.list_options:
            run_id = resolve_run_id(args.base_dir, args.run_id)
            print(f"Experiment: {run_id}")
            for name, label in PLOT_LABELS.items():
                print(f"  {name}: {label}")
            for column, values in display_options(args.base_dir, run_id).items():
                print(f"{column}: {', '.join(values)}")
            return
        run_all(args.base_dir, args.run_id, args.out_dir, args.top_n, args.timing_top_n, args.show,
                plots=args.plots, selectors=args.selectors, exclude_selectors=args.exclude_selectors,
                models=args.models, imbalances=args.imbalances)
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parser.exit(2, f"Cannot create figures: {exc}\n")


if __name__ == "__main__":
    main()
