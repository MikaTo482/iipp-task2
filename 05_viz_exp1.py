"""Reproduce experiment figures from saved JSON without running model fits.

Run with --run-id to select an experiment and --out-dir to change the destination.
Importing this module does not load experiments or create output files.
"""

import argparse
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
    "vote": ("Voting", "#D55E00", "*")
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
    "kuncheva": "Kuncheva index",
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
        bottom += 0.3
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
def plot_tradeoff(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None):
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    summary = read_json_records(base_dir / f"{run_id}_summary.json")
    stability = read_json_records(base_dir / f"{run_id}_stability.json")

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
        Line2D([], [], color=SELECTOR_STYLES[s][1], marker=SELECTOR_STYLES[s][2],
               linestyle="none", markersize=7, label=SELECTOR_STYLES[s][0])
        for s in selectors
    ]

    fig.suptitle("Feature reduction and quality anomaly prediction", fontsize=15, y=0.99)
    add_figure_legends(fig, handles)

    out = output_directory(base_dir, run_id, out_dir)
    save_figure(fig, out, f"{run_id}_rq1_tradeoff")

    return fig, axes, data


@plt.rc_context(PLOT_STYLE)
def plot_ap_difference(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None):
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    data = read_json_records(base_dir / f"{run_id}_results.json")
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
                _, color, _ = SELECTOR_STYLES[selector]
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
        Line2D([], [], color=SELECTOR_STYLES[s][1], marker="o",
               linestyle="none", markersize=7, label=SELECTOR_STYLES[s][0])
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


@plt.rc_context(PLOT_STYLE)
def plot_recall_precision(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None):
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    data = read_json_records(base_dir / f"{run_id}_results.json")
    config = read_config(base_dir, run_id)

    selectors = ordered_present(data["selector"], SELECTOR_STYLES)
    imbalances = ordered_present(data["imbalance"], IMBALANCE_LABELS)
    models = ordered_present(data["model"], MODEL_LABELS)
    keys = ["repeat", "fold", "imbalance", "selector", "model"]

    validate_splits(data, config)
    if data[["recall", "precision", "tp", "fp", "fn"]].isna().any().any():
        raise PlotUnavailable("Recall and precision need recorded scores and confusion counts for every split")

    for metric in ["recall", "precision"]:
        if not np.isfinite(data[metric]).all() or not data[metric].between(0, 1).all():
            raise ValueError(f"Invalid {metric}")

    check_complete_grid(data, imbalances, selectors, models, "Missing or unexpected configurations")

    for metric, denominator in [("recall", data.tp + data.fn), ("precision", data.tp + data.fp)]:
        calculated = np.divide(
            data.tp.to_numpy(dtype=float),
            denominator.to_numpy(dtype=float),
            out=np.zeros(len(data)),
            where=denominator.to_numpy() != 0,
        )
        np.testing.assert_allclose(data[metric], calculated, atol=1e-12)

    data["no_predicted_failures"] = (data.tp + data.fp).eq(0)
    repeat_means = data.groupby(
        ["repeat", "imbalance", "selector", "model"],
        as_index=False,
    ).agg(
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        n_folds=("fold", "nunique"),
        folds_without_predicted_failures=("no_predicted_failures", "sum"),
    )

    assert repeat_means["n_folds"].eq(config["n_splits"]).all()

    out = output_directory(base_dir, run_id, out_dir)
    repeat_means.to_csv(out / f"{run_id}_rq1_recall_precision_repeat_means.csv", index=False)

    figures = {}
    for metric in ["recall", "precision"]:
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

                for x, selector in enumerate(selectors):
                    _, color, _ = SELECTOR_STYLES[selector]
                    values = repeat_means.loc[
                        repeat_means["imbalance"].eq(imbalance)
                        & repeat_means["model"].eq(model)
                        & repeat_means["selector"].eq(selector)
                    ].sort_values("repeat")[metric].to_numpy()

                    ax.scatter(x + np.linspace(-0.15, 0.15, len(values)), values,
                               s=24, color=color, alpha=0.65, zorder=3)
                    ax.scatter(x, values.mean(), marker="D", s=65, color=color,
                               edgecolor="black", linewidth=0.8, zorder=4)

                ax.set_xticks([])
                ax.set_xlim(-0.5, len(selectors) - 0.5)
                ax.set_ylim(-0.025, 1.025)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(axis="y", alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)

                if i == 0:
                    ax.set_title(MODEL_LABELS.get(model, model))
                if j == 0:
                    ax.set_ylabel(f"{IMBALANCE_LABELS.get(imbalance, imbalance)}\n{metric.capitalize()}")

        color_handles = [
            Line2D([], [], marker="o", color=SELECTOR_STYLES[s][1],
                   linestyle="none", label=SELECTOR_STYLES[s][0])
            for s in selectors
        ]
        shape_handles = [
            Line2D([], [], marker="o", color="#555555", linestyle="none",
                   label="Mean of fold scores within a repeat"),
            Line2D([], [], marker="D", color="#555555", markeredgecolor="black",
                   linestyle="none", label="Mean across repeats"),
        ]

        fig.suptitle(
            f'Quality anomaly {metric} after feature selection\nDecision threshold {config["decision_threshold"]:g}',
            fontsize=14,
        )
        note = "Precision is recorded as zero when a fold has no predicted failures" if metric == "precision" else None
        add_figure_legends(fig, color_handles, shape_handles, note)
        save_figure(fig, out, f"{run_id}_rq1_{metric}")

        figures[metric] = (fig, axes)

    return figures, repeat_means


@plt.rc_context(PLOT_STYLE)
def plot_rq2a_nogueira_stability(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None):
    """Plot saved stability metrics. Supports MI, FDR, L1, XGB, Boruta, and Voting."""
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)

    df = read_json_records(base_dir / f'{run_id}_stability.json')

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
def plot_rq2b_top_feature_frequency(base_dir=BASE_DIR, run_id=RUN_ID, top_n=10, out_dir=None):
    """Plot top selected features per imbalance/selector configuration. Supports FDR."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError('top_n must be a positive integer')

    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)

    df = read_json_records(base_dir / f'{run_id}_feature_frequency.json')

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

def read_json_records(path):
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{Path(path).name}: expected a nonempty list of records"
        )

    return pd.DataFrame(records)


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
def plot_rq3_main_tradeoff(base_dir=BASE_DIR, run_id=RUN_ID, out_dir=None):
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
        base_dir / f"{run_id}_summary.json"
    )
    stability = read_json_records(
        base_dir / f"{run_id}_stability.json"
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


def load_timing(base_dir, run_id):
    """One row per fold x imbalance x selector x model, with selection, fit and predict time."""
    base_dir = Path(base_dir)
    results = read_json_records(base_dir / f"{run_id}_results.json")
    selections = read_json_records(base_dir / f"{run_id}_selections.json")

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


def run_timing_analysis(base_dir, run_id, out_dir, top_n=15, show=False):
    """Run every timing table and figure; everything is written to out_dir."""
    df = load_timing(base_dir, run_id)
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


def run_all(base_dir=BASE_DIR, run_id=None, out_dir=None, top_n=10,
            timing_top_n=15, show=False):
    """Render one run using the same labels, style and destination throughout."""
    base_dir = Path(base_dir)
    run_id = resolve_run_id(base_dir, run_id)
    out_dir = output_directory(base_dir, run_id, out_dir)
    print(f"Experiment: {run_id}\nOutput: {out_dir}")
    plots = (plot_tradeoff, plot_ap_difference, plot_recall_precision,
             plot_rq2a_nogueira_stability, plot_rq2b_top_feature_frequency,
             plot_rq3_main_tradeoff)
    for plot in plots:
        options = {"top_n": top_n} if plot is plot_rq2b_top_feature_frequency else {}
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

    results = read_json_records(base_dir / f"{run_id}_results.json")
    selection_path = base_dir / f"{run_id}_selections.json"
    selections = read_json_records(selection_path) if selection_path.exists() else pd.DataFrame()
    has_timing = (
        {"fit_time(s)", "predict_time_per_observation(ms)"}.issubset(results.columns)
        and {"selection_time(s)", "n_features_after_preprocessing"}.issubset(selections.columns)
    )
    if has_timing:
        run_timing_analysis(base_dir, run_id, out_dir / "timing", timing_top_n, show)
    else:
        print("Skipped timing analysis: this experiment has no complete timing fields")
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
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    args = parser.parse_args(argv)
    if not args.show:
        plt.switch_backend("Agg")
    try:
        run_all(args.base_dir, args.run_id, args.out_dir, args.top_n, args.timing_top_n, args.show)
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parser.exit(2, f"Cannot create figures: {exc}\n")


if __name__ == "__main__":
    main()
