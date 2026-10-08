"""Notebook-ready version of the approved RQ3 heatmap."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter


SELECTOR_ORDER = ["none", "mi", "fdr", "l1", "xgb", "boruta"]
SELECTOR_LABELS = {
    "none": "All features", "mi": "MI", "fdr": "FDR",
    "l1": "L1", "xgb": "XGB selector", "boruta": "Boruta",
}
IMBALANCE_ORDER = ["none", "weight", "smote"]
IMBALANCE_LABELS = {"none": "None", "weight": "Weight", "smote": "SMOTE"}
MODEL_ORDER = ["lr", "rf", "xgb", "knn", "nb"]
MODEL_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGBoost", "knn": "KNN", "nb": "NB"}

FIGURE_TITLE = (
    "Trade-offs among Average Precision, Feature Count, and Nogueira Stability"
)


def ordered_levels(values, preferred):
    values = set(values)
    return [v for v in preferred if v in values] + sorted(values - set(preferred))


def read_json_records(path):
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{Path(path).name}: expected a nonempty list of records")
    return pd.DataFrame(records)


def validate_frame(df, keys, metrics, filename):
    missing = set(keys + metrics) - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    if df[keys].isna().any().any():
        raise ValueError(f"{filename}: missing configuration identifiers")
    df = df.copy()
    df[keys] = df[keys].astype(str)
    if df.duplicated(keys).any():
        raise ValueError(f"{filename}: duplicate configuration rows")
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="raise").astype(float)
    return df


def plot_rq3_main_tradeoff(base_dir, run_id, out_dir=None):
    """Plot the approved heatmap; return fig, three main axes, and merged data.

    Uses saved AP/count means and Nogueira values without recomputing metrics.
    Figure dimensions grow with the number of configurations and models.
    """
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir is not None else base_dir / "figures" / "rq3_main" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    score_col = "average_precision_mean"
    count_col = "n_features_mean"
    keys = ["selector", "imbalance"]
    model_keys = keys + ["model"]

    summary = read_json_records(base_dir / f"{run_id}_summary.json")
    stability = read_json_records(base_dir / f"{run_id}_stability.json")
    stability_keys = model_keys if "model" in stability.columns else keys
    summary = validate_frame(summary, model_keys, [score_col, count_col], "summary.json")
    stability = validate_frame(stability, stability_keys, ["nogueira"], "stability.json")

    data = summary.merge(
        stability[stability_keys + ["nogueira"]],
        on=stability_keys, how="left",
        validate="one_to_one" if "model" in stability_keys else "many_to_one",
    )

    selectors = ordered_levels(set(summary.selector) | set(stability.selector) | {"fdr"}, SELECTOR_ORDER)
    imbalances = ordered_levels(set(summary.imbalance) | set(stability.imbalance), IMBALANCE_ORDER)
    model_values = set(summary.model)
    if "model" in stability.columns:
        model_values |= set(stability.model)
    models = ordered_levels(model_values, MODEL_ORDER)
    rows = pd.MultiIndex.from_product([selectors, imbalances], names=keys)

    ap_matrix = summary.pivot(index=keys, columns="model", values=score_col).reindex(index=rows, columns=models)
    count_matrix = summary.pivot(index=keys, columns="model", values=count_col).reindex(index=rows, columns=models)

    # Collapse identical saved counts only; never average across models.
    shared_count = all(count_matrix[m].equals(count_matrix.iloc[:, 0]) for m in models)
    if shared_count:
        count_matrix = count_matrix.iloc[:, [0]].copy()
        count_matrix.columns = ["Shared"]

    if "model" in stability.columns:
        stability_matrix = stability.pivot(index=keys, columns="model", values="nogueira").reindex(index=rows, columns=models)
    else:
        stability_matrix = stability.set_index(keys)[["nogueira"]].reindex(rows)
        stability_matrix.columns = ["Shared"]

    matrices = [ap_matrix, count_matrix, stability_matrix]
    arrays = [matrix.to_numpy(dtype=float) for matrix in matrices]
    finite = [values[np.isfinite(values)] for values in arrays]

    ap_min = np.floor(finite[0].min() / .01) * .01 if finite[0].size else 0.0
    ap_max = np.ceil(finite[0].max() / .01) * .01 if finite[0].size else 1.0
    if ap_min == ap_max:
        ap_min, ap_max = max(0.0, ap_min - .01), min(1.0, ap_max + .01)
    count_max = max(1.0, float(finite[1].max())) if finite[1].size else 1.0
    stability_min = min(0.0, float(finite[2].min())) if finite[2].size else 0.0
    stability_max = max(1.0, float(finite[2].max())) if finite[2].size else 1.0

    norms = [plt.Normalize(ap_min, ap_max), plt.Normalize(0, count_max),
             plt.Normalize(stability_min, stability_max)]
    cmaps = ["YlGnBu", "YlOrBr_r", "RdPu"]
    title_colors = ["#135d88", "#a64b08", "#a80f68"]
    titles = ["Average precision ↑", "Mean selected\nfeatures ↓", "Nogueira\nstability ↑"]
    formats = [".3f", ".1f", ".3f"]
    labels = [textwrap.fill(f"{SELECTOR_LABELS.get(s, s)} | {IMBALANCE_LABELS.get(i, i)}", 29)
              for s, i in rows]

    widths = [len(matrix.columns) for matrix in matrices]
    fig_width = max(8.0, 3.0 + sum(widths))
    label_lines = max(label.count("\n") + 1 for label in labels)
    fig_height = max(4.0, 1.3 + len(rows) * (.245 + .095 * (label_lines - 1)))

    style = {"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
             "text.color": "#243247", "xtick.color": "#243247", "ytick.color": "#243247"}

    with plt.rc_context(style):
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
        fig.suptitle(textwrap.fill(FIGURE_TITLE, 60), x=.02, y=.985,
                     ha="left", fontsize=13, fontweight="bold")
        grid = fig.add_gridspec(
            2, 3, left=2.0 / fig_width, right=1 - .24 / fig_width,
            top=1 - 1.25 / fig_height, bottom=.46 / fig_height,
            height_ratios=[20, .65], width_ratios=widths, wspace=.25, hspace=.18,
        )
        axes = np.empty(3, dtype=object)

        for j, (matrix, arr) in enumerate(zip(matrices, arrays)):
            ax = axes[j] = fig.add_subplot(grid[0, j])
            cmap = plt.get_cmap(cmaps[j]).copy()
            cmap.set_bad("#e5e7eb")
            im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norms[j], aspect="auto")
            ax.set_title(titles[j], fontsize=10, fontweight="bold", pad=25, color=title_colors[j])
            column_labels = [textwrap.fill(MODEL_LABELS.get(m, m), 15) for m in matrix.columns]
            ax.set_xticks(np.arange(arr.shape[1]), column_labels)
            ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=6)
            ax.set_yticks(np.arange(len(rows)), labels)
            ax.tick_params(axis="y", labelleft=j == 0, length=0, pad=7)
            ax.set_ylim(len(rows) - .5, -.5)

            for r in range(1, len(rows)):
                if rows[r][0] != rows[r - 1][0]:
                    ax.axhline(r - .5, color="#c7ced6", linewidth=.7, zorder=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.3)
            ax.tick_params(which="minor", length=0)

            for (r, c), value in np.ndenumerate(arr):
                valid = np.isfinite(value)
                rgba = cmap(norms[j](value)) if valid else (229 / 255, 231 / 255, 235 / 255, 1)
                rgb = np.asarray(rgba[:3])
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = np.dot(linear, [.2126, .7152, .0722])
                color = "black" if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else "white"
                ax.text(c, r, format(value, formats[j]) if valid else "NA",
                        ha="center", va="center", fontsize=8.5, color=color)

            cax = fig.add_subplot(grid[1, j])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.locator = FixedLocator(np.linspace(norms[j].vmin, norms[j].vmax, 3))
            cbar.formatter = FormatStrFormatter("%.3f" if j == 0 else "%.1f")
            cbar.update_ticks()
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=7, length=2)

        stem = f"{run_id}_rq3_main_heatmap"
        fig.savefig(out_dir / f"{stem}.png", dpi=300, facecolor="white")
        fig.savefig(out_dir / f"{stem}.pdf", facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})

    data.to_csv(out_dir / f"{stem}.csv", index=False)
    for name, matrix in zip(["ap", "feature_count", "nogueira"], matrices):
        matrix.to_csv(out_dir / f"{stem}_{name}.csv")
    print(f"Saved: {out_dir / stem} (.png, .pdf, .csv)")
    return fig, axes, data
