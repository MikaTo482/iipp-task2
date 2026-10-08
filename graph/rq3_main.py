"""Standalone approved RQ3 heatmap using saved JSON values only.

Run from the project root:
    .venv/bin/python graph/rq3_main.py --run-id 20261006_211727

Notebook usage:
    from graph.rq3_main import plot_rq3_main_heatmap
    output_dir = plot_rq3_main_heatmap(run_id="20261006_211727")

Dependencies: numpy, pandas, matplotlib. No other project modules are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parents[1]
SELECTORS = ["none", "mi", "fdr", "l1", "xgb", "boruta"]
IMBALANCES = ["none", "weight", "smote"]
MODELS = ["lr", "rf", "xgb", "knn", "nb"]
SELECTOR_LABELS = {"none": "All features", "mi": "MI", "fdr": "FDR",
                   "l1": "L1", "xgb": "XGB selector", "boruta": "Boruta"}
IMBALANCE_LABELS = {"none": "None", "weight": "Weight", "smote": "SMOTE"}
MODEL_LABELS = {"lr": "LR", "rf": "RF", "xgb": "XGBoost", "knn": "KNN", "nb": "NB"}
KEYS = ["selector", "imbalance"]
AP = "average_precision_mean"
COUNT = "n_features_mean"
STABILITY = "nogueira"
BG = "#e5e7eb"
INK = "#243247"



def levels(values, preferred):
    observed = set(values)
    return [x for x in preferred if x in observed] + sorted(observed - set(preferred))


def read_records(path, keys, numeric):
    # json.load preserves the original float representation; pandas.read_json can round it.
    with path.open() as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path.name}: expected a nonempty list of records")
    df = pd.DataFrame(records)
    missing = set(keys + numeric) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if df[keys].isna().any().any():
        raise ValueError(f"{path.name}: missing configuration identifiers")
    df[keys] = df[keys].astype(str)
    if df.duplicated(keys).any():
        raise ValueError(f"{path.name}: duplicate configuration keys")
    for field in numeric:
        df[field] = pd.to_numeric(df[field], errors="raise").astype(float)
    return df


def load_run(base_dir, run_id, show_missing_fdr=True):
    base_dir = Path(base_dir)
    summary_path = base_dir / f"{run_id}_summary.json"
    stability_path = base_dir / f"{run_id}_stability.json"
    summary = read_records(summary_path, KEYS + ["model"], [AP, COUNT])
    with stability_path.open() as stream:
        stability_records = json.load(stream)
    by_model = any("model" in row for row in stability_records)
    stability_keys = KEYS + (["model"] if by_model else [])
    stability = read_records(stability_path, stability_keys, [STABILITY, COUNT])
    selectors = levels(set(summary.selector) | set(stability.selector) |
                       ({"fdr"} if show_missing_fdr else set()), SELECTORS)
    imbalances = levels(set(summary.imbalance) | set(stability.imbalance), IMBALANCES)
    models = levels(set(summary.model) |
                    (set(stability.model) if by_model else set()), MODELS)
    rows = pd.MultiIndex.from_product([selectors, imbalances], names=KEYS)
    index = pd.MultiIndex.from_product([selectors, imbalances, models], names=KEYS + ["model"])
    full = summary.set_index(KEYS + ["model"])[[AP, COUNT]].reindex(index).reset_index()
    full = full.merge(stability[stability_keys + [STABILITY]], on=stability_keys,
                      how="left", validate="one_to_one" if by_model else "many_to_one")
    # Share feature counts only when saved summary and stability means really agree.
    check = summary.merge(stability[stability_keys + [COUNT]], on=stability_keys,
                          how="left", suffixes=("_summary", "_stability"),
                          validate="one_to_one" if by_model else "many_to_one")
    shared_count = not by_model and np.array_equal(
        check[f"{COUNT}_summary"].to_numpy(), check[f"{COUNT}_stability"].to_numpy(),
        equal_nan=True)
    return {"summary": summary, "stability": stability, "full": full,
            "rows": rows, "models": models, "shared_count": shared_count,
            "shared_stability": not by_model, "run_id": run_id,
            "inputs": [summary_path, stability_path],
            "missing_fdr": "fdr" not in set(summary.selector) | set(stability.selector)}


def matrix(data, metric, rows, models, shared=False):
    if shared:
        return data["stability"].set_index(KEYS)[metric].reindex(rows).to_numpy()[:, None]
    return (data["full"].pivot(index=KEYS, columns="model", values=metric)
            .reindex(index=rows, columns=models).to_numpy(dtype=float))


def metric_specs(data):
    ap_values = data["summary"][AP].to_numpy()
    counts = np.r_[data["summary"][COUNT], data["stability"][COUNT]]
    stable = data["stability"][STABILITY].to_numpy()
    finite_ap = ap_values[np.isfinite(ap_values)]
    finite_count = counts[np.isfinite(counts)]
    finite_stable = stable[np.isfinite(stable)]
    ap_min = np.floor(float(finite_ap.min()) / .01) * .01 if finite_ap.size else 0.
    ap_max = np.ceil(float(finite_ap.max()) / .01) * .01 if finite_ap.size else 1.
    if ap_min == ap_max:
        ap_min, ap_max = max(0., ap_min - .01), min(1., ap_max + .01)
    count_max = max(1., float(finite_count.max()) if finite_count.size else 1.)
    stability_min = min(0., float(finite_stable.min()) if finite_stable.size else 0.)
    stability_max = max(1., float(finite_stable.max()) if finite_stable.size else 1.)
    return [
        {"key": AP, "title": "Average precision ↑", "cmap": "YlGnBu", "color": "#135d88",
         "norm": colors.Normalize(ap_min, ap_max), "shared": False, "fmt": ".3f"},
        {"key": COUNT, "title": "Mean features ↓", "cmap": "YlOrBr_r", "color": "#a64b08",
         "norm": colors.Normalize(0., count_max), "shared": data["shared_count"], "fmt": ".1f"},
        {"key": STABILITY, "title": "Nogueira stability ↑", "cmap": "RdPu", "color": "#a80f68",
         "norm": colors.Normalize(stability_min, stability_max),
         "shared": data["shared_stability"], "fmt": ".3f"},
    ]


def row_labels(rows):
    return [textwrap.fill(f"{SELECTOR_LABELS.get(s, s)} | {IMBALANCE_LABELS.get(i, i)}", 29)
            for s, i in rows]


def decorate_rows(ax, rows, show_labels=False, heatmap=False):
    ax.set_yticks(np.arange(len(rows)), row_labels(rows))
    ax.tick_params(axis="y", labelleft=show_labels, length=0, pad=7)
    ax.set_ylim(len(rows) - .5, -.5)
    for row in range(1, len(rows)):
        if rows[row][0] != rows[row - 1][0]:
            ax.axhline(row - .5, color="#c7ced6", linewidth=.7, zorder=0 if not heatmap else 2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def canvas(title, rows):
    row_lines = max(label.count("\n") + 1 for label in row_labels(rows))
    height = 1.3 + len(rows) * (.245 + .095 * (row_lines - 1))
    fig = plt.figure(figsize=(8.0, height), facecolor="white")
    fig.suptitle(textwrap.fill(title, 60), x=.02, y=.985, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    return fig


def plot_heatmaps(data, rows, models, specs, *, title):
    fig = canvas(title, rows)
    widths = [1 if s["shared"] else len(models) for s in specs]
    grid = fig.add_gridspec(2, 3, left=.25, right=.97,
                           top=.78, bottom=.08,
                           height_ratios=[20, .65], width_ratios=widths, wspace=.25, hspace=.18)
    for j, spec in enumerate(specs):
        ax = fig.add_subplot(grid[0, j])
        arr = matrix(data, spec["key"], rows, models, spec["shared"])
        cmap = plt.get_cmap(spec["cmap"]).copy()
        cmap.set_bad(BG)
        im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, norm=spec["norm"])
        title = ["Average precision ↑", "Mean selected\nfeatures ↓", "Nogueira\nstability ↑"][j]
        ax.set_title(title, fontsize=10, fontweight="bold", pad=25, color=spec["color"])
        labels = ["Shared"] if spec["shared"] else [MODEL_LABELS.get(m, m) for m in models]
        ax.set_xticks(range(len(labels)), [textwrap.fill(x, 15) for x in labels])
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=6)
        decorate_rows(ax, rows, j == 0, True)
        ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", length=0)
        for (r, c), value in np.ndenumerate(arr):
            valid = np.isfinite(value)
            rgba = cmap(spec["norm"](value)) if valid else colors.to_rgba(BG)
            # Choose black/white text using linear-light sRGB contrast.
            rgb = np.asarray(rgba[:3])
            linear_rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = np.dot(linear_rgb, [.2126, .7152, .0722])
            text_color = "black" if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else "white"
            ax.text(c, r, format(value, spec["fmt"]) if valid else "NA", ha="center", va="center",
                    fontsize=8.5, color=text_color)
        cax = fig.add_subplot(grid[1, j])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.locator = ticker.FixedLocator(np.linspace(spec["norm"].vmin, spec["norm"].vmax, 3))
        bar.formatter = ticker.FormatStrFormatter("%.3f" if spec["key"] == AP else "%.1f")
        bar.update_ticks()
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, length=2)
    return fig


def save_fig(fig, out_dir, stem):
    for suffix in ["png", "pdf"]:
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=240, facecolor="white", metadata=metadata)
    plt.close(fig)


def plot_rq3_main_heatmap(base_dir=None, run_id=None, out_dir=None, *, show_missing_fdr=True):
    """Save the selected main figure, supporting CSVs, caption and manifest.

    Uses the latest matching JSON pair if run_id is omitted. Returns the output
    directory. Additional models/rows are paginated with shared colour scales.
    """
    base_dir = Path(base_dir) if base_dir is not None else ROOT / "experiments"
    if run_id is None:
        runs = sorted(path.name.removesuffix("_summary.json")
                      for path in base_dir.glob("*_summary.json")
                      if (base_dir / path.name.replace("_summary.json", "_stability.json")).exists())
        if not runs:
            raise ValueError(f"No matching summary/stability JSON pair in {base_dir}")
        run_id = runs[-1]
    data = load_run(base_dir, run_id, show_missing_fdr=show_missing_fdr)
    out_dir = Path(out_dir) if out_dir is not None else base_dir / "figures" / "rq3_main" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = metric_specs(data)
    pages = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42}):
        for row_start in range(0, len(data["rows"]), 18):
            rows = data["rows"][row_start:row_start + 18]
            for model_start in range(0, len(data["models"]), 3):
                models = data["models"][model_start:model_start + 3]
                stem = f"{run_id}_rq3_main_heatmap"
                if len(data["rows"]) > 18 or len(data["models"]) > 3:
                    stem += f"_r{row_start // 18 + 1}_m{model_start // 3 + 1}"
                fig = plot_heatmaps(
                    data, rows, models, specs,
                    title="Trade-offs among Average Precision, Feature Count, and Nogueira Stability")
                save_fig(fig, out_dir, stem)
                pages.append({"png": stem + ".png", "pdf": stem + ".pdf",
                              "rows": [list(row) for row in rows], "models": models})
    data["full"].to_csv(out_dir / "plotted_values.csv", index=False)
    data["stability"].to_csv(out_dir / "saved_stability.csv", index=False)
    manifest = {
        "run_id": run_id, "selected_design": "A: aligned heatmaps",
        "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in data["inputs"]},
        "metrics": [AP, COUNT, STABILITY],
        "shared_feature_counts": data["shared_count"],
        "shared_stability": data["shared_stability"],
        "fdr_missing_in_input": data["missing_fdr"],
        "max_rows_per_page": 18, "max_models_per_page": 3, "pages": pages,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "caption.md").write_text(
        "Performance–reduction–stability trade-off across feature-selection and imbalance-handling "
        "configurations. The aligned panels report saved mean average precision (AP) for each model, "
        "saved mean selected feature count, and Nogueira stability. Rows represent the same "
        "configurations across all panels. Darker cells indicate higher AP, fewer features, or "
        "higher stability, respectively. Each metric has a separate colour scale, shared across "
        "models and pages. The AP colour scale is zoomed to the observed range.\n\n"
        "Shared denotes values recorded at the selector–imbalance level and used across models. "
        "The all-feature reference uses variables surviving preprocessing; its saved stability "
        "therefore describes preprocessing. NA denotes unavailable values, including FDR when "
        "absent from the source run. Feature counts measure retained dimensionality, not a "
        "newly calculated reduction percentage. The panels describe trade-offs in saved values; "
        "they do not establish statistical significance or a combined ranking.\n",
        encoding="utf-8")
    print(f"RQ3 main figure: {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--hide-missing-fdr", action="store_true")
    args = parser.parse_args(argv)
    return plot_rq3_main_heatmap(args.base_dir, args.run_id, args.out_dir,
                                 show_missing_fdr=not args.hide_missing_fdr)


if __name__ == "__main__":
    main()
