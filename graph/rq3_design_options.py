"""Render three RQ3 design alternatives using saved JSON values only.

Run from the project root:
    .venv/bin/python graph/rq3_design_options.py --run-id 20261006_211727

No training, resampling, averaging, jitter, composite scores, or ranking.
Missing FDR is an explicit placeholder; --hide-missing-fdr removes it.
Extra models and configurations are paginated instead of shrinking the text.
"""
from __future__ import annotations

import argparse
import hashlib
import html
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
SELECTOR_LABELS = {"none": "All features*", "mi": "MI", "fdr": "FDR",
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


def footer(data):
    notes = ["Saved means and stability; NA = unavailable. AP display scale is zoomed.",
             "*All features: after preprocessing; its stability describes preprocessing."]
    if data["shared_stability"]:
        notes.append("Stability is shared across models.")
    if data["missing_fdr"] and "fdr" in data["rows"].get_level_values("selector"):
        notes.append("FDR was not recorded in this run.")
    return "\n".join(textwrap.wrap(" ".join(notes), 118))


def canvas(data, title, rows, width=8.0, *, show_notes=True):
    row_lines = max(x.count("\n") + 1 for x in row_labels(rows))
    height = (1.85 if show_notes else 1.3) + len(rows) * (.245 + .095 * (row_lines - 1))
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.suptitle(title if show_notes else textwrap.fill(title, 60),
                 x=.02, y=.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    if show_notes:
        fig.text(.02, .936, f"RQ3 · Run {data['run_id']} · Same configurations in every column",
                 fontsize=8.5, color="#617083", ha="left")
        fig.text(.02, .016, footer(data), fontsize=7, color="#596677", va="bottom", linespacing=1.5)
    return fig


def plot_heatmaps(data, rows, models, specs, *, title="A  |  Aligned heatmaps", show_notes=True):
    fig = canvas(data, title, rows, show_notes=show_notes)
    widths = [1 if s["shared"] else len(models) for s in specs]
    grid = fig.add_gridspec(2, 3, left=.25, right=.97,
                           top=.80 if show_notes else .78, bottom=.17 if show_notes else .08,
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
        if not show_notes:
            ax.set_yticks(np.arange(len(rows)),
                         [label.replace("All features*", "All features") for label in row_labels(rows)])
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


def plot_dots(data, rows, models, specs):
    fig = canvas(data, "B  |  Aligned dot plots", rows)
    widths = [1 if s["shared"] else len(models) for s in specs]
    grid = fig.add_gridspec(1, 3, left=.25, right=.97, top=.82, bottom=.17,
                           width_ratios=widths, wspace=.30)
    markers = ["o", "s", "D"]
    for j, spec in enumerate(specs):
        arr = matrix(data, spec["key"], rows, models, spec["shared"])
        subgrid = grid[0, j].subgridspec(1, arr.shape[1], wspace=.18)
        for col in range(arr.shape[1]):
            ax = fig.add_subplot(subgrid[0, col])
            vals = arr[:, col]
            valid = np.isfinite(vals)
            for r in range(len(rows)):
                if r % 2 == 0:
                    ax.axhspan(r - .5, r + .5, color="#f6f8fa", zorder=0)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.grid(axis="x", color="#e0e5eb", linewidth=.7)
            ax.scatter(vals[valid], np.flatnonzero(valid), s=24, color=spec["color"],
                       marker=markers[j], edgecolor="white", linewidth=.4, zorder=3)
            low, high = spec["norm"].vmin, spec["norm"].vmax
            margin = (high - low) * .07
            ax.set_xlim(low - margin, high + margin)
            for r in np.flatnonzero(~valid):
                ax.text(.5, r, "NA", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        color="#83909c", fontsize=7)
            model = "Shared" if spec["shared"] else MODEL_LABELS.get(models[col], models[col])
            title = spec["title"].replace(" ", "\n", 1) if j != 0 else "Average\nprecision ↑"
            ax.set_title(f"{title}\n{model}", fontsize=9, pad=12, linespacing=1.5)
            decorate_rows(ax, rows, j == 0 and col == 0)
            ax.tick_params(axis="x", labelsize=7, length=2)
    return fig


def plot_bubbles(data, rows, models, specs):
    fig = canvas(data, "C  |  Bubble plot with one row per configuration", rows)
    grid = fig.add_gridspec(1, len(models), left=.25, right=.97, top=.83, bottom=.27, wspace=.18)
    ap = matrix(data, AP, rows, models)
    count = matrix(data, COUNT, rows, models)
    stable = matrix(data, STABILITY, rows, models)
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad(BG)
    max_area = 110.
    factor = max_area / specs[1]["norm"].vmax
    finite_ap = data["summary"][AP].to_numpy()
    finite_ap = finite_ap[np.isfinite(finite_ap)]
    low, high = (float(finite_ap.min()), float(finite_ap.max())) if finite_ap.size else (0., 1.)
    pad = max((high - low) * .16, .005)
    for col, model in enumerate(models):
        ax = fig.add_subplot(grid[0, col])
        ax.set_axisbelow(True)
        for r in range(len(rows)):
            if r % 2 == 0:
                ax.axhspan(r - .5, r + .5, color="#f6f8fa", zorder=0)
            a, n, s = ap[r, col], count[r, col], stable[r, col]
            if not np.isfinite([a, n, s]).all():
                ax.text(.5, r, "NA", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        color="#83909c", fontsize=7)
                continue
            if n == 0:
                ax.plot(a, r, marker="x", markersize=5, color=cmap(specs[2]["norm"](s)))
            else:
                ax.scatter([a], [r], s=[n * factor], c=[s], cmap=cmap, norm=specs[2]["norm"],
                           edgecolor="#47355c", linewidth=.45, zorder=3)
        ax.set_title(textwrap.fill(MODEL_LABELS.get(model, model), 24), fontsize=10, fontweight="bold", pad=13)
        ax.set_xlim(max(0, low - pad), min(1, high + pad))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color="#e0e5eb", linewidth=.7)
        ax.set_xlabel("Average precision ↑ (zoomed)", fontsize=8, labelpad=8)
        ax.tick_params(axis="x", labelsize=7)
        decorate_rows(ax, rows, col == 0)
    cax = fig.add_axes([.25, .135, .26, .018])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=specs[2]["norm"], cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_label("Colour: Nogueira stability ↑", fontsize=8, labelpad=4)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    legend_ax = fig.add_axes([.59, .102, .36, .068])
    legend_ax.set_axis_off()
    max_count = specs[1]["norm"].vmax
    sizes = [max_count * v for v in [.05, .25, .5, 1.]]
    handles = [legend_ax.scatter([], [], s=n * factor, facecolor="#ddd4e8", edgecolor="#47355c", linewidth=.45)
               for n in sizes]
    legend_ax.legend(handles, [f"{n:.0f}" for n in sizes], title="Area: mean selected features ↓",
                     loc="center", ncol=4, frameon=False, fontsize=8, title_fontsize=8,
                     handletextpad=.5, columnspacing=1.1)
    return fig


def save_fig(fig, out_dir, stem):
    for suffix in ["png", "pdf"]:
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=240, facecolor="white", metadata=metadata)
    plt.close(fig)


def write_gallery(out_dir, pages, data):
    descriptions = {
        "A": "อ่านตัวเลขได้ครบในภาพเดียว สีแยกตาม metric และแสดงค่า shared เพียงครั้งเดียว",
        "B": "อ่านระยะห่างของค่าบนแกนจริง แต่ละวิธีมีแถวของตัวเอง ไม่มีจุดบังกัน",
        "C": "ตำแหน่ง = AP, สี = stability, ขนาดพื้นที่วงกลม = จำนวนฟีเจอร์ โดยแยกแถวให้ทุกวิธี",
    }
    cards = []
    for page in pages:
        stem, option = page["stem"], page["option"]
        cards.append(f'<section><h2>แบบ {option} · {html.escape(page["title"])}</h2>'
                     f'<p>{descriptions[option]}</p><a href="{stem}.png"><img src="{stem}.png" '
                     f'alt="RQ3 option {option}" loading="lazy"></a><p><a href="{stem}.png">PNG</a> · '
                     f'<a href="{stem}.pdf">PDF สำหรับ paper</a></p></section>')
    document = '<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += '<title>RQ3 · Graph design options</title><style>body{margin:0;background:#eef1f5;color:#243247;font:16px system-ui,sans-serif}main{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:30px}p{line-height:1.7}section{background:white;margin:28px 0;padding:24px;border:1px solid #dbe1e8;border-radius:12px}img{width:100%;height:auto}h2{margin:0}a{color:#245fa2}</style><main>'
    document += f'<h1>RQ3: เลือกรูปแบบกราฟ</h1><p>ข้อมูลจริงจาก run {data["run_id"]} · ใช้เฉพาะค่าใน JSON<br>'
    document += 'FDR แสดง NA เมื่อยังไม่มีผลใน run นี้ ขนาดฟีเจอร์และ stability ที่ใช้ร่วมกันระหว่างโมเดลระบุเป็น Shared<br>'
    document += 'All features เป็น baseline หลัง preprocessing; stability ของแถวนี้จึงอธิบาย preprocessing</p>'
    document += ''.join(cards) + '<p><a href="plotted_values.csv">ข้อมูลเบื้องหลังกราฟ (CSV)</a> · <a href="captions.md">คำอธิบายและข้อจำกัด</a></p></main></html>'
    (out_dir / "index.html").write_text(document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--hide-missing-fdr", action="store_true")
    args = parser.parse_args(argv)
    run_id = args.run_id
    if run_id is None:
        candidates = sorted(p.name.removesuffix("_summary.json") for p in args.base_dir.glob("*_summary.json")
                            if p.with_name(p.name.replace("_summary.json", "_stability.json")).exists())
        if not candidates:
            parser.error("No matching summary/stability JSON pair")
        run_id = candidates[-1]
    data = load_run(args.base_dir, run_id, not args.hide_missing_fdr)
    out_dir = args.out_dir or args.base_dir / "figures" / "rq3_options" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "svg.hashsalt": "rq3-options"})
    specs = metric_specs(data)
    pages = []
    for rstart in range(0, len(data["rows"]), 18):
        rows = data["rows"][rstart:rstart + 18]
        for mstart in range(0, len(data["models"]), 3):
            models = data["models"][mstart:mstart + 3]
            for option, title, plot in [("A", "Aligned heatmaps", plot_heatmaps),
                                        ("B", "Aligned dot plots", plot_dots),
                                        ("C", "Row-separated bubbles", plot_bubbles)]:
                stem = f"{option.lower()}_{title.lower().replace(' ', '_').replace('-', '_')}"
                if len(data["rows"]) > 18 or len(data["models"]) > 3:
                    stem += f"_r{rstart // 18 + 1}_m{mstart // 3 + 1}"
                save_fig(plot(data, rows, models, specs), out_dir, stem)
                pages.append(dict(option=option, title=title, stem=stem,
                                  rows=[list(r) for r in rows], models=models))
    data["full"].to_csv(out_dir / "plotted_values.csv", index=False)
    data["stability"].to_csv(out_dir / "saved_stability.csv", index=False)
    manifest = {"run_id": run_id, "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data["inputs"]},
                "metrics": [AP, COUNT, STABILITY], "shared_feature_counts": data["shared_count"],
                "shared_stability": data["shared_stability"], "fdr_missing_in_input": data["missing_fdr"],
                "pages": pages, "max_rows_per_page": 18, "max_models_per_page": 3}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "captions.md").write_text(
        f"RQ3 design alternatives — run {run_id}\n\n"
        "A. Aligned heatmaps compare saved mean average precision (AP), mean selected feature count, "
        "and Nogueira stability. Darker cells mean higher AP/stability and fewer selected features. "
        "Each metric has a separate colour scale; equal colours across metrics do not imply equal scores. "
        "The AP colour scale is zoomed to the observed range, with the same scale for all models and pages.\n\n"
        "B. Aligned dot plots show the same values on separate numeric axes. Rows remain in the same order. "
        "Each configuration has its own row, so equal values never hide another configuration. "
        "The AP axes are zoomed and shared across models/pages.\n\n"
        "C. Row-separated bubbles place mean AP on a zoomed horizontal axis; colour represents Nogueira "
        "stability and circle area is proportional to the saved mean feature count. A cross indicates "
        "a zero feature count, if present. NA means one or more required values are unavailable.\n\n"
        "All plots use saved values, without recomputing means, stability, or a composite metric. "
        "Smaller feature count indicates greater dimensionality reduction only relative to a comparable "
        "candidate set; no reduction percentage is inferred. AP evaluates the positive class as defined "
        "by the experiment. Summary means are descriptive, not evidence of statistical significance. "
        "NA is never zero. The all-feature reference is retained after preprocessing; its saved "
        "stability reflects preprocessing. Model-independent stability is reused for display, not "
        "interpreted as an independent model-level estimate. Feature counts are shared only when "
        "the summary and stability files agree exactly. FDR placeholders are unobserved configurations. "
        "Plots describe the saved run; changes to experiment code after that run are not reflected here.\n")
    write_gallery(out_dir, pages, data)
    print(f"Saved {len(pages)} design pages (PNG + PDF): {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
