"""Regression checks for saved experiment figures, without fitting models."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "05_viz_exp1.py"
spec = importlib.util.spec_from_file_location("viz_exp1", SCRIPT)
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)


def write_bundle(folder, run_id="sample"):
    """A smaller valid experiment with variable feature sets and no timings."""
    records = []
    for repeat in (1, 2):
        for fold in (1, 2):
            for selector, score in (("none", 0.2), ("mi", 0.3)):
                records.append(dict(repeat=repeat, fold=fold, selector=selector,
                                    imbalance="none", model="lr", ap=score,
                                    recall=0.5, precision=0.5, tp=1, fp=1, fn=1, tn=7,
                                    n_features=10 if selector == "none" else 2))
    summary = [dict(selector=s, imbalance="none", model="lr",
                    average_precision_mean=score, average_precision_std=0.01,
                    n_features_mean=count)
               for s, score, count in (("none", 0.2, 10), ("mi", 0.3, 2))]
    stability = [dict(selector=s, imbalance="none", n_features_std=0,
                      nogueira=0.6, mean_jaccard=0.7, kuncheva=None)
                 for s in ("none", "mi")]
    frequency = [dict(selector="mi", imbalance="none", feature=f,
                      selection_frequency=0.5, selected_count=2, n_runs=4)
                 for f in ("f10", "f2", "f1")]
    tables = dict(config=dict(n_splits=2, n_repeats=2, decision_threshold=0.5),
                  results=records, summary=summary, stability=stability,
                  feature_frequency=frequency)
    for name, data in tables.items():
        (folder / f"{run_id}_{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return tables


class ExperimentVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plt.close, "all")
        self.folder = Path(self.temp.name)
        self.tables = write_bundle(self.folder)

    def test_import_does_not_read_experiments_or_print(self):
        module = importlib.util.module_from_spec(spec)
        output = io.StringIO()
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected file read")):
            with contextlib.redirect_stdout(output):
                spec.loader.exec_module(module)
        self.assertEqual(output.getvalue(), "")

    def write_imbalance_bundle(self):
        for table in ("results", "summary", "stability"):
            baseline = list(self.tables[table])
            for condition, delta in (("weight", 0.04), ("smote", -0.02)):
                for row in baseline:
                    new = {**row, "imbalance": condition}
                    if table == "results":
                        new["ap"] += delta + 0.01 * row["repeat"]
                    elif table == "summary":
                        new["n_features_mean"] += 1
                    else:
                        new["nogueira"] += delta
                    self.tables[table].append(new)
        self.flush_tables()

    def test_imbalance_ap_pairs_conditions_not_feature_baselines(self):
        self.write_imbalance_bundle()
        with patch.object(viz, "save_figure"):
            _, _, repeats, paired = viz.plot_imbalance_ap(
                self.folder, "sample", filters={"selector": ["mi"], "imbalance": ["weight"]})
        self.assertEqual(set(paired.imbalance), {"weight"})
        self.assertEqual(set(paired.selector), {"mi"})
        np.testing.assert_allclose(paired.baseline_ap, 0.3)
        np.testing.assert_allclose(repeats.delta_ap, [0.05, 0.06])
        summary = viz.pd.read_csv(self.folder / "graph/sample/sample_rq4_imbalance_ap_summary.csv")
        np.testing.assert_allclose(summary.delta_ap_mean, 0.055)
        self.assertEqual(summary.n_complete_repeats.tolist(), [2])

    def test_imbalance_ap_missing_scores_exclude_whole_repeat(self):
        self.write_imbalance_bundle()
        row = next(r for r in self.tables["results"] if r["imbalance"] == "weight"
                   and r["selector"] == "mi" and r["repeat"] == 1)
        row["ap"] = None
        self.flush_tables()
        with patch.object(viz, "save_figure"):
            _, _, repeats, _ = viz.plot_imbalance_ap(self.folder, "sample", filters={"selector": ["mi"]})
        incomplete = repeats.loc[repeats.imbalance.eq("weight") & repeats.repeat.eq(1)]
        self.assertTrue(incomplete.delta_ap.isna().all())
        self.assertEqual(incomplete.n_paired_folds.tolist(), [1])

    def test_imbalance_comparison_requires_baseline_and_unique_pairs(self):
        self.write_imbalance_bundle()
        records = self.tables["results"]
        records.append(records[0].copy())
        self.flush_tables()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            viz.plot_imbalance_ap(self.folder, "sample")
        records.pop()
        self.tables["results"] = [r for r in records if r["imbalance"] != "none"]
        self.flush_tables()
        with self.assertRaisesRegex(viz.PlotUnavailable, "baseline"):
            viz.plot_imbalance_ap(self.folder, "sample")

    def test_imbalance_stability_keeps_reference_and_signed_changes(self):
        self.write_imbalance_bundle()
        with patch.object(viz, "save_figure"):
            _, _, paired = viz.plot_imbalance_stability(
                self.folder, "sample", filters={"selector": ["mi"], "imbalance": ["smote"]})
        self.assertEqual(set(paired.imbalance), {"smote"})
        np.testing.assert_allclose(paired.baseline_nogueira, 0.6)
        np.testing.assert_allclose(paired.delta_nogueira, -0.02)
        np.testing.assert_allclose(paired.baseline_n_features_mean, 2)
        np.testing.assert_allclose(paired.n_features_mean, 3)
        np.testing.assert_allclose(paired.delta_n_features_mean, 1)

    def test_imbalance_ap_all_missing_remains_unavailable_not_zero(self):
        self.write_imbalance_bundle()
        for row in self.tables["results"]:
            row["ap"] = None
        self.flush_tables()
        with patch.object(viz, "save_figure"):
            _, _, repeats, _ = viz.plot_imbalance_ap(self.folder, "sample")
        self.assertTrue(repeats.delta_ap.isna().all())
        self.assertTrue(repeats.n_paired_folds.eq(0).all())

    def test_imbalance_ap_keeps_learners_separate(self):
        self.write_imbalance_bundle()
        other_model = []
        for row in self.tables["results"]:
            new = {**row, "model": "rf"}
            new["ap"] = 0.7 if row["imbalance"] == "none" else 0.6
            other_model.append(new)
        self.tables["results"].extend(other_model)
        self.flush_tables()
        with patch.object(viz, "save_figure"):
            _, _, repeats, paired = viz.plot_imbalance_ap(self.folder, "sample")
        np.testing.assert_allclose(paired.loc[paired.model.eq("rf"), "baseline_ap"], 0.7)
        np.testing.assert_allclose(repeats.loc[repeats.model.eq("rf"), "delta_ap"], -0.1)

    def test_imbalance_stability_missing_and_model_specific_scores(self):
        self.write_imbalance_bundle()
        for row in self.tables["stability"]:
            row.update(model="lr", n_features_mean=2)
            if row["imbalance"] == "weight":
                row["nogueira"] = None
        self.flush_tables()
        with patch.object(viz, "save_figure"):
            _, axes, paired = viz.plot_imbalance_stability(self.folder, "sample")
        self.assertEqual(axes.shape, (2, 2))
        self.assertEqual(set(paired.model), {"lr"})
        self.assertTrue(paired.loc[paired.imbalance.eq("weight"), "delta_nogueira"].isna().all())

    def test_imbalance_figures_repeat_identically_and_support_selected_views(self):
        self.write_imbalance_bundle()
        before = {p.name: p.read_bytes() for p in self.folder.glob("*.json")}
        with contextlib.redirect_stdout(io.StringIO()):
            first = viz.run_all(self.folder, "sample", plots=["imbalance_ap", "imbalance_stability"],
                                selectors=["mi"], imbalances=["smote"])
            snapshots = {p.name: p.read_bytes() for p in first.iterdir()}
            second = viz.run_all(self.folder, "sample", plots=["imbalance_ap", "imbalance_stability"],
                                 selectors=["mi"], imbalances=["smote"])
        self.assertEqual(first, second)
        self.assertEqual(snapshots, {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertEqual(len(list(second.glob("*.png"))), 2)
        self.assertEqual(len(list(second.glob("*.pdf"))), 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.glob("*.json")})
        self.assertEqual(plt.get_fignums(), [])

    def test_imbalance_figures_have_legends_but_no_explanatory_footnotes(self):
        self.write_imbalance_bundle()
        with patch.object(viz, "save_figure"):
            for plot in (viz.plot_imbalance_ap, viz.plot_imbalance_stability):
                fig = plot(self.folder, "sample")[0]
                self.assertEqual(fig.texts, [fig._suptitle])
                self.assertEqual(len(fig.legends), 1)

    def write_threshold_bundle(self, grid=(0.2, 0.5, 0.8)):
        rows = []
        for record in self.tables["results"]:
            for threshold in grid:
                tp, fp = (2, 3) if threshold < 0.5 else ((1, 1) if threshold == 0.5 else (0, 0))
                row = {**record, "threshold": threshold, "tp": tp, "fp": fp,
                       "fn": 2 - tp, "tn": 8 - fp,
                       "recall": tp / 2, "precision": tp / (tp + fp) if tp + fp else 0}
                rows.append(row)
        self.tables["thresholds"] = rows
        self.tables["config"].pop("decision_threshold", None)
        self.tables["config"]["threshold_grid"] = list(grid)
        for record in self.tables["results"]:
            for key in ("recall", "precision", "tp", "fp", "fn", "tn"):
                record.pop(key, None)
        self.flush_tables()
        return rows

    def flush_tables(self):
        for name, data in self.tables.items():
            (self.folder / f"sample_{name}.json").write_text(json.dumps(data), encoding="utf-8")

    def test_vote_can_render_even_without_a_registered_style(self):
        for records in self.tables.values():
            if isinstance(records, list):
                for row in records:
                    if row.get("selector") == "mi":
                        row["selector"] = "vote"
        self.flush_tables()
        styles = {key: value for key, value in viz.SELECTOR_STYLES.items() if key != "vote"}
        with patch.dict(viz.SELECTOR_STYLES, styles, clear=True), patch.object(viz, "save_figure") as save:
            viz.run_all(self.folder, "sample")
        self.assertEqual(save.call_count, 5)
        self.assertEqual(plt.get_fignums(), [])

    def test_filters_reach_every_plot_and_table_and_preserve_ap_baseline(self):
        for records in self.tables.values():
            if isinstance(records, list):
                records.extend([{**row, "selector": "vote"} for row in records if row.get("selector") == "mi"])
                records.extend([{**row, "model": "rf"} for row in records if "model" in row])
                records.extend([{**row, "imbalance": "weight"} for row in records])
        for row in self.tables["results"]:
            row.update({"fit_time(s)": 0.2, "predict_time_per_observation(ms)": 0.01})
        self.tables["selections"] = [
            {key: row[key] for key in ("fold", "repeat", "imbalance", "selector")}
            | {"n_features_after_preprocessing": 10, "selection_time(s)": 0.1}
            for row in self.tables["results"] if row["model"] == "lr"
        ]
        self.write_threshold_bundle()
        before = {p.name: p.read_bytes() for p in self.folder.glob("*.json")}
        with patch.object(viz, "save_figure") as save:
            out = viz.run_all(self.folder, "sample", selectors=["mi"],
                              exclude_selectors=["none", "vote"], models=["rf"], imbalances=["weight"])
        for path in out.rglob("*.csv"):
            table = viz.pd.read_csv(path)
            for column, expected in (("selector", "mi"), ("model", "rf"), ("imbalance", "weight")):
                if column in table and not (column == "model" and set(table[column]) == {"all_models"}):
                    self.assertEqual(set(table[column]), {expected}, str(path))
        ap = viz.pd.read_csv(out / "sample_rq1_ap_difference_repeat_means.csv")
        np.testing.assert_allclose(ap.delta_ap, 0.1)
        stems = [call.args[2] for call in save.call_args_list]
        self.assertEqual(stems.count("sample_threshold_comparison_weight_rf"), 1)
        self.assertIn("sample_slowest_configs", stems)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.glob("*.json")})
        self.assertEqual(plt.get_fignums(), [])

    def test_plot_selection_does_not_read_unrequested_threshold_or_timing_files(self):
        (self.folder / "sample_thresholds.json").write_text("not JSON")
        (self.folder / "sample_selections.json").write_text("not JSON")
        with patch.object(viz, "save_figure") as save:
            out = viz.run_all(self.folder, "sample", plots=["stability"], selectors=["mi"])
        self.assertEqual([call.args[2] for call in save.call_args_list], ["sample_rq2a_stability_metrics"])
        self.assertEqual({p.name for p in out.glob("*.csv")}, {"sample_rq2a_stability_metrics.csv"})

    def test_custom_view_reruns_without_overwriting_other_views(self):
        destination = self.folder / "output"
        destination.mkdir()
        existing = destination / "sample_rq2a_stability_metrics.png"
        existing.write_bytes(b"original view")
        with contextlib.redirect_stdout(io.StringIO()):
            first = viz.run_all(self.folder, "sample", destination, plots=["stability"], selectors=["mi"])
            snapshots = {p.name: p.read_bytes() for p in first.iterdir()}
            second = viz.run_all(self.folder, "sample", destination, plots=["stability"], selectors=["mi"])
        self.assertEqual(first, second)
        self.assertEqual(snapshots, {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertEqual(existing.read_bytes(), b"original view")
        self.assertEqual(json.loads((first / "view_config.json").read_text())["selector"], ["mi"])

    def test_unavailable_and_empty_display_selections_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Available: none, mi"):
            viz.run_all(self.folder, "sample", selectors=["typo"])
        with self.assertRaisesRegex(viz.PlotUnavailable, "no records match"):
            viz.run_all(self.folder, "sample", exclude_selectors=["none", "mi"])
        with self.assertRaisesRegex(ValueError, "Choose plots"):
            viz.run_all(self.folder, "sample", plots=[])
        self.assertFalse((self.folder / "graph").exists())

    def test_cli_lists_options_and_passes_display_choices(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            viz.main(["--base-dir", str(self.folder), "--list-options"])
        self.assertIn("selector: none, mi", output.getvalue())
        self.assertFalse((self.folder / "graph").exists())
        with patch.object(viz, "run_all") as run:
            viz.main(["--base-dir", str(self.folder), "--plots", "stability", "heatmap",
                      "--selectors", "mi", "--exclude-selectors", "vote",
                      "--models", "lr", "--imbalances", "none"])
        self.assertEqual(run.call_args.kwargs, dict(plots=["stability", "heatmap"], selectors=["mi"],
                         exclude_selectors=["vote"], models=["lr"], imbalances=["none"]))

    def test_new_schema_only_renders_threshold_overviews(self):
        self.write_threshold_bundle()
        destination = self.folder / "output"
        with patch.object(viz, "save_figure") as save:
            viz.run_all(self.folder, "sample", destination)
        stems = [call.args[2] for call in save.call_args_list]
        for threshold in (0.2, 0.5, 0.8):
            for metric in ("recall", "precision"):
                self.assertNotIn(f"sample_rq1_{metric}_threshold_{threshold}", stems)
        self.assertEqual(list(destination.glob("sample_rq1_recall_precision*")), [])
        self.assertNotIn("sample_rq1_precision_recall_curve", stems)
        self.assertIn("sample_threshold_comparison_none_lr", stems)
        self.assertTrue((destination / "sample_threshold_comparison.csv").exists())
        self.assertEqual(plt.get_fignums(), [])

    def test_retired_figures_and_tables_are_removed_without_touching_other_outputs(self):
        self.write_threshold_bundle()
        destination = self.folder / "output"
        destination.mkdir()
        for threshold in (0.2, 0.5, 0.65, 0.8):
            for metric in ("recall", "precision"):
                for extension in ("png", "pdf"):
                    (destination / f"sample_rq1_{metric}_threshold_{threshold}.{extension}").write_bytes(b"obsolete")
            (destination / f"sample_rq1_recall_precision_threshold_{threshold}_repeat_means.csv").write_text("obsolete")
        for metric in ("recall", "precision"):
            for extension in ("png", "pdf"):
                (destination / f"sample_rq1_{metric}.{extension}").write_bytes(b"obsolete")
        (destination / "sample_rq1_recall_precision_repeat_means.csv").write_text("obsolete")
        unrelated = destination / "another_run_rq1_recall_threshold_0.2.png"
        unrelated.write_bytes(b"preserve")
        retired_pr = [destination / f"sample_rq1_precision_recall_curve{suffix}"
                      for suffix in (".png", ".pdf", ".csv", "_repeat_means.csv")]
        for path in retired_pr:
            path.write_bytes(b"obsolete")
        with patch.object(viz, "save_figure") as save:
            viz.run_all(self.folder, "sample", destination)
        stems = [call.args[2] for call in save.call_args_list]
        self.assertFalse(any(stem.startswith("sample_rq1_recall") for stem in stems))
        self.assertEqual(list(destination.glob("sample_rq1_recall*")), [])
        self.assertEqual(list(destination.glob("sample_rq1_precision_threshold_*")), [])
        self.assertFalse((destination / "sample_rq1_precision.png").exists())
        self.assertFalse((destination / "sample_rq1_precision.pdf").exists())
        self.assertEqual(unrelated.read_bytes(), b"preserve")
        self.assertFalse(any(path.exists() for path in retired_pr))

    def test_threshold_metrics_and_grid_validation(self):
        rows = self.write_threshold_bundle()
        data, _ = viz.load_threshold_results(self.folder, "sample")
        self.assertTrue(data.loc[data.threshold.eq(0.8), "precision"].eq(0).all())
        rows[0]["recall"] = 0.123
        self.flush_tables()
        with self.assertRaisesRegex(ValueError, "confusion counts"):
            viz.load_threshold_results(self.folder, "sample")
        rows[0]["recall"] = 1.0
        self.tables["thresholds"] = rows[:-1]
        self.flush_tables()
        with self.assertRaisesRegex(ValueError, "identical splits"):
            viz.load_threshold_results(self.folder, "sample")

    def test_missing_scores_preserve_coverage_without_partial_repeat_mean(self):
        rows = self.write_threshold_bundle()
        for row in rows:
            if row["selector"] == "mi" and row["repeat"] == 1 and row["fold"] == 1:
                for key in ("recall", "precision", "tp", "fp", "fn", "tn"):
                    row[key] = None
        self.flush_tables()
        data, config = viz.load_threshold_results(self.folder, "sample")
        repeats = viz.threshold_repeat_means(data)
        partial = repeats.loc[repeats.selector.eq("mi") & repeats.repeat.eq(1)]
        self.assertTrue(partial.recall.isna().all())
        self.assertTrue(partial.n_valid_folds.eq(1).all())
        with patch.object(viz, "save_figure"):
            summary = viz.plot_threshold_sweep(data, config, self.folder, "sample")
        self.assertTrue(summary.loc[summary.selector.eq("mi"), "recall_mean"].isna().all())

    def test_all_missing_and_single_threshold_are_supported(self):
        rows = self.write_threshold_bundle(grid=(0.35,))
        for row in rows:
            for key in ("recall", "precision", "tp", "fp", "fn", "tn"):
                row[key] = None
        self.flush_tables()
        data, config = viz.load_threshold_results(self.folder, "sample")
        repeats, summary = viz.summarize_threshold_scores(data, config)
        self.assertTrue(summary.recall_mean.isna().all())
        self.assertTrue(repeats.recall.isna().all())

    def test_threshold_sweep_reproduces_actual_image_files(self):
        self.write_threshold_bundle()
        data, config = viz.load_threshold_results(self.folder, "sample")
        viz.plot_threshold_sweep(data, config, self.folder, "sample")
        paths = list(self.folder.glob("sample_threshold_comparison*"))
        first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        viz.plot_threshold_sweep(data, config, self.folder, "sample")
        self.assertEqual(first, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        self.assertEqual(plt.get_fignums(), [])


    def test_new_schema_full_rerun_is_deterministic(self):
        self.write_threshold_bundle(grid=(0.2, 0.8))
        destination = self.folder / "new_output"
        def fingerprints():
            return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in destination.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            viz.run_all(self.folder, "sample", destination)
            first = fingerprints()
            viz.run_all(self.folder, "sample", destination)
        self.assertEqual(first, fingerprints())
        self.assertEqual(sum(name.endswith(".png") for name in first), 6)
        self.assertEqual(sum(name.endswith(".pdf") for name in first), 6)
        self.assertEqual(plt.get_fignums(), [])

    def test_latest_bundle_ignores_incomplete_run(self):
        write_bundle(self.folder, "20261001")
        write_bundle(self.folder, "20261002")
        (self.folder / "20990101_config.json").write_text("{}")
        # The timestamp ordering is lexical, so use only timestamped candidates here.
        for path in self.folder.glob("sample_*.json"):
            path.unlink()
        self.assertEqual(viz.resolve_run_id(self.folder), "20261002")

    def test_paired_ap_and_numeric_frequency_tie_order(self):
        with patch.object(viz, "save_figure"):
            _, _, repeats, paired = viz.plot_ap_difference(self.folder, "sample")
            _, _, top = viz.plot_rq2b_top_feature_frequency(self.folder, "sample", top_n=2)
        np.testing.assert_allclose(paired.delta_ap, 0.1)
        np.testing.assert_allclose(repeats.delta_ap, 0.1)
        self.assertEqual(top.feature.tolist(), ["f1", "f2"])
        self.assertEqual(top.cutoff_tie_total.tolist(), [3, 3])

    def test_duplicate_splits_are_rejected(self):
        records = self.tables["results"]
        (self.folder / "sample_results.json").write_text(json.dumps(records + records[:1]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            viz.plot_ap_difference(self.folder, "sample")

    def test_heatmap_keeps_missing_values_and_only_present_selectors(self):
        with patch.object(viz, "save_figure"):
            fig, axes, _ = viz.plot_rq3_main_tradeoff(self.folder, "sample")
            _, _, stability = viz.plot_rq2a_nogueira_stability(self.folder, "sample")
        self.assertTrue(stability.kuncheva.isna().all())
        self.assertEqual(axes[0].images[0].get_array().shape, (2, 1))
        self.assertFalse(any("FDR" in t.get_text() for t in axes[0].get_yticklabels()))
        np.testing.assert_allclose(axes[0].images[0].get_array()[:, 0], [0.2, 0.3])

    def test_repeated_execution_has_identical_outputs_and_no_open_figures(self):
        destination = self.folder / "output"
        def fingerprints():
            return {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in destination.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            viz.run_all(self.folder, "sample", destination, top_n=2)
            first = fingerprints()
            viz.run_all(self.folder, "sample", destination, top_n=2)
        self.assertEqual(first, fingerprints())
        self.assertEqual(len([p for p in first if p.endswith(".png")]), 5)
        self.assertEqual(len([p for p in first if p.endswith(".pdf")]), 5)
        self.assertIn("Skipped timing analysis", output.getvalue())
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
