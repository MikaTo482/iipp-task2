"""Development-only RQ5 preparation without fitting any predictive models."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "06_exp_2.py"
spec = importlib.util.spec_from_file_location("rq5_exp2", SCRIPT)
rq5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq5)


class RQ5InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("secom", "splits/random", "experiments"):
            (self.root / folder).mkdir(parents=True)
        data = pd.DataFrame({0: range(16), 1: np.arange(16, dtype=float) + 0.5})
        data.loc[0, 1] = np.nan
        data.to_csv(self.root / "secom/secom.data", sep=" ", header=False, index=False, na_rep="NaN")
        labels = pd.DataFrame({"label": [-1, 1] * 8, "timestamp": ["19/07/2008 11:55:00"] * 16})
        labels.to_csv(self.root / "secom/secom_labels.data", sep=" ", header=False, index=False)
        self.dev = [9, 0, 11, 2, 13, 4, 15, 6, 1, 8, 3, 10]
        self.test = sorted(set(range(16)) - set(self.dev))
        self.write_indices("dev", self.dev)
        self.write_indices("test", self.test)
        self.write_json("splits/random/split_config.json", dict(
            split_type="random", n_total=16, n_dev=12, n_test=4, random_seed=4))
        self.config = dict(run_id="sample", random_seed=17, algorithm_seed=99,
                           n_splits=2, n_repeats=2, n_features_original=2,
                           selectors=["none", "mi"], imbalance=["none", "weight", "smote"],
                           models=["lr", "rf"], selector_params={"mi": {"threshold": "mean_score"}},
                           model_params={"lr": {"C": 0.7}, "rf": {"n_estimators": 7}},
                           missing_threshold=50, correlation_threshold=0.8, scaler="RobustScaler",
                           learner_weighting={"condition": "weight", "lr": "balanced", "rf": "balanced"})
        self.tables = {name: [] for name in rq5.TABLE_KEYS}
        for fold in range(1, 5):
            for imbalance in self.config["imbalance"]:
                for selector in self.config["selectors"]:
                    features = ["f0", "f1"] if selector == "none" else ["f0"]
                    row = dict(fold=fold, repeat=(fold - 1) // 2 + 1, imbalance=imbalance, selector=selector)
                    self.tables["selections"].append(dict(
                        **row, features=features, candidate_features=["f0", "f1"],
                        n_features_selected=len(features), n_features_after_preprocessing=2))
                    for model in self.config["models"]:
                        self.tables["results"].append(dict(**row, model=model, ap=0.2, n_features=len(features)))
        for imbalance in self.config["imbalance"]:
            for selector in self.config["selectors"]:
                self.tables["stability"].append(dict(imbalance=imbalance, selector=selector, nogueira=0.5))
                for feature in ("f0", "f1"):
                    self.tables["feature_frequency"].append(dict(
                        imbalance=imbalance, selector=selector, feature=feature,
                        selection_frequency=float(feature == "f0" or selector == "none")))
        self.flush()

    def write_json(self, relative, value):
        (self.root / relative).write_text(json.dumps(value), encoding="utf-8")

    def write_indices(self, name, values):
        pd.DataFrame({"index": values}).to_csv(
            self.root / f"splits/random/{name}_indices_random.csv", index=False)

    def flush(self):
        self.write_json("experiments/sample_config.json", self.config)
        for name, table in self.tables.items():
            self.write_json(f"experiments/sample_{name}.json", table)

    def load(self):
        return rq5.load_rq5_inputs("sample", project_dir=self.root)

    def test_load_preserves_development_order_labels_raw_values_and_config(self):
        inputs = self.load()
        self.assertEqual(inputs.samples.source_index.tolist(), self.dev)
        self.assertEqual(inputs.X.f0.tolist(), self.dev)
        self.assertEqual(inputs.X.columns.tolist(), ["f0", "f1"])
        self.assertEqual(inputs.y.tolist(), [int(i % 2 == 1) for i in self.dev])
        self.assertTrue(pd.isna(inputs.X.iloc[1].f1))
        self.assertFalse(set(inputs.samples.source_index) & set(self.test))
        self.assertEqual(inputs.config, self.config)
        self.assertFalse(inputs.manifest["historical_data_identity_verified"])
        self.assertFalse(inputs.manifest["protected_test_used_for_analysis"])

    def test_full_development_is_returned_without_creating_cv(self):
        with patch("sklearn.model_selection.RepeatedStratifiedKFold", side_effect=AssertionError("No new CV")):
            inputs = self.load()
        development = inputs.get_development()
        self.assertEqual(set(development), {"X", "y", "samples"})
        self.assertEqual(development["samples"].source_index.tolist(), self.dev)
        pd.testing.assert_frame_equal(development["X"], inputs.X)
        pd.testing.assert_series_equal(development["y"], inputs.y)
        self.assertFalse(set(development["samples"].source_index) & set(self.test))
        self.assertEqual(inputs.manifest["analysis_scope"], "full_development")
        self.assertFalse(inputs.manifest["new_cross_validation"])
        self.assertFalse(hasattr(inputs, "folds"))
        self.assertEqual(inputs.manifest["source_cv"]["role"], "historical_reference_only")
        self.assertEqual(set(inputs.selections.fold), {1, 2, 3, 4})
        development["X"].iloc[0, 0] = -999
        self.assertNotEqual(inputs.X.iloc[0, 0], -999)

    def test_snapshot_rerun_is_identical_and_source_is_not_modified(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        first = rq5.save_input_bundle(self.load(), self.root / "output")
        snapshot = {p.name: p.read_bytes() for p in first.iterdir()}
        second = rq5.save_input_bundle(self.load(), self.root / "output")
        self.assertEqual(first, second)
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        development = pd.read_csv(first / "development.csv")
        self.assertEqual(development.source_index.tolist(), self.dev)
        self.assertEqual(len(snapshot), 8)
        self.assertFalse((second / "cv_folds.json").exists())

    def test_changed_inputs_create_new_snapshot_without_overwriting_old(self):
        first = rq5.save_input_bundle(self.load(), self.root / "output")
        self.config["algorithm_seed"] = 7
        self.flush()
        second = rq5.save_input_bundle(self.load(), self.root / "output")
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads((first / "config.json").read_text())["algorithm_seed"], 99)

    def test_overlap_and_duplicate_indices_are_rejected(self):
        self.write_indices("test", [self.dev[0], *self.test[1:]])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.load()
        self.write_indices("test", self.test)
        self.write_indices("dev", [self.dev[0], *self.dev[:-1]])
        with self.assertRaisesRegex(ValueError, "unique integers"):
            self.load()

    def test_missing_and_duplicate_fold_records_are_rejected(self):
        row = self.tables["results"].pop()
        self.flush()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.load()
        self.tables["results"].extend([row, row])
        self.flush()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_feature_and_result_counts_are_validated(self):
        self.tables["selections"][0]["features"] = ["f_unknown"]
        self.flush()
        with self.assertRaisesRegex(ValueError, "invalid features"):
            self.load()
        self.tables["selections"][0]["features"] = ["f0", "f1"]
        self.tables["results"][0]["n_features"] = 1
        self.flush()
        with self.assertRaisesRegex(ValueError, "disagree on feature counts"):
            self.load()

    def test_missing_scores_remain_missing_and_old_weighting_is_not_invented(self):
        self.tables["results"][0]["ap"] = None
        self.config.pop("learner_weighting")
        self.flush()
        inputs = self.load()
        self.assertTrue(pd.isna(inputs.results.iloc[0].ap))
        self.assertNotIn("learner_weighting", inputs.config)
        self.assertTrue(any("no learner_weighting" in note for note in inputs.manifest["limitations"]))

    def test_invalid_config_and_labels_are_rejected(self):
        self.config["n_splits"] = 1
        self.flush()
        with self.assertRaisesRegex(ValueError, "n_splits"):
            self.load()
        self.config["n_splits"] = 2
        self.flush()
        pd.DataFrame({"label": [0] * 16, "timestamp": ["date"] * 16}).to_csv(
            self.root / "secom/secom_labels.data", sep=" ", index=False, header=False)
        with self.assertRaisesRegex(ValueError, "Development labels"):
            self.load()

    def test_cli_and_import_do_not_start_experiment(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rq5.main(["--project-dir", str(self.root), "--list-runs"])
        self.assertEqual(output.getvalue().strip(), "sample")
        self.assertFalse((self.root / "experiments/rq5_inputs").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            rq5.main(["--project-dir", str(self.root), "--run-id", "sample"])
        self.assertEqual(len(list((self.root / "experiments/rq5_inputs/sample").glob("*/manifest.json"))), 1)
        path = Path(rq5.__file__).with_name("06_exp_2.py")
        spec = importlib.util.spec_from_file_location("exp2_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.object(pd, "read_csv", side_effect=AssertionError("Unexpected input read")):
            spec.loader.exec_module(module)
        self.assertEqual(module.load_rq5_inputs.__module__, "exp2_test")
        self.assertFalse(path.with_name("rq5_inputs.py").exists())

    def test_missing_run_and_path_traversal_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            rq5.load_rq5_inputs("absent", project_dir=self.root)
        with self.assertRaisesRegex(ValueError, "filename prefix"):
            rq5.load_rq5_inputs("../sample", project_dir=self.root)


if __name__ == "__main__":
    unittest.main()
