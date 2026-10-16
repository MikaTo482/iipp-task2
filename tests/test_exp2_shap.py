import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp2_shap", ROOT / "06_exp_2.py")
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


class ExperimentTwoTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.X = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"f{i}" for i in range(4)])
        self.y = np.zeros(50, dtype=int)
        self.y[np.argsort(self.X.f0.iloc[:40])[-10:]] = 1
        self.y[40:43] = 1

    def fixture(self, root):
        (root / "secom").mkdir()
        (root / "splits/random").mkdir(parents=True)
        self.X.to_csv(root / "secom/secom.data", sep=" ", index=False, header=False)
        pd.DataFrame({"label": np.where(self.y == 1, 1, -1), "timestamp": ["01/01/2020 00:00:00"] * 50}).to_csv(
            root / "secom/secom_labels.data", sep=" ", index=False, header=False)
        for name, indices in (("dev", range(40)), ("test", range(40, 50))):
            pd.DataFrame({"index": indices}).to_csv(root / f"splits/random/{name}_indices_random.csv", index=False)
        exp.save_json(root / "splits/random/split_config.json", {
            "split_type": "random", "random_seed": 42, "created_at": "fixture",
            "n_total": 50, "n_dev": 40, "n_test": 10,
        })

    def run_fixture(self, root, **kwargs):
        return exp.run_experiment(project_dir=root, selector_name="none", missing_threshold=50,
                                  background_size=10, n_jobs=1, **kwargs)

    def test_metrics_use_probabilities_and_fail_positive(self):
        y, p = [0, 1, 1, 0], [.1, .5, .4, .7]
        metrics = exp.prediction_metrics(y, p, .5)
        self.assertEqual(metrics["ap"], average_precision_score(y, p))
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["precision"], .5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["ap"], exp.prediction_metrics(y, p, .8)["ap"])
        self.assertEqual(exp.prediction_metrics(y, p, .8)["precision"], 0)
        self.assertIsNone(exp.prediction_metrics([0, 0], [.2, .3])["ap"])
        with self.assertRaises(ValueError):
            exp.prediction_metrics(y, [.1, np.nan, .4, .7])

    def test_positive_probability_uses_class_identity(self):
        class ReverseModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return np.array([[.8, .2]])

        np.testing.assert_equal(exp.positive_probabilities(ReverseModel(), None), [.8])

    def test_shap_all_models_and_single_observation(self):
        small_params = {"lr": {"max_iter": 1000}, "rf": {"n_estimators": 8, "max_depth": 3},
                        "xgb": {"n_estimators": 8, "max_depth": 2, "tree_method": "hist"}}
        with patch.dict(exp.MODEL_PARAMS, small_params):
            for name in small_params:
                with self.subTest(model=name):
                    model = exp.build_model(name, 42, n_jobs=1).fit(self.X, self.y)
                    importance, status, arrays = exp.compute_shap(model, name, self.X, self.X, self.y, background_size=10)
                    self.assertEqual(status["status"], "success")
                    self.assertEqual(status["output_units"], "probability" if name == "rf" else "log_odds")
                    np.testing.assert_allclose(arrays["base_values"] + arrays["values"].sum(axis=1),
                                               arrays["model_output"], rtol=1e-4, atol=1e-5)
                    self.assertEqual(len(importance), 4)
                    one = exp.compute_shap(model, name, self.X.iloc[:1], self.X, self.y[:1], background_size=10)
                    self.assertEqual(one[2]["values"].shape, (1, 4))
                    if name == "lr":
                        expected = (self.X.to_numpy() - arrays["background_X"].mean(axis=0)) * model.coef_[0]
                        np.testing.assert_allclose(arrays["values"], expected)

    def test_background_size_not_silently_capped(self):
        X = pd.concat([self.X] * 3, ignore_index=True)
        y = np.tile(self.y, 3)
        model = exp.build_model("lr", 42).fit(X, y)
        _, status, _ = exp.compute_shap(model, "lr", X, X, y, background_size=110)
        self.assertEqual(status["n_background"], 110)

    def test_single_feature_shap_shapes(self):
        X = self.X[["f0"]]
        for name, params in (("lr", {"max_iter": 1000}), ("rf", {"n_estimators": 5}),
                             ("xgb", {"n_estimators": 5, "max_depth": 2})):
            with self.subTest(model=name), patch.dict(exp.MODEL_PARAMS, {name: params}):
                model = exp.build_model(name, 42, n_jobs=1).fit(X, self.y)
                _, _, arrays = exp.compute_shap(model, name, X.iloc[:1], X, self.y[:1], background_size=10)
                self.assertEqual(arrays["values"].shape, (1, 1))

    def test_weighted_tree_runs_save_serializable_results(self):
        for name in ("rf", "xgb"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                with patch.dict(exp.MODEL_PARAMS, {name: {"n_estimators": 5, "max_depth": 2}}):
                    out = self.run_fixture(root, model_name=name, imbalance="weight")
                self.assertEqual(json.loads((out / "shap_status.json").read_text())[0]["status"], "success")
                config = json.loads((out / "config.json").read_text())
                self.assertTrue(config["use_class_weight"])
                params = config["fitted_model_params"]
                if name == "xgb":
                    self.assertEqual(params["scale_pos_weight"], 3)
                else:
                    self.assertEqual(params["class_weight"], "balanced")

    def test_rerun_reproduces_values_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            first, second = self.run_fixture(root), self.run_fixture(root)
            self.assertNotEqual(first, second)
            a = np.load(first / "development_shap_values.npz", allow_pickle=False)
            b = np.load(second / "development_shap_values.npz", allow_pickle=False)
            np.testing.assert_array_equal(a["values"], b["values"])
            np.testing.assert_array_equal(a["source_indices"], np.arange(40))
            self.assertFalse(list(first.glob("protected_test*")))
            results = json.loads((first / "results.json").read_text())
            self.assertEqual(results[0]["evaluation_scope"], "in_sample")
            self.assertEqual(results[0]["n_observations"], 40)
            self.assertTrue((first / "fitted_pipeline.joblib").is_file())

    def test_smote_explains_only_original_development(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            out = self.run_fixture(root, imbalance="smote")
            selections = json.loads((out / "selections.json").read_text())[0]
            self.assertGreater(selections["n_training_after_resampling"], 40)
            data = np.load(out / "development_shap_values.npz", allow_pickle=False)
            self.assertEqual(data["values"].shape[0], 40)
            self.assertTrue(np.all(data["background_source_indices"] < 40))

    def test_test_evaluation_requires_opt_in_and_does_not_change_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            base = self.run_fixture(root)
            opted = self.run_fixture(root, evaluate_test=True)
            a = np.load(base / "development_shap_values.npz")
            b = np.load(opted / "development_shap_values.npz")
            np.testing.assert_array_equal(a["values"], b["values"])
            test = np.load(opted / "protected_test_shap_values.npz")
            np.testing.assert_array_equal(test["source_indices"], np.arange(40, 50))
            self.assertTrue(np.all(test["background_source_indices"] < 40))

    def test_shap_failure_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(exp, "compute_shap", side_effect=RuntimeError("test failure")):
                out = self.run_fixture(root)
            self.assertTrue((out / "results.csv").is_file())
            self.assertEqual(json.loads((out / "shap_status.json").read_text())[0]["status"], "failed")
            self.assertEqual(json.loads((out / "shap_importance.json").read_text()), [])
            self.assertFalse((out / "development_shap_values.npz").exists())

    def test_empty_selection_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(exp, "select_features", return_value=[]):
                out = exp.run_experiment(project_dir=root, selector_name="mi", missing_threshold=50, n_jobs=1)
            self.assertEqual(json.loads((out / "results.json").read_text())[0]["status"], "no_features")
            self.assertFalse((out / "fitted_pipeline.joblib").exists())

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            pd.DataFrame({"index": range(39, 49)}).to_csv(root / "splits/random/test_indices_random.csv", index=False)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                exp.load_development_data(root)


if __name__ == "__main__":
    unittest.main()
