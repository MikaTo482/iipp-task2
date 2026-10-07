"""SHAP regression checks using real small classifiers and an exhaustive oracle.

Load the producer's actual function bodies via AST to avoid training-only CTGAN
imports. No research data is trained on or changed by these tests.
"""
import ast
import contextlib
from datetime import datetime
import io
from itertools import combinations
import json
from math import factorial
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_track_evaluation.py"
TREE = ast.parse(SOURCE.read_text())


def producer_namespace():
    ns = dict(globals())
    nodes = [node for node in TREE.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def fixture(n_features=2):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(30, n_features)),
                     columns=[f"f{i}" for i in range(n_features)])
    y = (X.f0 + .6 * X.f1 > .8).astype(int)
    return X, y, X.iloc[[2, 7, 11]].copy()


def exhaustive_shap(model, background, explain):
    """Direct Shapley definition: all coalitions, empirical background average."""
    d = explain.shape[1]
    positive = list(model.classes_).index(1)
    output = np.zeros(explain.shape)
    for row_index, (_, row) in enumerate(explain.iterrows()):
        coalition_values = {}
        for size in range(d + 1):
            for coalition in combinations(range(d), size):
                masked = background.copy()
                for index in coalition:
                    masked.iloc[:, index] = row.iloc[index]
                coalition_values[frozenset(coalition)] = model.predict_proba(masked)[:, positive].mean()
        for feature in range(d):
            others = [index for index in range(d) if index != feature]
            for size in range(d):
                weight = factorial(size) * factorial(d - size - 1) / factorial(d)
                for coalition in combinations(others, size):
                    subset = frozenset(coalition)
                    output[row_index, feature] += weight * (
                        coalition_values[subset | {feature}] - coalition_values[subset])
    return output


class ShapTests(unittest.TestCase):
    def setUp(self):
        self.ns = producer_namespace()
        self.X, self.y, self.explain = fixture()
        self.background = self.X.iloc[:8]

    def models(self):
        return {"knn": KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree", n_jobs=1),
                "nb": GaussianNB()}

    def test_both_models_match_exhaustive_two_feature_shap(self):
        # Forward/reverse sampling is exact with two features, even with one cycle.
        self.ns["SHAP_PERMUTATION_CYCLES"] = 1
        for name, model in self.models().items():
            with self.subTest(model=name):
                model.fit(self.X, self.y)
                values, space = self.ns["compute_shap"](
                    model, name, self.background, self.explain, 42)
                expected = exhaustive_shap(model, self.background, self.explain)
                np.testing.assert_allclose(values, expected, atol=1e-10)
                self.assertEqual(space, "probability")

    def test_three_feature_estimates_approach_exhaustive_values(self):
        X, y, explain = fixture(3)
        background = X.iloc[:8]
        self.ns["SHAP_PERMUTATION_CYCLES"] = 400
        for name, model in self.models().items():
            with self.subTest(model=name):
                model.fit(X, y)
                values, _ = self.ns["compute_shap"](model, name, background, explain, 42)
                np.testing.assert_allclose(values, exhaustive_shap(model, background, explain),
                                           atol=.01, rtol=0)

    def test_positive_class_is_found_by_label_not_column_position(self):
        fitted = GaussianNB().fit(self.X, self.y)
        reverse = SimpleNamespace(classes_=np.array([1, 0]),
                                  feature_names_in_=fitted.feature_names_in_,
                                  predict_proba=lambda X: fitted.predict_proba(X)[:, ::-1])
        values = self.ns["probability_shap"](reverse, self.background, self.explain, 42)
        np.testing.assert_allclose(values, exhaustive_shap(fitted, self.background, self.explain),
                                   atol=1e-10)

    def test_seed_is_reproducible_and_global_rng_is_unchanged(self):
        X, y, explain = fixture(3)
        model = GaussianNB().fit(X, y)
        self.ns["SHAP_MAX_BACKGROUND"] = 8
        self.ns["SHAP_PERMUTATION_CYCLES"] = 3
        before = np.random.get_state()
        first = self.ns["probability_shap"](model, X, explain, 42)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        second = self.ns["probability_shap"](model, X, explain, 42)
        np.testing.assert_array_equal(first, second)
        sampled = X.sample(8, random_state=42)
        base = model.predict_proba(sampled)[:, 1].mean()
        np.testing.assert_allclose(base + first.sum(axis=1), model.predict_proba(explain)[:, 1])

    def test_feature_order_nonfinite_inputs_and_wrong_labels_are_rejected(self):
        model = GaussianNB().fit(self.X, self.y)
        bad_nan = self.explain.copy()
        bad_nan.iloc[0, 0] = np.nan
        cases = [self.explain.iloc[:, ::-1], bad_nan, self.explain.iloc[:0]]
        for data in cases:
            with self.subTest(columns=list(data), rows=len(data)), self.assertRaises(ValueError):
                self.ns["probability_shap"](model, self.background, data, 42)
        with self.assertRaisesRegex(ValueError, "fitted model"):
            self.ns["probability_shap"](model, self.background.iloc[:, ::-1],
                                        self.explain.iloc[:, ::-1], 42)
        model.classes_ = np.array([0, 2])
        with self.assertRaisesRegex(ValueError, "Fail labelled 1"):
            self.ns["probability_shap"](model, self.background, self.explain, 42)

    def test_corrupt_explanations_are_rejected(self):
        model = GaussianNB().fit(self.X, self.y)
        correct_base = model.predict_proba(self.background)[:, 1].mean()
        for value, base, message in [
                (np.zeros(self.explain.shape), np.zeros(len(self.explain)), "base values"),
                (np.zeros(self.explain.shape), np.full(len(self.explain), correct_base), "additivity"),
                (np.full(self.explain.shape, np.nan), np.full(len(self.explain), correct_base), "non-finite")]:
            result = SimpleNamespace(values=value, base_values=base)
            with self.subTest(message=message), patch.object(shap, "PermutationExplainer") as fake:
                fake.return_value.return_value = result
                with self.assertRaisesRegex(ValueError, message):
                    self.ns["probability_shap"](model, self.background, self.explain, 42)

    def test_high_dimensional_budget_scales_and_retains_all_features(self):
        X = pd.DataFrame(np.zeros((2, 331)), columns=[f"f{i}" for i in range(331)])
        model = SimpleNamespace(classes_=np.array([0, 1]),
                                predict_proba=lambda data: np.tile([.8, .2], (len(data), 1)))
        result = SimpleNamespace(values=np.zeros(X.shape), base_values=np.full(len(X), .2))
        with patch.object(shap, "PermutationExplainer") as fake:
            fake.return_value.return_value = result
            records, status = self.ns["compute_shap_for_fit"](model, "knn", X, X, 42, fold=1)
            self.assertEqual(fake.return_value.call_args.kwargs["max_evals"], 20 * (2 * 331 + 1))
        self.assertEqual(len(records), 331)
        self.assertEqual(status["status"], "ok")

    def test_constant_feature_has_zero_attribution(self):
        X = self.X.assign(constant=3.)
        model = GaussianNB().fit(X, self.y)
        values = self.ns["probability_shap"](model, X.iloc[:8], X.iloc[[2, 7]], 42)
        np.testing.assert_array_equal(values[:, -1], 0.)

    def test_success_records_use_existing_summary_fields_and_audit_metadata(self):
        model = GaussianNB().fit(self.X, self.y)
        records, status = self.ns["compute_shap_for_fit"](
            model, "nb", self.background, self.explain, 42,
            fold=2, repeat=1, selector="mi", imbalance="none")
        expected = exhaustive_shap(model, self.background, self.explain)
        self.assertEqual([row["feature"] for row in records], list(self.X.columns))
        np.testing.assert_allclose([r["mean_abs_shap"] for r in records], abs(expected).mean(axis=0))
        np.testing.assert_allclose([r["mean_shap"] for r in records], expected.mean(axis=0))
        np.testing.assert_array_equal([r["shap_rank"] for r in records],
                                      pd.Series(abs(expected).mean(axis=0)).rank(ascending=False))
        self.assertEqual(status["status"], "ok")
        self.assertTrue(status["additivity_checked"])
        self.assertTrue(status["approximate"])
        self.assertEqual(status["output_space"], "probability")
        self.assertEqual(status["explained_class"], 1)
        self.assertEqual(status["fold"], 2)

    def test_failure_has_reason_and_no_fabricated_shap_rows(self):
        def broken(*args, **kwargs):
            raise ValueError("intentional explainer failure")
        with patch.dict(self.ns, compute_shap=broken), self.assertWarnsRegex(RuntimeWarning, "SHAP failed"):
            records, status = self.ns["compute_shap_for_fit"](
                None, "nb", self.background, self.explain, 42, fold=1)
        self.assertEqual(records, [])
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_type"], "ValueError")
        self.assertIn("intentional", status["error"])
        self.assertFalse(status["additivity_checked"])

    def test_existing_linear_and_tree_paths_still_work(self):
        for name, model, space in [
                ("lr", LogisticRegression(), "log_odds"),
                ("rf", RandomForestClassifier(n_estimators=5, random_state=42), "probability")]:
            with self.subTest(model=name):
                model.fit(self.X, self.y)
                values, actual_space = self.ns["compute_shap"](
                    model, name, self.background, self.explain, 42)
                self.assertEqual(values.shape, self.explain.shape)
                self.assertTrue(np.isfinite(values).all())
                self.assertEqual(actual_space, space)

    def test_actual_main_loop_saves_knn_and_nb_for_every_fold_and_condition(self):
        data = self.X.rename(columns={"f0": "0", "f1": "1"}).assign(label=self.y, timestamp=0)
        main = next(node for node in TREE.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        with tempfile.TemporaryDirectory() as directory:
            self.ns.update(
                RESULTS_DIR=directory, N_SPLITS=2, N_REPEATS=1,
                MODELS=["knn", "nb"], SELECTORS=["none"], IMBALANCE=["none", "weight"],
                SHAP_PERMUTATION_CYCLES=1,
                load_secom_data=lambda: data,
                load_protected_split=lambda *args: (np.arange(28), np.arange(28, 30)),
                preprocess_fold=lambda train, val: (train.copy(), val.copy()))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=main.body, type_ignores=[]), str(SOURCE), "exec"), self.ns)
            def read(suffix):
                return json.loads(next(Path(directory).glob(f"*_{suffix}.json")).read_text())
            statuses, records, results = read("shap_status"), read("shap"), read("results")
            self.assertEqual(len(statuses), 8)
            self.assertEqual(len(results), 8)
            self.assertEqual(len(records), 16)
            self.assertTrue(all(row["status"] == "ok" for row in statuses))
            self.assertTrue(all(row["additivity_checked"] for row in statuses))
            keys = lambda rows: {(r["fold"], r["imbalance"], r["selector"], r["model"]) for r in rows}
            self.assertEqual(keys(results), keys(statuses))
            self.assertEqual(keys(results), keys(records))
            config = read("config")
            self.assertEqual(config["shap_model_agnostic_background"], "training_fold_before_resampling")
            self.assertEqual(config["shap_explain_data"], "validation_fold")


if __name__ == "__main__":
    unittest.main()
