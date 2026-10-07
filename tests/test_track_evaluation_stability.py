"""Stability regression checks without loading data or fitting ML models.

Execute the actual producer's pure functions and final JSON-writing block via
AST so these tests do not require the training-only CTGAN/SHAP dependencies.
"""
import ast
import contextlib
from datetime import datetime
import io
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_track_evaluation.py"
TREE = ast.parse(SOURCE.read_text())
PURE = {"to_matrix_from_names", "selection_frequency", "nogueira_stability",
        "kuncheva_index", "mean_jaccard", "summarize_feature_stability"}


def producer_functions():
    ns = {"np": np, "pd": pd, "combinations": combinations}
    body = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in PURE]
    assert {node.name for node in body} == PURE
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def toy_selections():
    return [dict(repeat=1, fold=1, imbalance="none", selector="mi",
                 candidate_features=["a", "b"], n_features_after_preprocessing=2,
                 features=["a"], n_features_selected=1),
            dict(repeat=1, fold=2, imbalance="none", selector="mi",
                 candidate_features=["a", "b", "c"], n_features_after_preprocessing=3,
                 features=["b"], n_features_selected=1)]


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.fn = producer_functions()

    def test_fixed_k_known_value_and_kuncheva_agreement(self):
        z = np.array([[1, 0, 0], [0, 1, 0]])
        self.assertAlmostEqual(self.fn["nogueira_stability"](z), -.5)
        self.assertAlmostEqual(self.fn["kuncheva_index"]([["a"], ["b"]], 3), -.5)

    def test_identical_nontrivial_sets_have_stability_one(self):
        self.assertEqual(self.fn["nogueira_stability"](np.array([[1, 0, 1]] * 5)), 1.)
        self.assertEqual(self.fn["kuncheva_index"]([["a"], ["a"]], 3), 1.)

    def test_varying_subset_sizes_and_finite_sample_bound(self):
        z = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        value = self.fn["nogueira_stability"](z)
        self.assertAlmostEqual(value, -.35)
        self.assertGreaterEqual(value, -1 / (len(z) - 1))
        with self.assertRaises(ValueError):
            self.fn["kuncheva_index"]([["a"], ["a", "b"]], 3)

    def test_unavailable_estimates_remain_nan(self):
        for z in [np.array([[1, 0]]), np.zeros((2, 3)), np.ones((2, 3)), np.zeros((2, 0))]:
            self.assertTrue(np.isnan(self.fn["nogueira_stability"](z)))
        self.assertTrue(np.isnan(self.fn["kuncheva_index"]([["a"]], 3)))

    def test_invalid_selection_matrices_are_rejected(self):
        for z in [np.array([0, 1]), np.array([[0, 2]]), np.array([[0, np.nan]])]:
            with self.assertRaises(ValueError):
                self.fn["nogueira_stability"](z)

    def test_external_dimension_cannot_override_matrix_or_universe(self):
        with self.assertRaises(TypeError):
            self.fn["nogueira_stability"](np.array([[1, 0], [0, 1]]), d=1.5)
        with self.assertRaises(TypeError):
            self.fn["summarize_feature_stability"](pd.DataFrame(toy_selections()), ["a", "b", "c"], 2.5)
        for bad_p in [227.14, 0, -2, True]:
            with self.assertRaises(ValueError):
                self.fn["kuncheva_index"]([["a"], ["b"]], bad_p)

    def test_duplicate_or_unknown_names_are_not_silently_dropped(self):
        make = self.fn["to_matrix_from_names"]
        for subsets, universe in [([["a"]], ["a", "a"]), ([["a", "a"]], ["a"]),
                                   ([["b"]], ["a"])]:
            with self.assertRaises(ValueError):
                make(subsets, universe)

    def test_candidate_union_includes_never_selected_features(self):
        rows = pd.DataFrame(toy_selections())
        universe = sorted(set().union(*rows.candidate_features))
        stability, frequency = self.fn["summarize_feature_stability"](rows, universe)
        self.assertEqual(universe, ["a", "b", "c"])
        self.assertAlmostEqual(stability.nogueira.iloc[0], -.5)
        self.assertAlmostEqual(stability.kuncheva.iloc[0], -.5)
        never = frequency.set_index("feature").loc["c"]
        self.assertEqual(never.selection_frequency, 0.)
        self.assertEqual(never.n_available, 1)

    def test_summary_leaves_kuncheva_missing_when_k_varies(self):
        rows = toy_selections()
        rows[1].update(features=["a", "b"], n_features_selected=2)
        stability, _ = self.fn["summarize_feature_stability"](pd.DataFrame(rows), ["a", "b", "c"])
        self.assertTrue(np.isnan(stability.kuncheva.iloc[0]))
        self.assertTrue(np.isfinite(stability.nogueira.iloc[0]))

    def test_actual_save_block_persists_consistent_dimension_and_provenance(self):
        ns = self.fn | {"os": os, "json": json, "datetime": datetime,
                        "shap": SimpleNamespace(__version__="test")}
        assignments = [n for n in TREE.body if isinstance(n, ast.Assign)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), "exec"), ns)
        main = next(n for n in TREE.body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "run_id" for t in n.targets))
        results = [dict(repeat=1, fold=i, imbalance="none", selector="mi", model="lr",
                        pr_auc=.2, roc_auc=.7, recall=.1, precision=.2, f1=.15, mcc=.1, n_features=1)
                   for i in (1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            ns.update(RESULTS_DIR=directory, results=results, selections=toy_selections(),
                      shap_rows=[], shap_status=[], fold_id=2, n_features_original=3,
                      dev_df=pd.DataFrame(index=range(4)), y=pd.Series([0, 0, 0, 1]))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=main.body[start:], type_ignores=[]), str(SOURCE), "exec"), ns)
            config = json.loads(next(Path(directory).glob("*_config.json")).read_text())
            saved = json.loads(next(Path(directory).glob("*_stability.json")).read_text())[0]
            self.assertEqual(config["nogueira_d"], 3)
            self.assertIsInstance(config["nogueira_d"], int)
            self.assertEqual(config["stability_feature_universe"], ["a", "b", "c"])
            self.assertEqual(config["stability_universe"], "union_of_candidate_features_across_all_folds")
            self.assertEqual(config["stability_scope"], "preprocessing_and_feature_selection")
            self.assertAlmostEqual(saved["nogueira"], -.5)
            self.assertAlmostEqual(saved["kuncheva"], -.5)
            self.assertEqual(json.loads(next(Path(directory).glob("*_shap_status.json")).read_text()), [])
            self.assertTrue({"knn", "nb"}.issubset(config["shap_models"]))
            self.assertEqual(config["shap_output_spaces"]["nb"], "probability")

    @unittest.skipUnless((ROOT / "experiments/20261004_110720_selections.json").exists(),
                         "local research artifacts unavailable")
    def test_real_run_matches_independent_diagnosis_and_other_fields_unchanged(self):
        def read(name):
            return pd.DataFrame(json.loads((ROOT / f"experiments/20261004_110720_{name}.json").read_text()))
        selections = read("selections")
        universe = sorted(set().union(*selections.candidate_features))
        self.assertEqual(len(universe), 331)
        stability, frequency = self.fn["summarize_feature_stability"](selections, universe)
        keys = ["imbalance", "selector"]
        actual = stability.set_index(keys).sort_index()
        saved = read("stability").set_index(keys).sort_index()
        diagnosis = pd.read_csv(ROOT / "audits/nogueira_20261004_110720/diagnostic_comparison.csv",
                                float_precision="round_trip").set_index(keys).sort_index()
        self.assertEqual(len(actual), 40)
        for field in ("nogueira", "kuncheva"):
            np.testing.assert_allclose(actual[field], diagnosis[f"diagnostic_{field}_d_equals_columns"],
                                       rtol=0, atol=1e-12, equal_nan=True)
        for field in saved.columns.difference(["nogueira", "kuncheva"]):
            np.testing.assert_allclose(actual[field], saved[field], rtol=0, atol=1e-12, equal_nan=True)
        self.assertTrue((actual.nogueira >= -1 / (actual.n_runs - 1) - 1e-12).all())
        self.assertTrue((actual.nogueira <= 1).all())
        fkeys = keys + ["feature"]
        actual_freq = frequency.set_index(fkeys).sort_index()
        saved_freq = read("feature_frequency").set_index(fkeys).sort_index()
        for field in saved_freq:
            np.testing.assert_allclose(actual_freq[field], saved_freq[field], rtol=0, atol=1e-12, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
