"""Boyer-Moore voting, reporting, and the experiment's selection entry point."""
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from importlib import import_module

# The numbered script is importable without executing its __main__ loop.
workflow = import_module("04_experiment_workflow")


class BoyerMooreTests(unittest.TestCase):
    def test_every_binary_sequence_up_to_ten_voters(self):
        # Includes every tie, majority-0, majority-1, and voter ordering.
        for n_voters in range(1, 11):
            with self.subTest(n_voters=n_voters):
                ballots = np.array(list(product([0, 1], repeat=n_voters))).T
                expected = ballots.sum(axis=0) > n_voters / 2
                np.testing.assert_array_equal(workflow._boyer_moore_support(ballots), expected)

    def test_invalid_ballot_matrices(self):
        for ballots in ([], [[0, 2]], [[np.nan]], [[0.5]], np.empty((0, 2)),
                        np.empty((2, 0)), np.zeros((2, 2, 2))):
            with self.subTest(ballots=ballots), self.assertRaises(ValueError):
                workflow._boyer_moore_support(ballots)


class VoteScoresTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.zeros((4, 6)),
                              columns=["all", "four", "three", "two", "one", "never"])
        self.y = np.array([0, 0, 0, 1])
        self.picks = {name: list(self.X.columns[:5 - i])[::-1]
                      for i, name in enumerate(workflow.VOTERS)}

    def calculate(self, **kwargs):
        with patch.object(workflow, "select_features",
                          side_effect=lambda name, *a, **k: self.picks[name]) as voter:
            result = workflow.vote_scores(self.X, self.y, True, 42, n_jobs=2, **kwargs)
        return result, voter

    def test_selection_and_training_parameters(self):
        result, voter = self.calculate()
        pd.testing.assert_series_equal(result["support"],
                                       pd.Series([True, True, True, False, False, False],
                                                 index=self.X.columns))
        self.assertEqual(result["selected_by"], self.picks)
        self.assertEqual(set(result), {"support", "selected_by"})
        self.assertEqual(voter.call_count, len(workflow.VOTERS))
        for name, call in zip(workflow.VOTERS, voter.call_args_list):
            self.assertEqual(call.args[0], name)
            self.assertIs(call.args[1], self.X)
            self.assertIs(call.args[2], self.y)
            self.assertEqual(call.args[3:], (True, 42))
            self.assertEqual(call.kwargs, {"n_jobs": 2})

    def test_reversing_voters_preserves_decisions(self):
        forward, _ = self.calculate()
        reverse, _ = self.calculate(methods=workflow.VOTERS[::-1])
        pd.testing.assert_series_equal(forward["support"], reverse["support"])

    def test_each_feature_uses_the_original_algorithm(self):
        with patch.object(workflow, "boyer_moore_majority",
                          wraps=workflow.boyer_moore_majority) as majority:
            self.calculate()
        self.assertEqual(majority.call_count, self.X.shape[1])
        for column, call in zip(self.X.columns, majority.call_args_list):
            expected = [int(column in self.picks[name]) for name in workflow.VOTERS]
            np.testing.assert_array_equal(call.args[0], expected)

    def test_default_worker_count_follows_workflow_setting(self):
        select_features = workflow.select_features
        with patch.object(workflow, "N_JOBS", 3), \
                patch.object(workflow, "select_features",
                             side_effect=lambda name, *a, **k: self.picks[name]) as voter:
            for entry in ("vote_scores", "select_features"):
                with self.subTest(entry=entry):
                    voter.reset_mock()
                    if entry == "vote_scores":
                        workflow.vote_scores(self.X, self.y, False, 42)
                    else:
                        select_features("vote", self.X, self.y, False, 42)
                    self.assertEqual(voter.call_count, len(workflow.VOTERS))
                    for call in voter.call_args_list:
                        self.assertEqual(call.kwargs, {"n_jobs": 3})

    def test_empty_and_unanimous_subsets_have_no_fallback(self):
        for picked in ([], list(self.X.columns)):
            with self.subTest(picked=picked):
                self.picks = dict.fromkeys(workflow.VOTERS, picked)
                result, _ = self.calculate()
                self.assertEqual(list(self.X.columns[result["support"]]), picked)

    def test_single_voter_keeps_exactly_its_subset(self):
        result, voter = self.calculate(methods=("mi",))
        self.assertEqual(set(self.X.columns[result["support"]]), set(self.picks["mi"]))
        self.assertEqual(voter.call_count, 1)

    def test_experiment_entry_point_rejects_ties(self):
        original_select_features = workflow.select_features
        methods = ("mi", "fdr", "l1", "xgb")
        self.picks = {"mi": ["all", "two"], "fdr": ["all", "two"],
                      "l1": ["all"], "xgb": []}
        with patch.dict(workflow.SELECTOR_PARAMS, {"vote": dict(methods=methods)}), \
                patch.object(workflow, "select_features", side_effect=lambda name, *a, **k: self.picks[name]):
            selected = original_select_features("vote", self.X, self.y, False, 42)
        self.assertEqual(selected, ["all"])  # 3/4 survives; 2/4 does not.

    def test_bad_configuration_fails_before_fitting(self):
        cases = [({"methods": []}, ValueError), ({"methods": "mi"}, TypeError),
                 ({"methods": ("mi", "mi")}, ValueError),
                 ({"methods": ("none",)}, ValueError),
                 ({"methods": ("vote",)}, ValueError),
                 ({"methods": ("unknown",)}, ValueError),
                 ({"weighting": "size"}, TypeError)]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), patch.object(workflow, "select_features") as voter:
                with self.assertRaises(error):
                    workflow.vote_scores(self.X, self.y, False, 42, **kwargs)
                voter.assert_not_called()

    def test_invalid_feature_columns(self):
        duplicate = self.X.copy()
        duplicate.columns = ["same"] * len(duplicate.columns)
        for X, error in ((duplicate, ValueError), (self.X.iloc[:0], ValueError),
                         (self.X.iloc[:, :0], ValueError), (self.X.to_numpy(), TypeError)):
            with self.subTest(shape=X.shape), self.assertRaises(error):
                workflow.vote_scores(X, self.y, False, 42)

    def test_duplicate_or_unknown_voter_picks_are_rejected(self):
        for picked in (["all", "all"], ["not_a_column"]):
            with self.subTest(picked=picked), patch.object(workflow, "select_features", return_value=picked):
                with self.assertRaises(ValueError):
                    workflow.vote_scores(self.X, self.y, False, 42, methods=("mi",))

    def test_voter_errors_propagate_instead_of_becoming_no_votes(self):
        with patch.object(workflow, "select_features", side_effect=RuntimeError("fit failed")):
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                workflow.vote_scores(self.X, self.y, False, 42)

    def test_real_selectors_smoke(self):
        X, y = make_classification(n_samples=100, n_features=8, n_informative=3,
                                   n_redundant=0, weights=[0.8, 0.2], random_state=42)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        result = workflow.vote_scores(X, y, False, 42, methods=("mi", "fdr"))
        # With two voters, strict majority is their intersection.
        expected = set(result["selected_by"]["mi"]) & set(result["selected_by"]["fdr"])
        self.assertEqual(set(X.columns[result["support"]]), expected)


if __name__ == "__main__":
    unittest.main()
