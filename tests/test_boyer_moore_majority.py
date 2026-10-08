"""Verify the original single-sequence Boyer-Moore majority algorithm."""
from collections import Counter
from importlib import import_module
from itertools import product
import unittest

import numpy as np

workflow = import_module("04_experiment_workflow")


class BoyerMooreMajorityTests(unittest.TestCase):
    def test_all_sequences_with_three_values_up_to_length_seven(self):
        for length in range(8):
            for values in product((0, 1, 2), repeat=length):
                with self.subTest(values=values):
                    expected = next((value for value, count in Counter(values).items()
                                     if count > length / 2), None)
                    self.assertEqual(workflow.boyer_moore_majority(values), expected)

    def test_majority_with_different_input_types(self):
        for values, expected in ((["A", "B", "A"], "A"),
                                 ((-1, 3, -1), -1),
                                 (np.array([0, 1, 0, 1, 1]), 1)):
            with self.subTest(values=values):
                self.assertEqual(workflow.boyer_moore_majority(values), expected)

    def test_no_majority_in_article_counterexample(self):
        self.assertIsNone(workflow.boyer_moore_majority([0, 3, 4, 0]))

    def test_plurality_is_not_a_majority(self):
        self.assertIsNone(workflow.boyer_moore_majority([1, 2, 3, 1, 2]))


if __name__ == "__main__":
    unittest.main()
