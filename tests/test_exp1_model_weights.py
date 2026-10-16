"""Learner weighting checks without running the full experiment."""

import copy
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


EXP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "04_exp_1.py"))
build_model = EXP["build_model"]


class ModelWeightTests(unittest.TestCase):
    def test_balanced_lr_and_rf(self):
        for name in ("lr", "rf"):
            with self.subTest(model=name):
                self.assertEqual(build_model(name, 42, use_weights=True).class_weight, "balanced")
                self.assertIsNone(build_model(name, 42).class_weight)

    def test_xgb_uses_training_class_ratio(self):
        for labels, expected in [([0] * 8 + [1] * 2, 4.0), ([0] * 3 + [1] * 2, 1.5)]:
            model = build_model("xgb", 42, use_weights=True, y_train=pd.Series(labels))
            self.assertEqual(model.scale_pos_weight, expected)
        self.assertEqual(build_model("xgb", 42).scale_pos_weight, 1.0)

    def test_weighted_xgb_rejects_missing_or_invalid_training_labels(self):
        for labels in (None, [], [0, 0], [1, 1], [-1, 1], [0, np.nan, 1], [[0], [1]]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "y_train"):
                build_model("xgb", 42, use_weights=True, y_train=labels)

    def test_model_params_are_not_modified(self):
        before = copy.deepcopy(EXP["MODEL_PARAMS"])
        for name in ("lr", "rf", "xgb"):
            build_model(name, 42, use_weights=True, y_train=[0, 0, 1])
        self.assertEqual(before, EXP["MODEL_PARAMS"])

    def test_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("unknown", 42)

    def test_all_conditions_fit_and_only_weight_condition_enables_weights(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(30, 4)))
        y = pd.Series([0] * 20 + [1] * 10)
        params = copy.deepcopy(EXP["MODEL_PARAMS"])
        params["rf"]["n_estimators"] = 2
        params["xgb"]["n_estimators"] = 2
        with patch.dict(EXP["MODEL_PARAMS"], params):
            for condition in ("none", "weight", "smote"):
                X_fit, y_fit, enabled = EXP["apply_imbalance"](X, y, condition, 42)
                for name in ("lr", "rf", "xgb"):
                    with self.subTest(condition=condition, model=name):
                        model = build_model(name, 42, use_weights=enabled, y_train=y_fit)
                        if name == "xgb":
                            self.assertEqual(model.scale_pos_weight, 2.0 if condition == "weight" else 1.0)
                        else:
                            self.assertEqual(model.class_weight, "balanced" if condition == "weight" else None)
                        model.fit(X_fit, y_fit)
                        self.assertTrue(np.isfinite(model.predict_proba(X)).all())


if __name__ == "__main__":
    unittest.main()
