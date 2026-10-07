"""Exercise timing and serialization without launching the expensive CV loop."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "04_experiment_workflow.py"


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)


class WorkflowTimingTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text())

    def test_imported_clock_supports_elapsed_time(self):
        imports = [n for n in self.tree.body
                   if (isinstance(n, ast.ImportFrom) and n.module == "datetime")
                   or (isinstance(n, ast.Import) and any(a.name == "time" for a in n.names))]
        ns = {}
        execute(imports, ns)
        start = ns["time"].perf_counter()
        self.assertGreaterEqual(ns["time"].perf_counter() - start, 0)

    def check_summary(self, empty_features=False):
        # Evaluate the actual record expressions, not a duplicate of their schema.
        records = {"selections": [], "results": []}
        values = dict(np=np, fold_id=1, repeat=1, imb="none", selector="none",
                      mdl="lr", features=[] if empty_features else ["f0"],
                      X_train_processed=pd.DataFrame({"f0": [0, 1]}),
                      y_val=[0, 1], selection_time=2.0, fit_time=4.0,
                      predict_time=0.02, y_proba=[0.1, 0.9], y_pred=[0, 1],
                      tn=1, fp=0, fn=0, tp=1,
                      average_precision_score=lambda *a: 1.0,
                      recall_score=lambda *a, **k: 1.0,
                      precision_score=lambda *a, **k: 1.0)
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in records):
                continue
            record = node.args[0]
            assert isinstance(record, ast.Dict)
            is_empty = any(isinstance(k, ast.Constant) and k.value == "n_features"
                           and isinstance(v, ast.Constant) and v.value == 0
                           for k, v in zip(record.keys, record.values))
            if node.func.value.id == "results" and is_empty != empty_features:
                continue
            records[node.func.value.id].append(eval(compile(ast.Expression(record), str(SOURCE), "eval"), values))
        self.assertEqual(len(records["selections"]), 1)
        self.assertEqual(len(records["results"]), 1)
        ns = dict(selections_df=pd.DataFrame(records["selections"]),
                  results_df=pd.DataFrame(records["results"]))
        main = next(n for n in self.tree.body if isinstance(n, ast.If)
                    and "__main__" in ast.unparse(n.test))
        summary_nodes = [n for n in main.body
                         if (isinstance(n, ast.FunctionDef) and n.name in ("q1", "q3"))
                         or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                             and t.id in ("selection_time_df", "model_time_df") for t in n.targets))]
        execute(summary_nodes, ns)
        self.assertEqual(ns["selection_time_df"].iloc[0]["selection_time_median"], 2.0)
        model = ns["model_time_df"].iloc[0]
        if empty_features:
            self.assertTrue(pd.isna(model["fit_time_median"]))
            self.assertTrue(pd.isna(model["predict_time_per_sample_ms_median"]))
        else:
            self.assertEqual(model["fit_time_median"], 4.0)
            self.assertEqual(model["fit_time_q1"], 4.0)
            self.assertEqual(model["fit_time_q3"], 4.0)
            self.assertEqual(model["predict_time_per_sample_ms_median"], 10.0)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("selection_time_df", "model_time_df"):
                path = Path(tmp) / (name + ".json")
                ns[name].to_json(path, orient="records", indent=2, double_precision=15)
                data = json.loads(path.read_text())
                self.assertEqual(len(data), 1)
                if empty_features and name == "model_time_df":
                    self.assertIsNone(data[0]["fit_time_median"])

    def test_recorded_timings_can_be_aggregated_and_saved(self):
        self.check_summary()

    def test_empty_selection_timings_remain_missing(self):
        self.check_summary(empty_features=True)


if __name__ == "__main__":
    unittest.main()
