"""Check data fidelity and row separation in the optional RQ3 design previews."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rq3_options", ROOT / "graph/rq3_design_options.py")
rq3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rq3)


class RQ3Options(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.summary = [dict(selector="mi", imbalance="none", model=model,
                             average_precision_mean=.12345678901234566,
                             n_features_mean=20.) for model in ["lr", "rf"]]
        self.stability = [dict(selector="mi", imbalance="none", nogueira=-.25,
                               n_features_mean=20.)]
        self.write()

    def tearDown(self):
        plt.close("all")
        self.temp.cleanup()

    def write(self):
        for name, records in [("summary", self.summary), ("stability", self.stability)]:
            (self.path / f"run_{name}.json").write_text(json.dumps(records))

    def load(self):
        return rq3.load_run(self.path, "run")

    def test_saved_numbers_and_missing_fdr_are_not_reestimated(self):
        data = self.load()
        ap = rq3.matrix(data, rq3.AP, data["rows"], data["models"])
        self.assertEqual(ap[0, 0], self.summary[0][rq3.AP])
        self.assertTrue(np.isnan(ap[1]).all())
        stability = rq3.matrix(data, rq3.STABILITY, data["rows"], data["models"], True)
        self.assertEqual(stability[0, 0], -.25)
        self.assertTrue(np.isnan(stability[1, 0]))
        self.assertEqual(stability.shape, (2, 1))
        self.assertEqual(rq3.metric_specs(data)[2]["norm"].vmin, -.25)

    def test_duplicate_keys_are_rejected(self):
        self.summary.append(self.summary[0].copy())
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_different_saved_count_populations_are_not_merged(self):
        self.summary[1][rq3.COUNT] = 19.
        self.write()
        data = self.load()
        self.assertFalse(data["shared_count"])
        values = rq3.matrix(data, rq3.COUNT, data["rows"], data["models"])
        np.testing.assert_array_equal(values[0], [20., 19.])

    def test_model_specific_stability_is_not_collapsed(self):
        self.stability = [dict(self.stability[0], model="lr"),
                          dict(self.stability[0], model="rf", nogueira=.8)]
        self.write()
        data = self.load()
        self.assertFalse(data["shared_stability"])
        values = rq3.matrix(data, rq3.STABILITY, data["rows"], data["models"])
        np.testing.assert_array_equal(values[0], [-.25, .8])

    def test_unknown_categories_and_fdr_are_preserved(self):
        self.summary.append(dict(self.summary[0], selector="fdr", imbalance="new_method", model="new_model"))
        self.stability.append(dict(self.stability[0], selector="fdr", imbalance="new_method", nogueira=None))
        self.write()
        data = self.load()
        self.assertIn("new_model", data["models"])
        self.assertIn(("fdr", "new_method"), data["rows"])
        self.assertFalse(data["missing_fdr"])

    def test_equal_values_still_have_different_rows_and_inputs_unchanged(self):
        self.summary += [dict(row, selector="l1") for row in self.summary[:]]
        self.stability.append(dict(self.stability[0], selector="l1"))
        self.write()
        before = {p.name: p.read_bytes() for p in self.path.glob("*.json")}
        data = self.load()
        specs = rq3.metric_specs(data)
        for plot in [rq3.plot_heatmaps, rq3.plot_dots, rq3.plot_bubbles]:
            fig = plot(data, data["rows"], data["models"], specs)
            fig.canvas.draw()
            if plot == rq3.plot_dots:
                offsets = fig.axes[0].collections[0].get_offsets()
                self.assertEqual(len(set(offsets[:, 1])), 2)
                self.assertEqual(len(set(offsets[:, 0])), 1)
            plt.close(fig)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.glob("*.json")})


if __name__ == "__main__":
    unittest.main()
