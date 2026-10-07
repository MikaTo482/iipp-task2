"""Saved-value and layout regression checks for the basic 05 renderer.

Load the numbered standalone script from the project root.
No models are trained and no experiment data is changed by these tests.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('basic_visualization', ROOT / '05_visualize_result.py')
basic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = basic
SPEC.loader.exec_module(basic)
RUN = '20990101_000000'


def fixture(path, expanded=False):
    selectors = ['none', 'mi', 'l1']
    imbalances = ['none', 'smote']
    models = ['lr', 'rf']
    if expanded:
        selectors += ['selector_' + 'long_' * 15 + str(i) for i in range(10)]
        imbalances += ['imbalance_' + 'long_' * 15 + str(i) for i in range(5)]
        models += ['model_' + 'long_' * 15 + str(i) for i in range(5)]
    summary, stability, selections = [], [], []
    for selector in selectors:
        for imbalance in imbalances:
            keys = dict(selector=selector, imbalance=imbalance)
            stability.append(dict(**keys, nogueira=None if selector == 'none' else -7.812345,
                                  mean_jaccard=.314159, kuncheva=None, n_runs=2,
                                  n_features_mean=9.5, n_features_std=.7, n_features_min=9, n_features_max=10))
            # Deliberately differs from saved means: the renderer must not recalculate.
            for fold in (1, 2):
                selections.append(dict(**keys, repeat=1, fold=fold, n_features_selected=3,
                                       n_features_after_preprocessing=4, candidate_features=['a','b','c','d']))
            for model in models:
                summary.append(dict(**keys, model=model, pr_auc_mean=.123456789,
                                    recall_mean=0, precision_mean=.2, f1_mean=.15,
                                    mcc_mean=-.0123456, roc_auc_mean=.6, n_features_mean=9.75))
    for name, records in [('summary', summary), ('stability', stability), ('selections', selections),
                          ('config', {'nogueira_d': 3.5})]:
        (path / f'{RUN}_{name}.json').write_text(json.dumps(records))
    return summary, stability, selections


class BasicGraphs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.raw = fixture(self.path)

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def load(self):
        return basic.load_run(self.path, RUN)[1]

    def test_saved_means_negative_and_missing_stay_unchanged(self):
        frames = self.load()
        sample = frames['summary'].query("model == 'lr'")
        values = basic.matrix(sample, 'pr_auc_mean', ['none', 'mi', 'absent'], ['none', 'smote'])
        self.assertEqual(values[0, 0], .123456789)
        self.assertTrue(np.isnan(values[2]).all())
        stable = basic.matrix(frames['stability'], 'nogueira', ['none', 'mi'], ['none'])
        self.assertTrue(np.isnan(stable[0, 0]))
        self.assertEqual(stable[1, 0], -7.812345)
        self.assertEqual(basic.scale(frames['stability'], 'nogueira')[0], -7.812345)
        self.assertEqual(basic.scale(sample, 'mcc_mean'), (-1, 1))

    def test_duplicate_keys_fail_instead_of_silent_averaging(self):
        (self.path / f'{RUN}_summary.json').write_text(json.dumps(self.raw[0] + [self.raw[0][0]]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_bad_numeric_value_fails(self):
        self.raw[0][0]['recall_mean'] = 'unknown'
        (self.path / f'{RUN}_summary.json').write_text(json.dumps(self.raw[0]))
        with self.assertRaisesRegex(ValueError, 'invalid numeric'):
            self.load()

    def test_legacy_denominator_warning_preserves_data(self):
        _, frames, config, _ = basic.load_run(self.path, RUN)
        warnings = basic.warnings_for(frames, config)
        self.assertTrue(any('d=3.5, candidate union=4' in warning for warning in warnings))
        self.assertEqual(frames['stability'].iloc[2].nogueira, -7.812345)
        config.update(nogueira_d=4, stability_universe='union_of_candidate_features_across_all_folds')
        self.assertFalse(any('denominator' in warning for warning in basic.warnings_for(frames, config)))

    def test_all_three_families_and_exact_csv_values(self):
        before = {p.name: p.read_bytes() for p in self.path.glob('*.json')}
        out_path = self.path / 'out'
        with mock.patch.object(pd.Series, 'mean', side_effect=AssertionError('No re-estimated means')):
            self.assertEqual(basic.main(['--results-dir', str(self.path), '--out-dir', str(out_path)]), 0)
        manifest = json.loads((out_path / 'manifest.json').read_text())
        self.assertEqual([v['family'] for v in manifest['views']], ['metrics', 'metrics', 'stability', 'features'])
        self.assertEqual(manifest['views'][2]['models'], 'shared_across_models')
        saved = pd.read_csv(out_path / 'tables/summary.csv', float_precision='round_trip')
        for _, row in saved.iterrows():
            original = self.raw[0][int(row.source_row)]
            for field in [f'{key}_mean' for key in basic.METRICS] + ['n_features_mean']:
                self.assertEqual(row[field], original[field])
        after = {p.name: p.read_bytes() for p in self.path.glob('*.json')}
        self.assertEqual(before, after)
        self.assertGreater((out_path / 'overview.pdf').stat().st_size, 1000)

    def test_missing_summary_does_not_trigger_reestimation(self):
        (self.path / f'{RUN}_summary.json').unlink()
        frames = self.load()
        self.assertTrue(frames['summary'].empty)
        _, labels, _ = basic.category_labels(frames)
        seen = []
        def save(name, fig, family, fields, scope):
            seen.append(family)
            plt.close(fig)
        out = mock.Mock(save=save)
        basic.render(out, frames, labels, list(basic.METRICS), 10, 5, [])
        self.assertEqual(seen, ['stability'])

    def test_expanded_grid_is_paginated_without_losing_combinations(self):
        fixture(self.path, expanded=True)
        frames = self.load()
        levels, labels, mapping = basic.category_labels(frames)
        self.assertEqual([len(levels[k]) for k in ('selector', 'imbalance', 'model')], [13, 7, 7])
        for factor in labels:
            self.assertEqual(len(set(labels[factor].values())), len(labels[factor]))
        seen = {'metrics': set(), 'features': set(), 'stability': set()}
        def save(name, fig, family, fields, scope):
            self.assertLessEqual(len(scope['selectors']), 10)
            self.assertLessEqual(len(scope['imbalances']), 5)
            models = scope['models'] if family != 'stability' else [None]
            for model in models:
                for selector in scope['selectors']:
                    for imbalance in scope['imbalances']:
                        seen[family].add((model, selector, imbalance))
            plt.close(fig)
        basic.render(mock.Mock(save=save), frames, labels, list(basic.METRICS), 10, 5, [])
        self.assertEqual(len(seen['metrics']), 13 * 7 * 7)
        self.assertEqual(seen['features'], seen['metrics'])
        self.assertEqual(len(seen['stability']), 13 * 7)
        # Actually render the fullest page of long labels; Output.save checks all text bounds.
        fig, axes = basic.figure(6, 'Long categories', 'Pagination keeps the canvas bounded.', 10)
        for ax, model in zip(axes, levels['model'][-6:]):
            basic.heatmap(fig, ax, frames['summary'].query('model == @model'), 'n_features_mean',
                          labels['model'][model], levels['selector'][-10:], levels['imbalance'][-5:], labels, (0, 10))
        out = basic.Output(self.path / 'long', ['png'])
        try:
            out.save('long', fig, 'features', ['n_features_mean'], {})
        finally:
            out.pdf.close()

    def test_explicit_model_stability_keeps_model_dimension(self):
        rows = [dict(row, model=model) for model in ['lr', 'rf'] for row in self.raw[1]]
        (self.path / f'{RUN}_stability.json').write_text(json.dumps(rows))
        frames = self.load()
        _, labels, _ = basic.category_labels(frames)
        seen = []
        def save(name, fig, family, fields, scope):
            if family == 'stability': seen.extend(scope['models'])
            plt.close(fig)
        basic.render(mock.Mock(save=save), frames, labels, ['recall'], 10, 5, [])
        self.assertEqual(seen, ['lr', 'rf'])

    def test_single_metric_layout_determinism_and_owned_cleanup(self):
        outputs = []
        for name in ['a', 'b']:
            target = self.path / name
            self.assertEqual(basic.main(['--results-dir', str(self.path), '--out-dir', str(target),
                                        '--metric', 'recall', '--formats', 'png,pdf,svg']), 0)
            outputs.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in target.glob('*') if p.is_file()})
        self.assertEqual(outputs[0], outputs[1])
        stale = self.path / 'a' / 'stale.png'
        stale.write_text('renderer-owned')
        unrelated = self.path / 'a' / 'my_notes.txt'
        unrelated.write_text('keep')
        manifest_path = self.path / 'a' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['outputs'].append('stale.png')
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(basic.main(['--results-dir', str(self.path), '--out-dir', str(self.path / 'a'),
                                    '--metric', 'recall']), 0)
        self.assertFalse(stale.exists())
        self.assertTrue(unrelated.exists())
        self.assertFalse((self.path / 'a' / '01_metrics_01.svg').exists())


if __name__ == '__main__':
    unittest.main()
