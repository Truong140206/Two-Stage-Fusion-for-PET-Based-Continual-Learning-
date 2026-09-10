"""Small, dependency-free regression tests for result verification."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import audit_weight_evidence as audit
from tools import verify_paper_results as verify


def name(seed=42, w='0p7', beta='0p5', dim=10000, suffix=''):
    return ('imr_lora_rank8_baseline_10tasks_seed%d_eval_rp_lora_d%d_relu_l10000'
            '_nnone_t0_b0p0_p1_inone_c0_ra0ls0_f1d1w%slsw0p0c0ca0'
            'cw%ssh1p0m1gmargin%s.log') % (seed, dim, w, beta, suffix)


def row(stage, acc=75.0):
    tail = ' Forgetting: 3.0 Backward: -2.0' if stage > 1 else ''
    return ('[Average accuracy till task%d] Acc@task: 79.0 Acc@1: %.4f '
            'Acc@5: 88.0 Loss: 1.2%s\n') % (stage, acc, tail)


class AuditTests(unittest.TestCase):
    def test_config_key_preserves_all_nonvaried_fields(self):
        cfg = audit.parse_name(name())
        for changed in [name(dim=5000), name().replace('_c0_', '_c1_'),
                        name().replace('_p1_', '_p0_'),
                        name().replace('_nnone_', '_nl2_'),
                        name().replace('sh1p0', 'sh2p0'),
                        name().replace('m1g', 'm2g'),
                        name(suffix='r0p5'), name(suffix='bmocov3'),
                        name(suffix='__maskfix_v1')]:
            self.assertNotEqual(audit.configuration_key(cfg),
                                audit.configuration_key(audit.parse_name(changed)))

    def test_pair_key_ignores_only_seed_and_w(self):
        self.assertEqual(audit.configuration_key(audit.parse_name(name())),
                         audit.configuration_key(audit.parse_name(name(seed=43, w='0p6'))))
        self.assertNotEqual(audit.configuration_key(audit.parse_name(name())),
                            audit.configuration_key(audit.parse_name(name(beta='0p3'))))

    def test_grid_key_ignores_both_weights_but_keeps_seed(self):
        key = lambda n: audit.configuration_key(audit.parse_name(n),
                                                varying=('w', 'beta'), across_seeds=False)
        self.assertEqual(key(name()), key(name(w='0p6', beta='0p3')))
        self.assertNotEqual(key(name()), key(name(seed=43)))

    def test_partial_run_is_not_final(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / name()
            path.write_text(row(3), encoding='utf-8')
            self.assertEqual(audit.read_final(path), (None, {}))
            path.write_text(row(10), encoding='utf-8')
            self.assertEqual(audit.read_final(path)[0], 10)

    def test_missing_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / name()
            path.write_text(row(10).replace(' Loss: 1.2', ''), encoding='utf-8')
            self.assertEqual(audit.read_final(path), (None, {}))

    def test_unknown_task_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'unknown.log'
            path.write_text(row(10), encoding='utf-8')
            self.assertEqual(audit.read_final(path), (None, {}))
            self.assertEqual(audit.read_final(path, 10)[0], 10)

    def test_confidence_interval_depends_on_sample_size(self):
        mean, sd, lo, hi = audit.paired([0.0, 1.0])
        self.assertAlmostEqual(mean, .5)
        self.assertAlmostEqual(hi - mean, 12.706204736432095 / 2)
        mean, sd, lo, hi = audit.paired([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(hi - mean, 3.182446305284263 * sd / 2)
        self.assertEqual(audit.paired([1.0]), (1.0, 0.0, None, None))
        with self.assertRaises(ValueError):
            audit.paired([])

    def test_identity_not_mislabelled_routing_only(self):
        self.assertEqual(audit.stage(audit.parse_name(name(w='1p0', beta='0p0'))),
                         'baseline-identity')

    def test_grid_separates_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            for dim, acc in [(5000, 1), (10000, 2)]:
                (Path(root) / name(dim=dim)).write_text(row(10, acc), encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(audit.main(root), 0)
            self.assertIn('d5000', out.getvalue())
            self.assertIn('d10000', out.getvalue())


class VerificationTests(unittest.TestCase):
    def test_incomplete_stage_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'test.log'
            path.write_text(row(10), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify.read_stages(path)

    def test_conflicting_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'test.log'
            text = ''.join(row(i) for i in range(1, 11))
            path.write_text(text + row(10, 2), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify.read_stages(path)
            path.write_text(text + row(10), encoding='utf-8')
            self.assertEqual(len(verify.read_stages(path)), 10)

    def test_checkpoint_dataset_mismatch_fails(self):
        saved = SimpleNamespace(dataset='Split-Imagenet-R', seed=42,
                                num_tasks=10, model='vit_base_patch16_224')
        with self.assertRaisesRegex(ValueError, 'dataset'):
            verify.check_saved_args(saved, 'Split-CUB200', 42, 'TII')
        saved.dataset = 'Split-CUB200'
        verify.check_saved_args(saved, 'Split-CUB200', 42, 'TII')
        with self.assertRaisesRegex(ValueError, 'seed'):
            verify.check_saved_args(saved, 'Split-CUB200', 43, 'TII')

    def test_arms_use_same_dataset_checkpoint_and_seed(self):
        args = SimpleNamespace(dataset='cub200', data_root=Path('/data'),
                               port=29558, batch_size=24)
        commands = [verify.command(args, 42, arm, Path('/tii'), Path('/lora'))
                    for arm in ('baseline', 'identity', 'full')]
        for cmd in commands:
            self.assertIn('--eval', cmd)
            for flag, value in [('--dataset', 'Split-CUB200'), ('--seed', '42'),
                                ('--trained_original_model', str(Path('/tii'))),
                                ('--output_dir', str(Path('/lora')))]:
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn('--rp_head', commands[0])
        self.assertEqual(commands[1][commands[1].index('--rp_class_fusion_weight') + 1], '0.0')
        self.assertEqual(commands[2][commands[2].index('--rp_class_fusion_weight') + 1], '0.5')

    def test_existing_unverified_log_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'test.log'
            path.write_text('historical', encoding='utf-8')
            with patch.object(verify, 'check_gpu') as gpu:
                with self.assertRaises(ValueError):
                    verify.run_or_read(path, [], {}, True)
                gpu.assert_not_called()
            self.assertEqual(path.read_text(), 'historical')

    def test_new_log_names_are_distinct_from_historical(self):
        run = 'imr_lora_rank8_baseline_10tasks_seed42'
        for arm in ('baseline', 'identity', 'full'):
            self.assertNotEqual(verify.log_name(run, arm), verify.log_name(run, arm, 'maskfix'))
            self.assertNotEqual(audit.parse_name(verify.log_name(run, arm, 'maskfix'))['kind'], 'other')

    def test_tii_training_command_passes_dataset(self):
        script = (verify.REPO / 'training_scripts' / 'train_any_4090.sh').read_text()
        tii_command = script.split('main.py "${CFG_TII}"', 1)[1].split('printf', 1)[0]
        self.assertIn('--dataset "${DS}"', tii_command)


if __name__ == '__main__':
    unittest.main()
