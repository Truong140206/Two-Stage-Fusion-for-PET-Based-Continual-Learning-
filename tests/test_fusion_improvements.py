"""Runner tests plus real-tensor regression tests (require torch, not a GPU)."""
import argparse
import ast
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from configs.fusion_experiments import add_fusion_experiments
from tools import try_fusion_improvements as probe

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'engines/hrm_lora_wtp_and_tap_engine.py'


class ProbeTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(dataset='imr', data_root=Path('/data'),
                               resolved_data_path=Path('/data'), batch_size=24,
                               port=29558, output_root=Path('/out'))

    def cmd(self, variant):
        return probe.variant_command(self.args(), 42, Path('/tii'), Path('/lora'), variant)

    def test_reference_command_identical(self):
        self.assertEqual(self.cmd('reference'), probe.verify.command(
            self.args(), 42, 'full', Path('/tii'), Path('/lora')))

    def test_single_factor_route(self):
        self.assertEqual(self.cmd('route_class_zmax'),
                         self.cmd('reference') + ['--rp_route_score_mode', 'class_zmax'])

    def test_single_factor_gate(self):
        self.assertEqual(self.cmd('gate_floor025'),
                         self.cmd('reference') + ['--rp_class_gate_floor', '0.25'])

    def test_single_factor_dimension(self):
        cmd = self.cmd('rp5000')
        index = cmd.index('--rp_dim') + 1
        self.assertEqual(cmd[index], '5000')
        cmd[index] = '10000'
        self.assertEqual(cmd, self.cmd('reference'))

    def test_no_training_or_checkpoint_destination_change(self):
        for variant in ('reference', *probe.VARIANTS):
            cmd = self.cmd(variant)
            self.assertIn('--eval', cmd)
            self.assertIn('--strict_exemplar_free', cmd)
            self.assertEqual(cmd[cmd.index('--output_dir') + 1], str(Path('/lora')))

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            self.cmd('typo')

    def test_parser_defaults_and_bounds(self):
        parser = argparse.ArgumentParser()
        add_fusion_experiments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.rp_route_score_mode, 'task_z')
        self.assertEqual(args.rp_class_gate_floor, 0.0)
        for value in ('nan', 'inf', '-0.1', '1.1'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--rp_class_gate_floor', value])

    def test_both_configs_accept_experiments(self):
        from configs import imr_lora, cifar100_lora
        for module in (imr_lora, cifar100_lora):
            parser = argparse.ArgumentParser()
            module.get_args_parser(parser)
            args = parser.parse_args(['--rp_route_score_mode', 'class_zmax',
                                      '--rp_class_gate_floor', '0.25'])
            self.assertEqual(args.rp_class_gate_floor, 0.25)

    def test_old_evidence_is_read_only_with_original_source(self):
        with patch.object(probe.verify, 'run_or_read', return_value={}) as run:
            probe.old_reference(self.args(), 42, Path('/tii'), Path('/lora'), [])
        self.assertFalse(run.call_args.kwargs['execute'])
        self.assertEqual(run.call_args.args[2]['source_sha256'], probe.BASELINE_SOURCE)
        self.assertIn('maskfix_verify_v2', str(run.call_args.args[0]))

    def test_reference_regression_stops(self):
        with patch.object(probe.verify, 'compare', return_value=False) as compare:
            with self.assertRaisesRegex(ValueError, 'regression'):
                probe.require_reference_match({}, {})
        self.assertEqual(list(compare.call_args.args[2]), list(range(1, 11)))

    def test_default_route_expression_unchanged_from_backup(self):
        old = subprocess.check_output(['git', 'show', probe.BASELINE_COMMIT + ':' + ENGINE],
                                      cwd=ROOT, text=True, encoding='utf-8')
        def expression(source):
            tree = ast.parse(source)
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fuse_routers')
            return next(ast.dump(n.value) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'fused' for t in n.targets))
        self.assertEqual(expression(old), expression((ROOT / ENGINE).read_text(encoding='utf-8')))


HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'Real tensor tests require torch; run on lab CPU before GPU eval')
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        names = {'_stage_ramp', '_task_scores_from_class_scores', 'fuse_routers',
                 '_valid_moments', '_standardize_valid', '_top2_margin',
                 '_fusion_gate', 'fuse_class_scores'}
        def load(source):
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            env = dict(torch=torch, math=math, args_ref=[None], gate_stats={})
            exec(compile(tree, ENGINE, 'exec'), env)
            return env
        cls.new = load((ROOT / ENGINE).read_text(encoding='utf-8'))
        cls.old = load(subprocess.check_output(
            ['git', 'show', probe.BASELINE_COMMIT + ':' + ENGINE], cwd=ROOT,
            text=True, encoding='utf-8'))

    def test_default_route_and_class_bit_exact_all_stages(self):
        t = self.torch
        rng = t.Generator().manual_seed(1701)
        args = SimpleNamespace(rp_route_fusion_weight=0.7, rp_class_fusion_gate='margin')
        for tasks in range(1, 11):
            r = t.randn(16, 200, generator=rng) * 5
            s = t.randn(16, 200, generator=rng) * 0.2
            r[:, tasks*20:] = -float('inf')
            s[:, tasks*20:] = -float('inf')
            mask = [list(range(i*20, (i+1)*20)) for i in range(10)]
            routes, outputs = [], []
            for env in (self.old, self.new):
                env['args_ref'][0] = args
                routes.append(env['fuse_routers'](s, r, mask, tasks, args, r.device))
                outputs.append(env['fuse_class_scores'](r, s, 0.5, tasks))
            self.assertTrue(t.equal(*routes))
            self.assertTrue(t.equal(*outputs))

    def test_two_task_margin_no_longer_collapses(self):
        t = self.torch
        r, s = t.tensor([[1.01, -10., 1., -10.]]), t.tensor([[0., 0., 100., 0.]])
        args = SimpleNamespace(rp_route_fusion_weight=0.7)
        route = self.new['fuse_routers']
        self.assertEqual(route(s, r, [[0, 1], [2, 3]], 2, args, r.device).item(), 0)
        args.rp_route_score_mode = 'class_zmax'
        self.assertEqual(route(s, r, [[0, 1], [2, 3]], 2, args, r.device).item(), 1)

    def test_new_route_mask_scale_and_endpoints(self):
        t = self.torch
        r, s = t.tensor([[1.01, -10., 1., -10., 1e6]]), t.tensor([[0., 0., 100., 0., -1e6]])
        args = SimpleNamespace(rp_route_fusion_weight=0.7, rp_route_score_mode='class_zmax')
        route = self.new['fuse_routers']
        def run(a, b):
            return route(b, a, [[0, 1], [2, 3], [4]], 2, args, a.device).item()
        self.assertEqual(run(r, s), run(r * 10 + 2, s * 3 - 7))
        r[0, 4], s[0, 4] = -1e9, 1e9
        self.assertEqual(run(r, s), 1)
        args.rp_route_fusion_weight = 1.0
        self.assertEqual(run(r, s), 0)
        args.rp_route_fusion_weight = 0.0
        self.assertEqual(run(r, s), 1)

    def test_floor_endpoints_and_seen_mask(self):
        t = self.torch
        r = t.tensor([[5., 1., 0., -float('inf')]])
        s = t.tensor([[0., 1., 3., -float('inf')]])
        args = SimpleNamespace(rp_class_fusion_gate='margin', rp_class_gate_floor=0.0)
        self.new['args_ref'][0] = args
        fn = self.new['fuse_class_scores']
        default = fn(r, s, 0.5)
        args.rp_class_gate_floor = 0.25
        middle = fn(r, s, 0.5)
        self.assertGreaterEqual(self.new['gate_stats']['gate'], 0.25)
        self.assertTrue(t.isneginf(middle[0, 3]))
        args.rp_class_gate_floor = 1.0
        full = fn(r, s, 0.5)
        args.rp_class_gate_floor, args.rp_class_fusion_gate = 0.0, 'none'
        self.assertTrue(t.equal(full, fn(r, s, 0.5)))
        self.assertFalse(t.equal(default, middle))

    def test_invalid_floor_or_mode_rejected(self):
        t = self.torch
        r, s = t.tensor([[3., 1.]]), t.tensor([[1., 3.]])
        for floor, mode in ((float('nan'), 'margin'), (-1., 'margin'), (0.25, 'none')):
            self.new['args_ref'][0] = SimpleNamespace(rp_class_gate_floor=floor, rp_class_fusion_gate=mode)
            with self.assertRaises(ValueError):
                self.new['fuse_class_scores'](r, s, 0.5)


if __name__ == '__main__':
    unittest.main()
