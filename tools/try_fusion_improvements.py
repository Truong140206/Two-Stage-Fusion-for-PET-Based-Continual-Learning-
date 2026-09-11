#!/usr/bin/env python3
"""Exploratory evaluations only; never select a winner automatically.

Runs the unchanged full reference first and checks ALL stages against verified
pre-experiment logs, then changes one factor at a time. No training or test-label
tuning; these are development-set probes, not new confirmatory evidence.
Old logs/checkpoints are never overwritten. GPU execution requires --run.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import finish_soict_checks as finish
from tools import verify_paper_results as verify

BASELINE_COMMIT = '4989ab866b83bafe04ed7aa2d23ec4f7235e22c4'
BASELINE_SOURCE = '6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9'
VARIANTS = ('route_class_zmax', 'gate_floor025', 'rp5000')


def variant_command(args, seed, tii, lora, variant):
    cmd = verify.command(args, seed, 'full', tii, lora)
    if variant == 'route_class_zmax':
        cmd += ['--rp_route_score_mode', 'class_zmax']
    elif variant == 'gate_floor025':
        cmd += ['--rp_class_gate_floor', '0.25']
    elif variant == 'rp5000':
        finish.set_flag(cmd, '--rp_dim', 5000)
    elif variant != 'reference':
        raise ValueError('Unknown variant: ' + variant)
    return cmd


def old_reference(args, seed, tii, lora, records):
    tag = ('paperfix_verify_v3' if args.dataset == 'cub200' and seed != 42
           else 'maskfix_verify_v2')
    cmd = verify.command(args, seed, 'full', tii, lora)
    meta = dict(source_sha256=BASELINE_SOURCE, checkpoints=records, command=cmd)
    return verify.run_or_read(args.output_root / verify.log_name(lora.name, 'full', tag),
                              cmd, meta, execute=False)


def driver_digest():
    # Include dependencies omitted by the model-source digest.
    return {name: finish.file_hash(ROOT / 'tools' / name) for name in
            ('try_fusion_improvements.py', 'finish_soict_checks.py',
             'verify_paper_results.py', 'audit_weight_evidence.py')}


def run_variant(args, seed, tii, lora, records, source, drivers, variant):
    if verify.source_digest() != source or driver_digest() != drivers:
        raise ValueError('Source changed during this batch; stop and use a new tag')
    cmd = variant_command(args, seed, tii, lora, variant)
    path = args.output_root / (lora.name + '__' + args.tag + '__' + variant + '.log')
    meta = dict(source_sha256=source, checkpoints=records, command=cmd,
                driver_sha256=drivers, baseline_commit=BASELINE_COMMIT,
                purpose='exploratory; no automatic parameter selection')
    if args.run and not path.exists():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        free, _ = torch.cuda.mem_get_info(0)
        if free < args.min_free_gib * 1024 ** 3:
            raise RuntimeError('GPU busy; retry later, do not stop another job')
    rows = verify.run_or_read(path, cmd, meta, execute=args.run)
    if verify.source_digest() != source or driver_digest() != drivers:
        raise ValueError('Source changed while evaluating; do not use this result')
    print('EXPERIMENT_FINAL', path.name, json.dumps(rows[10], sort_keys=True), flush=True)
    return rows


def require_reference_match(old, current):
    if not verify.compare(old, current, range(1, 11), finish.METRICS,
                          label='REFERENCE_ALL_STAGES'):
        raise ValueError('Reference regression: experiments were not started')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=verify.DATASETS, default='imr')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='fusion_probe_v1')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=16.0)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {42, 43, 44, 45}:
        parser.error('Use unique verified seeds from 42, 43, 44, 45')
    if len(set(args.variants)) != len(args.variants):
        parser.error('Duplicate variant')
    if not 5 <= args.min_free_gib <= 24:
        parser.error('--min-free-gib must be between 5 and 24')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output/data directories required')
    args.batch_size, args.port, args.data_path = 24, 29558, None
    args.resolved_data_path = (verify.resolve_imr_data_path(args.data_root)
                              if args.dataset == 'imr' else args.data_root)
    source, drivers = verify.source_digest(), driver_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    print('EXPLORATORY_ONLY; fixed probes, no test-selected winner', flush=True)
    results = {v: [] for v in ('reference', *args.variants)}
    for seed in args.seeds:
        lora = args.output_root / ('%s_lora_rank8_baseline_10tasks_seed%d' % (args.dataset, seed))
        tii = args.output_root / ('%s_tii_original_10tasks_seed%d' % (args.dataset, seed))
        records = finish.manifest(tii, lora, verify.DATASETS[args.dataset][0],
                                  'vit_base_patch16_224', seed)
        old = old_reference(args, seed, tii, lora, records)
        reference = run_variant(args, seed, tii, lora, records, source, drivers, 'reference')
        require_reference_match(old, reference)
        results['reference'].append(reference[10])
        for variant in args.variants:
            rows = run_variant(args, seed, tii, lora, records, source, drivers, variant)
            results[variant].append(rows[10])
    for variant in args.variants:
        finish.contrast(args.dataset + ' ' + variant + '_MINUS_REFERENCE',
                        results[variant], results['reference'])
    if len(args.seeds) == 1:
        print('n=1: SD=0 is a placeholder; no confidence interval or stability claim')
    print('FUSION_PROBE_COMPLETE=' + args.dataset, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit('FUSION_PROBE_STOP: ' + str(error))
