#!/usr/bin/env python3
"""Audit the fusion-weight evidence in the paper against the evaluation logs.

The paper's Sect. 4.6 makes four claims that can only be checked against the
runs themselves: the nine cells of the w-by-beta grid, the four-seed comparison
between w = 0.6 and w = 0.7, the provenance of the two self-run rows of the
placement table, and whether a class-plus-gate control without routing was ever
run. This reads every log under the output root, reconstructs each run's
configuration from its filename, and reports what exists.

The filename is the record: eval_rp_head_any_4090.sh encodes source, dimension,
lambda, the fusion flags, w, beta, the gate and the ramp into the log path, with
dots written as 'p'. Nothing here re-runs anything or writes to the run
directories.

Usage:  python3 audit_weight_evidence.py [OUTPUT_ROOT]
        default OUTPUT_ROOT is ~/hrm-pet-output
"""
import glob
import os
import re
import sys
from collections import defaultdict

FINAL = re.compile(r'\[Average accuracy till task(\d+)\]\s*(.*)')
NUM = re.compile(r'([A-Za-z@0-9/]+):\s*(-?\d+(?:\.\d+)?)')
T3 = 3.182  # two-sided 95% Student-t, three degrees of freedom


def undot(s):
    return s.replace('p', '.')


def read_final(path):
    """Return (num_tasks, {metric: value}) from the last final-average line."""
    last = None
    try:
        with open(path, 'r', errors='replace') as handle:
            for line in handle:
                found = FINAL.search(line)
                if found:
                    last = found
    except OSError:
        return None, {}
    if last is None:
        return None, {}
    return int(last.group(1)), {k: float(v) for k, v in NUM.findall(last.group(2))}


def parse_name(name):
    """Reconstruct the run configuration from the log filename."""
    cfg = {'file': name, 'run': name.split('_eval_')[0]}
    seed = re.search(r'seed(\d+)', name)
    cfg['seed'] = seed.group(1) if seed else '?'
    if '_eval_conventional' in name:
        cfg['kind'] = 'baseline'
        return cfg
    if '_eval_rp_' not in name:
        cfg['kind'] = 'other'
        return cfg
    cfg['kind'] = 'rp'
    fuse = re.search(r'_f(\d)d(\d)w([0-9p]+)lsw', name)
    if fuse:
        cfg['route_fusion'] = fuse.group(1) == '1'
        cfg['route_drm'] = fuse.group(2) == '1'
        cfg['w'] = undot(fuse.group(3))
    beta = re.search(r'cw([0-9p]+)sh', name)
    cfg['beta'] = undot(beta.group(1)) if beta else '?'
    cfg['gate'] = 'margin_both' if 'gmargin_both' in name else (
        'margin' if 'gmargin' in name else (
            'entropy' if 'gentropy' in name else 'none'))
    cfg['ramp'] = 'yes' if re.search(r'm\d+(?:g[a-z_]+)?r\d+p\d+', name) else 'no'
    src = re.search(r'_eval_rp_([a-z]+)_d(\d+)_([a-z]+)_l([0-9p]+)', name)
    if src:
        cfg['source'], cfg['dim'] = src.group(1), src.group(2)
        cfg['act'], cfg['lambda'] = src.group(3), undot(src.group(4))
    return cfg


def stage(cfg):
    """Name the experimental configuration in the paper's own terms."""
    if cfg['kind'] != 'rp':
        return cfg['kind']
    w = cfg.get('w', '?')
    beta = cfg.get('beta', '?')
    routed = cfg.get('route_fusion') or cfg.get('route_drm')
    if not routed and beta in ('0.0', '?'):
        return 'RP-only'
    if beta == '0.0':
        return 'routing-only'
    if w == '1.0':
        return 'class-only' + ('+gate' if cfg['gate'] != 'none' else '')
    return 'full' + ('+gate' if cfg['gate'] != 'none' else '')


def paired(pairs):
    """Mean, sample SD and 95% t interval of a list of paired differences."""
    n = len(pairs)
    mean = sum(pairs) / n
    if n < 2:
        return mean, 0.0, None, None
    var = sum((d - mean) ** 2 for d in pairs) / (n - 1)
    sd = var ** 0.5
    half = T3 * sd / (n ** 0.5)
    return mean, sd, mean - half, mean + half


def main(root):
    logs = sorted(glob.glob(os.path.join(root, '*.log')))
    print('thu muc :', root)
    print('so log  :', len(logs))
    if not logs:
        print('KHONG THAY LOG. Truyen duong dan that lam tham so.')
        return 1

    rows = []
    for path in logs:
        cfg = parse_name(os.path.basename(path))
        tasks, met = read_final(path)
        if not met:
            continue
        cfg['tasks'] = tasks
        cfg['met'] = met
        cfg['stage'] = stage(cfg)
        rows.append(cfg)
    print('log co dong ket qua cuoi:', len(rows))
    print()

    # ---- 1. inventory ---------------------------------------------------
    print('=' * 78)
    print('1. KIEM KE: moi cau hinh da chay')
    print('=' * 78)
    print('%-26s %-5s %-16s %-5s %-5s %-7s %8s %8s'
          % ('run', 'seed', 'stage', 'w', 'beta', 'gate', 'Acc@task', 'Acc@1'))
    for r in sorted(rows, key=lambda r: (r['run'], r['stage'],
                                         r.get('w', ''), r.get('beta', ''))):
        print('%-26s %-5s %-16s %-5s %-5s %-7s %8.4f %8.4f'
              % (r['run'][:26], r['seed'], r['stage'], r.get('w', '-'),
                 r.get('beta', '-'), r.get('gate', '-'),
                 r['met'].get('Acc@task', float('nan')),
                 r['met'].get('Acc@1', float('nan'))))
    print()

    # ---- 2. the w-by-beta grid -----------------------------------------
    print('=' * 78)
    print('2. LUOI w x beta (bang 3 trong bai): can du 9 o, mot seed, gate margin')
    print('=' * 78)
    grid = defaultdict(dict)
    for r in rows:
        if r['stage'].startswith('full') and r.get('gate') == 'margin':
            grid[(r['run'], r['seed'])][(r.get('w'), r.get('beta'))] = \
                r['met'].get('Acc@1')
    if not grid:
        print('KHONG co o nao khop (full + gate margin).')
    for (run, seed), cells in sorted(grid.items()):
        print('%s seed %s: %d o' % (run, seed, len(cells)))
        ws = sorted({w for w, _ in cells})
        bs = sorted({b for _, b in cells})
        print('       ' + ''.join('%10s' % ('beta=' + b) for b in bs))
        for w in ws:
            line = '  w=%-4s' % w
            for b in bs:
                v = cells.get((w, b))
                line += '%10s' % ('-' if v is None else '%.2f' % v)
            print(line)
        if len(cells) < 9:
            print('  THIEU %d o so voi bang trong bai.' % (9 - len(cells)))
    print()

    # ---- 3. w = 0.6 against w = 0.7, paired over seeds -------------------
    print('=' * 78)
    print('3. w=0.6 SO w=0.7 GHEP CAP THEO SEED (cung beta, cung gate, cung mode)')
    print('=' * 78)
    buckets = defaultdict(dict)
    for r in rows:
        if r['kind'] != 'rp' or r.get('w') not in ('0.6', '0.7'):
            continue
        dataset = r['run'].rsplit('_seed', 1)[0]
        key = (dataset, r['stage'], r.get('beta'), r.get('gate'))
        buckets[key].setdefault(r['seed'], {})[r['w']] = r['met']
    if not buckets:
        print('KHONG co cap w nao de so.')
    for key, by_seed in sorted(buckets.items()):
        dataset, st, beta, gate = key
        seeds = sorted(s for s, d in by_seed.items() if '0.6' in d and '0.7' in d)
        print('%s | %s | beta=%s gate=%s : %d seed ghep duoc %s'
              % (dataset, st, beta, gate, len(seeds), seeds))
        if len(seeds) < 2:
            continue
        for name in ('Acc@1', 'Acc@task', 'Forgetting', 'Backward'):
            diffs = []
            for s in seeds:
                a = by_seed[s]['0.6'].get(name)
                b = by_seed[s]['0.7'].get(name)
                if a is not None and b is not None:
                    diffs.append(a - b)
            if len(diffs) != len(seeds):
                continue
            mean, sd, lo, hi = paired(diffs)
            span = '' if lo is None else '  CI95 [%+.3f, %+.3f]' % (lo, hi)
            print('   %-11s 0.6 - 0.7 = %+.4f +- %.4f%s' % (name, mean, sd, span))
    print()

    # ---- 4. controls the paper says it does not have --------------------
    print('=' * 78)
    print('4. HAI DOI CHUNG BAI DANG THIEU')
    print('=' * 78)
    for want, label in (('class-only+gate', 'class+gate, khong routing'),
                        ('RP-only', 'RP-only / RanPAC cung protocol')):
        got = defaultdict(set)
        for r in rows:
            if r['stage'] == want:
                got[r['run'].rsplit('_seed', 1)[0]].add(r['seed'])
        if not got:
            print('%-34s: KHONG CO log nao' % label)
        for dataset, seeds in sorted(got.items()):
            print('%-34s: %s -> seed %s'
                  % (label, dataset, sorted(seeds)))
    print()

    # ---- 5. what the placement table quotes -----------------------------
    print('=' * 78)
    print('5. HAI HANG TU CHAY CUA BANG 2 (ImageNet-A va 5-Datasets)')
    print('=' * 78)
    for r in sorted(rows, key=lambda r: r['run']):
        low = r['run'].lower()
        if 'imagenet-a' in low or 'imr_a' in low or '5-dataset' in low \
                or 'five' in low or 'inat' in low:
            print('%-30s seed %-4s %-16s tasks=%s Acc@1=%.4f'
                  % (r['run'][:30], r['seed'], r['stage'], r['tasks'],
                     r['met'].get('Acc@1', float('nan'))))
            print('     %s' % r['file'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else os.path.expanduser('~/hrm-pet-output')))
