"""Opt-in inference experiments; defaults preserve the verified paper path."""
import argparse
import math


def unit_interval(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError('Expected a finite value in [0, 1]')
    return value


def add_fusion_experiments(parser):
    parser.add_argument('--rp_route_score_mode', default='task_z',
                        choices=('task_z', 'class_zmax'),
                        help='task_z: verified max-then-standardize; class_zmax: '
                             'standardize seen class scores before task max (experimental)')
    parser.add_argument('--rp_class_gate_floor', type=unit_interval, default=0.0,
                        help='Experimental margin gate floor a: a+(1-a)*gate; '
                             '0 preserves the paper, 1 gives ungated fusion')
