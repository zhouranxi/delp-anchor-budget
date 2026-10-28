"""Compare a server-side re-run against the local CPU result.

Usage:
    python compare_server_cpu.py --server results_s5/server_X.json \
                                 --local  results_s5/X.json

Prints per-arm means for ell / final_avg / bwt, the absolute and relative
difference, and a PASS/FAIL verdict against a tolerance.  A systematic
offset between the two backends would mean the paper's numbers are
backend-specific and should say so; random scatter of the size of the
float32 seed noise is fine.
"""
import argparse
import json
import math
import os
from collections import defaultdict


def load(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d, dict):
        for k in ('rows', 'results'):
            if k in d and isinstance(d[k], list):
                return d[k]
        return d
    return d


# fields that hold SCALAR measurements (not per-run records)
SCALAR_FIELDS = ('ell', 'final_avg', 'bwt', 'gate_mean', 'gate_min', 'gate_max',
                  'own_segment_win_frac', 'offdiag_block_rel_move',
                  'offdiag_blocks_frozen_frac', 'diag_block_rel_move',
                  'col_cos_mean_trained', 'col_cos_min_trained',
                  'col_cos_mean_init', 'col_cos_min_init',
                  'col_cos_trained_vs_init', 'flip_mean', 'flip_max',
                  'flip_min')


def group(recs, key_fields, metrics=None):
    """Group records by key fields and collect the scalar measurements.

    Handles two shapes:
      * per-run records  (rows with 'seed' and one value per metric)
      * already-aggregated rows (one value per metric, no seed)
    """
    out = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if not isinstance(r, dict):
            continue
        key = tuple(r.get(k) for k in key_fields)
        ms = metrics if metrics is not None else SCALAR_FIELDS
        for m in ms:
            if r.get(m) is not None:
                out[key][m].append(r[m])
    return out


def mean(v):
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--local', required=True)
    ap.add_argument('--keys', default='dataset,arm',
                    help='comma-separated record fields used to group arms')
    ap.add_argument('--tol', type=float, default=0.1,
                    help='absolute tolerance on the per-arm means')
    ap.add_argument('--rel-tol', type=float, default=0.01)
    args = ap.parse_args()

    if not os.path.exists(args.server):
        print(f'FAIL: server file not found: {args.server}')
        return 1
    if not os.path.exists(args.local):
        print(f'FAIL: local file not found: {args.local}')
        return 1

    keys = [k.strip() for k in args.keys.split(',')]
    s = group(load(args.server), keys)
    l = group(load(args.local), keys)

    print(f'{"group":<38}{"metric":<12}{"server":>10}{"local":>10}'
          f'{"abs diff":>11}{"rel":>10}  verdict')
    print('-' * 96)
    worst_abs = 0.0
    worst_rel = 0.0
    n = 0
    n_bad = 0
    for key in sorted(set(s) | set(l), key=str):
        sv = s.get(key, {})
        lv = l.get(key, {})
        for m in sorted(set(sv) | set(lv)):
            sm, lm = mean(sv.get(m, [])), mean(lv.get(m, []))
            if sm is None or lm is None:
                print(f'{str(key):<38}{m:<12}{("-" if sm is None else f"{sm:.4f}"):>10}'
                      f'{("-" if lm is None else f"{lm:.4f}"):>10}   MISSING')
                continue
            d = sm - lm
            rel = abs(d) / max(abs(lm), 1e-9)
            worst_abs = max(worst_abs, abs(d))
            worst_rel = max(worst_rel, rel)
            n += 1
            ok = abs(d) <= args.tol or rel <= args.rel_tol
            if not ok:
                n_bad += 1
            print(f'{str(key):<38}{m:<12}{sm:>10.4f}{lm:>10.4f}'
                  f'{d:>11.4f}{rel:>9.2%}  {"OK" if ok else "DIFF"}')
    print('-' * 96)
    print(f'comparisons: {n}   outside tolerance: {n_bad}   '
          f'worst |abs diff|: {worst_abs:.4f}   worst |rel|: {worst_rel:.2%}')
    if n_bad == 0:
        print('VERDICT: PASS -- every comparison is within tolerance '
              '(no systematic difference between backends)')
        return 0
    print(f'VERDICT: {n_bad} comparison(s) outside tolerance -- investigate '
          f'before trusting either backend')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
