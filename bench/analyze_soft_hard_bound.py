#!/usr/bin/env python3
"""Soft-vs-hard bound collapse: learning term vs protected-mass horizon.

Reproduces the paper's single-curve result (Sec. 'Single-anchor references'):
across plain EWC (h=1), online EWC with gamma in {0.5, 0.9} (h ~= 1/(1-gamma)),
DeLP-fifo M=5 (h=5) and unbounded DeLP (h=T), the learning term ell is
log-linear in the protected-mass horizon at T=20, lambda=1000, n=10 seeds.

Usage: run from anywhere; resolves records relative to this file
(bench/results/ in the committed layout, results/ next to the script in
the server scratch layout).
"""
import json, math, os, statistics as st
from collections import defaultdict

LAM, T, SEEDS = 1000.0, 20, 10

_HERE = os.path.dirname(os.path.abspath(__file__))

def _p(fname):
    """Resolve a records file under both layouts (repo vs server scratch)."""
    for c in (os.path.join(_HERE, 'results', fname),   # repo layout
              os.path.join(_HERE, fname)):             # server scratch
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(fname)

def load_ewc():
    recs = []
    for f in ['r8_gamma05.json', 'r8_gamma09.json']:
        recs += [r for r in json.load(open(_p(f)))['results'] if 'error' not in r]
    for f in ['r7_ewc.json.part0', 'r7_ewc.json.part1']:
        recs += [r for r in json.load(open(_p(f)))['results'] if 'error' not in r]
    return recs

def main():
    ewc = load_ewc()
    mn = json.load(open(_p('anchor_budget_all.json')))['results']

    arms = {}
    arms['plain EWC (h=1)'] = {r['seed']: r for r in ewc
                               if r['mode'] == 'plain' and r['ntask'] == T and float(r['lam']) == LAM}
    arms['online g=0.5 (h=2)'] = {r['seed']: r for r in ewc
                                  if r['mode'] == 'online' and float(r.get('gamma', 1.0)) == 0.5
                                  and r['ntask'] == T and float(r['lam']) == LAM}
    arms['delp-fifo M5 (h=5)'] = {r['seed']: r for r in mn
                                  if r.get('dataset', 'mnist') == 'mnist' and r.get('policy') == 'fifo'
                                  and r.get('M') == 5 and r['ntask'] == T
                                  and float(r.get('lam') or 0) == LAM}
    arms['online g=0.9 (h=10)'] = {r['seed']: r for r in ewc
                                   if r['mode'] == 'online' and float(r.get('gamma', 1.0)) == 0.9
                                   and r['ntask'] == T and float(r['lam']) == LAM}
    arms['unbounded (h=20)'] = {r['seed']: r for r in mn
                                if r.get('dataset', 'mnist') == 'mnist' and r.get('policy') == 'unbounded'
                                and r['ntask'] == T and float(r.get('lam') or 0) == LAM}

    print(f"{'arm':22s} {'ell':>12s} {'acc':>12s} {'BWT':>12s}")
    hs, ells = [], []
    for name, S in arms.items():
        seeds = sorted(S)[:SEEDS]
        e = st.mean([S[s]['ell'] for s in seeds])
        a = st.mean([S[s]['final_avg'] for s in seeds])
        b = st.mean([S[s]['bwt'] for s in seeds])
        h = int(name.rsplit('h=', 1)[1].rstrip(')'))
        hs.append(h); ells.append(e)
        print(f"{name:22s} {e:6.1f}±{st.stdev([S[s]['ell'] for s in seeds]):4.1f}"
              f" {a:6.1f} {b:6.1f}  (n={len(seeds)})")

    xs = [math.log(h) for h in hs]
    mx, my = st.mean(xs), st.mean(ells)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ells)) / sum((x - mx) ** 2 for x in xs)
    alpha = my - beta * mx
    pred = [alpha + beta * x for x in xs]
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ells, pred)) / sum((y - my) ** 2 for y in ells)
    print(f"\nell = {alpha:.1f} {beta:+.1f}*ln(h)   R^2 = {r2:.3f}")

if __name__ == '__main__':
    main()
