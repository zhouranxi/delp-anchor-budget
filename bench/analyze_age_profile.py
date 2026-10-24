#!/usr/bin/env python3
"""Forgetting age-profile analysis (paper Sec. 'Single-anchor references').

Computes per-task forgetting (peak minus final accuracy) and its
correlation with task age, from the acc_matrix fields already present
in the released records. Reproduces the numbers quoted in the paper:
  drop-age correlations at T=20, lambda=1000, n=10:
    plain -0.94 | delp-fifo M5 -0.79 | online EWC -0.70 | unbounded -0.43
  fifo recent-task (15-19) drop 8.6 vs unbounded 7.3; oldest (0-4) drop 27.7.
"""
import json, statistics as st

def drops(am):
    final, T = am[-1], len(am)
    return [max(am[t][j] for t in range(j, T)) - final[j] for j in range(T)]

def load(name, parts):
    recs = []
    for f in parts:
        recs += [r for r in json.load(open(f))['results'] if 'error' not in r]
    return recs

ewc = load('ewc', ['results/r7_ewc.json.part0', 'results/r7_ewc.json.part1'])
mn  = load('mn',  ['results/anchor_budget_all.json'])

arms = {
    'plain':   {r['seed']: r for r in ewc if r['mode']=='plain'  and r['ntask']==20 and float(r['lam'])==1000},
    'online':  {r['seed']: r for r in ewc if r['mode']=='online' and r['ntask']==20 and float(r['lam'])==1000},
    'fifo':    {r['seed']: r for r in mn  if r.get('dataset','mnist')=='mnist' and r.get('policy')=='fifo'
                and r.get('M')==5 and float(r.get('lam',0))==1000 and r['ntask']==20},
    'unbounded': {r['seed']: r for r in mn if r.get('dataset','mnist')=='mnist'
                and r.get('policy')=='unbounded' and float(r.get('lam',0))==1000 and r['ntask']==20},
}

print(f"{'arm':10s} {'drop 0-4':>12s} {'drop 15-19':>12s} {'corr(drop,age)':>15s}")
for name, d in arms.items():
    seeds = sorted(d)
    profs = [drops(d[s]['acc_matrix']) for s in seeds]
    T = len(profs[0])
    def seg(ix):
        v = [st.mean([p[i] for i in ix]) for p in profs]
        return st.mean(v)
    cors = []
    for p in profs:
        ages = list(range(T)); ma, mp = st.mean(ages), st.mean(p)
        cov = sum((a-ma)*(x-mp) for a, x in zip(ages, p))
        va, vp = sum((a-ma)**2 for a in ages), sum((x-mp)**2 for x in p)
        cors.append(cov/(va*vp)**.5 if vp > 0 else 0.0)
    print(f"{name:10s} {seg(range(0,5)):8.2f}  {seg(range(15,20)):8.2f}  {st.mean(cors):+9.2f}  (n={len(seeds)})")
