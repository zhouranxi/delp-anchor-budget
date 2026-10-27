# -*- coding: utf-8 -*-
"""S4 (Repair Bypasses the Gate) 数据一致性审计。"""
import json, glob, os, math
from statistics import mean, stdev

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, '..')

def _find_results():
    cands = [
        os.path.join(_REPO, 'S4_gate_bypass', 'results'),
        os.path.join(_REPO, 'papers/s4_zenodo/S4_repair_bypasses_gate/results'),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit('results dir not found; looked in: %r' % cands)

R = _find_results()

def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        d = json.load(open(f))
        out += d['results'] if isinstance(d, dict) else d
    return out

def tstat(d):
    n = len(d)
    if n < 2: return float('nan')
    m = mean(d); sd = stdev(d)
    return m / (sd / math.sqrt(n)) if sd else 0.0

# ---- hard noise (E1/E2)
hard = load('s4ctx_hard.part*')
print('=' * 80)
print('S4 审计：hard-noise 数据集')
print('=' * 80)
from collections import defaultdict
cells = defaultdict(dict)
for x in hard:
    cells[(x['dataset'], x['bias'], x['soft'])][x['seed']] = x
for k in sorted(cells):
    v = cells[k]
    seeds = sorted(v)
    print(f'  ds={k[0]:<8} bias={k[1]:<5} soft={k[2]} n={len(seeds)} seeds={seeds}')

RHOS = ['0.0', '0.1', '0.2', '0.4']

print('\n' + '=' * 80)
print('表 1  Table main：final_avg 按 rho')
print('=' * 80)
paper_main = {
    ('mnist', 0.0):  [71.41, 71.41, 71.42, 71.40],
    ('mnist', 10.0): [72.63, 72.63, 72.63, 72.63],
    ('fashion', 0.0):  [66.38, 66.39, 66.40, 66.41],
    ('fashion', 10.0): [66.84, 66.84, 66.84, 66.84],
}
means = {}
for (ds, b), pv in paper_main.items():
    v = cells[(ds, b, False)]
    got = [round(mean([v[s]['final_avg_by_rho'][r] for s in sorted(v)]), 2) for r in RHOS]
    ok = all(abs(g - p) < 0.006 for g, p in zip(got, pv))
    means[(ds, b)] = {r: mean([v[s]['final_avg_by_rho'][r] for s in sorted(v)]) for r in RHOS}
    print(f'  {"OK " if ok else "XX "}{ds:<8} b={b:<5} got={got}  paper={pv}')

print('\n  --- 逐 seed 差异（配对 b=10 vs b=0）---')
for ds in ('mnist', 'fashion'):
    z = cells[(ds, 0.0, False)]; a = cells[(ds, 10.0, False)]
    common = sorted(set(z) & set(a))
    print(f'  {ds}: common seeds n={len(common)}')
    for r in RHOS:
        d = [a[s]['final_avg_by_rho'][r] - z[s]['final_avg_by_rho'][r] for s in common]
        t = tstat(d)
        print(f'    rho={r}: Δa={mean(d):+.3f} (t={t:+.2f}) {"*" if abs(t) > 2 else " "}  '
              f'per-seed={[round(x,2) for x in d]}')

print('\n  --- retention (b=10 gain 在 rho 下 / rho=0) ---')
for ds in ('mnist', 'fashion'):
    z = cells[(ds, 0.0, False)]; a = cells[(ds, 10.0, False)]
    common = sorted(set(z) & set(a))
    g0 = mean([a[s]['final_avg_by_rho']['0.0'] - z[s]['final_avg_by_rho']['0.0'] for s in common])
    for r in RHOS[1:]:
        g = mean([a[s]['final_avg_by_rho'][r] - z[s]['final_avg_by_rho'][r] for s in common])
        print(f'    {ds:<8} rho={r}: gain {g:+.3f} / {g0:+.3f} = {g/g0*100:.1f}%')

print('\n  --- 退化幅度（rho>0 相对 rho=0 的绝对差）---')
for ds in ('mnist', 'fashion'):
    for b in (0.0, 10.0):
        v = cells[(ds, b, False)]
        for r in RHOS[1:]:
            d = [v[s]['final_avg_by_rho'][r] - v[s]['final_avg_by_rho']['0.0'] for s in sorted(v)]
            print(f'    {ds:<8} b={b:<5} rho={r}: Δ={mean(d):+.3f} (max|Δ| per seed {max(abs(x) for x in d):.2f})')

print('\n' + '=' * 80)
print('表 2  Table ablation：no-gate control vs repair vs base')
print('=' * 80)
nod1 = {x['seed']: x for x in load('s4nodend.json') if x['dataset'] == 'mnist'}
nod2 = {x['seed']: x for x in load('s4nodend2.json') if x['dataset'] == 'mnist'}
for nm, d in (('s4nodend.json (v1 无效版)', nod1), ('s4nodend2.json (v2 修正版)', nod2)):
    print(f'  {nm}: n={len(d)} seeds={sorted(d)}')

print('\n  --- 各臂 final_avg ---')
paper_ab = {
    'mnist':   dict(rep=72.63, nod=73.22, base=71.41, diff=+1.81),
    'fashion': dict(rep=66.84, nod=64.30, base=66.38, diff=-2.09),
}
for ds in ('mnist', 'fashion'):
    z = cells[(ds, 0.0, False)]; a = cells[(ds, 10.0, False)]
    nod = {x['seed']: x for x in load('s4nodend2.json') if x['dataset'] == ds}
    nod1 = {x['seed']: x for x in load('s4nodend.json') if x['dataset'] == ds}
    common = sorted(set(z) & set(a) & set(nod))
    print(f'\n  === {ds} ===  common seeds n={len(common)}')
    rep = mean([a[s]['final_avg_by_rho']['0.0'] for s in common])
    bas = mean([z[s]['final_avg_by_rho']['0.0'] for s in common])
    n2 = mean([nod[s]['final_avg'] for s in common])
    n1 = mean([nod1[s]['final_avg'] for s in common])
    p = paper_ab[ds]
    print(f'    base   (b=0)      = {bas:.2f}  (paper {p["base"]})')
    print(f'    repaired(b=10)    = {rep:.2f}  (paper {p["rep"]})')
    print(f'    no-gate v2        = {n2:.2f}  (paper {p["nod"]})')
    print(f'    no-gate v1(无效)  = {n1:.2f}')
    d_nb = [nod[s]['final_avg'] - z[s]['final_avg_by_rho']['0.0'] for s in common]
    d_nr = [nod[s]['final_avg'] - a[s]['final_avg_by_rho']['0.0'] for s in common]
    d_rb = [a[s]['final_avg_by_rho']['0.0'] - z[s]['final_avg_by_rho']['0.0'] for s in common]
    print(f'    no-gate - base  = {mean(d_nb):+.2f} (t={tstat(d_nb):+.2f}) {"*" if abs(tstat(d_nb))>2 else " "}  (paper {p["diff"]:+.2f})')
    print(f'    no-gate - repair= {mean(d_nr):+.2f} (t={tstat(d_nr):+.2f}) {"*" if abs(tstat(d_nr))>2 else " "}')
    print(f'    repair - base   = {mean(d_rb):+.2f} (t={tstat(d_rb):+.2f}) {"*" if abs(tstat(d_rb))>2 else " "}')

print('\n' + '=' * 80)
print('E3 soft mixture')
print('=' * 80)
soft = load('s4ctx_soft.json')
scells = defaultdict(dict)
for x in soft:
    scells[(x['dataset'], x['bias'])][x['seed']] = x
for k in sorted(scells):
    v = scells[k]
    got = [round(mean([v[s]['final_avg_by_rho'][r] for s in sorted(v)]), 3) for r in RHOS]
    d = [v[s]['final_avg_by_rho']['0.4'] - v[s]['final_avg_by_rho']['0.0'] for s in sorted(v)]
    print(f'  ds={k[0]:<8} b={k[1]:<5} means={got}  |Δ(0.4-0)| max={max(abs(x) for x in d):.3f}')

print('\n  --- soft vs hard 逐 seed 配对（b=0, rho=0.4）---')
for ds in ('mnist', 'fashion'):
    h = cells[(ds, 0.0, False)]; s = scells[(ds, 0.0)]
    common = sorted(set(h) & set(s))
    d = [s[k]['final_avg_by_rho']['0.4'] - h[k]['final_avg_by_rho']['0.4'] for k in common]
    print(f'    {ds}: soft-hard Δ={mean(d):+.3f} (t={tstat(d):+.2f}) n={len(common)}')
