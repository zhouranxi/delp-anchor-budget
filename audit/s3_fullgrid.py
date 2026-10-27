# -*- coding: utf-8 -*-
"""S1 剂量表全 22 格（含 branch 粒度）三列重算，用于核对 S1 正文分母与 S3 的转述。"""
import json, glob, os, math
from statistics import mean, stdev

ROOT = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27'
RES = os.path.join(ROOT, 'dab_t/delp-anchor-budget-main/bench/results')
RES2 = os.path.join(ROOT, 'dab_t/delp-anchor-budget-main/S1_剂量响应审计/results')

def load(pat, base=RES):
    out = []
    for f in sorted(glob.glob(os.path.join(base, pat))):
        d = json.load(open(f))
        out += d['results'] if isinstance(d, dict) else d
    return out

def ob(x):
    return x.get('own_bias', 0) or 0

def tstat(d):
    n = len(d); m = mean(d); sd = stdev(d)
    return m / (sd / math.sqrt(n)) if sd else 0.0

# 收集所有 neuron/branch 剂量记录
recs = []
for ds in ('mnist', 'fashion'):
    for b in (0, 1, 3, 10):
        for pat in (f'c2_{ds}_b{b}.json.part*', f'c2t20_{ds}_b{b}.json.part*',
                    f'c2_{ds}_b{b}.json', f'c2t20_{ds}_b{b}_neuron.json',
                    f'c2t20_{ds}_b{b}.json'):
            recs += load(pat)
            recs += load(pat, RES2)

cells = {}
for x in recs:
    if x.get('kind') != 'gran' and x.get('gran') not in ('neuron', 'branch'):
        continue
    if 'error' in x:
        continue
    g = x.get('gran'); T = x['ntask']; b = ob(x)
    cells.setdefault((x['dataset'], T, g, b), {})[x['seed']] = x

print('=' * 96)
print('S1 剂量网格：全部可用 cell（neuron + branch）')
print('=' * 96)
rows = []
for ds in ('mnist', 'fashion'):
    for T in (10, 20):
        for g in ('neuron', 'branch'):
            for b in (1, 3, 10):
                a = cells.get((ds, T, g, b), {})
                z = cells.get((ds, T, g, 0), {})
                common = sorted(set(a) & set(z))
                if len(common) < 2:
                    print(f'  {ds:<8} T={T:<3} {g:<7} b={b:<3} : 数据不足 (n={len(common)})')
                    continue
                dl = [a[s]['ell'] - z[s]['ell'] for s in common]
                da = [a[s]['final_avg'] - z[s]['final_avg'] for s in common]
                db = [a[s]['bwt'] - z[s]['bwt'] for s in common]
                tl, ta, tb = tstat(dl), tstat(da), tstat(db)
                rows.append(dict(ds=ds, T=T, g=g, b=b, n=len(common),
                                 dl=mean(dl), tl=tl, da=mean(da), ta=ta, db=mean(db), tb=tb))
                print(f'  {ds:<8} T={T:<3} {g:<7} b={b:<3} n={len(common)} '
                      f'Δell={mean(dl):+6.3f}(t{tl:+6.2f}){"*" if abs(tl)>2 else " "} '
                      f'Δavg={mean(da):+6.3f}(t{ta:+6.2f}){"*" if abs(ta)>2 else " "} '
                      f'Δbwt={mean(db):+6.3f}(t{tb:+6.2f}){"*" if abs(tb)>2 else " "}')

n = len(rows)
npos = sum(1 for r in rows if r['dl'] > 0)
nsigl = sum(1 for r in rows if abs(r['tl']) > 2)
nboth = sum(1 for r in rows if abs(r['tl']) > 2 and abs(r['ta']) > 2)
print(f'\n  总格数 {n}')
print(f'  Δell > 0        : {npos}/{n}')
print(f'  Δell* (|t|>2)   : {nsigl}/{n}')
print(f'  Δell* & Δavg*   : {nboth}/{n}')
print(f'  |t| 最大(Δell)  : {max(abs(r["tl"]) for r in rows):.2f}  '
      f'(cell: {[(r["ds"],r["T"],r["g"],r["b"]) for r in rows if abs(r["tl"])==max(abs(x["tl"]) for x in rows)]})')
print(f'  |t| 最大(Δavg)  : {max(abs(r["ta"]) for r in rows):.2f}')
print(f'  |t| 最大(Δbwt)  : {max(abs(r["tb"]) for r in rows):.2f}')
print(f'  全表最大 |t|    : {max(max(abs(r["tl"]),abs(r["ta"]),abs(r["tb"])) for r in rows):.2f}')
print(f'  Δell 最小       : {min(r["dl"] for r in rows):+.3f}  '
      f'(cell: {[(r["ds"],r["T"],r["g"],r["b"],round(r["tl"],2)) for r in rows if r["dl"]==min(x["dl"] for x in rows)]})')
print(f'  Δell 最大       : {max(r["dl"] for r in rows):+.3f}  '
      f'(cell: {[(r["ds"],r["T"],r["g"],r["b"],round(r["tl"],2)) for r in rows if r["dl"]==max(x["dl"] for x in rows)]})')

print('\n  --- T=10 ΔBWT 统计 ---')
t10 = [r for r in rows if r['T'] == 10]
print(f'  Δbwt 显著数: {sum(1 for r in t10 if abs(r["tb"])>2)}/{len(t10)}')
print('\n  --- T=20 ΔBWT 显著负 ---')
t20 = [r for r in rows if r['T'] == 20]
neg = [r for r in t20 if r['db'] < 0 and abs(r['tb']) > 2]
print(f'  {len(neg)}/{len(t20)} : {[(r["ds"],r["g"],r["b"],round(r["db"],2),round(r["tb"],2)) for r in neg]}')
print(f'  其余 {len(t20)-len(neg)} 格与基线无显著差异')

print('\n  --- neuron 粒度单独统计（S3 引用口径） ---')
neur = [r for r in rows if r['g'] == 'neuron']
print(f'  neuron {len(neur)} 格: Δell>0 {sum(1 for r in neur if r["dl"]>0)}/{len(neur)}, '
      f'Δell* {sum(1 for r in neur if abs(r["tl"])>2)}/{len(neur)}, '
      f'双显著 {sum(1 for r in neur if abs(r["tl"])>2 and abs(r["ta"])>2)}/{len(neur)}')
print(f'  neuron Δell 范围: {min(r["dl"] for r in neur):+.3f} .. {max(r["dl"] for r in neur):+.3f}')
print(f'  neuron Δell* 的最小值: {min(r["dl"] for r in neur if abs(r["tl"])>2):+.3f}')
print(f'  neuron 非显著格的最小 |t|: {min(abs(r["tl"]) for r in neur if abs(r["tl"])<=2):.2f}')
