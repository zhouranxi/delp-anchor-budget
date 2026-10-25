# -*- coding: utf-8 -*-
"""S1 (Branch-DeLP) 配对分析：branch/neuron/plain 三臂，逐 (dataset, T, lam) 组合配对 t 检验。"""
import json, math, os, itertools, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def _resolve_base():
    """Locate results/: bench/results (committed) -> CWD/results (server scratch) -> legacy path."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "results"), "results"):
        if os.path.isdir(cand):
            return cand
    return r'D:/WorkBuddy/2026-10-17-14-29-40/delp-anchor-budget/bench/results'

BASE = _resolve_base()
FILES = ['s1_mnist.json', 's1_fashion.json', 's1_refs.json', 's1_l1000_t10.json', 's1_l1000_t20.json']

records, seen = [], set()
for f in FILES:
    for r in json.load(open(f'{BASE}/{f}', encoding='utf-8'))['results']:
        if r.get('error'):
            continue
        if 'ell' not in r:  # si/mas 无存储 ell，按恒等式派生：ell = final_avg - (T-1)/T*BWT
            r['ell'] = round(r['final_avg'] - (r['ntask'] - 1) / r['ntask'] * r['bwt'], 2)
        key = (r['dataset'], r['kind'], r.get('gran'), r['ntask'], r['lam'], r['seed'], round(r['ell'], 4),
               round(r['final_avg'], 4), round(r['bwt'], 4))
        if key in seen:
            continue
        seen.add(key)
        records.append(r)

def group_stats(rs):
    n = len(rs)
    ell = sum(r['ell'] for r in rs) / n
    favg = sum(r['final_avg'] for r in rs) / n
    bwt = sum(r['bwt'] for r in rs) / n
    sd = lambda xs: (sum((x - sum(xs)/len(xs))**2 for x in xs) / (len(xs)-1)) ** 0.5 if n > 1 else float('nan')
    return dict(n=n, ell=ell, favg=favg, bwt=bwt,
                sd_ell=sd([r['ell'] for r in rs]), sd_favg=sd([r['final_avg'] for r in rs]), sd_bwt=sd([r['bwt'] for r in rs]))

def paired_t(xs, ys):
    """t = mean(d)/(sd(d)/sqrt(n)), d = x - y; 配对自由度 n-1."""
    n = len(xs)
    d = [x - y for x, y in zip(xs, ys)]
    m = sum(d) / n
    s = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5
    t = m / (s / math.sqrt(n)) if s > 0 else float('inf') * (1 if m else 0)
    return m, t

# 按 (dataset, T, lam) 分组，gran 方法内部按 seed 配对
combos = {}
for r in records:
    if r['kind'] != 'gran':
        continue
    combos.setdefault((r['dataset'], r['ntask'], r['lam']), {}).setdefault(r['gran'], {})[r['seed']] = r

print(f'{"dataset":8s} {"T":>3s} {"lam":>5s} | {"对比":14s} {"n":>2s} | '
      f'{"Δℓ":>7s} {"t(ℓ)":>7s} {"Δfavg":>7s} {"t(favg)":>8s} {"ΔBWT":>7s} {"t(BWT)":>7s} | '
      f'{"ℓ(b/n/p)":26s} {"BWT(b/n/p)":24s}')
rows = []
for (ds, T, lam) in sorted(combos, key=str):
    g = combos[(ds, T, lam)]
    for a, b in [('branch', 'neuron'), ('plain', 'neuron')]:
        if a not in g or b not in g:
            continue
        sa, sb = g[a], g[b]
        seeds = sorted(set(sa) & set(sb))
        ra = [sa[s] for s in seeds]; rb = [sb[s] for s in seeds]
        m_ell, t_ell = paired_t([r['ell'] for r in ra], [r['ell'] for r in rb])
        m_fav, t_fav = paired_t([r['final_avg'] for r in ra], [r['final_avg'] for r in rb])
        m_bwt, t_bwt = paired_t([r['bwt'] for r in ra], [r['bwt'] for r in rb])
        Sta, Stb = group_stats(ra), group_stats(rb)
        rows.append(dict(dataset=ds, T=T, lam=lam, cmp=f'{a} vs {b}', n=len(seeds),
                         d_ell=m_ell, t_ell=t_ell, d_favg=m_fav, t_favg=t_fav, d_bwt=m_bwt, t_bwt=t_bwt))
        print(f'{ds:8s} {T:3d} {lam:5.0f} | {a+" vs "+b:14s} {len(seeds):2d} | '
              f'{m_ell:+7.2f} {t_ell:+7.2f} {m_fav:+7.2f} {t_fav:+8.2f} {m_bwt:+7.2f} {t_bwt:+7.2f} | '
              f'{Sta["ell"]:5.2f}/{Stb["ell"]:5.2f}/{group_stats(list(g.get("plain",{}).values())).get("ell",float("nan")) if "plain" in g else float("nan"):5.2f} '
              f'({Sta["sd_ell"]:4.2f}/{Stb["sd_ell"]:4.2f}) '
              f'{Sta["bwt"]:+6.2f}/{Stb["bwt"]:+6.2f}/{group_stats(list(g.get("plain",{}).values())).get("bwt",float("nan")) if "plain" in g else float("nan"):+6.2f}')

# 基线参考（si/mas，均值仅供参考，seed 数不同不做配对）
print('\n--- 基线参考 (si / mas，非配对) ---')
for r in records:
    if r['kind'] in ('si', 'mas'):
        pass
baselines = {}
for r in records:
    if r['kind'] in ('si', 'mas'):
        baselines.setdefault((r['dataset'], r['ntask'], r['lam'], r['kind']), []).append(r)
for k in sorted(baselines, key=str):
    st = group_stats(baselines[k])
    print(f'{k[0]:8s} T{k[1]:<3d} lam{k[2]:<6.0f} {k[3]:>4s}: n={st["n"]:2d}  ℓ={st["ell"]:6.2f}±{st["sd_ell"]:4.2f}  '
          f'final_avg={st["favg"]:6.2f}±{st["sd_favg"]:4.2f}  BWT={st["bwt"]:+6.2f}±{st["sd_bwt"]:4.2f}')

json.dump(rows, open(f'{BASE}/s1_paired_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved -> s1_paired_stats.json')
