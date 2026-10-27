# -*- coding: utf-8 -*-
"""S3 (Repair Headroom) 数据一致性审计：从 results/ 原始记录重算论文 main.tex 全部数字。"""
import json, glob, os, math, sys
from statistics import mean, pstdev, stdev

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, '..')

def _find_results():
    """在仓库内定位 S3 的 results 目录（兼容两种布局）"""
    cands = [
        os.path.join(_REPO, 'S3_修复边界', 'results'),
        os.path.join(_REPO, 'papers', 's3_zenodo', 'S3_repair_headroom', 'results'),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit('results dir not found; looked in: %r' % cands)

R = _find_results()

def load(f):
    p = os.path.join(R, f)
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    return d['results'] if isinstance(d, dict) else d

def load_parts(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pat))):
        out += load(os.path.basename(f))
    return out

def paired(a, b):
    """a,b 为 {seed: rec}；返回 (diff列表, t, mean_diff)"""
    seeds = sorted(set(a) & set(b))
    d = [a[s]['ell'] - b[s]['ell'] for s in seeds]
    return d

def mean_diff(a, b, key):
    seeds = sorted(set(a) & set(b))
    return mean([a[s][key] - b[s][key] for s in seeds])

def tstat(d):
    n = len(d)
    if n < 2:
        return float('nan')
    m = mean(d)
    sd = stdev(d)
    if sd == 0:
        return float('inf') if m else 0.0
    return m / (sd / math.sqrt(n))

def sig(t):
    return abs(t) > 2

def r_(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/ (n-1))
    sy = math.sqrt(sum((y-my)**2 for y in ys)/ (n-1))
    if sx == 0 or sy == 0:
        return float('nan')
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / ((n-1)*sx*sy)

# ---------------------------------------------------------------- 载入
# Conv: cnnc2_* (T=10) / cnnc2t20_* (T=20)
conv = {}
for ds in ('mnist', 'fashion'):
    for b in (0, 1, 3, 10):
        for t in (10, 20):
            pre = 'cnnc2' if t == 10 else 'cnnc2t20'
            recs = load(f'{pre}_{ds}_b{b}.json')
            conv[(ds, t, b)] = {x['seed']: x for x in recs if x['ntask'] == t}

# GN-MLP: s5bgn_* (T=10 only)
gn = {}
for ds in ('mnist', 'fashion'):
    for b in (0, 1, 3, 10):
        recs = load(f's5bgn_{ds}_b{b}.json')
        gn[(ds, b)] = {x['seed']: x for x in recs}

# bare MLP: c2_* dose family (own_bias in field, filename bias)
def load_bare(ds, b):
    recs = load_parts(f'c2_{ds}_b{b}.json.part*')
    # 合并版 c2_<ds>_b1.json / b3.json 实际含 T=20 neuron（S1 审计结论），单独处理
    out = {}
    for x in recs:
        if x.get('gran') != 'neuron':
            continue
        out[(x['ntask'], x['seed'])] = x
    return out

def load_bare_merged(ds, b):
    """S1 审计结论：c2_<ds>_b{1,3}.json 实际装的是 T=20 neuron"""
    recs = load(f'c2_{ds}_b{b}.json')
    out = {}
    for x in recs:
        if x.get('gran') != 'neuron':
            continue
        out[(x['ntask'], x['seed'])] = x
    return out

print('=' * 78)
print('S3 审计：加载统计')
print('=' * 78)
for k, v in sorted(conv.items()):
    print(f'  conv {k}: n={len(v)} seeds={sorted(v)}')
for k, v in sorted(gn.items()):
    print(f'  gn   {k}: n={len(v)} seeds={sorted(v)}')

print('\n--- bare MLP dose 数据文件族 ---')
bare_parts = {}
for ds in ('mnist', 'fashion'):
    for b in (0, 1, 3, 10):
        parts = load_parts(f'c2_{ds}_b{b}.json.part*')
        neur = [(x['ntask'], x['seed']) for x in parts if x.get('gran') == 'neuron']
        print(f'  parts c2_{ds}_b{b}: {len(parts)} recs, neuron keys={sorted(set(neur))}')
        bare_parts[(ds, b)] = parts

# ---------------------------------------------------------------- 论文声明
PAPER = {}

# Table convdose (main.tex L233-244)
# 每行： ds, T, b, Δell, Δavg, ΔBWT, (star_ell, star_avg, star_bwt)
paper_convdose = [
    ('MNIST',    10, 1,  +0.17, -1.64, -2.02, (False, False, False)),
    ('MNIST',    10, 3,  +0.30, -0.99, -1.43, (False, False, False)),
    ('MNIST',    10, 10, +0.07, -0.84, -1.01, (False, False, False)),
    ('Fashion',  10, 1,  -0.06, +0.22, +0.31, (False, False, False)),
    ('Fashion',  10, 3,  +0.02, -0.89, -1.01, (False, False, False)),
    ('Fashion',  10, 10, +0.36, -2.29, -2.95, (False, True,  True)),
    ('MNIST',    20, 1,  +0.63, -0.24, -0.92, (False, False, False)),
    ('MNIST',    20, 3,  +1.46, -0.26, -1.81, (True,  False, False)),
    ('MNIST',    20, 10, +1.28, +0.30, -1.04, (False, False, False)),
    ('Fashion',  20, 1,  +0.31, +0.66, +0.36, (True,  False, False)),
    ('Fashion',  20, 3,  +0.29, -0.51, -0.84, (True,  False, False)),
    ('Fashion',  20, 10, +0.31, -0.87, -1.24, (True,  False, False)),
]

print('\n' + '=' * 78)
print('表 1  ConvADNet dose 表 (Table convdose)')
print('=' * 78)
print(f'{"cell":<18}{"b":>3} | {"d_ell":>16} | {"d_avg":>16} | {"d_bwt":>16} | star')
sigcount = 0
signcount = 0
for ds, T, b, pe, pa, pb, stars in paper_convdose:
    a = conv[(ds.lower(), T, b)]
    z = conv[(ds.lower(), T, 0)]
    dl = [a[s]['ell'] - z[s]['ell'] for s in sorted(set(a) & set(z))]
    da = [a[s]['final_avg'] - z[s]['final_avg'] for s in sorted(set(a) & set(z))]
    db = [a[s]['bwt'] - z[s]['bwt'] for s in sorted(set(a) & set(z))]
    tl, ta, tb = tstat(dl), tstat(da), tstat(db)
    sl, sa, sb = sig(tl), sig(ta), sig(tb)
    ok = (abs(round(mean(dl), 2) - pe) < 0.005 and abs(round(mean(da), 2) - pa) < 0.005
          and abs(round(mean(db), 2) - pb) < 0.005
          and (sl, sa, sb) == stars)
    flag = 'OK ' if ok else 'XX '
    if sl: sigcount += 1
    if mean(dl) > 0: signcount += 1
    print(f'{flag}{ds+" T="+str(T):<18}{b:>3} | {mean(dl):+6.2f} (t{tl:+6.2f}){"*" if sl else " "} | '
          f'{mean(da):+6.2f} (t{ta:+6.2f}){"*" if sa else " "} | '
          f'{mean(db):+6.2f} (t{tb:+6.2f}){"*" if sb else " "} | '
          f'paper {pe:+6.2f} {pa:+6.2f} {pb:+6.2f} star={stars}')
print(f'\n  论文: 4/12 cells significant (Δell*) ; 11/12 sign-consistent')
print(f'  重算: Δell* {sigcount}/12 ; Δell>0 {signcount}/12')

# ---------------------------------------------------------------- Table three
print('\n' + '=' * 78)
print('表 2  三配置对比表 (Table three, T=10)')
print('=' * 78)

def dead_frac_mean(recs):
    v = []
    for x in recs:
        for ph in x.get('dom_per_phase', []):
            if ph.get('dead_frac') is not None:
                v.append(ph['dead_frac'])
    return mean(v) if v else float('nan')

def naive_active_share(recs):
    """naive = dom_hist 众数占比(按 phase 平均), active = dom_hist_active"""
    naive, active = [], []
    for x in recs:
        dh = [ph.get('dom_hist', []) for ph in x.get('dom_per_phase', [])]
        ah = [ph.get('dom_hist_active', []) for ph in x.get('dom_per_phase', [])]
        n = max((len(h) for h in dh), default=0)
        if not n: continue
        for h in dh:
            if h and sum(h) > 0:
                naive.append(max(h) / sum(h))
        for h in ah:
            if h and sum(h) > 0:
                active.append(max(h) / sum(h))
    return (mean(naive) if naive else float('nan'),
            mean(active) if active else float('nan'))

# bare MLP b=0 T=10：来自 S1 剂量族
def bare_cell(ds, T, b):
    ds = ds.lower()
    out = {}
    for x in bare_parts[(ds, b)]:
        if x.get('gran') == 'neuron' and x['ntask'] == T:
            out[x['seed']] = x
    if not out and b in (1, 3):
        for (nt, sd), x in load_bare_merged(ds, b).items():
            if nt == T:
                out[sd] = x
    return out

paper_three = [
    ('MNIST',   'bare-MLP', 74.3, 25.7, 0.572, +3.32, True,  +4.80, True),
    ('MNIST',   'GN-MLP',   93.5,  6.5, 0.765, +0.57, True,  -3.88, True),
    ('MNIST',   'ConvADNet',84.1, 15.9, 0.000, +0.07, False, -0.84, False),
    ('Fashion', 'bare-MLP', 56.8, 43.2, 0.603, +1.39, True,  +0.69, False),
    ('Fashion', 'GN-MLP',   79.3, 20.7, 0.855, +0.36, True,  -6.55, True),
    ('Fashion', 'ConvADNet',75.3, 24.7, 0.000, +0.36, False, -2.29, True),
]

def get_recs(ds, cfg, b, T=10):
    ds = ds.lower()
    if cfg == 'ConvADNet':
        return list(conv[(ds, T, b)].values())
    if cfg == 'GN-MLP':
        return list(gn[(ds, b)].values())
    return list(bare_cell(ds, T, b).values())

cells_for_r = []
for ds, cfg, p_l0, p_hr, p_dead, p_dl, p_sdl, p_da, p_sda in paper_three:
    base = get_recs(ds, cfg, 0)
    arm = get_recs(ds, cfg, 10)
    if not base or not arm:
        print(f'XX {ds} {cfg}: 数据缺失 base={len(base)} arm={len(arm)}')
        continue
    z = {x['seed']: x for x in base}
    a = {x['seed']: x for x in arm}
    common = sorted(set(z) & set(a))
    dl = [a[s]['ell'] - z[s]['ell'] for s in common]
    da = [a[s]['final_avg'] - z[s]['final_avg'] for s in common]
    l0 = mean([z[s]['ell'] for s in common])
    dead = dead_frac_mean(base)
    tl, ta = tstat(dl), tstat(da)
    ok = (abs(round(l0, 1) - p_l0) < 0.05 and abs(round(dead, 3) - p_dead) < 0.0006
          and abs(round(mean(dl), 2) - p_dl) < 0.005 and sig(tl) == p_sdl
          and abs(round(mean(da), 2) - p_da) < 0.005 and sig(ta) == p_sda)
    flag = 'OK ' if ok else 'XX '
    print(f'{flag}{ds:<8}{cfg:<11} n={len(common):<3} l0={l0:6.2f}(p{p_l0:5.1f}) '
          f'dead={dead:.4f}(p{p_dead:.3f}) dell={mean(dl):+5.2f}(t{tl:+6.2f}){"*" if sig(tl) else " "}(p{p_dl:+5.2f}) '
          f'davg={mean(da):+5.2f}(t{ta:+6.2f}){"*" if sig(ta) else " "}(p{p_da:+5.2f})')
    cells_for_r.append((p_hr, mean(dl), l0, dead))

# ---------------------------------------------------------------- 正文声明
print('\n' + '=' * 78)
print('正文数字核对')
print('=' * 78)

# conv baseline: ell_0 84.1/75.3, BWT_0 -50.4/-32.8 ; MLP -11.6/-16.2
for ds in ('mnist', 'fashion'):
    cv = list(conv[(ds, 10, 0)].values())
    l0 = mean([x['ell'] for x in cv]); bwt = mean([x['bwt'] for x in cv])
    bv = list(bare_cell(ds, 10, 0).values())
    bl0 = mean([x['ell'] for x in bv]); bbwt = mean([x['bwt'] for x in bv])
    print(f'  {ds}: conv l0={l0:.2f} bwt={bwt:.2f} | mlp l0={bl0:.2f} bwt={bbwt:.2f} '
          f'| forgetting ratio {abs(bwt)/abs(bbwt):.2f}x')

# conv mortality: exactly 0 in every phase of every run
allmax = 0.0
for (ds, T, b), recs in conv.items():
    for x in recs.values():
        for ph in x.get('dom_per_phase', []):
            allmax = max(allmax, ph.get('dead_frac') or 0.0)
print(f'  conv max dead_frac over all runs/phases = {allmax}')

# GN mortality
for ds in ('mnist', 'fashion'):
    print(f'  GN {ds} dead = {dead_frac_mean(list(gn[(ds,0)].values())):.4f}')

# bare MLP dead
for ds in ('mnist', 'fashion'):
    print(f'  bare {ds} dead = {dead_frac_mean(list(bare_cell(ds,10,0).values())):.4f}')

# naive vs active share
print('\n  --- naive vs active share (G2 artifact) ---')
for ds in ('mnist', 'fashion'):
    for cfg in ('bare-MLP', 'GN-MLP', 'ConvADNet'):
        nv, ac = naive_active_share(get_recs(ds, cfg, 0))
        print(f'   {ds:<8}{cfg:<11} naive={nv:.4f} active={ac:.4f} diff={nv-ac:+.4f} '
              f'{"G2 PASS" if nv-ac >= 0.05 else "G2 FAIL"}')

# correlation HR vs gain
hrs = [c[0] for c in cells_for_r]
gains = [c[1] for c in cells_for_r]
print(f'\n  Pearson r (cell means, paper-claimed HR) = {r_(hrs, gains):.4f}  (paper: 0.38)')

# pooled per-seed r: HR is per-cell constant; pool 10 per-seed deltas per cell
pool_x, pool_y = [], []
for ds, cfg, p_hr, p_dl, p_sdl, p_da, p_sda, _s in paper_three:
    base = get_recs(ds, cfg, 0); arm = get_recs(ds, cfg, 10)
    z = {x['seed']: x for x in base}; a = {x['seed']: x for x in arm}
    for s in sorted(set(z) & set(a)):
        pool_x.append(p_hr); pool_y.append(a[s]['ell'] - z[s]['ell'])
print(f'  Pearson r (pooled per-seed) = {r_(pool_x, pool_y):.4f}  (paper: 0.31)')

# GN-MLP gain retention percentages
for ds, cfg, p_hr, p_dl, p_sdl, p_da, p_sda, _s in paper_three:
    if cfg != 'bare-MLP':
        continue
    print(f'  bare {ds} gain = {p_dl}')

print('\n' + '=' * 78)
print('MLP 侧（S1 声称）: Δell +1.4..+4.6, 20/20 positive, 18/20 significant, final-avg 17/20')
print('=' * 78)
for ds in ('mnist', 'fashion'):
    for T in (10, 20):
        z = bare_cell(ds, T, 0)
        if not z:
            print(f'  {ds} T={T}: b=0 数据缺失')
            continue
        for b in (1, 3, 10):
            a = bare_cell(ds, T, b)
            if not a:
                print(f'  {ds} T={T} b={b}: 数据缺失')
                continue
            common = sorted(set(z) & set(a))
            dl = [a[s]['ell'] - z[s]['ell'] for s in common]
            da = [a[s]['final_avg'] - z[s]['final_avg'] for s in common]
            print(f'  {ds} T={T} b={b:>2}: n={len(common)} dell={mean(dl):+6.3f} (t{tstat(dl):+6.2f})'
                  f'{"*" if sig(tstat(dl)) else " "} davg={mean(da):+6.3f}'
                  f'{"+" if mean(da)>0 else "-"}')
