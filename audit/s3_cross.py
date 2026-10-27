# -*- coding: utf-8 -*-
"""交叉核查：S1 论文 dose 表（Table 5/6）的格数与星号 vs S1 正文声称的 18/18、16/18、15/18
   以及 S3 论文引用的 S1 数字（20/20、18/20、17/20、+1.4..+4.6、t>=2.4、t up to 26、3-30x）。"""
import re, os, json, glob, math
from statistics import mean, stdev

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_REPO = os.path.join(ROOT, 'dab_t/delp-anchor-budget-main')
if not os.path.isdir(_REPO):
    _REPO = os.path.join(_HERE, '..')
S1TEX = os.path.join(_REPO, 'S1_剂量响应审计', 'paper', 'main.tex')
RES = os.path.join(_REPO, 'bench', 'results')
RES2 = os.path.join(_REPO, 'S1_剂量响应审计', 'results')

src = open(S1TEX, encoding='utf-8').read()
lines = src.split('\n')

def grab_table(label):
    """从 tex 中抽出 table 环境的行"""
    i = src.index('\\label{%s}' % label)
    j = src.index('\\end{table}', i)
    body = src[src.index('\\begin{tabular}', i):j]
    rows = []
    for ln in body.split('\n'):
        ln = ln.strip()
        if ln.startswith('\\midrule') or ln.startswith('\\toprule') or ln.startswith('\\bottomrule') or '&' not in ln:
            continue
        if 'dataset' in ln or 'arm (' in ln:
            continue
        cells = [c.strip() for c in ln.replace('\\\\', '').split('&')]
        rows.append(cells)
    return rows

def num(s):
    s = s.replace('$', '').replace('{', '').replace('}', '').replace('+', '')
    s = s.replace('--', '-').replace('\\', '').replace(' ', '')
    s = s.replace('*', '').replace('^', '')
    if s in ('', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def starred(s):
    return '*' in s

print('=' * 78)
print('A. S1 Table 5 / Table 6 逐行星号统计')
print('=' * 78)
all_rows = []
for lab, tname in (('tab:dose10', 'T=10'), ('tab:dose20', 'T=20')):
    rows = grab_table(lab)
    print(f'\n-- {tname} ({lab}) : {len(rows)} 行 --')
    nsig_l = nsig_both = npos = 0
    for r in rows:
        ds, arm, dl, da, db, own = r[0], r[1], r[2], r[3], r[4], r[5]
        sl = starred(dl); sa = starred(da)
        v = num(dl)
        if v is not None and v > 0: npos += 1
        if sl: nsig_l += 1
        if sl and sa: nsig_both += 1
        all_rows.append((tname, ds, arm, dl, da, db, sl, sa))
        print(f'   {ds:<8}{arm:<16} {dl:>10} {da:>10} {db:>10}  ell*={int(sl)} avg*={int(sa)}')
    print(f'   -> {tname}: Δell>0 {npos}/{len(rows)}; Δell* {nsig_l}/{len(rows)}; 双显著 {nsig_both}/{len(rows)}')

n = len(all_rows)
npos = sum(1 for r in all_rows if num(r[3]) and num(r[3]) > 0)
nsigl = sum(1 for r in all_rows if r[6])
nsigb = sum(1 for r in all_rows if r[6] and r[7])
print(f'\n  合计 {n} 格: Δell>0 {npos}/{n}; Δell* {nsigl}/{n}; Δell&avg 双显著 {nsigb}/{n}')
print(f'  S1 正文声称: 18/18 positive, 16/18 significant, 15/18 双显著, 24 格报 22 格')

print('\n' + '=' * 78)
print('B. 从原始数据重算 S1 剂量格（确认表格本身无误）')
print('=' * 78)

def load(pat, base=None):
    out = []
    for b_ in ([base] if base else [RES, RES2]):
        for f in sorted(glob.glob(os.path.join(b_, pat))):
            d = json.load(open(f))
            out += d['results'] if isinstance(d, dict) else d
    return out

def own_bias(x):
    return x.get('own_bias', 0) or 0

def tstat(d):
    n = len(d); m = mean(d); sd = stdev(d)
    return m / (sd / math.sqrt(n)) if sd else 0.0

cells = {}
for ds in ('mnist', 'fashion'):
    for b in (0, 1, 3, 10):
        for T in (10, 20):
            recs = []
            recs += load(f'c2_{ds}_b{b}.json.part*')
            recs += load(f'c2t20_{ds}_b{b}.json.part*')
            recs += load(f'c2_{ds}_b{b}.json')
            for x in recs:
                if x.get('gran') != 'neuron' or x['ntask'] != T:
                    continue
                if abs(own_bias(x) - b) > 1e-9:
                    continue
                cells.setdefault((ds, T, b), {})[x['seed']] = x

print('  可用 cell 统计:')
for k in sorted(cells):
    print(f'   {k}: n={len(cells[k])}')

print('\n  --- neuron 粒度逐格重算 ---')
rows_calc = []
for ds in ('mnist', 'fashion'):
    for T in (10, 20):
        z = cells.get((ds, T, 0), {})
        for b in (1, 3, 10):
            a = cells.get((ds, T, b), {})
            common = sorted(set(z) & set(a))
            if len(common) < 2:
                print(f'   {ds} T={T} b={b:>2}: 数据不足 (n={len(common)})')
                continue
            dl = [a[s]['ell'] - z[s]['ell'] for s in common]
            da = [a[s]['final_avg'] - z[s]['final_avg'] for s in common]
            db = [a[s]['bwt'] - z[s]['bwt'] for s in common]
            tl, ta, tb = tstat(dl), tstat(da), tstat(db)
            rows_calc.append((ds, T, b, mean(dl), tl, mean(da), ta, mean(db), tb, len(common)))
            print(f'   {ds} T={T} b={b:>2}: n={len(common)} Δell={mean(dl):+6.3f}(t{tl:+6.2f}){"*" if abs(tl)>2 else " "} '
                  f'Δavg={mean(da):+6.3f}(t{ta:+6.2f}){"*" if abs(ta)>2 else " "} '
                  f'Δbwt={mean(db):+6.3f}(t{tb:+6.2f}){"*" if abs(tb)>2 else " "}')

npos = sum(1 for r in rows_calc if r[3] > 0)
nsig = sum(1 for r in rows_calc if abs(r[4]) > 2)
nboth = sum(1 for r in rows_calc if abs(r[4]) > 2 and abs(r[5+1]) > 2)
print(f'\n  neuron 粒度: {len(rows_calc)} 格, Δell>0 {npos}, Δell* {nsig}, 双显著 {nboth}')
print(f'  Δell 最大 = {max(r[3] for r in rows_calc):+.3f} (t={max(rows_calc, key=lambda r: r[3])[4]:+.2f})')
print(f'  |t| 最大 (Δell) = {max(abs(r[4]) for r in rows_calc):.2f}')
print(f'  Δell 最小 = {min(r[3] for r in rows_calc):+.3f}')

print('\n' + '=' * 78)
print('C. S3 论文引用的 S1 数字 vs S1 现值')
print('=' * 78)
s3 = open(os.path.join(ROOT, 'papers/s3_zenodo/S3_repair_headroom/main.tex'), encoding='utf-8').read()
checks = [
    ('20/20 dose cells positive', '20/20'),
    ('18/20 significant', '18/20'),
    ('17/20 cells', '17/20'),
    ('Δell +1.4 to +4.6', '+4.6'),
    ('all paired t >= 2.4', '2.4'),
    ('t up to 26', '26'),
    ('3--30x larger', '30'),
]
for label, token in checks:
    print(f'  S3 中出现 "{token}": {token in s3}   <- {label}')
