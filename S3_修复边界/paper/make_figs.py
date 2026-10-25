"""S5 paper figures. Reads cnnc2/s5bgn/c2 result JSONs, writes PDFs to figs/."""
import json, os, sys, glob
import statistics as st

SP = 'C:/Users/Lance/.workbuddy/binaries/python/envs/default/Lib/site-packages'
if SP not in sys.path:
    sys.path.insert(0, SP)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = FIG = None
for _cand in (os.path.join(HERE, '..', 'results'),
              os.path.join(HERE, '..', 'delp-anchor-budget', 'bench', 'results'),
              os.path.join(HERE, '..', 'bench', 'results'),
              os.path.join(HERE, 'results')):
    if os.path.isdir(_cand):
        RES = _cand
        break
if RES is None:
    sys.exit('results dir not found')
FIG = os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})

CFGS = [('bare-MLP', 's3ref_c2', '#4878d0', 'o'),
        ('GN-MLP', 's5bgn', '#ee854a', 's'),
        ('ConvADNet', 'cnnc2', '#6acc64', '^')]

def load(pfx, ds, b, T=10):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, f'{pfx}_{ds}_b{b}.json*'))):
        d = json.load(open(f))
        for r in (d['results'] if isinstance(d, dict) else d):
            if ('error' not in r and r.get('gran') == 'neuron'
                    and r.get('lam') == 1000 and r.get('ntask') == T):
                rows[r['seed']] = r
    return rows

def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, m / (sd / n ** 0.5) if sd > 0 else 0.0

# ------------------------------------------------ Fig 1: dose x 3 configs
fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
for ax, ds in zip(axes, ('mnist', 'fashion')):
    for name, pfx, c, mk in CFGS:
        b0 = load(pfx, ds, 0)
        xs, ys = [], []
        for b in (1.0, 3.0, 10.0):
            arm = load(pfx, ds, b)
            seeds = sorted(set(b0) & set(arm))
            if not seeds:
                continue
            m, _ = paired([arm[s]['ell'] for s in seeds],
                          [b0[s]['ell'] for s in seeds])
            xs.append(b); ys.append(m)
        if xs:
            ax.plot(xs, ys, mk + '-', ms=4, color=c, label=name)
    ax.axhline(0, lw=0.8, color='gray')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([1, 3, 10]); ax.set_xticklabels(['1', '3', '10'])
    ax.set_xlabel('own-segment bias $b$')
    ax.set_title({'mnist': 'MNIST', 'fashion': 'Fashion'}[ds] + '  $T{=}10$', fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel(r'$\Delta\ell$ vs $b{=}0$')
        ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_dose3.pdf'))
plt.close(fig)

# ------------------------------------------------ Fig 2: mortality bars
fig, ax = plt.subplots(figsize=(4.6, 2.7))
labels, vals, colors = [], [], []
for ds in ('mnist', 'fashion'):
    for name, pfx, c, mk in CFGS:
        b0 = load(pfx, ds, 0)
        if not b0:
            continue
        dead = st.mean([st.mean([p['dead_frac'] for p in r['dom_per_phase']])
                        for r in b0.values()])
        labels.append(f'{name}\n{ds[:4].capitalize()}')
        vals.append(dead); colors.append(c)
ax.bar(range(len(vals)), vals, color=colors, width=0.62)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel('dead-gate fraction')
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_mortality.pdf'))
plt.close(fig)

# ------------------------------------------------ Fig 3: baseline vs gain
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for name, pfx, c, mk in CFGS:
    for ds in ('mnist', 'fashion'):
        b0 = load(pfx, ds, 0); b10 = load(pfx, ds, 10)
        seeds = sorted(set(b0) & set(b10))
        if not seeds:
            continue
        ell0 = st.mean([r['ell'] for r in b0.values()])
        m, _ = paired([b10[s]['ell'] for s in seeds], [b0[s]['ell'] for s in seeds])
        ax.scatter(ell0, m, color=c, marker=mk, s=45, zorder=3,
                   label=name if ds == 'mnist' else None)
ax.axhline(0, lw=0.8, color='gray')
ax.set_xlabel(r'baseline learning term $\ell_0$ (weaker $\to$ right)')
ax.set_ylabel(r'$\Delta\ell$ at $b{=}10$')
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_headroom.pdf'))
plt.close(fig)

# ------------------------------------------------ Fig 4: dEll vs dAvg
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for name, pfx, c, mk in CFGS:
    for ds in ('mnist', 'fashion'):
        b0 = load(pfx, ds, 0); b10 = load(pfx, ds, 10)
        seeds = sorted(set(b0) & set(b10))
        if not seeds:
            continue
        mL, _ = paired([b10[s]['ell'] for s in seeds], [b0[s]['ell'] for s in seeds])
        mA, _ = paired([b10[s]['final_avg'] for s in seeds],
                       [b0[s]['final_avg'] for s in seeds])
        ax.scatter(mL, mA, color=c, marker=mk, s=45, zorder=3,
                   label=name if ds == 'mnist' else None)
ax.axhline(0, lw=0.8, color='gray'); ax.axvline(0, lw=0.8, color='gray')
ax.set_xlabel(r'$\Delta\ell$ at $b{=}10$')
ax.set_ylabel(r'$\Delta$final avg at $b{=}10$')
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_tradeoff.pdf'))
plt.close(fig)

print('figs written:', sorted(os.listdir(FIG)))
