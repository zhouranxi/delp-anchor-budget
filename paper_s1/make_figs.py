"""S1 paper figures. Reads result JSONs from bench/results, writes PDFs to figs/."""
import json, os, sys, glob
import statistics as st

SP = 'C:/Users/Lance/.workbuddy/binaries/python/envs/default/Lib/site-packages'
if SP not in sys.path:
    sys.path.insert(0, SP)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'delp-anchor-budget', 'bench', 'results')
FIG = os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})

# ---------------------------------------------------------------- Fig 1: evolution
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7))
for ax, (f, ds, T, n) in zip(axes, [
        ('gevo_mnist_t10.json', 'Permuted-MNIST', 10, 10),
        ('gevo_mnist_t20.json', 'Permuted-MNIST', 20, 20),
        ('gevo_fashion_t10.json', 'Permuted-Fashion', 10, 10)]):
    r = json.load(open(os.path.join(RES, f)))['results']
    nph = len(r[0]['phases'])
    starts, ends, deads = [], [], []
    for p in range(nph):
        st_m = [x['phases'][p]['measures'][0] for x in r]
        en_m = [x['phases'][p]['measures'][-1] for x in r]
        starts.append(st.mean([m['p_own_active'] for m in st_m if m['p_own_active'] is not None]))
        ends.append(st.mean([m['p_own_active'] for m in en_m if m['p_own_active'] is not None]))
        deads.append(st.mean([m['dead_frac'] for m in en_m]))
    xs = range(1, nph + 1)
    ax.plot(xs, starts, 'o-', ms=3, color='#4878d0', label='phase start')
    ax.plot(xs, ends, 's-', ms=3, color='#d65f5f', label='phase end')
    ax.axhline(1.0 / T, ls='--', lw=0.8, color='gray', label='chance $1/T$')
    ax.set_title(f'{ds}, $T$={T}', fontsize=9)
    ax.set_xlabel('task phase')
    ax.set_ylim(-0.01, 0.16 if T == 10 else 0.09)
    if ax is axes[0]:
        ax.set_ylabel('$P$(own seg. dominant | active)')
        ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_evolution.pdf'))
plt.close(fig)

# dead fraction companion panel
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for f, ds, T, c in [('gevo_mnist_t10.json', 'MNIST $T{=}10$', 10, '#4878d0'),
                    ('gevo_mnist_t20.json', 'MNIST $T{=}20$', 20, '#ee854a'),
                    ('gevo_fashion_t10.json', 'Fashion $T{=}10$', 10, '#6acc64')]:
    r = json.load(open(os.path.join(RES, f)))['results']
    nph = len(r[0]['phases'])
    deads = [st.mean([x['phases'][p]['measures'][-1]['dead_frac'] for x in r])
             for p in range(nph)]
    ax.plot(range(1, nph + 1), deads, '-', color=c, label=ds)
ax.set_xlabel('task phase')
ax.set_ylabel('inactive-neuron fraction')
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_deadfrac.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: artifact bar
# All values computed from data: untrained probe JSON + c2 b=0 anchor instrumentation.
fig, ax = plt.subplots(figsize=(4.4, 2.7))
probe = json.load(open(os.path.join(RES, 'untrained_probe.json')))
un_naive = st.mean(probe['naive'])
un_act = st.mean(probe['active'])
b0 = []
for f in glob.glob(os.path.join(RES, 'c2_mnist_b0*.json.part*')):
    for r in json.load(open(f))['results']:
        if 'error' not in r and r.get('kind') == 'gran' \
           and abs(r.get('own_bias', 0)) < 1e-9:
            b0.append(r)
tr_naive = tr_act = None
if b0:
    n0 = t0 = a0 = ta0 = 0
    for r in b0:
        for d in r.get('dom_per_phase', []):
            n0 += d['dom_hist'][0]; t0 += sum(d['dom_hist'])
            a0 += d['dom_hist_active'][0]; ta0 += sum(d['dom_hist_active'])
    tr_naive = n0 / max(t0, 1); tr_act = a0 / max(ta0, 1)
vals = [un_naive, tr_naive, un_act, tr_act]
labels = ['untrained\n(naive)', 'trained\n(naive)', 'untrained\n(active)', 'trained\n(active)']
colors = ['#bbbbbb', '#d65f5f', '#bbbbbb', '#4878d0']
ax.bar(range(4), vals, color=colors, width=0.62)
ax.axhline(0.10, ls='--', lw=0.8, color='gray')
ax.text(3.35, 0.115, 'chance', fontsize=7, color='gray', ha='right')
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('$P$(segment 0 dominant)')
ax.set_ylim(0, max(vals) * 1.18)
print(f'fig_artifact values: un_naive={un_naive:.3f} tr_naive={tr_naive:.3f} '
      f'un_act={un_act:.3f} tr_act={tr_act:.3f}')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_artifact.pdf'))
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: dose response
def sel(recs, ds, b, g):
    return {r['seed']: r for r in recs if r['dataset'] == ds
            and abs(r.get('own_bias', 0) - b) < 1e-9 and r['gran'] == g}

def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, m / (sd / n ** 0.5) if sd > 0 else 0.0

fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7))
metnames = [('ell', r'$\Delta$ learning term $\ell$'),
            ('final_avg', r'$\Delta$ final accuracy'),
            ('bwt', r'$\Delta$ BWT')]
for pidx, ax in zip(range(3), axes):
    met, ylab = metnames[pidx]
    for ds, mk, c in [('mnist', 'o', '#4878d0'), ('fashion', 's', '#6acc64')]:
        for T, ls in [(10, '-'), (20, '--')]:
            recs = []
            for f in glob.glob(os.path.join(RES, f'c2{"" if T == 10 else "t20"}_*_b*.json.part*')):
                for r in json.load(open(f))['results']:
                    if 'error' not in r and r.get('kind') == 'gran':
                        recs.append(r)
            xs_, ys_ = [], []
            for b in (0.0, 1.0, 3.0, 10.0):
                sb, s0 = sel(recs, ds, b, 'neuron'), sel(recs, ds, 0.0, 'neuron')
                seeds = sorted(set(sb) & set(s0))
                if not seeds:
                    continue
                m, t = paired([sb[k][met] for k in seeds], [s0[k][met] for k in seeds])
                xs_.append(b); ys_.append(m)
            if xs_:
                ax.plot(xs_, ys_, mk + ls, ms=4, color=c,
                        label=f'{ds.replace("mnist", "MNIST").replace("fashion", "Fashion")} $T$={T}')
    ax.axhline(0, lw=0.8, color='gray')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([0, 1, 3, 10]); ax.set_xticklabels(['0', '1', '3', '10'])
    ax.set_xlabel('own-segment bias $b$')
    ax.set_ylabel(ylab)
    if pidx == 0:
        ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_dose.pdf'))
plt.close(fig)

print('figs written:', sorted(os.listdir(FIG)))
