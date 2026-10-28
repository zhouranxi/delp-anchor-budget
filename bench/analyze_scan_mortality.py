"""Analyse the A5 mortality scan: does the repair gain track neuron mortality?

The paper claims (Sec. "the dying-gate hypothesis") that mortality and
gain do NOT track each other: the configuration with the highest mortality
is not the one with the largest repair gain.  The archived 2x2 factorial
scan (width x normalization) recorded accuracy but not mortality, so the
claim could not be checked against it.

This script reads the ten-seed server re-run
(bench/results_s5/server/server_scan_mortality.json, 160 runs) and reports,
for every (dataset, width, normalization) cell:

  * the mean neuron mortality at the end of the sequence
  * the paired repair gain (b=10 minus b=0) in final accuracy
  * the paired t over seeds for that gain

and then the correlation between mortality and gain ACROSS cells, which is
the quantity the paper's claim is about.

Output: paper/scan_mortality.json + paper/scan_mortality.md
"""
import json
import math
import os
from collections import defaultdict

try:
    from scipy import stats as sps
except ImportError:
    print('需要 scipy'); raise SystemExit(1)

ROOT = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main'
PAPER = os.path.join(ROOT, 'S123_合并稿', 'paper')
SRC = os.path.join(ROOT, 'bench', 'results_s5', 'server',
                   'server_scan_mortality.json')

rows = json.load(open(SRC, encoding='utf-8'))['rows']

# ---- group by cell, pairing b=10 against b=0 within the same seed ----
cells = defaultdict(dict)
for r in rows:
    key = (r['dataset'], r['nh'], r['gn'])
    cells[key].setdefault(r['bias'], {})[r['seed']] = r

out_cells = []
for key in sorted(cells, key=str):
    ds, nh, gn = key
    arms = cells[key]
    base, rep = arms.get(0.0), arms.get(10.0)
    if not base or not rep:
        continue
    seeds = sorted(set(base) & set(rep))
    if len(seeds) < 3:
        continue
    d_avg = [rep[s]['final_avg'] - base[s]['final_avg'] for s in seeds]
    d_ell = [rep[s]['ell'] - base[s]['ell'] for s in seeds]
    t_avg, p_avg = sps.ttest_rel([rep[s]['final_avg'] for s in seeds],
                                 [base[s]['final_avg'] for s in seeds])
    t_ell, p_ell = sps.ttest_rel([rep[s]['ell'] for s in seeds],
                                 [base[s]['ell'] for s in seeds])
    sd = (sum((x - sum(d_avg) / len(d_avg)) ** 2 for x in d_avg)
          / (len(d_avg) - 1)) ** 0.5
    out_cells.append({
        'dataset': ds, 'nh': nh, 'gn': gn, 'n': len(seeds),
        'dead_frac_base': round(sum(base[s]['dead_frac_final']
                                    for s in seeds) / len(seeds), 4),
        'dead_frac_repaired': round(sum(rep[s]['dead_frac_final']
                                        for s in seeds) / len(seeds), 4),
        'dead_frac_mean_repaired': round(
            sum(rep[s]['dead_frac_mean'] for s in seeds) / len(seeds), 4),
        'd_final_avg': round(sum(d_avg) / len(d_avg), 3),
        'd_ell': round(sum(d_ell) / len(d_ell), 3),
        't_final_avg': round(float(t_avg), 2),
        'p_final_avg': float(p_avg),
        't_ell': round(float(t_ell), 2),
        'p_ell': float(p_ell),
        'd_z': round((sum(d_avg) / len(d_avg)) / sd, 2) if sd > 0 else None,
        'ci95': [round(sum(d_avg) / len(d_avg) - 1.96 * sd / len(seeds) ** 0.5, 3),
                  round(sum(d_avg) / len(d_avg) + 1.96 * sd / len(seeds) ** 0.5, 3)],
    })

# ---- the correlation the claim is about: mortality vs gain, across cells ----
corr = {}
for ds in sorted(set(c['dataset'] for c in out_cells)):
    sel = [c for c in out_cells if c['dataset'] == ds]
    mort = [c['dead_frac_repaired'] for c in sel]
    gain = [c['d_final_avg'] for c in sel]
    if len(sel) >= 3:
        r_p, p_p = sps.pearsonr(mort, gain)
        r_s, p_s = sps.spearmanr(mort, gain)
        corr[ds] = {
            'n_cells': len(sel),
            'pearson_r': round(float(r_p), 3), 'pearson_p': float(p_p),
            'spearman_rho': round(float(r_s), 3), 'spearman_p': float(p_s),
        }
    # also across all cells pooled
allm = [c['dead_frac_repaired'] for c in out_cells]
allg = [c['d_final_avg'] for c in out_cells]
if len(out_cells) >= 3:
    r_p, p_p = sps.pearsonr(allm, allg)
    r_s, p_s = sps.spearmanr(allm, allg)
    corr['pooled'] = {
        'n_cells': len(out_cells),
        'pearson_r': round(float(r_p), 3), 'pearson_p': float(p_p),
        'spearman_rho': round(float(r_s), 3), 'spearman_p': float(p_s),
    }

report = {'cells': out_cells, 'corr': corr}

L = []
w = L.append
w('# A5: mortality and repair gain in the 2x2 width x normalization scan')
w('')
w('Ten seeds per cell, paired b=10 against b=0 within the same seed.')
w('')
w('| dataset | width | GroupNorm | n | mortality (b=0) | mortality (b=10) | '
  'Δ final acc | t | p | d_z | 95% CI |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for c in out_cells:
    w(f"| {c['dataset']} | {c['nh']} | {'yes' if c['gn'] else 'no'} | {c['n']} | "
      f"{c['dead_frac_base']:.3f} | {c['dead_frac_repaired']:.3f} | "
      f"{c['d_final_avg']:+.2f} | {c['t_final_avg']:+.1f} | {c['p_final_avg']:.3f} | "
      f"{(c['d_z'] if c['d_z'] is not None else float('nan')):+.2f} | "
      f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
w('')
w('## Correlation between mortality and repair gain across cells')
w('')
w('| subset | cells | Pearson r | p | Spearman rho | p |')
w('|---|---|---|---|---|---|')
for k, v in corr.items():
    w(f"| {k} | {v['n_cells']} | {v['pearson_r']:+.3f} | {v['pearson_p']:.3f} | "
      f"{v['spearman_rho']:+.3f} | {v['spearman_p']:.3f} |")

open(os.path.join(PAPER, 'scan_mortality.md'), 'w', encoding='utf-8',
     newline='\n').write('\n'.join(L) + '\n')
json.dump(report, open(os.path.join(PAPER, 'scan_mortality.json'), 'w',
                       encoding='utf-8'), indent=1, ensure_ascii=False)

print('\n'.join(L))
print()
print('写入', os.path.join(PAPER, 'scan_mortality.md'))
