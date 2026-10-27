# -*- coding: utf-8 -*-
import json, re, glob, os
D = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/papers/s4_zenodo/S4_repair_bypasses_gate/results'
rows = []
for f in sorted(glob.glob(os.path.join(D, 's4diag_*.log'))):
    for o in re.findall(r'\{[^{}]*\}', open(f).read(), re.S):
        rows.append((os.path.basename(f), json.loads(o)))

print('=== 全部诊断记录 ===')
for f, d in rows:
    print(f'  {f:<22} ds={d["ds"]:<8} seed={d["seed"]} b={d["bias"]:<5} '
          f'flip_mean={d["pred_flip_rate_mean"]*100:.3f}% flip_max={d["pred_flip_rate_max"]*100:.3f}% '
          f'gate={d["gate_mean_own"]:.6f}/{d["gate_mean_other"]:.6f} '
          f'cos_mean={d["dend_col_cosine_mean"]:.6f} cos_min={d["dend_col_cosine_min"]:.6f}')

b0 = [d for f, d in rows if d['bias'] == 0.0]
b10 = [d for f, d in rows if d['bias'] == 10.0]
print()
print('b=0  flip mean 范围: %.2f%% .. %.2f%%   论文称 6.2--6.9%%' % (
    min(d['pred_flip_rate_mean'] for d in b0) * 100,
    max(d['pred_flip_rate_mean'] for d in b0) * 100))
print('b=0  flip max  范围: %.2f%% .. %.2f%%   论文称 14.9--19.0%%' % (
    min(d['pred_flip_rate_max'] for d in b0) * 100,
    max(d['pred_flip_rate_max'] for d in b0) * 100))
print('b=10 flip 值集合: %s   论文称 exactly 0.0' % sorted(set(d['pred_flip_rate_mean'] for d in b10)))
print('gate own==other (全部): %s' % all(abs(d['gate_mean_own'] - d['gate_mean_other']) < 1e-12 for d in rows))
print('gate 值范围: %.4f .. %.4f' % (
    min(min(d['gate_mean_own'], d['gate_mean_other']) for d in rows),
    max(max(d['gate_mean_own'], d['gate_mean_other']) for d in rows)))
print('cos_mean 范围: %.6f .. %.6f   论文称 [0.001, 0.003]' % (
    min(d['dend_col_cosine_mean'] for d in rows),
    max(d['dend_col_cosine_mean'] for d in rows)))
print('cos_min  范围: %.6f .. %.6f   论文称 min -0.034' % (
    min(d['dend_col_cosine_min'] for d in rows),
    max(d['dend_col_cosine_min'] for d in rows)))
print()
print('mnist seed0 (论文头条):')
m0 = [d for f, d in rows if d['ds'] == 'mnist' and d['seed'] == 0 and d['bias'] == 0.0]
if m0:
    d = m0[0]
    print(f'  flip_mean={d["pred_flip_rate_mean"]*100:.2f}% (论文 6.2%) '
          f'flip_max={d["pred_flip_rate_max"]*100:.2f}% (论文 17.7%) '
          f'gate={d["gate_mean_own"]:.4f} (论文 0.5649) '
          f'cos_mean={d["dend_col_cosine_mean"]:.4f} (论文 0.003) '
          f'cos_min={d["dend_col_cosine_min"]:.4f} (论文 -0.034)')
