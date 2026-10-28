# -*- coding: utf-8 -*-
r"""重做 task-relevance 探针（合并稿报告了它的结果，但原始数据与脚本都不在仓库）。

设计（与 S1 正文报告一致）：
  10 个交错 phase，分三组：
    高相关 (2 个): 域内数字/奇偶任务
    中相关 (2 个): 旋转计数任务
    零相关 (6 个): 随机标签任务
  剂量 b ∈ {0,1,3,10}，n=10 paired seeds，Permuted-MNIST，n_h=256，λ=1000。

指标：每个 phase 的 per-task diagnostic accuracy（用该任务自己的训练标签测），
按三组汇总。

关键设计点（审稿人 B22 指出原文的探针是同义反复）：
  零相关任务用随机标签 -> 其准确率按定义就是 10%，"在所有剂量下都是机会
  水平"是**先验必然**，不携带证据量。因此这里加一个**非平凡的零相关
  对照**：把标签在任务间随机重排但**保持类内结构**（每个类内的样本标签
  在任务之间随机轮换），这样任务在标签空间上仍然可学（类内结构未破坏），
  但任务-标签映射在不同任务间不同。若 bias 对这类任务也无影响，则
  "只对可学结构有效"的结论才是有内容的。

用法：python s5_task_relevance.py --out results_s5/s5_taskrel.json
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute, kwta, ADNet
from branch_delp import (make_anchor, anchor_penalty, flat_params,
                         BiasADNet)
from anchor_budget import load_data

NH = 256
LAM = 1000.0
NPHASE = 10
GROUPS = {'high': [0, 1], 'medium': [2, 3], 'zero': [4, 5, 6, 7, 8, 9]}


def build_tasks(xtr, ytr, perms, seed, nphase=NPHASE):
    """Ten interleaved task streams with three relevance levels.

    high   : in-domain digit task (identity permutation, original labels)
    medium : rotated-count task (label = (y + 1) mod 10) -- learnable but
             not the canonical mapping
    zero   : random-label task; two variants:
              'shuffled'  -- labels permuted across the whole training set
                              (the original, degenerate control)
              'rotated'   -- labels permuted WITHIN each class across tasks
                              (non-trivial: class structure preserved, so the
                              task is still learnable, only the task-label
                              mapping differs)
    """
    torch.manual_seed(seed + 7777)
    n = xtr.size(0)
    out = []
    for t in range(nphase):
        xs = permute(xtr, perms[t % len(perms)])
        ys = ytr.clone()
        if t in GROUPS['high']:
            pass                                     # in-domain
        elif t in GROUPS['medium']:
            ys = (ys + 1) % 10                       # rotated count
        else:
            variant = 'shuffled' if t % 2 == 0 else 'rotated'
            if variant == 'shuffled':
                perm = torch.randperm(n)
                ys = ys[perm]
            else:
                # within-class label rotation, different per task
                shift = 1 + (t % 9)
                ys = torch.empty_like(ys)
                for c in range(10):
                    m = (ytr == c)
                    vals = torch.arange(int(m.sum()), device=ys.device) % 10
                    rot = (vals + shift) % 10
                    ys[m] = rot
        out.append((xs, ys))
    return out


def run_arm(seed, data, perms, bias, lam=LAM, nh=NH, nphase=NPHASE):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = BiasADNet(nphase, nphase, own_bias=bias,
                       bias_mode='fixed', nh=nh).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    tasks = build_tasks(xtr, ytr, perms, seed, nphase)
    per_task_diag = []          # per phase: accuracy on own training labels
    for phase in range(nphase):
        xs, ys = tasks[phase]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                nphase).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, 'neuron')
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, nphase, 'neuron'))
        # diagnostic: accuracy on this task's own training labels
        model.eval()
        with torch.no_grad():
            ctx = F.one_hot(torch.full((min(2000, xs.size(0)),), phase,
                                       device=DEV), nphase).float()
            o, _ = model(xs[:min(2000, xs.size(0))].to(DEV), ctx)
            diag = float((o.argmax(1) == ys[:min(2000, xs.size(0))].to(DEV))
                         .float().mean()) * 100
        per_task_diag.append(round(diag, 2))
        a = eval_model(model, xte, yte, perms, phase, nphase, True)
        acc.update(a)
    bwt, favg = finish(acc, nphase)
    ell = sum(acc[(t, t)] for t in range(nphase)) / nphase
    groups = {g: [per_task_diag[t] for t in ts] for g, ts in GROUPS.items()}
    return {"seed": seed, "bias": bias, "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "per_task_diag": per_task_diag, "groups": groups,
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(nphase)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_s5/s5_taskrel.json')
    ap.add_argument('--biases', default='0,1,3,10')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--lam', type=float, default=LAM)
    args = ap.parse_args()

    biases = [float(b) for b in args.biases.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out, encoding='utf-8')).get('results', [])
        except Exception:
            results = []
    done = {(r['bias'], r['seed']) for r in results}

    data = load_data('mnist')
    for seed in seeds:
        perms = get_perms(seed, NPHASE)
        for b in biases:
            if (b, seed) in done:
                continue
            t0 = time.time()
            rec = run_arm(seed, data, perms, b, lam=args.lam)
            rec['elapsed'] = round(time.time() - t0, 1)
            results.append(rec)
            done.add((b, seed))
            g = rec['groups']
            print(f"  s{seed} b={b:.0f}: high={sum(g['high'])/2:.1f} "
                  f"med={sum(g['medium'])/2:.1f} zero={sum(g['zero'])/6:.1f} "
                  f"({rec['elapsed']}s)", flush=True)
            json.dump({"results": results, "meta": vars(args)},
                      open(args.out, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out} ({len(results)} records)')


if __name__ == '__main__':
    main()
