# -*- coding: utf-8 -*-
r"""P0-1: 现代持续学习基线（DER / ER）—— 填补"零个 2019 年后方法"的硬伤。

背景：DeLP 五篇（及合并稿）全部只与 EWC / SI / MAS / PackNet 对比，
全部是 2016-2018 年的方法。审稿人会直接问："结论在 2019 年后的
replay-based 方法上还有效吗？"

这里实现两个当代基线，跑在与 DeLP 完全相同的协议下
（Permuted-MNIST / Permuted-Fashion, T=10, n_h=256, n=10 paired seeds,
Adam lr=1e-3, 2 epochs/task, 同一 perms、同一 eval）：

  ER   (Experience Replay, 2019 baseline form)
       存一个大小为 K 的 buffer，训练时每 batch 混入 K 个回放样本。
  DER  (Dark Experience Replay, Buzzega et al. 2020)
       存 (logits, label) 对，训练时在回放样本上加 MSE 蒸馏项。

两者都用与 DeLP 相同形状的骨干（这里用 ADNet 的前向形式但不带 segment
bias，n_h=256），因此比较是"保护方式"的比较而不是"架构"的比较。
若 replay 方法的表现远超 DeLP，那么"DeLP 改善了 retention-plasticity
权衡"的相对主张需要重新表述；若 DeLP 与它们相当，则原文的基线不足
这一指责被直接驳回。

用法：python s5_modern_baselines.py --out results_s5/s5_baselines.json
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
from branch_delp import make_anchor, anchor_penalty, flat_params
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = 256
BUFFER = 500          # ER buffer size (standard for MNIST CL)
DER_ALPHA = 0.5       # DER loss weight
DER_BETA = 0.5        # DER logit-distillation weight


def run_er(seed, data, perms, ntask, lam=LAM, nh=NH, buf_size=BUFFER):
    """Experience Replay with a reservoir buffer of (x, y) pairs."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    buf_x, buf_y = [], []
    seen = 0
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                if buf_x:
                    # reservoir sampling into the buffer
                    for i in range(xb.size(0)):
                        if len(buf_x) < buf_size:
                            buf_x.append(xb[i].cpu()); buf_y.append(yb[i].cpu())
                        else:
                            j = int(torch.randint(0, seen + 1, (1,)).item())
                            if j < buf_size:
                                buf_x[j] = xb[i].cpu(); buf_y[j] = yb[i].cpu()
                        seen += 1
                if buf_x:
                    ridx = torch.randint(0, len(buf_x), (BS,))
                    rx = torch.stack([buf_x[i] for i in ridx]).to(DEV)
                    ry = torch.stack([buf_y[i] for i in ridx]).to(DEV)
                    xb = torch.cat([xb, rx], 0)
                    yb = torch.cat([yb, ry], 0)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                # NOTE: no DeLP anchor chain here.  Replay-based methods are
                # an ALTERNATIVE protection mechanism, not an addition to one:
                # stacking the lambda=1000 elastic penalty on top of a replay
                # buffer would suppress exactly the gradient the buffer is
                # there to provide, and the comparison would be meaningless.
                loss = F.cross_entropy(out, yb)
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "method": "ER", "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def run_der(seed, data, perms, ntask, lam=LAM, nh=NH, buf_size=BUFFER,
            alpha=DER_ALPHA, beta=DER_BETA):
    """Dark Experience Replay: replay (x, y, logits) with a logit MSE term."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    buf_x, buf_y, buf_z = [], [], []
    seen = 0
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                # no DeLP anchor chain: replay is an alternative protection
                # mechanism (see the note in run_er)
                loss = F.cross_entropy(out, yb)
                # DER: store the current batch's logits, then replay
                with torch.no_grad():
                    z = out.detach().cpu()
                for i in range(xb.size(0)):
                    if len(buf_x) < buf_size:
                        buf_x.append(xb[i].cpu()); buf_y.append(yb[i].cpu())
                        buf_z.append(z[i].cpu())
                    else:
                        j = int(torch.randint(0, seen + 1, (1,)).item())
                        if j < buf_size:
                            buf_x[j] = xb[i].cpu(); buf_y[j] = yb[i].cpu()
                            buf_z[j] = z[i].cpu()
                    seen += 1
                if buf_x:
                    ridx = torch.randint(0, len(buf_x), (BS,))
                    rx = torch.stack([buf_x[i] for i in ridx]).to(DEV)
                    ry = torch.stack([buf_y[i] for i in ridx]).to(DEV)
                    rz = torch.stack([buf_z[i] for i in ridx]).to(DEV)
                    rctx = F.one_hot(torch.full((rx.size(0),), phase, device=DEV),
                                     ntask).float()
                    rout, _ = model(rx, rctx)
                    der_loss = F.mse_loss(rout, rz)
                    loss = loss + alpha * der_loss + beta * F.cross_entropy(rout, ry)
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "method": "DER", "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_s5/s5_baselines.json')
    ap.add_argument('--methods', default='ER,DER')
    ap.add_argument('--datasets', default='mnist,fashion')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--ntask', type=int, default=NTASK)
    ap.add_argument('--lam', type=float, default=LAM)
    ap.add_argument('--buf', type=int, default=BUFFER)
    args = ap.parse_args()

    methods = args.methods.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out, encoding='utf-8')).get('results', [])
        except Exception:
            results = []
    done = {(r['dataset'], r['method'], r['seed']) for r in results}

    for ds in args.datasets.split(','):
        data = load_data(ds)
        for seed in seeds:
            perms = get_perms(seed, args.ntask)
            for meth in methods:
                if (ds, meth, seed) in done:
                    continue
                t0 = time.time()
                fn = run_er if meth == 'ER' else run_der
                rec = fn(seed, data, perms, args.ntask, lam=args.lam,
                         buf_size=args.buf)
                rec['dataset'] = ds
                rec['elapsed'] = round(time.time() - t0, 1)
                results.append(rec)
                done.add((ds, meth, seed))
                print(f"  {ds} s{seed} {meth}: ell={rec['ell']:.2f} "
                      f"avg={rec['final_avg']:.2f} bwt={rec['bwt']:.2f} "
                      f"({rec['elapsed']}s)", flush=True)
                json.dump({"results": results, "meta": vars(args)},
                          open(args.out, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out} ({len(results)} records)')


if __name__ == '__main__':
    main()
