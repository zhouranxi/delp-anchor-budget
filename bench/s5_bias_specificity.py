# -*- coding: utf-8 -*-
r"""A1: own-segment 特异性对照（能判死中心解释的最便宜实验）。

动机（S4 机制结论的必然推论，论文此前没做）：
若 own-segment bias 的作用机制是"把一个本来与上下文无关的门饱和到 ≈1"，
那么把同样大小的 bias 加在 *任意* segment 上、或加在 *所有* segment 上，
应产生同等增益。若机制是"恢复 own-segment 路由"，则只有加在 own 上才有效。

四个臂（同一协议：T=10, n_h=256, λ=1000, usage-weighted Fisher anchors,
n=10 paired seeds, Permuted-MNIST + Permuted-Fashion）：

  base      b=0，无 bias
  repaired  b=10 加在 own segment（现有的第三点对照）
  random    b=10 加在一个 *随机选定* 的 segment（每个 seed 抽一次并固定）
  uniform   b=10 加在 *所有* segment 上

判据（预声明）：
  - 若 random 或 uniform 的 Δℓ 与 repaired 的差异不显著（Holm 校正后
    90% CI 落在 ±0.5 点内），则 "own" 这一限定是冗余的，own-segment 语义
    不成立，全文须改写为 "一个常数去阻尼干预"。
  - 若 repaired 显著优于 random 与 uniform，则 "own" 有特异性。

另附 rescale 臂：h = kwta(t / 0.565)（不改门，只把 trunk 输出按常数缩放），
用来区分"bias 抬高门值"与"等效于放大 trunk 信号"。

用法：python s5_bias_specificity.py --out results_s5/s5_spec.json
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute, kwta
from branch_delp import ADNet, make_anchor, anchor_penalty, flat_params
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = 256


class SegBiasADNet(ADNet):
    """ADNet + a constant bias on a *specified* set of segments.

    bias_mode:
      'own'    -> bias added to the context-selected (own) segment only
      'random' -> bias added to one fixed random segment per (task, neuron)
      'uniform'-> bias added to every segment
    """
    def __init__(self, ntask, nseg, bias=0.0, bias_mode='own', nh=NH,
                 rand_map=None):
        super().__init__(nh=nh, protected=True, nseg=nseg, ntask=ntask)
        self.bias = bias
        self.bias_mode = bias_mode
        # rand_map: (ntask, nh) int -> which segment gets the bias for that
        # (task, neuron). Fixed per seed so the arm is deterministic.
        self.rand_map = rand_map

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.bias != 0.0:
            if self.bias_mode == 'own':
                tid = ctx.argmax(1)
                bmask = torch.zeros_like(u)
                bmask[torch.arange(u.size(0), device=u.device), tid] = self.bias
                u = u + bmask
            elif self.bias_mode == 'uniform':
                u = u + self.bias
            elif self.bias_mode == 'random':
                # for each (neuron, segment) pair, bias applies if that
                # segment is the chosen one for the current task
                tid = ctx.argmax(1)                     # (batch,)
                m = self.rand_map[tid]                  # (batch, nh) segment ids
                sel = (torch.arange(self.nh, device=u.device).unsqueeze(0)
                       == m)                            # (batch, nh)
                bmask = torch.zeros_like(u)
                bmask.scatter_(1, m.unsqueeze(1),
                               self.bias * sel.unsqueeze(1).float())
                u = u + bmask
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id


class RescaleADNet(ADNet):
    """ADNet with the gate bypassed by a constant rescale of the trunk.

    h = kwta(t / c).  This is the 'de-damping' arm: it does not touch the
    gate at all, it just removes a constant factor of ~0.565 from the
    effective signal.  If it reproduces the repair's gain, the repair is a
    constant rescale in disguise.
    """
    def __init__(self, ntask, nseg, c=0.565, nh=NH):
        super().__init__(nh=nh, protected=True, nseg=nseg, ntask=ntask)
        self.c = c

    def forward(self, x, ctx):
        t = self.ff(x)
        h = kwta(t / self.c)
        return self.out(h), None


def run_arm(seed, data, perms, ntask, arm, lam=LAM, bias=10.0, nh=NH):
    """One training run of one arm. Returns the same record shape as
    branch_delp.run_gran so the analysis code is shared."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)

    if arm == 'base':
        model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    elif arm == 'rescale':
        model = RescaleADNet(ntask, ntask, c=0.565, nh=nh).to(DEV)
    else:
        rand_map = None
        mode = 'own' if arm == 'repaired' else arm
        if arm == 'random':
            g = torch.Generator().manual_seed(10000 + seed)
            rand_map = torch.randint(0, ntask, (ntask, nh), generator=g).to(DEV)
        model = SegBiasADNet(ntask, ntask, bias=(0.0 if arm == 'base' else bias),
                             bias_mode=mode, nh=nh, rand_map=rand_map).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
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
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, 'neuron')
                    loss = loss + lam * pen
                loss.backward()
                if hasattr(model, 'protect_grads'):
                    model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, ntask, 'neuron'))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "arm": arm, "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_s5/s5_spec.json')
    ap.add_argument('--arms', default='base,repaired,random,uniform,rescale')
    ap.add_argument('--datasets', default='mnist,fashion')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--ntask', type=int, default=NTASK)
    ap.add_argument('--lam', type=float, default=LAM)
    ap.add_argument('--bias', type=float, default=10.0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    arms = args.arms.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.smoke:
        seeds = seeds[:2]
        arms = arms[:2]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    results = []
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path, encoding='utf-8')).get('results', [])
        except Exception:
            results = []
    done = {(r['dataset'], r['arm'], r['seed']) for r in results}

    for ds in args.datasets.split(','):
        data = load_data(ds)
        for seed in seeds:
            perms = get_perms(seed, args.ntask)
            for arm in arms:
                key = (ds, arm, seed)
                if key in done:
                    continue
                t0 = time.time()
                rec = run_arm(seed, data, perms, args.ntask, arm,
                              lam=args.lam, bias=args.bias)
                rec['dataset'] = ds
                rec['elapsed'] = round(time.time() - t0, 1)
                results.append(rec)
                done.add(key)
                print(f"  {ds} s{seed} {arm}: ell={rec['ell']:.2f} "
                      f"avg={rec['final_avg']:.2f} bwt={rec['bwt']:.2f} "
                      f"({rec['elapsed']}s)", flush=True)
                json.dump({"results": results, "meta": vars(args)},
                          open(out_path, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
    print(f'\n写入 {out_path} ({len(results)} records)')


if __name__ == '__main__':
    main()
