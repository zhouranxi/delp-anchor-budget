# -*- coding: utf-8 -*-
r"""A2: dominant-segment 仪器的正对照（oracle）。

动机：论文的标题级主张是 "specialization does not emerge"，而其否定读数
全部来自 dominant-segment 仪器 —— 一个论文自己承认"在本架构族中不可靠"的
仪器。一个只会返回负值的仪器无法区分"没有专门化"与"仪器瞎了"。

正对照：训练/推理中强制 own segment 主导
    u_{j,t} := +M   (own segment)
    u_{j,k} := -M   (k != t)
若该模型在 active-only 仪器下 own rate ≈ 1.0，则仪器本身能检出专门化，
S4/S1 的否定读数成为有意义的负结果；若检不出，则"专门化不存在"不可证伪。

三臂：oracle (M=5) / weak-oracle (M=1) / base。
M=1 检验检出阈值（若 M=1 都检不出，说明门值尺度下仪器不灵敏）。

指标：active-only own-segment rate 的逐 phase 曲线（+ Wilson CI）、
训练后 own rate 的配对检验。

用法：python s5_oracle_control.py --out results_s5/s5_oracle.json
"""
import argparse, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute, kwta
from branch_delp import (ADNet, make_anchor, anchor_penalty, flat_params,
                        dom_instr, usage_stats)
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = 256


class OracleADNet(ADNet):
    """ADNet with the own segment forced to dominate by a margin M.

    IMPORTANT (found while running this control): the paper's dominant-segment
    instrument (`branch_delp.usage_stats`) reads `model.dend(ctx)` directly,
    i.e. the RAW gate, not `forward`.  An override that lives only inside
    `forward` is therefore invisible to the instrument -- the oracle would
    "fail" for a reason that has nothing to do with the instrument's
    sensitivity.  To make the positive control a fair test, the override is
    applied in a single place (`gate_u`) that BOTH the instrument and the
    forward pass use.

    We achieve this by replacing the `dend` Linear with a small wrapper
    module whose forward applies the override.
    """
    def __init__(self, ntask, nseg, M=5.0, nh=NH):
        super().__init__(nh=nh, protected=True, nseg=nseg, ntask=ntask)
        self.M = M
        self._dend_lin = self.dend
        self.dend = _OracleDend(self._dend_lin, M, nseg, nh)

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id


class _OracleDend(nn.Module):
    """Wraps the context->dendrite Linear and forces own-segment dominance."""
    def __init__(self, lin, M, nseg, nh):
        super().__init__()
        self.lin = lin
        self.M = M
        self.nseg = nseg
        self.nh = nh

    def forward(self, ctx):
        u = self.lin(ctx).view(-1, self.nseg, self.nh)
        if self.M > 0:
            tid = ctx.argmax(1)
            u = torch.full_like(u, -self.M)
            u[torch.arange(u.size(0), device=u.device), tid] = self.M
        return u.view(-1, self.nh * self.nseg)

    @property
    def weight(self):
        # ADNet.protect_grads reaches for self.dend.weight
        return self.lin.weight


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def run_arm(seed, data, perms, ntask, arm, lam=LAM, M=5.0, nh=NH):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    if arm == 'oracle':
        model = OracleADNet(ntask, ntask, M=M, nh=nh).to(DEV)
    elif arm == 'weak_oracle':
        model = OracleADNet(ntask, ntask, M=1.0, nh=nh).to(DEV)
    else:
        model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    own_per_phase = []
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

        # --- instrument the dominant-segment statistic on this phase ---
        # 用论文完全相同的仪器：branch_delp.usage_stats -> dom_instr
        o, o_seg = usage_stats(model, xs, phase, ntask)
        instr = dom_instr(o, o_seg, ntask)
        ha = instr['dom_hist_active']
        tot_a = sum(ha)
        own_active = ha[phase] if tot_a else 0
        rec_ph = {
            'phase': phase,
            'own_rate_active': round(own_active / tot_a, 4) if tot_a else None,
            'own_count_active': own_active,
            'n_active': tot_a,
            'own_rate_naive': round(instr['dom_hist'][phase] / max(1, sum(instr['dom_hist'])), 4),
            'dead_frac': instr['dead_frac'],
        }
        if tot_a:
            lo, hi = wilson(own_active, tot_a)
            rec_ph['own_rate_ci95'] = [round(lo, 4), round(hi, 4)]

        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        own_per_phase.append(rec_ph)

    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "arm": arm, "M": (M if arm != 'base' else 0.0),
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2), "own_per_phase": own_per_phase,
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_s5/s5_oracle.json')
    ap.add_argument('--arms', default='base,weak_oracle,oracle')
    ap.add_argument('--datasets', default='mnist,fashion')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--ntask', type=int, default=NTASK)
    ap.add_argument('--lam', type=float, default=LAM)
    ap.add_argument('--M', type=float, default=5.0)
    args = ap.parse_args()

    arms = args.arms.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out, encoding='utf-8')).get('results', [])
        except Exception:
            results = []
    done = {(r['dataset'], r['arm'], r['seed']) for r in results}

    for ds in args.datasets.split(','):
        data = load_data(ds)
        for seed in seeds:
            perms = get_perms(seed, args.ntask)
            for arm in arms:
                if (ds, arm, seed) in done:
                    continue
                t0 = time.time()
                rec = run_arm(seed, data, perms, args.ntask, arm,
                              lam=args.lam, M=args.M)
                rec['dataset'] = ds
                rec['elapsed'] = round(time.time() - t0, 1)
                results.append(rec)
                done.add((ds, arm, seed))
                print(f"  {ds} s{seed} {arm}: ell={rec['ell']:.2f} "
                      f"avg={rec['final_avg']:.2f} "
                      f"own_rate_naive(end)={rec['own_per_phase'][-1]['own_rate_naive']:.3f} "
                      f"({rec['elapsed']}s)", flush=True)
                json.dump({"results": results, "meta": vars(args)},
                          open(args.out, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out} ({len(results)} records)')


if __name__ == '__main__':
    main()
