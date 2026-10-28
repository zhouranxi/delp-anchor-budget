"""A5: re-run the S3 2x2 (width x normalization) scan WITH a mortality column.

Why
---
The reviewer's A5 asked for the factorial scan to report neuron mortality
alongside the accuracy gain, because the paper's mechanism claim is that
mortality and gain do not track each other, and the scan is the table where
a reader would look for that.  The archived scan
(S3_修复边界/results/s3_scan_fixed_*.json) records accuracy but not
mortality, so the two claims could not be checked against it.

What this adds
--------------
`make_anchor` (branch_delp.py) already returns `dead_frac` alongside the
dominance histogram, so the measurement exists; the archived scan simply
did not write it out.  This script is the archived scan with one extra
field per record and nothing else changed -- same model, same protocol,
same anchors, same seed handling.

Protocol (identical to the archived scan):
  widths   : 256 (bare-MLP default) and 2000 (GN-MLP default)
  gn       : off / on
  bias     : 0 and 10
  datasets : mnist, fashion
  ntask    : 10
  lam      : 1000
  seeds    : 10 per cell

Output: results_s5/server/s3_scan_mortality.json
"""
import argparse
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import exp6_baselines as eb
from exp6_baselines import get_perms, eval_model, finish
from exp4_branch_protection import permute
from branch_delp import make_anchor, anchor_penalty, flat_params
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
WIDTHS = [256, 2000]
SEEDS = list(range(10))


class ScanNet(torch.nn.Module):
    """Same network as the archived scan: optional GroupNorm, optional bias."""

    def __init__(self, ntask, nseg, nh=256, gn=False, own_bias=0.0):
        super().__init__()
        self.ff = torch.nn.Linear(784, nh)
        self.gn = torch.nn.GroupNorm(16, nh) if gn else None
        self.dend = torch.nn.Linear(ntask, nh * nseg, bias=False)
        self.out = torch.nn.Linear(nh, 10)
        self.nseg, self.nh = nseg, nh
        self.own_bias, self.gn_on = own_bias, gn

    def forward(self, x, ctx):
        t = self.ff(x)
        if self.gn is not None:
            t = self.gn(t)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.own_bias != 0.0:
            bias = torch.zeros_like(u)
            tid = ctx.argmax(1)
            bias[torch.arange(u.size(0), device=u.device), tid] = self.own_bias
            u = u + bias
        d, seg_id = u.max(dim=1)
        h = eb.kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id

    def protect_grads(self, ctx_label):
        if self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad
        Wv = W.view(self.nseg, self.nh, -1)
        mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        mask[ctx_label] = 1.0
        Wv.mul_(mask)


def run(ds, seed, nh, gn, own_bias, ntask=NTASK, lam=LAM):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, ntask)
    model = ScanNet(ntask, ntask, nh=nh, gn=gn, own_bias=own_bias).to(eb.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=eb.LR)
    anchors = []
    dead = []
    acc = {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:eb.NTRAIN], perms[phase]), ytr[:eb.NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=eb.BS, shuffle=True)
        model.train()
        for ep in range(eb.EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(eb.DEV), yb.to(eb.DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=eb.DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, "neuron")
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anc = make_anchor(model, xs, phase, ntask, "neuron")
        anchors.append(anc)
        dead.append(float(anc.get("dead_frac", float("nan"))))
        # evaluate after EVERY phase, exactly as the archived scan does
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {
        "kind": "scan_mortality", "seed": seed, "dataset": ds, "gran": "neuron",
        "ntask": ntask, "lam": lam, "nh": nh, "gn": gn, "bias": own_bias,
        "bwt": round(bwt, 2), "final_avg": round(favg, 2),
        "ell": round(ell, 2),
        # NEW: mortality per phase, and its mean over the sequence
        "dead_frac_per_phase": [round(d, 4) for d in dead],
        "dead_frac_mean": round(sum(dead) / len(dead), 4),
        "dead_frac_final": round(dead[-1], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_s5/s3_scan_mortality.json")
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    jobs = []
    for ds in datasets:
        for nh in WIDTHS:
            for gn in (False, True):
                for bias in (0.0, 10.0):
                    for seed in seeds:
                        jobs.append((ds, seed, nh, gn, bias))
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    out = {'rows': [], 'config': {
        'widths': WIDTHS, 'gn': [False, True], 'bias': [0.0, 10.0],
        'datasets': datasets, 'seeds': seeds, 'lam': LAM, 'ntask': NTASK,
        'shard': args.shard, 'n_shards': args.n_shards}}
    t0 = time.time()
    for k, (ds, seed, nh, gn, bias) in enumerate(jobs):
        r = run(ds, seed, nh, gn, bias)
        out['rows'].append(r)
        print(f"[{k+1}/{len(jobs)}] {ds} s{seed} nh={nh} gn={gn} b={bias}: "
              f"ell={r['ell']} favg={r['final_avg']} "
              f"dead_final={r['dead_frac_final']} ({time.time()-t0:.0f}s)",
              flush=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
    print(f"WROTE {args.out} ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
