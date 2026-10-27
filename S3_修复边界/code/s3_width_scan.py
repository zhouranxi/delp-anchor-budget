"""S3 FIX: width x normalization factorial scan (2x2).

WHY THIS EXISTS (data-consistency audit, 2026-10-27)
------------------------------------------------------
S3's decisive control is GN-MLP: "the original single-hidden-layer network
with one GroupNorm layer inserted between the feature projection and the
dendritic gate ... the only delta".  That claim was FALSE in the width
dimension.  From the raw records (sum of dom_hist per phase = n_h):

    bare MLP (c2)        n_h =  256
    GN-MLP   (s5bgn)     n_h = 2000
    ConvADNet(cnnc2)     n_h = 2048

So the GN-MLP arm differed from the bare-MLP arm by BOTH GroupNorm AND a
7.8x wider hidden layer.  S3's causal claim -- "mortality rises but the
repair gain collapses anyway, therefore gate death is not the mediating
variable" -- could be explained by width alone.

THIS SCRIPT separates the two factors with a 2x2 design:

    width      : 256 (bare-MLP default)  vs  2000 (GN-MLP default)
    normaliz.  : none (plain ADNet)      vs  GroupNorm(16, nh)

    4 cells x 2 datasets x 3 doses {1,3,10} x 10 seeds = 240 runs

Cell naming: w{width}_{plain|gn}   (b=0 arm included for the paired baseline)

The key readout is the 2x2 interaction:
  - If the GN effect on the repair gain survives at MATCHED width, the
    dying-gate refutation stands.
  - If the gain collapse tracks WIDTH (wider => stronger baseline => less
    headroom => smaller gain, exactly S3's own "repair headroom" law),
    then S3's mechanism claim must be restated: what is being refuted is
    not "gate death" but "baseline headroom".
  - If both factors act independently, that is the honest, stronger result.

Everything else (protocol, lambda, anchors, seeds, permutations) is byte-for-byte
the same as s5b_gnmlp.py / branch_delp.run_gran.
"""

import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, _here)

import exp4_branch_protection as e4
import exp6_baselines as eb
from exp6_baselines import get_perms, eval_model, finish
from exp4_branch_protection import kwta, permute
from branch_delp import make_anchor, anchor_penalty, flat_params
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
SEEDS = list(range(10))
DOSES = [0.0, 1.0, 3.0, 10.0]
WIDTHS = [256, 2000]


class ScanNet(nn.Module):
    """ADNet with an optional GroupNorm on the ff features.

    width and normalization are the two independent factors of the 2x2.
    """
    def __init__(self, ntask, nseg, nh=256, gn=False, own_bias=0.0):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.gn = nn.GroupNorm(16, nh) if gn else None
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh = nseg, nh
        self.protected = True
        self.own_bias = own_bias

    def forward(self, x, ctx):
        t = self.ff(x)
        if self.gn is not None:
            t = self.gn(t)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.own_bias != 0.0:
            tid = ctx.argmax(1)
            bias = torch.zeros_like(u)
            bias[torch.arange(u.size(0), device=u.device), tid] = self.own_bias
            u = u + bias
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
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
    anchors, acc = [], {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:eb.NTRAIN], perms[phase]), ytr[:eb.NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=eb.BS, shuffle=True)
        model.train()
        for ep in range(eb.EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(eb.DEV), yb.to(eb.DEV)
                ctx = F.one_hot(
                    torch.full((xb.size(0),), phase, device=eb.DEV),
                    ntask).float()
                opt.zero_grad()
                out, seg = model(xb, ctx)
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
        anchors.append(make_anchor(model, xs, phase, ntask, "neuron"))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"kind": "scan", "seed": seed, "dataset": ds, "gran": "neuron",
            "ntask": ntask, "lam": lam, "nh": nh, "gn": gn,
            "bias": own_bias, "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default="s3_scan.json")
    args = ap.parse_args()

    jobs = []
    for ds in ("mnist", "fashion"):
        for nh in WIDTHS:
            for gn in (False, True):
                for b in DOSES:
                    for s in SEEDS:
                        jobs.append((ds, s, nh, gn, b))
    shard = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    res = []
    t0 = time.time()
    for i, (ds, s, nh, gn, b) in enumerate(shard):
        r = run(ds, s, nh, gn, b)
        res.append(r)
        print(f"[{i+1}/{len(shard)}] {ds} seed={s} nh={nh} gn={int(gn)} b={b} "
              f"ell={r['ell']} favg={r['final_avg']} bwt={r['bwt']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({"results": res}, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}: {len(res)} records")


if __name__ == "__main__":
    main()
