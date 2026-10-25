#!/usr/bin/env python
"""S5b/E4: GN-MLP — the decisive dying-gate test.

Hypothesis (from S5/CNN double failure): the own-bias repair works only
where kWTA mortality exists. Insert GroupNorm between the MLP ff layer
and the dendritic gate -> features stay alive -> mortality should drop
to ~0 AND the repair effect should vanish, all else identical to S1.

Design: exact mirror of branch_delp.run_gran (neuron arm) with ONE delta:
    t = GN(Linear(x))   instead of   t = Linear(x)
Arms: b in {0,1,3,10} x {mnist,fashion} x T=10 x seeds 0-9 = 80 runs.
Reference: existing c2_* (bare MLP) b=0/b=10 data.
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

import exp4_branch_protection as e4
import anchor_budget
import exp6_baselines as eb
import branch_delp as bd
from branch_delp import make_anchor, anchor_penalty, _clone
from exp6_baselines import get_perms, eval_model, finish
from exp4_branch_protection import kwta
from anchor_budget import load_data

NH = 2000
LAM = 1000.0
SEEDS = list(range(10))
DOSES = [0.0, 1.0, 3.0, 10.0]


class GNFFNet(nn.Module):
    """ADNet with GroupNorm on the ff features (the ONLY delta vs exp4.ADNet)."""
    def __init__(self, ntask, nseg, nh=NH, own_bias=0.0):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.gn = nn.GroupNorm(16, nh)                 # <-- the delta
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh = nseg, nh
        self.protected = True
        self.own_bias = own_bias

    def forward(self, x, ctx):
        t = self.gn(self.ff(x))
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


def flat_params(model):
    return {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
            "gn.w": model.gn.weight, "gn.b": model.gn.bias,
            "dend.w": model.dend.weight,
            "out.w": model.out.weight, "out.b": model.out.bias}


# make_anchor/fisher_raw resolve flat_params at call time from the
# branch_delp module namespace -> patch it so GN keys are visible there.
# fisher/usage only touch model(x, ctx) and model.dend (GN-compatible).
bd.flat_params = flat_params


def run_gn(seed, data, perms, ntask, lam, own_bias):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = GNFFNet(ntask, ntask, own_bias=own_bias).to(eb.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=eb.LR)
    anchors, acc = [], {}
    dom_all = []
    for phase in range(ntask):
        xs, ys = e4.permute(xtr[:eb.NTRAIN], perms[phase]), ytr[:eb.NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=eb.BS, shuffle=True)
        model.train()
        for _ep in range(eb.EPOCHS):
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
        dom_all.append({k: anchors[-1][k] for k in
                        ("dom_hist", "dom_hist_active", "dead_frac")})
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "gran": "neuron", "bias": own_bias,
            "ntask": ntask, "lam": lam,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "dom_per_phase": dom_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s5bgn_mnist_b0.json")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--ntask", type=int, default=10)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    seed_list = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        eb.EPOCHS = 1
        eb.NTRAIN = 2000
        seed_list = seed_list[:2]

    data = load_data(a.dataset)
    results, t0 = [], time.time()
    for seed in seed_list:
        rec = {"kind": "gnmlp", "dataset": a.dataset, "seed": seed}
        try:
            perms = get_perms(seed, a.ntask)
            rec.update(run_gn(seed, data, perms, a.ntask, LAM, a.bias))
        except Exception as exc:                            # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL s{seed}: {exc}", flush=True)
        results.append(rec)
        print(f"  DONE s{seed} b={a.bias}: ell={rec.get('ell')} "
              f"avg={rec.get('final_avg')} bwt={rec.get('bwt')} "
              f"dead={rec.get('dom_per_phase',[{}])[-1].get('dead_frac')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(a.out, "w") as f:
        json.dump({"results": results, "elapsed": time.time() - t0},
                  f, indent=1)
    print(f"\nDONE: {len(results)} runs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
