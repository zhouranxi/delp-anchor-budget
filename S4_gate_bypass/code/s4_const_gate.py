"""S4 follow-up v3: the constant-gate control (2026-10-27).

WHY: after the lambda/width fix, the no-gate control (h = kwta(t)) NO LONGER
matches the repaired network on MNIST (no-gate 63.63 vs repair 68.67,
t=-13.6), so the "repair = gate deletion" reading is refuted. But the
mechanism probes show the repaired gate is not context-free-random either:
the winning-segment gate value is a CONSTANT ~0.565 under every context,
and the flip rate collapses to ~0.008-0.02%.

Hypothesis under test: the repair does not DELETE the gate, it FLATTENS it
to a per-network constant multiplicative gain:

    h = kwta(t * c)      with c ~= mean sigmoid(d) ~= 0.565

If a ConstantGateNet (c fixed at the measured 0.565, width 256, lam 1000
actually applied) matches the repaired arm, the mechanism claim becomes
PRECISE: "the repair replaces context routing with a constant multiplicative
gain" -- strictly weaker than deletion, and exactly what the probes read.

Control arms in the same run (paired seeds):
  - repaired   BiasADNet b=10   (the real repair)
  - const-gate ConstantGateNet  (h = kwta(t * c), c=0.565)
  - no-gate    NoDendNet        (h = kwta(t))         [replicates fixed run]
  - base       ADNet b=0        (reference)

All arms: T=10, lam=1000 (applied), nh=256, seeds 0-9, both datasets.
"""

import argparse, json, os, sys, time
import torch, torch.nn.functional as F
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
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute, kwta
from branch_delp import (ADNet, BiasADNet, make_anchor, anchor_penalty,
                         flat_params)
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = e4.NH            # 256
CONST_C = 0.565       # measured mean sigmoid(d) of repaired nets


class ConstantGateNet(torch.nn.Module):
    """h = kwta(t * c) with a fixed constant gate c. Dendrite weights are
    frozen at the S4 initialization (never trained); only ff/out move.

    Keeps the branch_delp interface (ff/dend/out, nseg/nh, protect_grads)
    so make_anchor/usage_stats work unmodified.
    """
    def __init__(self, nh=NH, ntask=NTASK, c=CONST_C):
        super().__init__()
        self.ff = nn_Linear(784, nh)
        self.out = nn_Linear(nh, 10)
        # dend exists for the anchor interface; frozen, and its output is
        # overridden to the constant c in forward().
        self.dend = nn_Linear(ntask, nh * ntask, bias=False)
        for p in self.dend.parameters():
            p.requires_grad_(False)
        self.nseg, self.nh = ntask, nh
        self.protected = True
        self.c = c

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d = u.max(dim=1).values          # real winning value (for usage_stats)
        # CONSTANT gate: replace the winning-segment sigmoid by c
        h = kwta(t * self.c)
        seg = u.max(dim=1).indices
        return self.out(h), seg

    def protect_grads(self, ctx_label):
        pass  # dend frozen; ff/out are shared (no segment masking needed)


import torch.nn as nn
nn_Linear = nn.Linear


class NoDendNet(torch.nn.Module):
    """h = kwta(t). Width matched to the dendritic arms (256)."""
    def __init__(self, nh=NH):
        super().__init__()
        self.ff = nn_Linear(784, nh)
        self.out = nn_Linear(nh, 10)
        self.dend = nn_Linear(NTASK, nh * NTASK, bias=False)
        for p in self.dend.parameters():
            p.requires_grad_(False)
        self.nseg, self.nh = NTASK, nh
        self.protected = True

    def forward(self, x, ctx=None):
        h = kwta(self.ff(x))
        return self.out(h), None

    def protect_grads(self, ctx_label):
        pass


def run_arm(arm, ds, seed, lam=LAM, nh=NH):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, NTASK)

    if arm == "repaired":
        model = BiasADNet(NTASK, NTASK, own_bias=10.0,
                          bias_mode="fixed", nh=nh).to(DEV)
    elif arm == "const":
        model = ConstantGateNet(nh=nh).to(DEV)
    elif arm == "nogate":
        model = NoDendNet(nh=nh).to(DEV)
    elif arm == "base":
        model = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=nh).to(DEV)
    else:
        raise ValueError(arm)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR)
    anchors, acc = [], {}
    for phase in range(NTASK):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                NTASK).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, NTASK, "neuron"))
        a = eval_model(model, xte, yte, perms, phase, NTASK, True)
        acc.update(a)
    bwt, favg = finish(acc, NTASK)
    ell = sum(acc[(t, t)] for t in range(NTASK)) / NTASK
    return {"kind": f"constgate_{arm}", "arm": arm, "seed": seed,
            "dataset": ds, "ntask": NTASK, "lam": lam, "nh": nh,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s4_constgate.json")
    ap.add_argument("--nh", type=int, default=NH)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    results = []
    t0 = time.time()
    for arm in ("repaired", "const", "nogate", "base"):
        for ds in ("mnist", "fashion"):
            for s in range(10):
                r = run_arm(arm, ds, s, nh=args.nh)
                results.append(r)
                print(f"DONE {arm} {ds} s{s}: lam={r['lam']} nh={r['nh']} "
                      f"ell={r['ell']} favg={r['final_avg']} bwt={r['bwt']} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump({"results": results}, open(args.out, "w"), indent=1)
    print("WROTE", args.out)
