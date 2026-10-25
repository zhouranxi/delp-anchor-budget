"""S4 follow-up v2 (corrected control): is the b=10 repair gate ablation?

Hypothesis from s4_diag: b=10 saturates sigmoid(d) to ~1 for all neurons
(gate_mean 0.565 -> ~1.0 region for biased segment, and winning-segment gate
becomes ~uniform), so the model degenerates to a context-free kWTA network:
h = kwta(t * ~1) ~= kwta(t).

Test: train the NO-DENDRITE degenerate ADNet (e4's naive variant:
h = kwta(sigmoid(ff(x)))) on the identical protocol (T10, lam=1000 anchor
chain on shared ff, seeds 0-9) and compare favg with the b=10 BiasADNet arm
from s4ctx. If they match within noise, the repair = gate ablation story
is confirmed; if b=10 > no-dendrite, the biased routing adds something.
"""

import argparse, json, os, sys
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, _here)

import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute
from branch_delp import (ADNet, make_anchor, anchor_penalty, flat_params)
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0


class NoDendNet(torch.nn.Module):
    """e4's degenerate ADNet: no dendrites, h = kwta(sigmoid(ff(x))).

    Keeps the SAME ff/out shapes as ADNet (nh=2000) and the same anchor
    chain interface (flat_params-compatible keys) so the protection penalty
    applies to ff/out exactly as in the dendritic nets.
    """

    def __init__(self, nh=2000):
        super().__init__()
        self.nh = nh
        self.ff = torch.nn.Linear(784, nh)
        self.out = torch.nn.Linear(nh, 10)
        self.nseg = 1

    def forward(self, x, ctx=None):
        # CORRECT degenerate control: gate -> 1 means h = kwta(t) with RAW t
        # (ADNet computes kwta(t * sigmoid(d)); sigmoid(10+u)~1 saturates it).
        # The earlier version wrongly applied sigmoid to t -- activation
        # structure mismatch, invalidating the comparison.
        return self.out(e4_kwta(self.ff(x))), None


def e4_kwta(h, k=64):
    vals, idx = h.topk(k, dim=1)
    gate = torch.zeros_like(h)
    gate.scatter_(1, idx, 1.0)
    return h * gate


def run(ds, seed, lam=LAM):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, NTASK)
    model = NoDendNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    for phase in range(NTASK):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    # anchor on ff/out keys only (no dend keys in this net)
                    P = {k: v for k, v in flat_params_like(model).items()}
                    for a in anchors:
                        loss = loss + anchor_penalty_like(a, P)
                loss.backward()
                opt.step()
        anchors.append(make_anchor_nodend(model, xs))
        a = eval_model(model, xte, yte, perms, phase, NTASK, False)
        acc.update(a)
    bwt, favg = finish(acc, NTASK)
    return {"kind": "nodend", "seed": seed, "dataset": ds, "ntask": NTASK,
            "lam": lam, "bwt": round(bwt, 2), "final_avg": round(favg, 2)}


def flat_params_like(model):
    return {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
            "out.w": model.out.weight, "out.b": model.out.bias}


def make_anchor_nodend(model, xs, ntask=NTASK):
    """Plain EWC anchor on ff/out (Fisher via sampled grads), no usage."""

    P = flat_params_like(model)
    theta = {k: v.detach().clone() for k, v in P.items()}
    model.eval()
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    n = min(len(xs), 1000)
    idx = torch.randperm(len(xs))[:n]
    for i in idx:
        x = xs[i:i + 1].to(DEV)
        model.zero_grad()
        out, _ = model(x)
        # Fisher w.r.t. predicted class (standard EWC practice)
        loss = -torch.log_softmax(out, 1)[0].max()
        loss.backward()
        for k, v in P.items():
            if v.grad is not None:
                fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= n
    model.zero_grad()
    return {"gran": "plain", "theta": theta, "fisher": fisher}


def anchor_penalty_like(a, P):
    pen = 0.0
    for k, v in P.items():
        pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
    return pen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s4nodend.json")
    args = ap.parse_args()
    results = []
    for ds in ("mnist", "fashion"):
        for s in range(10):
            r = run(ds, s)
            results.append(r)
            print(f"DONE {ds} s{s}: favg={r['final_avg']} bwt={r['bwt']}",
                  flush=True)
    json.dump({"results": results}, open(args.out, "w"), indent=1)
    print("WROTE", args.out)
