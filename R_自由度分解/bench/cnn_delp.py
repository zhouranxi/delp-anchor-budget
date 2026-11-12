#!/usr/bin/env python
"""S5/CNN: ConvADNet — tail-mounted dendritic layer on a conv stack.

Architecture (design doc, option A):
    x (B,1,28,28) -> [Conv-GN-ReLU x2 stride2, Conv-GN-ReLU x2] -> GAP -> t (B,nh)
    ctx (B,ntask) -> dend (ntask -> nh*nseg) -> u (B,nseg,nh)
    d, seg_id = u.max(1);  h = kWTA(t * sigmoid(d));  logits = out(h)

Interface-compatible with exp4.ADNet:
    model(x, ctx) -> (logits, seg_id); model.protect_grads(task_id);
    model.dend; model.nseg; model.nh.  flat_params() serialises the conv
    stack under "ff.*" keys so anchor construction is key-agnostic.
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
from branch_delp import (make_anchor, anchor_penalty, _clone)
from exp6_baselines import get_perms, eval_model, finish
from exp4_branch_protection import kwta
from anchor_budget import load_data

NH = 2048
KSEG = None  # set to ntask at construct time
LAM = 1000.0
SEEDS = list(range(10))
DOSES = [0.0, 1.0, 3.0, 10.0]


class ConvStack(nn.Module):
    """4-layer conv feature extractor -> (B, nh)."""
    def __init__(self, nh=NH, in_ch=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False)
        self.g1 = nn.GroupNorm(8, 32)
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.g2 = nn.GroupNorm(8, 64)
        self.c3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.g3 = nn.GroupNorm(8, 128)
        self.c4 = nn.Conv2d(128, nh, 3, 2, 1, bias=False)
        self.g4 = nn.GroupNorm(16, nh)

    def forward(self, x):                       # (B, C, H, W)
        if x.dim() == 2:                        # accept flat (B, 784)
            x = x.view(-1, 1, 28, 28)
        x = F.relu(self.g1(self.c1(x)))
        x = F.relu(self.g2(self.c2(x)))
        x = F.relu(self.g3(self.c3(x)))
        x = F.relu(self.g4(self.c4(x)))
        return x.flatten(2).mean(dim=2)         # GAP -> (B, nh)


class ConvADNet(nn.Module):
    """ADNet-interface conv model; dend/kWTA semantics identical to exp4.ADNet."""
    def __init__(self, ntask, nseg=None, nh=NH, in_ch=1):
        super().__init__()
        nseg = nseg or ntask
        self.ff = ConvStack(nh, in_ch)
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh, self.ntask = nseg, nh, ntask
        self.protected = True

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id

    def protect_grads(self, ctx_label):
        """Same semantics as exp4.ADNet: freeze non-current dendrite segments."""
        if self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad                 # (nh*nseg, ntask)
        Wv = W.view(self.nseg, self.nh, -1)
        mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        mask[ctx_label] = 1.0
        Wv.mul_(mask)


class BiasConvNet(ConvADNet):
    """ConvADNet + constant bias on the OWN segment (mirrors branch_delp.BiasADNet)."""
    def __init__(self, ntask, own_bias=0.0, nh=NH, in_ch=1):
        super().__init__(ntask, ntask, nh, in_ch)
        self.own_bias = own_bias

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.own_bias != 0.0:
            tid = ctx.argmax(1)
            bias = torch.zeros_like(u)
            bias[torch.arange(u.size(0), device=u.device), tid] = self.own_bias
            u = u + bias
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id


def flat_params(model):
    """Serialise parameters under ff.*/dend/out keys (anchor-compatible)."""
    d = {}
    for name, p in model.ff.named_parameters():
        d[f"ff.{name}"] = p
    d["dend.w"] = model.dend.weight
    d["out.w"] = model.out.weight
    d["out.b"] = model.out.bias
    return d


def usage_stats(model, xs, task_id, ntask):
    """Raw-gate routing stats, identical spirit to branch_delp.usage_stats:
    reads model.dend directly (bias-free instrument)."""
    with torch.no_grad():
        ctx1 = F.one_hot(torch.tensor([task_id], device=xs.device), ntask).float()
        u = model.dend(ctx1).view(model.nseg, model.nh)   # (nseg, nh)
        o_seg_all = u                                     # full table
        winners = u.argmax(dim=0)                         # (nh,) seg id per neuron
        o = (winners == task_id).float()                  # own-seg indicator
        active = u.max(dim=0).values.abs() > 0            # GN+conv rarely all-zero
    return o, o_seg_all, winners, active


def run_cnn(seed, data, perms, ntask, lam, own_bias):
    """Mirror of branch_delp.run_gran (neuron arm) with CNN model.
    Records dom instrumentation per phase."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    if own_bias:
        model = BiasConvNet(ntask, own_bias=own_bias).to(eb.DEV)
    else:
        model = ConvADNet(ntask).to(eb.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=eb.LR)
    anchors, acc = [], {}
    dom_all = []
    for phase in range(ntask):
        xs, ys = e4.permute(xtr[:eb.NTRAIN], perms[phase]), ytr[:eb.NTRAIN]
        xs4 = xs.view(-1, 1, 28, 28)
        dl = DataLoader(TensorDataset(xs4, ys), batch_size=eb.BS, shuffle=True)
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
        anchors.append(make_anchor_cnn(model, xs4.to(eb.DEV), phase, ntask))
        dom_all.append(anchors[-1]["dom"])
        # eval: eval_model permutes flat images; ConvStack accepts flat (B,784)
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} T{phase}: "
              + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase + 1)),
              flush=True)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "gran": "neuron", "bias": own_bias,
            "ntask": ntask, "lam": lam,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "dom_per_phase": dom_all}


def make_anchor_cnn(model, xs, task_id, ntask):
    """Neuron-granularity anchor: Fisher (via task CE) x usage, with
    dom instrumentation. Mirrors branch_delp.make_anchor(neuron) but with
    CNN flat_params."""
    model.eval()
    P = flat_params(model)
    theta = _clone(P)
    # Fisher via explicit CE on the batch (xf gradient squared mean)
    xs_grad = xs[:512].detach().requires_grad_(False)
    ctx1 = F.one_hot(torch.full((xs_grad.size(0),), task_id, device=xs.device),
                     ntask).float()
    out, _ = model(xs_grad, ctx1)
    pseudo = out.argmax(1)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    model.zero_grad()
    F.cross_entropy(out, pseudo).backward()
    for k in P:
        if P[k].grad is not None:
            fisher[k] = P[k].grad.detach() ** 2
    # usage: raw dend gate (bias-free instrument)
    with torch.no_grad():
        ctxr = F.one_hot(torch.tensor([task_id], device=xs.device), ntask).float()
        u = model.dend(ctxr).view(model.nseg, model.nh)
        winners = u.argmax(dim=0)
        o = (winners == task_id).float()
        active = u.max(dim=0).values.abs() > 0
        dom_hist = torch.bincount(winners, minlength=ntask).tolist()
        dom_hist_active = torch.bincount(winners[active], minlength=ntask).tolist()
        dead_frac = float((~active).float().mean())
    fisher["ff.c4.weight"] = fisher["ff.c4.weight"] * o.view(-1, 1, 1, 1)
    rec = {"gran": "neuron", "theta": theta, "fisher": fisher,
           "dom": {"dom_hist": dom_hist, "dom_hist_active": dom_hist_active,
                    "dead_frac": round(dead_frac, 6)}}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cnnc2_mnist_b0.json")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--ntask", type=int, default=10)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()

    seed_list = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        eb.EPOCHS = 1
        eb.NTRAIN = 2000
        seed_list = seed_list[:2]

    data = load_data(a.dataset)
    results, t0 = [], time.time()
    for i, seed in enumerate(seed_list):
        if i % a.n_shards != a.shard:
            continue
        rec = {"kind": "cnn", "dataset": a.dataset, "seed": seed}
        try:
            perms = get_perms(seed, a.ntask)
            rec.update(run_cnn(seed, data, perms, a.ntask, LAM, a.bias))
        except Exception as exc:                            # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL s{seed}: {exc}", flush=True)
        results.append(rec)
        print(f"  DONE s{seed} b={a.bias}: ell={rec.get('ell')} "
              f"avg={rec.get('final_avg')} bwt={rec.get('bwt')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(a.out, "w") as f:
        json.dump({"results": results, "elapsed": time.time() - t0},
                  f, indent=1)
    print(f"\nDONE: {len(results)} runs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
