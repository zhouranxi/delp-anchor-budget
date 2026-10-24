"""S1 Q8-1: signed drift of the own gating block under the causal probe.

Trains the neuron-arm DeLP on Permuted-MNIST T=10 (lam=1000) under
b=0 and b=10 (fixed bias), 3 seeds. Per phase, snapshots the context-t
column of dend.weight before/after training and records:
  d_col       signed mean displacement of the whole column t
  d_own       signed mean displacement of the own-segment block of col t
  col_norm    norm of column t at phase end
  u_own_raw   mean own-segment gate over the task's usage batch, NO bias
  u_own_eff   same WITH bias applied (b-arm only; = raw + b)
  dead_frac, p_own_active (context for continuity)

Adversarial compensation shows up as: d_own(b=10) systematically more
negative than d_own(b=0), and/or (u_own_raw(b=10) - u_own_raw(b=0)) << -b
offset, i.e. effective offset u_own_eff - u_own_raw(b=0) << b.

Usage: python signed_drift.py --out signeddrift.json [--smoke]
"""
import argparse, json, os, sys, time

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
for _cand in (os.path.join(_here, "..", "dendritic-lab"),   # server layout
              os.path.join(_here, "..", "..", "dendritic-lab")):  # local
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

from exp6_baselines import (EPOCHS, BS, NTRAIN, NH, LR, DEV,
                            get_perms, eval_model, finish)
from exp4_branch_protection import ADNet, permute, kwta
from branch_delp import BiasADNet, anchor_penalty, load_data

NFISHER, NUSAGE = 300, 500
BIASES = [0.0, 10.0]
SEEDS = [0, 1, 2]
NTASK = 10
LAM = 1000.0


def usage_gate(model, xs, task_id, ntask):
    """Mean own-segment gate u[:, task, task] over a usage batch (raw, no bias
    applied even in b-mode: temporarily zeroed)."""
    was = model.own_bias
    model.own_bias = 0.0
    with torch.no_grad():
        for i in range(0, len(xs), 250):
            xb = xs[i:i + 250].to(DEV)
            ctx = F.one_hot(torch.full((xb.size(0),), task_id),
                            ntask).float().to(DEV)
            u = model.dend(ctx).view(-1, model.nseg, model.nh)
            g = u[:, task_id, task_id].mean().item()
    model.own_bias = was
    return g


def col_instr(model, task_id, snap):
    col = model.dend.weight[:, task_id]
    own = col[task_id * NH:(task_id + 1) * NH]
    d = col - snap
    return {"d_col": (d.mean()).item(),
            "d_own": (d[task_id * NH:(task_id + 1) * NH].mean()).item(),
            "col_norm": col.norm().item()}


def run(seed, data, perms, bias):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = BiasADNet(NTASK, NTASK, own_bias=bias, bias_mode="fixed").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    phases = []
    for t in range(NTASK):
        xtr_t = permute(xtr, perms[t])[:NTRAIN]
        ytr_t = torch.full((len(xtr_t),), t, dtype=torch.long)
        xs = xtr_t[:NUSAGE]
        dl = DataLoader(TensorDataset(xtr_t, ytr_t), batch_size=BS,
                        shuffle=True)
        snap = model.dend.weight[:, t].detach().clone()
        u0 = usage_gate(model, xs, t, NTASK)
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb, F.one_hot(yb, NTASK).float())
                loss = F.cross_entropy(out, yb)
                if anchors:
                    from exp6_baselines import flat_params
                    Pf = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, Pf, a["gran"])
                    loss = loss + LAM * pen
                loss.backward()
                model.protect_grads(t)
                opt.step()
        anchors.append(make_anchor(model, xs, t))
        instr = col_instr(model, t, snap)
        instr["phase"] = t
        instr["u_own_raw_start"] = round(u0, 4)
        instr["u_own_raw_end"] = round(usage_gate(model, xs, t, NTASK), 4)
        instr["u_own_eff_end"] = round(instr["u_own_raw_end"] + bias, 4)
        # continuity measures (reuse corrected instrumentation)
        from branch_delp import usage_stats
        o, o_seg = usage_stats(model, xs, t, NTASK)
        act = o > 0
        dom = o_seg.argmax(0)
        instr["p_own_active"] = round(
            (dom[act] == t).float().mean().item(), 4) if act.any() else None
        instr["dead_frac"] = round((~act).float().mean().item(), 4)
        phases.append(instr)
        # eval (same protocol as branch_delp: flat (phase, class) keys)
        acc.update(eval_model(model, xte, yte, perms, t, NTASK, True))
        print(f"  b={bias} s{seed} T{t}: "
              + " ".join(f"t{c}={acc[(t,c)]:.0f}" for c in range(t + 1)),
              flush=True)
    bwt, favg = finish(acc, NTASK)
    ell = sum(acc[(t, t)] for t in range(NTASK)) / NTASK
    return {"seed": seed, "bias": bias, "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "phases": phases}


def make_anchor(model, xs, task_id):
    """Neuron-arm anchor identical to branch_delp.make_anchor gran='neuron'."""
    from exp6_baselines import flat_params
    from branch_delp import usage_stats
    P = flat_params(model)
    fisher = None
    from branch_delp import fisher_raw
    fisher = fisher_raw(model, xs, task_id, NTASK)
    o, o_seg = usage_stats(model, xs, task_id, NTASK)
    fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
    fisher["ff.b"] = fisher["ff.b"] * o
    fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
    return {"gran": "neuron", "theta": {k: v.clone() for k, v in P.items()},
            "fisher": fisher}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        import exp6_baselines as eb
        import exp4_branch_protection as e4
        import branch_delp as bd
        for mod in (eb, e4, bd):
            mod.EPOCHS = 1
            mod.NTRAIN = 2000
        global EPOCHS, NTRAIN, NFISHER
        EPOCHS, NTRAIN, NFISHER = 1, 2000, 20
    data = load_data("mnist")
    res, t0 = [], time.time()
    for bias in BIASES:
        for seed in SEEDS:
            print(f"bias={bias} seed={seed} ...", flush=True)
            r = run(seed, data, get_perms(seed, NTASK), bias)
            res.append(r)
            print(f"  bwt={r['bwt']} avg={r['final_avg']} "
                  f"ell={r['ell']}", flush=True)
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(a.out, "w"), indent=1)
    print(f"WROTE {a.out} ({round(time.time() - t0)}s)", flush=True)


if __name__ == "__main__":
    main()
