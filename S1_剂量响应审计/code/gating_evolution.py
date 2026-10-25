"""gating_evolution.py -- S1 C-1: when and how does segment gating anti-
specialize?  Trains the branch-arm DeLP (unbounded, lam configurable) and,
at EVERY epoch boundary of EVERY phase, measures gating with the artifact-
controlled instrumentation:

  p_own_active  P(dominant segment == current task) among ACTIVE (o>0) neurons
  p_own_all     same over all neurons (legacy, dead-neuron-inflated bin 0)
  dead_frac     fraction of neurons never active in NUSAGE samples
  p_seg0_active P(dominant == segment 0) among active neurons

Also per phase: dend column-t drift -- own-block (trained by protect_grads)
vs other-blocks (should be exactly frozen) -- the rich-get-richer check.

Usage (server):  CUDA_VISIBLE_DEVICES=0 python gating_evolution.py --out gating_evo_mnist_t10.json --seeds 0,1,2 --ntasks 10
"""
import argparse, json, os, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
from exp6_baselines import (EPOCHS, BS, NTRAIN, NH, LR, DEV,
                            get_perms, flat_params, eval_model, finish)
from exp4_branch_protection import ADNet, permute, kwta
from anchor_budget import load_data
from branch_delp import (usage_stats, make_anchor, attribute_drift,
                         _clone, dom_instr, anchor_penalty)


def measure(model, xs, task_id, ntask):
    o, o_seg = usage_stats(model, xs, task_id, ntask)
    d = dom_instr(o, o_seg, ntask)
    hist_a = d["dom_hist_active"]
    tot_a = sum(hist_a)
    d["p_own_active"] = round(hist_a[task_id] / tot_a, 4) if tot_a else None
    d["p_seg0_active"] = round(hist_a[0] / tot_a, 4) if tot_a else None
    tot = sum(d["dom_hist"])
    d["p_own_all"] = round(d["dom_hist"][task_id] / tot, 4) if tot else None
    return d


def run_evolve(seed, data, perms, ntask, lam):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    phases = []
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        W_start = _clone(flat_params(model))
        own0 = model.dend.weight[phase * model.nh:(phase + 1) * model.nh,
                                 phase].detach().clone()
        col0 = model.dend.weight[:, phase].detach().clone()
        win_cnt = torch.zeros(ntask, model.nh, device=DEV)
        n_seen = 0
        measures = [measure(model, xs, phase, ntask)]        # phase start
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, seg = model(xb, ctx)
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
                for k in range(ntask):
                    win_cnt[k] += (seg == k).float().sum(0)
                n_seen += xb.size(0)
            measures.append(measure(model, xs, phase, ntask))  # per epoch
        P_now = flat_params(model)
        dW = {k: P_now[k].detach() - W_start[k] for k in P_now}
        attribute_drift(anchors, win_cnt / max(n_seen, 1), dW, "branch")
        anchors.append(make_anchor(model, xs, phase, ntask, "branch"))
        own1 = model.dend.weight[phase * model.nh:(phase + 1) * model.nh,
                                 phase].detach().clone()
        col1 = model.dend.weight[:, phase].detach().clone()
        oth = torch.ones(ntask * model.nh, dtype=torch.bool, device=DEV)
        oth[phase * model.nh:(phase + 1) * model.nh] = False
        measures.append(measure(model, xs, phase, ntask))      # phase end
        phases.append({
            "phase": phase,
            "own_block_drift": round(float((own1 - own0).norm()), 5),
            "own_block_rel": round(float((own1 - own0).norm()
                                         / own0.norm().clamp(min=1e-12)), 5),
            "other_blocks_drift": round(float((col1 - col0)[oth].norm()), 8),
            "measures": measures})
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        m = phases[-1]["measures"][-1]
        print(f"  s{seed} T{phase}: p_own_act={m['p_own_active']} "
              f"dead={m['dead_frac']} p_seg0_act={m['p_seg0_active']} "
              f"own_drift={phases[-1]['own_block_rel']}", flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "phases": phases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gating_evolution.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ntasks", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        import exp6_baselines as eb
        import exp4_branch_protection as e4
        import anchor_budget as ab
        import branch_delp as bd
        for mod in (eb, e4, ab, bd):
            mod.EPOCHS, mod.NTRAIN = 1, 2000
        eb.NFISHER, eb.NUSAGE = 20, 50
        a.seeds, a.ntasks = "0", 4
    seed_list = [int(s) for s in a.seeds.split(",")]
    data = load_data(a.dataset)
    res, t0 = [], time.time()
    for seed in seed_list:
        r = run_evolve(seed, data, get_perms(seed, a.ntasks), a.ntasks, a.lam)
        r["dataset"], r["ntask"], r["lam"] = a.dataset, a.ntasks, a.lam
        res.append(r)
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(a.out, "w"), indent=1)
    print(f"WROTE {a.out} in {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    main()
