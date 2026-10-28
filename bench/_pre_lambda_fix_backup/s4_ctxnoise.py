"""S4: eval-time context noise robustness (T10, MNIST + Fashion).

Preregistration: S4_ctxnoise_预注册.md (locked 2026-10-25, before any data).

E1  gating robustness: noise rho in {0,.1,.2,.4} at eval; DeLP chain intact
    (anchors + protect_grads), b=0.
E2  repair survival: same rho grid x b in {0,10}; paired (rho,ds,seed).
E3  soft-mix mitigation (exploratory): ctx = (1-rho)*onehot + rho*uniform.

Noise model (worst-case): with prob rho per SAMPLE, the eval ctx one-hot is
replaced by a uniformly random OTHER task. Training always oracle ctx.

Arms: (rho, b, ds) grid, seeds 0-9, lam=1000, T=10, neuron granularity.
Mirrors branch_delp.run_gran exactly (anchors accumulated per phase, penalty
summed over the stack, DataLoader shuffle, protect_grads) -- the ONLY deltas
are (i) noisy final-phase eval and (ii) BiasADNet for b=10 arms.
"""

import argparse, json, os, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),   # server layout
              os.path.join(_here, "..", "..", "dendritic-lab")):  # local
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, _here)
import exp4_branch_protection as e4
import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute
from branch_delp import (ADNet, BiasADNet, make_anchor, anchor_penalty,
                         flat_params, _clone)
from anchor_budget import load_data

LAM = 1000.0
NTASK = 10
SEEDS = list(range(10))
RHOS = [0.0, 0.1, 0.2, 0.4]
BIASES = [0.0, 10.0]


def eval_noisy_final(model, xte, yte, perms, ntask, rho, soft, seed):
    """Final-phase eval with per-sample ctx noise; only (T-1, c) entries."""
    model.eval()
    acc = {}
    gen = torch.Generator().manual_seed(90000 + seed)
    with torch.no_grad():
        for cc in range(ntask):
            xt = permute(xte, perms[cc]).to(DEV)
            correct = 0
            for i in range(0, xt.size(0), 2000):
                n = min(2000, xt.size(0) - i)
                xb = xt[i:i+2000]
                ctx = F.one_hot(torch.full((n,), cc, device=DEV), ntask).float()
                if rho > 0 and not soft:
                    flip = (torch.rand(n, generator=gen) < rho).to(DEV)
                    other = (torch.randint(0, ntask - 1, (n,),
                                           generator=gen) + cc + 1) % ntask
                    ctx_new = F.one_hot(other.to(DEV), ntask).float()
                    ctx = torch.where(flip.unsqueeze(1), ctx_new, ctx)
                elif rho > 0 and soft:
                    ctx = (1 - rho) * ctx + rho / ntask
                out, _ = model(xb, ctx)
                correct += (out.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
            acc[(ntask - 1, cc)] = correct / xt.size(0) * 100
    return acc


def run_noise(ds, seed, own_bias, rhos, lam=LAM, soft=False):
    torch.manual_seed(seed)
    data = load_data(ds)
    xtr, ytr, xte, yte = data
    perms = get_perms(seed, NTASK)

    if own_bias:
        model = BiasADNet(NTASK, NTASK, own_bias=own_bias,
                          bias_mode="fixed").to(DEV)
    else:
        model = ADNet(protected=True, nseg=NTASK, ntask=NTASK).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
                    for a in anchors:
                        loss = loss + anchor_penalty(a, P, a["gran"])
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, NTASK, "neuron"))
        a = eval_model(model, xte, yte, perms, phase, NTASK, True)
        acc.update(a)

    favgs = {}
    accs_by_rho = {}
    for rho in rhos:
        if rho == 0.0:
            final_acc = {k: v for k, v in acc.items() if k[0] == NTASK - 1}
        else:
            final_acc = eval_noisy_final(model, xte, yte, perms, NTASK,
                                         rho, soft, seed)
        favg = sum(final_acc[(NTASK - 1, c)] for c in range(NTASK)) / NTASK
        favgs[rho] = favg
        accs_by_rho[f"{rho}"] = {f"{c}": round(final_acc[(NTASK - 1, c)], 2)
                                 for c in range(NTASK)}

    return {"kind": "ctxnoise", "seed": seed, "dataset": ds, "ntask": NTASK,
            "lam": lam, "bias": own_bias, "soft": soft,
            "final_avg_by_rho": {str(k): round(v, 2) for k, v in favgs.items()},
            "acc_final_by_rho": accs_by_rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.soft:
        jobs = [(ds, 0.0, s) for ds in ("mnist", "fashion") for s in SEEDS]
    else:
        jobs = [(ds, b, s) for ds in ("mnist", "fashion")
                for b in BIASES for s in SEEDS]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    results = []
    t0 = time.time()
    for ds, b, s in jobs:
        r = run_noise(ds, s, b, RHOS, soft=args.soft)
        results.append(r)
        print(f"DONE: ds={ds} b={b} s={s} "
              f"favg(rho)={r['final_avg_by_rho']} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    json.dump({"results": results}, open(args.out, "w"), indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
