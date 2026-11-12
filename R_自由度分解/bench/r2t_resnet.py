# -*- coding: utf-8 -*-
"""r2t_resnet.py -- R2T: transfer of R2's anchor-budget-direction finding to
ResNet-18+GN / Split-CIFAR-100 (E9 protocol, prereg results_r2t/R2T_预注册.md).

Reuses e9_resnet.py VERBATIM (same class split via import) and adds ONLY the
recency-weighted anchor penalty from r2_recency.recency_weights:

    pen = sum_j w_j * ||theta - a_j||_F^2 ,   w from recency_weights(v, alpha)

Everything else -- backbone, gate mount, SGD protocol, Fisher anchors,
masked softmax, oracle-ctx eval, seeds, class order -- is byte-identical to
E9 (imported, not copied).  base/renorm are NOT rerun: E9's grid is the
shared control arm (prereg §3).

Arms (prereg §3):
    base_rec10   forward=base   alpha=1.0 reverse=False
    renorm_rec10 forward=renorm alpha=1.0 reverse=False
    base_anti10  forward=base   alpha=1.0 reverse=True

Usage:
  CUDA_VISIBLE_DEVICES=2 python r2t_resnet.py --task-id 0 --n-shards 2 \
      --arms base_rec10,renorm_rec10,base_anti10 --seeds 0,...,9 \
      --out results_r2t/r2t_grid.part0.json
"""
import argparse, json, math, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import e9_resnet as E9          # VERBATIM reuse (protocol, model, anchors)
from r2_recency import recency_weights   # VERBATIM reuse (weights only)

LAM = E9.LAM                    # 0.1, ResNet-calibrated (S2 round-6)
TAU_DIV = 4.0                   # tau = ntask / 4 (prereg §2)

# arm -> (forward, alpha, reverse)   [prereg §3]
ARM_SPEC = {
    "base_rec10":  dict(forward="base",   alpha=1.0, reverse=False),
    "renorm_rec10": dict(forward="renorm", alpha=1.0, reverse=False),
    "base_anti10":  dict(forward="base",   alpha=1.0, reverse=True),
}
DEFAULT_ARMS = ["base_rec10", "renorm_rec10", "base_anti10"]


def run_one(seed, ntask, arm, d_tr, aux, lam=LAM, epochs=E9.EPOCHS):
    """One run: E9's loop with the ONLY change being per-anchor weights w_j.

    Runtime assertions (prereg G4): lam is dispatched exactly as written into
    the loss; constants are the E9 values; weights are logged every phase.
    """
    spec = ARM_SPEC[arm]
    alpha, reverse = spec["alpha"], spec["reverse"]
    ytr, yte, xte, xfx = aux
    g = torch.Generator().manual_seed(100 + seed)          # E9 class order
    order = torch.randperm(100, generator=g).tolist()
    cpt = 100 // ntask
    task_cls = [order[t * cpt:(t + 1) * cpt] for t in range(ntask)]
    torch.manual_seed(seed)                                 # E9 init seed
    model = E9.GatedResNet(ntask, arm=spec["forward"]).to(E9.DEV)
    opt = torch.optim.SGD(model.parameters(), lr=E9.LR, momentum=E9.MOM,
                          weight_decay=E9.WD)
    anchors, acc = [], {}
    phase_weights, weight_sums = [], []
    t0 = time.time()
    for phase in range(ntask):
        cls = task_cls[phase]
        lut = torch.tensor(cls, device=E9.DEV)
        idx = [i for i, y in enumerate(ytr.tolist()) if y in cls]
        dl = DataLoader(Subset(d_tr, idx), batch_size=E9.BS, shuffle=True,
                        num_workers=2, pin_memory=True)
        w = recency_weights(len(anchors), alpha, ntask, reverse)  # R2 verbatim
        assert abs(sum(w) - len(anchors)) < 1e-6 and all(x > 0 for x in w), \
            f"weight invariant violated at phase {phase}"          # prereg G1
        phase_weights.append([round(x, 6) for x in w])
        weight_sums.append(round(sum(w), 6))
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(E9.DEV), yb.to(E9.DEV)
                local = torch.zeros_like(yb)
                for ci, c in enumerate(cls):
                    local[yb == c] = ci
                ctx = F.one_hot(torch.full((xb.size(0),), phase,
                                           device=E9.DEV), ntask).float()
                opt.zero_grad()
                out = model(xb, ctx)[:, lut]
                loss = F.cross_entropy(out, local)
                if anchors and lam > 0:
                    P = dict(model.named_parameters())
                    pen = 0.0
                    for j, a in enumerate(anchors):
                        for k in a["fisher"]:          # anchored keys only
                            v = P[k]
                            pen = pen + w[j] * (
                                a["fisher"][k] * (v - a["theta"][k]) ** 2
                            ).sum()
                    loss = loss + lam * pen            # lam IS args.lam (G4)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), E9.CLIP_NORM)
                opt.step()
        if spec["forward"] != "plain":
            xs_probe = torch.stack([d_tr[i][0] for i in idx[:250]]).to(E9.DEV)
            model.gate_stats.append(E9.probe(model, xs_probe, phase))
        if lam > 0:
            fx_idx = idx[:E9.NFISHER * 2]
            fisher = E9.fisher_anchor(model, xfx, fx_idx)
            theta = {k: v.detach().clone() for k, v in
                     dict(model.named_parameters()).items() if k in fisher}
            anchors.append({"theta": theta, "fisher": fisher})
        acc.update(E9.eval_tasks(model, task_cls, yte, xte, phase))
        print(f"  s{seed} {arm} T{phase}: t{phase}="
              f"{acc[(phase, phase)]:.0f} ({time.time()-t0:.0f}s)", flush=True)
    bwt = sum(acc[(ntask - 1, c)] - acc[(c, c)]
              for c in range(ntask - 1)) / (ntask - 1)
    favg = sum(acc[(ntask - 1, c)] for c in range(ntask)) / ntask
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"kind": "r2t", "ntask": ntask, "seed": seed, "arm": arm,
            "alpha": alpha, "reverse": reverse,
            "lam": lam, "tau": ntask / TAU_DIV,
            "dataset": "cifar100", "backbone": "resnet18-gn",
            "protocol": "task-IL", "g": E9.G_CONST,
            "epochs": epochs, "bs": E9.BS, "k_keep": E9.K_KEEP, "nh": E9.NH,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "gate_stats": model.gate_stats,
            "phase_weights": phase_weights, "weight_sums": weight_sums}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default="r2t_result.json")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ntasks", type=int, default=10)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--epochs", type=int, default=E9.EPOCHS)
    args = ap.parse_args()

    d_tr, aux = E9.load_cifar100()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    jobs = [(a, s) for a in arms for s in seeds]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.task_id]
    print(f"R2T shard {args.task_id}/{args.n_shards}: {len(jobs)} runs "
          f"(arms={arms}, lam={args.lam}, tau={args.ntasks/TAU_DIV}, "
          f"DEV={E9.DEV})", flush=True)

    results = []
    for i, (arm, seed) in enumerate(jobs):
        rec = run_one(seed, args.ntasks, arm, d_tr, aux,
                      lam=args.lam, epochs=args.epochs)
        results.append(rec)
        # incremental checkpoint (prereg G5): shared cards may vanish
        json.dump({"prereg": "bench/results_r2t/R2T_预注册.md",
                   "shard": args.task_id, "n_shards": args.n_shards,
                   "seeds": seeds, "arms": arms, "lam": args.lam,
                   "results": results},
                  open(args.out, "w"), ensure_ascii=False)
    print(f"WROTE {args.out} {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
