"""ab_ewc_arms.py -- S2 reviewer-response arms: plain EWC and online EWC.

Same architecture (ADNet with frozen dendritic segments), same data,
permutations, optimiser, epochs and lambda as the DeLP arms of
anchor_budget.py -- only the protection state differs:

  plain   EWC (Kirkpatrick 2017): the anchor is the PREVIOUS task's
          posterior (theta_t, F_t); it is replaced after every task.
  online  EWC (Chaudhry et al. 2018, gamma=1): a single running posterior;
          the Fisher accumulates over tasks and theta_ref tracks the latest
          snapshot. No age structure survives this collapse -- it is the
          reference the related-work section contrasts bounded stacks with.

Usage (server):
  CUDA_VISIBLE_DEVICES=0 python ab_ewc_arms.py --task-id 0 --n-tasks 2 \
      --modes plain,online --ntasks 10,20 --lams 1000,5000,20000 \
      --seeds 0,1,2,3,4,5,6,7,8,9 --out r7_ewc.json
Usage (smoke):  python ab_ewc_arms.py --smoke
"""
import argparse, json, time
import sys, os
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "dendritic-lab"))
from exp6_baselines import (EPOCHS, BS, NTRAIN, LR, DEV, get_perms,
                            flat_params, eval_model, finish, fisher_usage)
from exp4_branch_protection import ADNet, permute
from anchor_budget import load_data


def run_ewc(seed, data, perms, ntask, lam, mode, gamma=1.0):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchor, acc = None, {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchor is not None:
                    P = flat_params(model)
                    pen = 0.0
                    for k, v in P.items():
                        pen = pen + (anchor["fisher"][k]
                                     * (v - anchor["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        new_anchor = fisher_usage(model, xs, phase, ntask)
        if mode == "plain" or anchor is None:
            anchor = new_anchor
        else:   # online (Chaudhry et al.): Fisher runs with discount gamma,
                # theta_ref tracks the latest snapshot
            anchor = {"theta": new_anchor["theta"],
                      "fisher": {k: gamma * anchor["fisher"][k]
                                 + new_anchor["fisher"][k]
                                 for k in new_anchor["fisher"]}}
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} {mode} T{phase}: "
              + " ".join(f"t{c}={acc[(phase, c)]:.0f}" for c in range(phase + 1)),
              flush=True)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="r7_ewc.json")
    ap.add_argument("--modes", default="plain,online")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lams", default="5000")
    ap.add_argument("--ntasks", default="10,20")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    a = ap.parse_args()

    modes = a.modes.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    lams = [float(s) for s in a.lams.split(",")]
    nts = [int(s) for s in a.ntasks.split(",")]

    if a.smoke:
        import exp6_baselines as eb
        import exp4_branch_protection as e4
        eb.EPOCHS = e4.EPOCHS = 1
        eb.NTRAIN = e4.NTRAIN = 2000
        eb.NFISHER = 20
        eb.NUSAGE = 50
        global EPOCHS, NTRAIN
        EPOCHS, NTRAIN = 1, 2000
        jobs = [(m, 4, 0, 5000.0) for m in modes]
    else:
        jobs = [(m, nt, s, lam) for m in modes for nt in nts
                for s in seeds for lam in lams]

    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs",
          flush=True)
    data = load_data(a.dataset)
    res, t0 = [], time.time()
    for mode, nt, seed, lam in mine:
        rec = {"kind": "ewc", "mode": mode, "ntask": nt, "seed": seed,
               "lam": lam, "gamma": a.gamma, "dataset": a.dataset}
        try:
            rec.update(run_ewc(seed, data, get_perms(seed, nt), nt, lam,
                               mode, a.gamma))
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        res.append(rec)
        print(f"  {mode} lam{lam:.0f} T{nt} s{seed}: bwt={rec.get('bwt')} "
              f"avg={rec.get('final_avg')} ell={rec.get('ell')}", flush=True)

    out = a.out if a.n_tasks == 1 else f"{a.out}.part{a.task_id}"
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print(f"WROTE {out} in {round(time.time()-t0,1)}s", flush=True)

    import statistics as st
    agg = {}
    for r in res:
        if "error" in r:
            continue
        agg.setdefault(f"{r['mode']}|lam{r['lam']:.0f}|T{r['ntask']}", []).append(r)
    print("\n==== aggregate (mean±sd over seeds) ====")
    for key, rs in sorted(agg.items()):
        b = [r["bwt"] for r in rs]
        f_ = [r["final_avg"] for r in rs]
        e = [r["ell"] for r in rs]
        print(f"{key:22s}  BWT {st.mean(b):7.2f}±{st.stdev(b) if len(b)>1 else 0:4.2f}"
              f"  acc {st.mean(f_):6.2f}±{st.stdev(f_) if len(f_)>1 else 0:4.2f}"
              f"  ell {st.mean(e):6.2f}")


if __name__ == "__main__":
    main()
