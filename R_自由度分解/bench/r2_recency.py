# -*- coding: utf-8 -*-
"""r2_recency.py -- R2: can the gain--retention frontier be moved by
changing WHAT is protected (rather than WHEN the intervention acts)?

Preregistration: bench/results_r2/R2_预注册.md (landed 2026-11-11 19:47:24,
before any R2 data).  The intervention is a *total-weight-conserving*
reallocation of the anchor-penalty budget towards recent tasks:

    tau   = ntask / 4
    raw_j = exp(-g_j / tau),   g_j = v-1-j  (gap of anchor j at phase v)
    w_j   = (1-alpha)*1 + alpha*(v*raw_j/sum_k raw_k)      sum_j w_j == v
    pen   = sum_j w_j * anchor_penalty(a_j, P, "neuron")
    loss  = CE + lam_eff * pen

alpha=0 is bit-equivalent to E5/E6/E7's uniform penalty.  Nothing else
changes: same ADNet/RenormADNet, same make_anchor plumbing (including the
patched fisher_raw for gate-free forwards), same protect_grads, same oracle
evaluation, same seeds/permutations/optimiser.

Usage (server GPU venv, shared cards 2/3 only):
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
    python r2_recency.py --shard 0 --n-shards 2 --out results_r2/r2_grid.part0.json

Pilot / implementation-gate probes:
  python r2_recency.py --shard 0 --n-shards 1 --seeds 0 \
      --ntasks 10 --arms base,renorm,renorm_rec00 --out /tmp/r2_pilot.json
"""
import argparse, json, math, os, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import ADNet, permute, kwta
import branch_delp as _bd
from anchor_budget import load_data

# Importing e5_arms applies the module-level patches that gate-free arms
# need (_patched_fisher_raw guards params with grad is None, plus the
# flat_params / usage_stats / make_anchor dispatch).  e6e7_arms.py uses the
# identical plumbing, so R2 stays bit-comparable with E5/E6/E7.
import e5_arms as _e5
from e5_arms import RenormADNet
_bd_flat_params_dispatch = _e5._bd_flat_params_dispatch
_bd_make_anchor_dispatch = _e5._bd_make_anchor_dispatch

LAM = 1000.0
LAM_UP = 1000.0 * 3.13          # 3130: protection STRENGTH control
GATE_CONST = 0.565
NH = 256

# arm -> (forward, alpha, lam_eff, reverse)
ARM_SPEC = {
    "base":          dict(forward="base",   alpha=0.0, lam=LAM,    reverse=False),
    "renorm":        dict(forward="renorm", alpha=0.0, lam=LAM,    reverse=False),
    # pilot-only: alpha=0 must be bit-identical to `renorm` (prereg G2)
    "renorm_rec00":  dict(forward="renorm", alpha=0.0, lam=LAM,    reverse=False),
    "renorm_rec05":  dict(forward="renorm", alpha=0.5, lam=LAM,    reverse=False),
    "renorm_rec10":  dict(forward="renorm", alpha=1.0, lam=LAM,    reverse=False),
    "base_rec10":    dict(forward="base",   alpha=1.0, lam=LAM,    reverse=False),
    "renorm_lamup":  dict(forward="renorm", alpha=0.0, lam=LAM_UP, reverse=False),
    # conditional-trigger arm only (prereg §3.1); NOT in the default grid
    "renorm_anti10": dict(forward="renorm", alpha=1.0, lam=LAM,    reverse=True),
}
DEFAULT_ARMS = ["base", "renorm", "renorm_rec05", "renorm_rec10",
                "base_rec10", "renorm_lamup"]


def recency_weights(v, alpha, ntask, reverse=False):
    """Weights for the v anchors a_0..a_{v-1} at phase v.

    Invariant (prereg G3): sum(w) == v for any alpha, and w_j > 0.
    alpha=0 returns exactly [1.0]*v -> bit-equivalent to E5/E6/E7.
    """
    if v <= 0:
        return []
    tau = ntask / 4.0
    raw = []
    for j in range(v):
        g = v - 1 - j
        e = g / tau if reverse else -g / tau
        raw.append(math.exp(e))
    s = sum(raw)
    rec = [v * x / s for x in raw]
    return [(1.0 - alpha) * 1.0 + alpha * r for r in rec]


def build_model(forward, ntask, seed, nh=NH):
    torch.manual_seed(seed)
    if forward == "renorm":
        return RenormADNet(nh=nh, protected=True,
                           nseg=ntask, ntask=ntask).to(DEV)
    return ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)


def run_one(ds, seed, ntask, arm):
    spec = ARM_SPEC[arm]
    alpha, lam_eff = spec["alpha"], spec["lam"]
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, ntask)
    model = build_model(spec["forward"], ntask, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    phase_weights, weight_sums = [], []

    for phase in range(ntask):
        xs = permute(xtr[:NTRAIN], perms[phase])
        ys = ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        w = recency_weights(len(anchors), alpha, ntask, spec["reverse"])
        phase_weights.append([round(x, 6) for x in w])
        weight_sums.append(round(sum(w), 6))
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = _bd_flat_params_dispatch(model)
                    pen = 0.0
                    for j, a in enumerate(anchors):
                        pen = pen + w[j] * _bd.anchor_penalty(a, P, a["gran"])
                    loss = loss + lam_eff * pen      # MUST carry lam_eff (G1)
                loss.backward()
                if hasattr(model, "protect_grads"):
                    model.protect_grads(phase)
                opt.step()
        anchors.append(_bd_make_anchor_dispatch(model, xs, phase, ntask,
                                                "neuron"))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)

    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {
        "arm": arm, "forward": spec["forward"], "alpha": alpha,
        "tau": ntask / 4.0, "lam_eff": lam_eff, "reverse": spec["reverse"],
        "ntask": ntask, "ds": ds, "seed": seed, "nh": NH,
        "gate_const": GATE_CONST,
        "ell": round(ell, 4), "avg": round(favg, 4), "bwt": round(bwt, 4),
        "acc_matrix": [[round(acc[(p, c)], 4) for c in range(p + 1)]
                       for p in range(ntask)],
        "phase_weights": phase_weights, "weight_sums": weight_sums,
    }


def build_jobs(arms, datasets, ntasks, seeds):
    return [(arm, ds, T, s) for arm in arms for ds in datasets
            for T in ntasks for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ntasks", default="10,20")
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.08,
                    help="cap at this fraction of the visible card; GPU 2/3 "
                         "are shared with llama-server")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=0)

    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARM_SPEC]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    datasets = args.datasets.split(",")
    ntasks = [int(x) for x in args.ntasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    jobs = build_jobs(arms, datasets, ntasks, seeds)
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
    print(f"R2 shard {args.shard}/{args.n_shards}: {len(jobs)}/{len(build_jobs(arms, datasets, ntasks, seeds))} runs "
          f"(arms={arms}, T={ntasks}, seeds={seeds}, DEV={DEV}, "
          f"LAM={LAM}, LAM_UP={LAM_UP}, NH={NH})", flush=True)

    results, t0 = [], time.time()
    meta = {"prereg": "bench/results_r2/R2_预注册.md",
            "shard": args.shard, "n_shards": args.n_shards,
            "arms": arms, "datasets": datasets, "ntasks": ntasks,
            "seeds": seeds, "lam": LAM, "lam_up": LAM_UP, "nh": NH,
            "gate_const": GATE_CONST}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i, (arm, ds, T, s) in enumerate(jobs):
        r = run_one(ds, s, T, arm)
        results.append(r)
        print(f"  [{i+1:3d}/{len(jobs)} {arm:<14s} {ds:<7s} T{T:<2d} "
              f"s{s:<2d}] ell={r['ell']:6.2f} avg={r['avg']:6.2f} "
              f"bwt={r['bwt']:7.2f}", flush=True)
        # Incremental checkpoint: shared-card runs must survive a reboot.
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({**meta, "elapsed": round(time.time() - t0, 1),
                       "results": results}, f)
    print(f"\nwrote {args.out} ({len(results)} runs, "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
