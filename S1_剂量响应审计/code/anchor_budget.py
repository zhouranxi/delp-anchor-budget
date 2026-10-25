"""anchor_budget.py -- S2 pilot: bounded anchor sets for DeLP.

Motivation (paper Sec 4.7): the 10-task deficit is concentrated in the LEARNING
term, and truncating the anchor stack to the M most recent anchors recovers part
of it (+6.5 ell at lambda=10000 for -1.6 BWT). exp11 only probed FIFO truncation.
This pilot turns the probe into a mechanism comparison.

Anchor policies (what happens when the stack exceeds budget M):
  unbounded  keep every anchor (original DeLP)            -- reference
  fifo       evict the oldest anchor                      -- exp11 probe
  random     evict a uniformly random anchor
  redun      evict the anchor with the smallest constraint mass at eviction
             time, i.e. sum_k fisher_k*(theta_now - theta_ref_k)^2; an anchor
             whose Fisher sits where theta already is constrains nothing and
             is the cheapest to drop
  merge      merge the two most similar anchors (cosine distance between
             flattened Fisher profiles): Fisher adds, theta becomes the
             elementwise Fisher-weighted barycenter (preserves the quadratic
             form to first order)

Protocol: identical to exp6_fixed (same data/permutations/seeds/optimiser/
architecture/lambda=5000). Arms: 5 policies x T in {10,20} x seeds {0,1,2};
plus corrected SI (lambda=100) at T=10 as reference. Metrics: BWT, final avg,
and ell = mean diagonal accuracy (the term S2 claims to recover).

Usage (server):  CUDA_VISIBLE_DEVICES=0 python anchor_budget.py --task-id 0 --n-tasks 3
Usage (smoke):   python anchor_budget.py --smoke
"""
import argparse, json, os, statistics as st, time
import sys
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "dendritic-lab"))  # local layout
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR, DEV,
                            get_perms, flat_params, eval_model, finish,
                            fisher_usage, run_si_impl, load_mnist)
from exp4_branch_protection import ADNet, permute

LAM_DELP = 5000.0
LAM_SI = 100.0
POLICIES = ["unbounded", "fifo", "random", "redun", "merge", "agew", "agew2"]


def load_data(name="mnist"):
    """MNIST or Fashion-MNIST, same 784-dim / 10-class interface."""
    from torchvision import datasets
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    tr = cls("./data", train=True, download=True)
    te = cls("./data", train=False, download=True)
    return (tr.data.float().view(-1, 784) / 255.0, tr.targets,
            te.data.float().view(-1, 784) / 255.0, te.targets)


# ---------------------------------------------------------------- eviction ops
def _evict_random(anchors, M, origin=None):
    import random as _r
    while len(anchors) > M:
        i = _r.randrange(len(anchors))
        anchors.pop(i)
        if origin is not None:
            origin.pop(i)


def _evict_fifo(anchors, M, origin=None):
    while len(anchors) > M:
        anchors.pop(0)
        if origin is not None:
            origin.pop(0)


def _redun_scores(anchors, theta_now):
    """Constraint mass of each anchor w.r.t. the CURRENT parameters."""
    out = []
    for a in anchors:
        s = 0.0
        for k in theta_now:
            s += (a["fisher"][k] * (theta_now[k] - a["theta"][k]) ** 2).sum().item()
        out.append(s)
    return out


def _evict_redun(anchors, M, theta_now, origin=None):
    while len(anchors) > M:
        scores = _redun_scores(anchors, theta_now)
        i = scores.index(min(scores))
        anchors.pop(i)
        if origin is not None:
            origin.pop(i)


def _merge_once(anchors):
    """Merge the most similar anchor pair (Fisher cosine); Fisher adds, theta
    becomes the elementwise Fisher-weighted barycenter."""
    flats = [torch.cat([a["fisher"][k].flatten() for k in sorted(a["fisher"])])
             for a in anchors]
    best = None
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            d = 1.0 - F.cosine_similarity(flats[i], flats[j], dim=0).item()
            if best is None or d < best[0]:
                best = (d, i, j)
    _, i, j = best
    a, b = anchors[i], anchors[j]
    merged = {"theta": {}, "fisher": {}}
    for k in a["fisher"]:
        f = a["fisher"][k] + b["fisher"][k]
        w = a["fisher"][k] / (f + 1e-12)
        merged["fisher"][k] = f
        merged["theta"][k] = w * a["theta"][k] + (1 - w) * b["theta"][k]
    anchors[i] = merged
    anchors.pop(j)
    return i, j


def _evict_agew(anchors, M, theta_now, origin=None):
    """Principled fifo: evict argmin(constraint_mass * age_rank). Older
    anchors get a smaller multiplier (cheap to drop); among similarly-aged
    anchors the weakest constrainer goes first. Softens blind fifo with the
    one bit of information redun used, without redun's failure mode of
    dropping young-but-weakly-constraining anchors that still protect."""
    while len(anchors) > M:
        scores = _redun_scores(anchors, theta_now)
        sc = [scores[i] * (i + 1) for i in range(len(anchors))]  # i=0 oldest
        i = sc.index(min(sc))
        anchors.pop(i)
        if origin is not None:
            origin.pop(i)


def _evict_agew2(anchors, M, theta_now, origin=None):
    """Age-first, mass-second: evict the OLDEST anchor among those with
    below-median eviction-time constraint mass; if every anchor is above
    median, fall back to pure fifo. Predicted to sit near fifo, testing
    whether the mass signal adds anything once age is the primary key."""
    while len(anchors) > M:
        scores = _redun_scores(anchors, theta_now)
        med = sorted(scores)[len(scores) // 2]
        for i in range(len(anchors)):          # ascending age: oldest first
            if scores[i] <= med:
                anchors.pop(i)
                if origin is not None:
                    origin.pop(i)
                break
        else:
            anchors.pop(0)
            if origin is not None:
                origin.pop(0)


def _stack_penalty(anchors, theta_now):
    """Total quadratic penalty of the current stack at theta_now (unit lambda)."""
    pen = 0.0
    for a in anchors:
        for k, v in theta_now.items():
            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
    return float(pen)


def _enforce(anchors, M, policy, theta_now, origin=None, stats=None):
    if policy == "unbounded":
        return
    if policy == "fifo":
        _evict_fifo(anchors, M, origin)
    elif policy == "random":
        _evict_random(anchors, M, origin)
    elif policy == "redun":
        _evict_redun(anchors, M, theta_now, origin)
    elif policy == "agew":
        _evict_agew(anchors, M, theta_now, origin)
    elif policy == "agew2":
        _evict_agew2(anchors, M, theta_now, origin)
    elif policy == "merge":
        instrument = stats is not None
        if instrument:
            stats["pen_before"] = _stack_penalty(anchors, theta_now)
        while len(anchors) > M:
            i, j = _merge_once(anchors)
            if stats is not None:
                stats["n_merges"] += 1
            if origin is not None:
                origin.pop(j)   # merged twin stored at i; slot j's origin dies
        if instrument:
            pen_after = _stack_penalty(anchors, theta_now)
            rel = abs(pen_after - stats["pen_before"]) / max(stats["pen_before"],
                                                             1e-12)
            stats["drifts"].append(rel)
            stats["max_rel_drift"] = max(stats["max_rel_drift"], rel)
    else:
        raise ValueError(policy)


# ---------------------------------------------------------------- one run
def run_ours_bounded(seed, data, perms, ntask, lam, policy, M):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    origin, survivor_history = [], []
    mstats = {"n_merges": 0, "max_rel_drift": 0.0, "drifts": []}
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
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k]
                                         * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        new_anchor = fisher_usage(model, xs, phase, ntask)
        anchors.append(new_anchor)
        origin.append(phase)
        theta_now = flat_params(model)
        _enforce(anchors, M, policy, theta_now, origin, mstats)
        survivor_history.append(sorted(origin))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} {policy}{M} T{phase}: "
              + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase + 1)),
              flush=True)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask   # just-learned mean
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2), "n_anchors_final": len(anchors),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "survivor_history": survivor_history,
            "n_merges": mstats["n_merges"],
            "max_rel_drift": round(mstats["max_rel_drift"], 8),
            "merge_drifts": [round(d, 8) for d in mstats["drifts"]]}


# ---------------------------------------------------------------- driver
def build_jobs(policies, seeds, budgets, lams, ntasks=(10, 20)):
    """job = ("delp", ntask, seed, policy, M, lam).

    NOTE: an arm never evicts when M >= ntask+1 (the stack can hold one anchor
    per task), so (policy, M, T) combos with M > T are equivalent to unbounded
    and are skipped to save compute."""
    jobs = []
    for nt in ntasks:
        for pol in policies:
            for M in budgets:
                if pol != "unbounded" and M > nt:
                    continue          # M >= nt+1 can never trigger eviction
                for seed in seeds:
                    for lam in lams:
                        jobs.append(("delp", nt, seed, pol, M, lam))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="anchor_budget_result.json")
    ap.add_argument("--policies", default="unbounded,fifo,random,redun,merge")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budgets", default="5")
    ap.add_argument("--lams", default="5000")
    ap.add_argument("--ntasks", default="10,20")   # task counts, comma list
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    a = ap.parse_args()

    pols = a.policies.split(",")
    seed_list = [int(s) for s in a.seeds.split(",")]
    bud_list = [int(s) for s in a.budgets.split(",")]
    lam_list = [float(s) for s in a.lams.split(",")]
    nt_list = [int(s) for s in a.ntasks.split(",")]

    if a.smoke:                                # tiny local correctness check
        import exp6_baselines as eb
        import exp4_branch_protection as e4
        eb.EPOCHS = e4.EPOCHS = 1              # functions read their OWN module globals
        eb.NTRAIN = e4.NTRAIN = 2000
        eb.NFISHER = 20
        eb.NUSAGE = 50
        global EPOCHS, NTRAIN                  # ...and so does this module
        EPOCHS, NTRAIN = 1, 2000
        jobs = [("delp", 4, 0, pol, 2, 5000.0) for pol in POLICIES]
    else:
        jobs = build_jobs(pols, seed_list, bud_list, lam_list, nt_list)

    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs",
          flush=True)
    data = load_data(a.dataset)
    res, t0 = [], time.time()
    for kind, nt, seed, pol, M, lam in mine:
        rec = {"kind": kind, "ntask": nt, "seed": seed, "policy": pol,
               "M": M, "lam": lam, "dataset": a.dataset}
        try:
            if kind == "delp":
                r = run_ours_bounded(seed, data, get_perms(seed, nt), nt,
                                     lam, pol, M)
            else:
                r = run_si_impl(seed, data, get_perms(seed, nt), nt, LAM_SI)
            rec.update(r)
        except Exception as exc:              # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        res.append(rec)
        print(f"  {kind} {pol}{M} lam{lam:.0f} T{nt} s{seed}: bwt={rec.get('bwt')} "
              f"avg={rec.get('final_avg')} ell={rec.get('ell')}", flush=True)

    out = a.out if a.n_tasks == 1 else f"{a.out}.part{a.task_id}"
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print(f"WROTE {out} in {round(time.time()-t0,1)}s", flush=True)

    # quick aggregate for the shard log
    agg = {}
    for r in res:
        if "error" in r or r["kind"] != "delp":
            continue
        key = f"{r['policy']}{r['M']}|lam{r.get('lam', LAM_DELP):.0f}|T{r['ntask']}"
        agg.setdefault(key, []).append(r)
    print("\n==== aggregate (mean±sd over seeds) ====")
    for key, rs in sorted(agg.items()):
        b = [r["bwt"] for r in rs]
        f_ = [r["final_avg"] for r in rs]
        e = [r["ell"] for r in rs]
        print(f"{key:14s}  BWT {st.mean(b):7.2f}±{st.stdev(b) if len(b)>1 else 0:4.2f}"
              f"  acc {st.mean(f_):6.2f}±{st.stdev(f_) if len(f_)>1 else 0:4.2f}"
              f"  ell {st.mean(e):6.2f}")


if __name__ == "__main__":
    main()
