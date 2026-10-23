"""anchor_budget_cifar.py -- S2 round-6: bounded anchor sets on Split-CIFAR-100.

Purpose: show the anchor accumulation problem (paper's central claim) is not an
artifact of the dendritic Permuted-MNIST setting. Backbone: CIFAR ResNet-18
(3x3 stem, no maxpool). Protocol: Split-CIFAR-100, T in {10, 20} tasks, task-IL
eval (argmax restricted to the current task's class slice). Anchors are plain
diagonal-Fisher EWC snapshots, one per finished task -- usage-localization is
not ported (no dendritic segments here); what we test is the ACCUMULATION
mechanism: unbounded stacks over-constrain, bounded sets with age-first
eviction recover plasticity, quality-only eviction (redun) fails.

Protocol notes (v2, after calibration rounds a/a'/a''):
  * masked softmax: each phase trains with CE over ITS OWN 10 classes only
    (logits restricted to the task's class slice). Head rows of other tasks
    receive no gradient and never drift -- forgetting can only happen through
    the shared backbone, which is exactly where the anchor accumulation
    mechanism lives. The classifier head is therefore EXCLUDED from anchors.
  * anchors cover backbone parameters only, Fisher globally normalized over
    the backbone (head Fisher is orders of magnitude larger than conv Fisher
    and made single-lambda calibration impossible: the head was pinned at
    every lambda >= 0.01 while new tasks sat at chance).

Eviction policies mirror anchor_budget.py exactly (same names, same semantics):
  unbounded  keep every anchor (reference)
  fifo       evict the oldest anchor
  random     evict a uniformly random anchor
  redun      evict argmin eviction-time constraint mass sum_k F_k (theta-theta_k)^2
  merge      merge the most Fisher-similar pair (cosine); F adds, theta is the
             Fisher-weighted barycenter
  agew       evict argmin(mass * age_rank)
  agew2      oldest anchor among below-median-mass ones, else fifo

Usage (server, one shard):
  CUDA_VISIBLE_DEVICES=0 python anchor_budget_cifar.py --task-id 0 --n-tasks 2 \
      --out s2r6_b.json --policies unbounded,fifo,redun,agew2 --budgets 2,5 \
      --lams 100 --ntasks 10,20 --seeds 0,1,2,3,4,5,6,7,8,9
Usage (smoke, ~1 min on GPU):
  python anchor_budget_cifar.py --smoke --out smoke_cifar.json
"""
import argparse, json, os, statistics as st, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
EPOCHS, BS, LR, WD, MOM = 5, 128, 0.05, 5e-4, 0.9
NFISHER = 50                       # per-sample Fisher draws, as in the MNIST code
CLIP_NORM = 5.0                    # safety net: collapse observed at anchor onset
POLICIES = ["unbounded", "fifo", "random", "redun", "merge", "agew", "agew2"]


# ---------------------------------------------------------------- data / model
def load_cifar100():
    from torchvision import datasets, transforms
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = datasets.CIFAR100
    d_tr = tr("./data", train=True, download=True, transform=tf_tr)
    d_te = tr("./data", train=False, download=True, transform=tf_te)
    d_fx = tr("./data", train=True, download=True, transform=tf_te)  # clean, for Fisher
    ytr = torch.tensor(d_tr.targets)
    yte = torch.tensor(d_te.targets)
    xte = torch.stack([d_te[i][0] for i in range(len(d_te))])
    xfx = torch.stack([d_fx[i][0] for i in range(len(d_fx))])
    return d_tr, (ytr, yte, xte, xfx)


def _bn_to_gn(module):
    """Replace BatchNorm2d with GroupNorm(32, C). EWC anchors cover only
    named_parameters; BN running mean/var are buffers that drift with each
    new task and wipe out earlier tasks even when all weights are pinned
    (observed in round A: final avg == chance at every lambda). GroupNorm has
    no running stats and is the standard fix in continual-learning papers."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name,
                    nn.GroupNorm(min(32, child.num_features), child.num_features))
        else:
            _bn_to_gn(child)


def make_model():
    from torchvision.models import resnet18
    m = resnet18(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    _bn_to_gn(m)
    return m.to(DEV)


# ------------------------------------------------------------- eviction ops
# (verbatim semantics from anchor_budget.py; duplicated so this file is
#  self-contained for reviewers)
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
    # Memory-light variant of anchor_budget.py's merge: computes each pair's
    # cosine by flattening lazily (no (A, P) matrix materialized) -- identical
    # selection rule, but peak transient memory is O(P) not O(A*P). Matters on
    # GPUs that share memory with another service.
    def flat(a):
        return torch.cat([a["fisher"][k].flatten() for k in sorted(a["fisher"])])
    best = None
    for i in range(len(anchors)):
        fi = flat(anchors[i])
        ni = fi.norm()
        for j in range(i + 1, len(anchors)):
            fj = flat(anchors[j])
            d = 1.0 - torch.dot(fi, fj) / (ni * fj.norm() + 1e-12)
            if best is None or d.item() < best[0]:
                best = (d.item(), i, j)
        del fi
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
    while len(anchors) > M:
        scores = _redun_scores(anchors, theta_now)
        sc = [scores[i] * (i + 1) for i in range(len(anchors))]  # i=0 oldest
        i = sc.index(min(sc))
        anchors.pop(i)
        if origin is not None:
            origin.pop(i)


def _evict_agew2(anchors, M, theta_now, origin=None):
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


def _enforce(anchors, M, policy, theta_now, origin=None):
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
        while len(anchors) > M:
            i, j = _merge_once(anchors)
            if origin is not None:
                origin.pop(j)   # merged twin stored at i; slot j's origin dies
    else:
        raise ValueError(policy)


# ---------------------------------------------------------------- one run
def anchor_keys(model):
    """Backbone parameter names (classifier head excluded)."""
    return [k for k, _ in model.named_parameters() if not k.startswith("fc.")]


def fisher_anchor(model, xfx, idx):
    """Diagonal Fisher of a finished task on the BACKBONE, NFISHER single-sample
    draws of the squared grad of the max log-softmax (same estimator as
    fisher_usage). Global-scale normalized over the backbone: one scalar (mean
    Fisher value over all backbone params) divides every tensor, preserving
    cross-layer structure while making lambda backbone-independent. Raw-scale
    Fisher made the penalty explode at anchor onset; head Fisher dominated the
    global scale before the head was excluded (rounds a'/a'')."""
    model.eval()
    P = {k: v for k, v in dict(model.named_parameters()).items()
         if not k.startswith("fc.")}
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    for i in range(NFISHER):
        xb = xfx[idx[i % len(idx)]].unsqueeze(0).to(DEV)
        out = model(xb)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    n = sum(f.numel() for f in fisher.values())
    scale = sum(f.sum() for f in fisher.values()) / max(n, 1)
    for k in fisher:
        fisher[k] = fisher[k] / (scale + 1e-12)
    return fisher


def eval_tasks(model, task_cls, yte, xte, phase):
    """Task-IL: for each seen task, argmax over its own class slice."""
    model.eval()
    acc = {}
    with torch.no_grad():
        for cc in range(phase + 1):
            cls = task_cls[cc]
            idx = torch.tensor([i for i, y in enumerate(yte.tolist()) if y in cls])
            xt = xte[idx].to(DEV)
            yt = yte[idx].to(DEV)
            lut = torch.tensor(cls, device=DEV)
            correct = 0
            for i in range(0, xt.size(0), 512):
                out = model(xt[i:i + 512])[:, lut]
                pred = lut[out.argmax(1)]
                correct += (pred == yt[i:i + 512]).sum().item()
            acc[(phase, cc)] = correct / xt.size(0) * 100
    return acc


def run_ours_bounded(seed, ntask, lam, policy, M, d_tr, aux, cpt, epochs, cap):
    ytr, yte, xte, xfx = aux
    g = torch.Generator().manual_seed(100 + seed)      # same convention as get_perms
    order = torch.randperm(100, generator=g).tolist()
    task_cls = [order[t * cpt:(t + 1) * cpt] for t in range(ntask)]
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    anchors, acc = [], {}
    origin, survivor_history = [], []
    for phase in range(ntask):
        cls = task_cls[phase]
        lut = torch.tensor(cls, device=DEV)             # masked-softmax slice
        idx = [i for i, y in enumerate(ytr.tolist()) if y in cls]
        if cap:
            idx = idx[:cap]
        dl = DataLoader(Subset(d_tr, idx), batch_size=BS, shuffle=True,
                        num_workers=2, pin_memory=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                local = torch.zeros_like(yb)
                for ci, c in enumerate(cls):
                    local[yb == c] = ci
                opt.zero_grad()
                out = model(xb)[:, lut]                 # CE over own classes only
                loss = F.cross_entropy(out, local)
                if anchors and lam > 0:   # lam==0: plain-SGD arm, skip the graph
                    P = {k: v for k, v in dict(model.named_parameters()).items()
                         if k in anchors[0]["theta"]}
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k]
                                         * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
        fx_idx = [i for i, y in enumerate(ytr.tolist()) if y in cls][:NFISHER * 2]
        fisher = fisher_anchor(model, xfx, fx_idx)
        theta = {k: v.detach().clone() for k, v in dict(model.named_parameters()).items()
                 if k in fisher}
        anchors.append({"theta": theta, "fisher": fisher})
        origin.append(phase)
        _enforce(anchors, M, policy, theta, origin)
        survivor_history.append(sorted(origin))
        a = eval_tasks(model, task_cls, yte, xte, phase)
        acc.update(a)
        print(f"  s{seed} {policy}{M} T{phase}: "
              + " ".join(f"t{c}={acc[(phase, c)]:.0f}" for c in range(phase + 1)),
              flush=True)
    bwt = sum(acc[(ntask - 1, c)] - acc[(c, c)] for c in range(ntask - 1)) / (ntask - 1)
    favg = sum(acc[(ntask - 1, c)] for c in range(ntask)) / ntask
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "ell": round(ell, 2), "n_anchors_final": len(anchors),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)],
            "survivor_history": survivor_history}


# ---------------------------------------------------------------- driver
def build_jobs(policies, seeds, budgets, lams, ntasks):
    """job = ("delp", ntask, seed, policy, M, lam). Dedupes unbounded across
    budgets (M is a no-op for it); skips (policy, M, T) with M >= ntask since
    the stack can then hold one anchor per task and never evicts."""
    jobs = []
    for nt in ntasks:
        for pol in policies:
            for mi, M in enumerate(budgets):
                if pol == "unbounded" and mi > 0:
                    continue                     # unbounded: run once per (T,seed,lam)
                if pol != "unbounded" and M >= nt:
                    continue                     # never triggers eviction
                for seed in seeds:
                    for lam in lams:
                        jobs.append(("delp", nt, seed, pol, M, lam))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="anchor_budget_cifar_result.json")
    ap.add_argument("--policies", default="unbounded,fifo,random,redun,merge,agew,agew2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budgets", default="2,5")
    ap.add_argument("--lams", default="100")
    ap.add_argument("--ntasks", default="10,20")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--cap", type=int, default=0)   # per-class train cap; 0 = all 500
    a = ap.parse_args()

    pols = a.policies.split(",")
    seed_list = [int(s) for s in a.seeds.split(",")]
    bud_list = [int(s) for s in a.budgets.split(",")]
    lam_list = [float(s) for s in a.lams.split(",")]
    nt_list = [int(s) for s in a.ntasks.split(",")]

    if a.smoke:                      # T=4 (25 cls/task), M=2 -> evictions from phase 2
        epochs, cap, nt_list = 1, 50, [4]
        jobs = [("delp", 4, 0, pol, 2, 100.0) for pol in POLICIES]
    else:
        epochs, cap = a.epochs, a.cap
        jobs = build_jobs(pols, seed_list, bud_list, lam_list, nt_list)

    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs", flush=True)
    d_tr, aux = load_cifar100()
    res, t0 = [], time.time()
    for kind, nt, seed, pol, M, lam in mine:
        cpt = 100 // nt
        rec = {"kind": kind, "ntask": nt, "seed": seed, "policy": pol,
               "M": M, "lam": lam, "dataset": "cifar100", "backbone": "resnet18-gn",
               "protocol": "masked-ce-backbone-anchors"}
        try:
            rec.update(run_ours_bounded(seed, nt, lam, pol, M, d_tr, aux,
                                        cpt, epochs, cap))
        except Exception as exc:                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        res.append(rec)
        print(f"  {kind} {pol}{M} lam{lam:.0f} T{nt} s{seed}: bwt={rec.get('bwt')} "
              f"avg={rec.get('final_avg')} ell={rec.get('ell')}", flush=True)
        json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
                  open(a.out + f".part{a.task_id}", "w"), indent=1)  # crash-safe checkpoint

    out = a.out if a.n_tasks == 1 else f"{a.out}.part{a.task_id}"
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print(f"WROTE {out} in {round(time.time()-t0,1)}s", flush=True)

    agg = {}
    for r in res:
        if "error" in r:
            continue
        key = f"{r['policy']}{r['M']}|lam{r['lam']:.0f}|T{r['ntask']}"
        agg.setdefault(key, []).append(r)
    print("\n==== aggregate (mean±sd over seeds) ====")
    for key, rs in sorted(agg.items()):
        b = [r["bwt"] for r in rs]
        f_ = [r["final_avg"] for r in rs]
        e = [r["ell"] for r in rs]
        print(f"{key:16s}  BWT {st.mean(b):7.2f}±{st.stdev(b) if len(b)>1 else 0:4.2f}"
              f"  acc {st.mean(f_):6.2f}±{st.stdev(f_) if len(f_)>1 else 0:4.2f}"
              f"  ell {st.mean(e):6.2f}")


if __name__ == "__main__":
    main()
