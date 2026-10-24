"""branch_delp.py -- S1: branch-level (segment-resolved) usage-localized EWC.

Granularity ladder (the paper's ablation axis):
  neuron  original DeLP: F_loc = o_j * F, single reference per anchor   -- baseline
  plain   unlocalized EWC on the same ADNet (o == 1)                    -- 3rd point
  branch  segment-resolved anchors:
            o_seg[k, j] = P(neuron j active AND segment k wins | task t samples),
            with sum_k o_seg[k, j] = o_j (refines, not redefines, usage),
            K per-segment references refs[k] per anchor, and segment-attributed
            drift tracking: after phase p with parameter change dW, anchor-a's
            reference for segment k moves by dW * w_pk[j] where w_pk[j] is the
            fraction of phase-p batches in which segment k won neuron j.
          Penalty from anchor a: sum_k o_seg[k] * F * (W - refs[k])^2.
          At anchor-creation time this equals the neuron penalty exactly
          (refs all == theta, sum_k o_seg == o); the two arms diverge only
          through attributed drift -> the mechanism is *where* old anchors
          keep fighting after later tasks move the shared weights.

B experiment (option B, pre-registered): granularity x budget axis.
  Job kind "bfifo": same run_gran but the anchor stack is capped at M
  anchors (fifo eviction of the oldest, identical to anchor_budget.fifo).
  Question: at matched anchor count M, does the finer-grained branch anchor
  buy more ell than the neuron anchor (budget savings), or do the two stay
  paired as in P1?  Sanity: bfifo with M > T never evicts and must
  bit-match the gran arm of the same seed.

Instrumentation (per run): seg_mass[t, k]  = Fisher mass of anchor t routed
to segment k; dom_hist[t] = per-neuron dominant segment under task t.

Equivalence smoke: --smoke also runs anchor_budget.run_ours_bounded
(policy=unbounded) at the same seed/protocol and asserts bit-equality with
the neuron arm (validates that the reimplementation changed nothing).

Usage (server):  CUDA_VISIBLE_DEVICES=0 python branch_delp.py --task-id 0 --n-tasks 4
Usage (smoke):   python branch_delp.py --smoke --out smoke_branch.json
"""
import argparse, json, os, statistics as st, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),   # server layout
              os.path.join(_here, "..", "..", "dendritic-lab")):  # local layout
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR, DEV,
                            get_perms, flat_params, eval_model, finish,
                            run_si_impl, run_mas_impl)
from exp4_branch_protection import ADNet, permute, kwta
from anchor_budget import load_data

NFISHER, NUSAGE = 300, 500
GRANS = ["neuron", "branch", "plain"]
PARAM_KEYS = ["ff.w", "ff.b", "out.w", "out.b"]
LAM_SI, LAM_MAS = 100.0, 1.0     # SI: S2-paper reference value; MAS: exp6 grid


# ------------------------------------------------------------ stats & anchors
def usage_stats(model, xs, task_id, ntask):
    """Replicates exp6.fisher_usage's usage loop exactly (same batching), and
    adds per-segment winner counts. Returns o (nh,), o_seg (nseg, nh) with
    o_seg.sum(0) == o."""
    model.eval()
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    ns, nh = model.nseg, model.nh
    o = torch.zeros(nh, device=DEV)
    o_seg = torch.zeros(ns, nh, device=DEV)
    with torch.no_grad():
        for i in range(0, NUSAGE, 250):
            xb = xs[i:i + 250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.ff(xb)
            u = model.dend(ctx).view(-1, ns, nh)
            d, seg = u.max(dim=1)              # seg: (batch, nh)
            h = kwta(t * torch.sigmoid(d))
            act = (h > 0).float()
            o += act.sum(0)
            for k in range(ns):
                o_seg[k] += (act * (seg == k).float()).sum(0)
    return o / NUSAGE, o_seg / NUSAGE


def fisher_raw(model, xs, task_id, ntask):
    """Unlocalized empirical Fisher (same estimator as exp6: grad of
    log-sum-exp logits on true-context forwards, NFISHER single samples)."""
    model.eval()
    P = flat_params(model)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    for i in range(NFISHER):
        xb = xs[i:i + 1].to(DEV)
        out, _ = model(xb, ctx1)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= NFISHER
    return fisher


def _clone(P):
    return {k: v.detach().clone() for k, v in P.items()}


def make_anchor(model, xs, task_id, ntask, gran):
    P = flat_params(model)
    theta = _clone(P)
    if gran in ("neuron", "plain"):
        fisher = fisher_raw(model, xs, task_id, ntask)
        if gran == "neuron":
            o, _ = usage_stats(model, xs, task_id, ntask)
            fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
            fisher["ff.b"] = fisher["ff.b"] * o
            fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
        return {"gran": gran, "theta": theta, "fisher": fisher}
    # branch
    fisher = fisher_raw(model, xs, task_id, ntask)
    o, o_seg = usage_stats(model, xs, task_id, ntask)
    kz = [k for k in range(o_seg.shape[0]) if o_seg[k].abs().max() > 0]
    refs = [_clone(P) for _ in range(o_seg.shape[0])]
    seg_mass = []
    for k in kz:
        m = (o_seg[k].unsqueeze(1) * fisher["ff.w"]).sum()
        m = m + (o_seg[k] * fisher["ff.b"]).sum()
        m = m + (o_seg[k].unsqueeze(0) * fisher["out.w"]).sum()
        seg_mass.append(float(m) + float(fisher["out.b"].sum()) / len(kz))
    dom = o_seg.argmax(0)
    return {"gran": gran, "theta": theta, "fisher": fisher, "o_seg": o_seg,
            "refs": refs, "kz": kz, "seg_mass": seg_mass,
            "dom_hist": torch.bincount(dom, minlength=o_seg.shape[0]).tolist()}


def anchor_penalty(a, P, gran):
    if gran in ("neuron", "plain"):
        pen = 0.0
        for k, v in P.items():
            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
        return pen
    pen = 0.0
    fw, fb, ow, ob = (a["fisher"][k] for k in PARAM_KEYS)
    for kk in a["kz"]:
        osk = a["o_seg"][kk]
        r = a["refs"][kk]
        pen = pen + (osk.unsqueeze(1) * fw * (P["ff.w"] - r["ff.w"]) ** 2).sum()
        pen = pen + (osk * fb * (P["ff.b"] - r["ff.b"]) ** 2).sum()
        pen = pen + (osk.unsqueeze(0) * ow * (P["out.w"] - r["out.w"]) ** 2).sum()
        pen = pen + (ob * (P["out.b"] - r["out.b"]) ** 2).sum()
    return pen


def attribute_drift(anchors, w_pk, dW, gran):
    """branch only: move per-segment references by the drift attributed to
    each segment. w_pk: (nseg, nh) winner fractions of the finished phase;
    dW: dict of parameter changes over the phase."""
    if gran != "branch":
        return
    wm = w_pk.mean()                       # scalar attribution for out.b
    for a in anchors:
        if a["gran"] != "branch":
            continue
        for kk in a["kz"]:
            w = w_pk[kk]
            r = a["refs"][kk]
            r["ff.w"] += dW["ff.w"] * w.unsqueeze(1)
            r["ff.b"] += dW["ff.b"] * w
            r["out.w"] += dW["out.w"] * w.unsqueeze(0)
            r["out.b"] += dW["out.b"] * wm


# ---------------------------------------------------------------- one run
def run_gran(seed, data, perms, ntask, lam, gran, M=None):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    seg_mass_all, dom_all = [], []
    evicted, n_created = [], 0
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        W_start = _clone(flat_params(model)) if gran == "branch" else None
        win_cnt = torch.zeros(ntask, model.nh, device=DEV) \
            if gran == "branch" else None
        n_seen = 0
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
                if gran == "branch":
                    for k in range(ntask):
                        win_cnt[k] += (seg == k).float().sum(0)
                    n_seen += xb.size(0)
        if gran == "branch":
            P_now = flat_params(model)
            dW = {k: P_now[k].detach() - W_start[k] for k in P_now}
            w_pk = win_cnt / max(n_seen, 1)
            attribute_drift(anchors, w_pk, dW, gran)
        anchors.append(make_anchor(model, xs, phase, ntask, gran))
        n_created += 1
        if gran == "branch":
            seg_mass_all.append(anchors[-1]["seg_mass"])
            dom_all.append(anchors[-1]["dom_hist"])
        if M is not None:
            while len(anchors) > M:          # fifo, identical to anchor_budget
                anchors.pop(0)
                evicted.append(n_created - len(anchors) - 1)  # origin task id
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} {gran} T{phase}: "
              + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase + 1)),
              flush=True)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    rec = {"seed": seed, "gran": gran, "bwt": round(bwt, 2),
           "final_avg": round(favg, 2), "ell": round(ell, 2),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                          for p in range(ntask)]}
    if gran == "branch":
        rec["seg_mass"] = [[round(m, 4) for m in row] for row in seg_mass_all]
        rec["dom_hist"] = dom_all
    if M is not None:
        rec["M"] = M
        rec["evicted"] = evicted
    return rec


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="branch_result.json")
    ap.add_argument("--granularities", default="neuron,branch,plain")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lams", default="5000")
    ap.add_argument("--ntasks", default="10,20")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--with-refs", action="store_true",
                    help="also queue corrected SI/MAS reference arms")
    ap.add_argument("--budgets", default="",
                    help="comma list of fifo budgets; non-empty queues bfifo "
                         "jobs for the granularities in --granularities")
    ap.add_argument("--skip-gran", action="store_true",
                    help="skip unbounded gran arms (e.g. when P1 data covers them)")
    a = ap.parse_args()

    grans = a.granularities.split(",")
    seed_list = [int(s) for s in a.seeds.split(",")]
    lam_list = [float(s) for s in a.lams.split(",")]
    nt_list = [int(s) for s in a.ntasks.split(",")]

    if a.smoke:
        import exp6_baselines as eb
        import exp4_branch_protection as e4
        import anchor_budget as ab
        for mod in (eb, e4, ab, sys.modules[__name__]):
            mod.EPOCHS = 1
            mod.NTRAIN = 2000
        eb.NFISHER, eb.NUSAGE = 20, 50
        sys.modules[__name__].NFISHER, sys.modules[__name__].NUSAGE = 20, 50
        jobs = [("gran", 4, 0, g, 5000.0) for g in GRANS]
        jobs += [("si", 4, 0, "si", LAM_SI), ("mas", 4, 0, "mas", LAM_MAS)]
        # budget-path sanity: M=10 >= T+1 never evicts -> must bit-match gran
        jobs += [("bfifo", 4, 0, "neuron", 5000.0, 10),
                 ("bfifo", 4, 0, "branch", 5000.0, 2)]   # M=2 exercises eviction
    else:
        jobs = []
        for nt in nt_list:
            if not a.skip_gran:
                for g in grans:
                    for seed in seed_list:
                        for lam in lam_list:
                            jobs.append(("gran", nt, seed, g, lam))
            if a.with_refs:
                for kind, lam in (("si", LAM_SI), ("mas", LAM_MAS)):
                    for seed in seed_list:
                        jobs.append((kind, nt, seed, kind, lam))
            if a.budgets:
                for g in grans:
                    for Mb in [int(s) for s in a.budgets.split(",")]:
                        if Mb > nt:
                            continue          # M >= T+1 can never trigger eviction
                        for seed in seed_list:
                            for lam in lam_list:
                                jobs.append(("bfifo", nt, seed, g, lam, Mb))

    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs",
          flush=True)
    data = load_data(a.dataset)
    res, t0 = [], time.time()
    for job in mine:
        kind, nt, seed, g, lam = job[:5]
        Mb = job[5] if len(job) > 5 else None
        rec = {"kind": kind, "ntask": nt, "seed": seed,
               "gran": g if kind in ("gran", "bfifo") else kind,
               "lam": lam, "dataset": a.dataset}
        try:
            if kind == "gran":
                r = run_gran(seed, data, get_perms(seed, nt), nt, lam, g)
            elif kind == "bfifo":
                r = run_gran(seed, data, get_perms(seed, nt), nt, lam, g, M=Mb)
            elif kind == "si":
                r = run_si_impl(seed, data, get_perms(seed, nt), nt, lam)
            elif kind == "mas":
                r = run_mas_impl(seed, data, get_perms(seed, nt), nt, lam)
            else:
                raise ValueError(kind)
            rec.update(r)
        except Exception as exc:                              # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        res.append(rec)
        print(f"  {kind} {g}{'M'+str(Mb) if Mb else ''} T{nt} s{seed} "
              f"lam{lam:.0f}: bwt={rec.get('bwt')} "
              f"avg={rec.get('final_avg')} ell={rec.get('ell')}", flush=True)

    if a.smoke:
        # equivalence check: neuron arm must bit-match anchor_budget unbounded
        import anchor_budget as ab
        nt = 4
        ref = ab.run_ours_bounded(0, data, get_perms(0, nt), nt, 5000.0,
                                  "unbounded", 2)
        neu = next(r for r in res if r.get("gran") == "neuron"
                   and r.get("kind") == "gran")
        same = ref["acc_matrix"] == neu["acc_matrix"]
        print(f"\nEQUIVALENCE neuron-vs-original: "
              f"{'PASS' if same else 'FAIL'}", flush=True)
        if not same:
            print("  orig:", ref["acc_matrix"])
            print("  neu :", neu["acc_matrix"])
        res.append({"kind": "equivalence_check", "pass": bool(same)})
        # budget path: M=10 >= T+1 never evicts -> bit-match gran neuron
        bneu = next(r for r in res if r.get("kind") == "bfifo"
                    and r.get("gran") == "neuron")
        same_b = bneu["acc_matrix"] == neu["acc_matrix"] \
            and bneu.get("evicted") == []
        print(f"EQUIVALENCE bfifo(M=10,no-evict)-vs-gran: "
              f"{'PASS' if same_b else 'FAIL'}", flush=True)
        res.append({"kind": "equivalence_check_bfifo", "pass": bool(same_b)})

    out = a.out if a.n_tasks == 1 else f"{a.out}.part{a.task_id}"
    json.dump({"results": res, "elapsed": round(time.time() - t0, 1)},
              open(out, "w"), indent=1)
    print(f"WROTE {out} in {round(time.time()-t0,1)}s", flush=True)

    agg = {}
    for r in res:
        if "error" in r or r.get("kind") not in ("gran", "bfifo"):
            continue
        mtag = f"M{r['M']}" if r["kind"] == "bfifo" else "unb"
        agg.setdefault(f"{r['gran']}|{mtag}|lam{r['lam']:.0f}|T{r['ntask']}", []) \
            .append(r)
    print("\n==== aggregate (mean±sd over seeds) ====")
    for key, rs in sorted(agg.items()):
        b = [r["bwt"] for r in rs]
        f_ = [r["final_avg"] for r in rs]
        e = [r["ell"] for r in rs]
        print(f"{key:18s}  BWT {st.mean(b):7.2f}"
              f"±{st.stdev(b) if len(b)>1 else 0:4.2f}"
              f"  acc {st.mean(f_):6.2f}"
              f"±{st.stdev(f_) if len(f_)>1 else 0:4.2f}"
              f"  ell {st.mean(e):6.2f}")


if __name__ == "__main__":
    main()
