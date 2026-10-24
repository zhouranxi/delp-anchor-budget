"""analyze_gating_census.py -- S1 option C: the gating-diffusion audit.

Question: do active-dendrite segments actually specialize in ADNet continual
learning, or is "which segment wins" essentially decided by random init?

Part 1 (existing P1 data): for every branch-arm anchor (task t), from
dom_hist[t] (per-neuron dominant-segment histogram under task-t samples):
  p_own[t]   = hist[t] / sum(hist)          P(dominant seg = own task)
  ddist[t]   = hist / sum(hist)             full dominant-identity distribution
  share_own[t] = seg_mass[t][t] / sum(seg_mass[t])   Fisher mass on own segment
Aggregated by (dataset, T); plus the dominant-identity matrix (how often the
dominant segment index equals d, pooled over anchors) and |d - t| distance.

Part 2 (null model, --null): an UNTRAINED ADNet measured with the identical
usage_stats protocol: for each task t, P(argmax segment = t) under task-t
context.  If the null matches the trained p_own, the trained network's gating
is indistinguishable from random-init competition.

Usage:  python analyze_gating_census.py            # part 1 (JSON only)
        python analyze_gating_census.py --null     # + part 2 (needs torch+data)
"""
import argparse, json, os, sys

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "results")


def part1():
    recs = []
    for f in ["s1_mnist.json", "s1_fashion.json"]:
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["results"]:
            if r.get("gran") == "branch" and "dom_hist" in r:
                recs.append(r)
    if not recs:
        print("no branch-arm P1 records found"); return

    print("=== Part 1: gating census from P1 branch arms ===")
    print(f'{"dataset":8s} {"T":>3s} {"lam":>6s} | {"p_own":>6s} | '
          f'{"fisher_top1":>11s} | {"p_dom0":>7s} {"p_dom_rand":>10s}')
    for ds in ("mnist", "fashion"):
        for lam in (1000.0, 5000.0):
            for T in (10, 20):
                sel = [r for r in recs if r["dataset"] == ds
                       and r["ntask"] == T and r["lam"] == lam]
                if not sel:
                    continue
                powns, shares, d0s, drands = [], [], [], []
                for r in sel:
                    for t, hist in enumerate(r["dom_hist"]):
                        tot = sum(hist)
                        if tot:
                            powns.append(hist[t] / tot)
                            d0s.append(hist[0] / tot)
                            # non-own mass spread over T-1 other segs
                            drands.append((tot - hist[t]) / tot / (T - 1))
                    for row in r["seg_mass"]:
                        s = sum(row)
                        if s:
                            # kz indices are not stored, so use the index-free
                            # top-1 concentration instead of an own-task share
                            shares.append(max(row) / s)
                n = len(powns)
                print(f"{ds:8s} {T:3d} {lam:6.0f} | {sum(powns)/n:6.3f} | "
                      f"{sum(shares)/len(shares):11.3f} | "
                      f"{sum(d0s)/n:7.3f} {sum(drands)/n:10.3f}")

    # dominant-identity distance structure, pooled
    print("\n--- P(|dominant - own| = k), pooled over all anchors ---")
    dist = {}
    for r in recs:
        T = r["ntask"]
        for t, hist in enumerate(r["dom_hist"]):
            tot = sum(hist)
            if not tot:
                continue
            for d, c in enumerate(hist):
                k = abs(d - t)
                dist[k] = dist.get(k, 0) + c
    tot = sum(dist.values())
    for k in sorted(dist):
        print(f"  |d-t|={k}: {dist[k]/tot:6.3f}")


def part2(ntask=10, nusage=500):
    for cand in (os.path.join(_here, "..", "dendritic-lab"),      # server
                 os.path.join(_here, "..", "..", "dendritic-lab")):  # local
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
            break
    import torch
    import torch.nn.functional as F
    from exp4_branch_protection import ADNet, kwta
    from anchor_budget import load_data

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data("mnist")
    xtr = data[0]
    torch.manual_seed(0)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(dev)
    model.eval()                       # untrained: random-init columns
    ctx1 = lambda t: F.one_hot(torch.tensor([t], device=dev), ntask).float()
    print(f"=== Part 2: null model (untrained ADNet, seed 0, {dev}) ===")
    print("P(dominant seg = own task) per task, task-t samples, task-t context:")
    ps = []
    for t in range(ntask):
        xs = xtr[t * 1000:(t + 1) * 1000]     # disjoint sample pools per task
        own = 0; tot = 0
        with torch.no_grad():
            for i in range(0, nusage, 250):
                xb = xs[i:i + 250].to(dev)
                ctx = ctx1(t).expand(xb.size(0), -1)
                tt = model.ff(xb)
                u = model.dend(ctx).view(-1, model.nseg, model.nh)
                d, seg = u.max(dim=1)          # d: (batch, nh), seg: (batch, nh)
                hh = kwta(tt * torch.sigmoid(d))
                act = (hh > 0).float()
                own += (act * (seg == t).float()).sum().item()
                tot += act.sum().item()
        p = own / tot if tot else float("nan")
        ps.append(p)
        print(f"  task {t:2d}: p_own = {p:.3f}")
    print(f"null mean p_own = {sum(ps)/len(ps):.3f}   "
          f"(chance level for uniform argmax over {ntask} segs = {1/ntask:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true")
    a = ap.parse_args()
    part1()
    if a.null:
        part2()
