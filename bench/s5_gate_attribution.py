"""Attribution test for the paper's mechanism section.

Three claims are under test:

  C1. "The dendritic columns remain mutually orthogonal after training ...
       indistinguishable from their random initialization."
       -> Is this a fact about TRAINING, or a structural consequence of the
          gradient mask?

  C2. "The winning segment's gate value is a constant of the training run
       (0.565), identical to four decimals under every alternative context."
       -> Is 0.565 produced by TRAINING, or already present in a freshly
          initialized, untrained network?

  C3. "the context pathway is present, protected per task, and unused as a
       modulator."
       -> Does the network learn ANY task-to-segment routing?

Mechanism of the suspected effect
---------------------------------
`protect_grads(ctx_label)` zeroes the gradient of every block of
`dend.weight` except block `ctx_label`, in the layout
`W.view(nseg, nh, ntask)`.  But the forward pass computes
`u = dend(ctx) = W[:, ctx]`, i.e. it reads ALL nseg blocks of column
`ctx`.  Block `s != ctx` is masked during task `ctx` (so it never gets a
gradient when it is read) and it is also masked during task `s` (because
only block `ctx=s` is opened then).  Therefore every off-diagonal block
`s != t` receives zero gradient in every phase and is frozen at its random
initialization for the whole run, while still shaping the gate of task `t`.

Arms
----
  fresh      : no training at all (initialization only)
  protected  : the paper's training (gradient mask on)
  unmasked   : identical training with the mask OFF (control)
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from exp6_baselines import get_perms, DEV, EPOCHS, BS, NTRAIN, LR
from exp4_branch_protection import permute
from branch_delp import ADNet, BiasADNet, make_anchor, anchor_penalty, flat_params
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = 256


def train(ds, seed, own_bias, masked):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, NTASK)
    if own_bias:
        model = BiasADNet(NTASK, NTASK, own_bias=own_bias, bias_mode="fixed", nh=NH).to(DEV)
    else:
        model = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=NH).to(DEV)
    if not masked:
        # turn the per-task gradient mask off without touching anything else
        model.protected = False
    W0 = model.dend.weight.detach().clone().view(NTASK, model.nh, NTASK)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors = []
    for phase in range(NTASK):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), NTASK).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    for a in anchors:
                        # lam factor required: the paper's protocol uses
                        # lambda=1000 (see the 2026-10-27 data-consistency fix
                        # in S4_gate_bypass/code/s4_diag.py).
                        loss = loss + LAM * anchor_penalty(a, P, a["gran"])
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, NTASK, "neuron"))
    return model, W0, perms, xte


def colcos(M):
    s = []
    for c in range(NTASK):
        for o in range(NTASK):
            if o == c:
                continue
            a, b = M[..., c].flatten(), M[..., o].flatten()
            s.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))))
    return s


def probe(model, W0, perms, xte, masked, own_bias):
    W = model.dend.weight.detach().view(NTASK, model.nh, NTASK)
    rel = ((W - W0).norm(dim=1) / W0.norm(dim=1).clamp_min(1e-12))
    diag = [float(rel[s, s]) for s in range(NTASK)]
    off = [float(rel[s, t]) for s in range(NTASK) for t in range(NTASK) if s != t]

    ct = colcos(W)
    ci = colcos(W0)

    dev = W.device
    gates, own_frac = [], []
    with torch.no_grad():
        for t in range(NTASK):
            xt = xte[:1500].to(dev)
            xt = xt[:, perms[t]]
            ctx = F.one_hot(torch.full((xt.size(0),), t, device=dev), NTASK).float()
            u = model.dend(ctx).view(-1, NTASK, model.nh)
            d, seg = u.max(dim=1)
            gates.append(float(torch.sigmoid(d).mean()))
            own_frac.append(float((seg == t).float().mean()))

    return {
        "masked": masked,
        "bias": own_bias,
        "diag_block_rel_move": sum(diag) / len(diag),
        "offdiag_block_rel_move": sum(off) / len(off),
        "offdiag_blocks_frozen_frac": sum(1 for x in off if x < 1e-6) / len(off),
        "col_cos_mean_trained": sum(ct) / len(ct),
        "col_cos_min_trained": min(ct),
        "col_cos_mean_init": sum(ci) / len(ci),
        "col_cos_min_init": min(ci),
        "col_cos_trained_vs_init": sum(
            float(F.cosine_similarity(W[..., c].flatten().unsqueeze(0),
                                      W0[..., c].flatten().unsqueeze(0)))
            for c in range(NTASK)) / NTASK,
        "gate_mean": sum(gates) / len(gates),
        "gate_min": min(gates),
        "gate_max": max(gates),
        "own_segment_win_frac": sum(own_frac) / len(own_frac),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results_s5/gate_attribution.json")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []

    for ds in datasets:
        for seed in seeds:
            # fresh (no training)
            torch.manual_seed(seed)
            fresh = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=NH).to(DEV)
            W0f = fresh.dend.weight.detach().clone().view(NTASK, NH, NTASK)
            xtr, ytr, xte, yte = load_data(ds)
            perms = get_perms(seed, NTASK)
            r = probe(fresh, W0f, perms, xte, True, 0.0)
            r.update({"ds": ds, "seed": seed, "arm": "fresh"})
            rows.append(r)
            print("  %-8s s%d fresh    gate=%.4f  own_win=%.4f"
                  % (ds, seed, r["gate_mean"], r["own_segment_win_frac"]), flush=True)

            for masked in (True, False):
                model, W0, perms2, xte2 = train(ds, seed, 0.0, masked)
                r = probe(model, W0, perms2, xte2, masked, 0.0)
                r.update({"ds": ds, "seed": seed,
                          "arm": "protected" if masked else "unmasked"})
                rows.append(r)
                print("  %-8s s%d %-10s gate=%.4f own_win=%.4f  offdiag_frozen=%.2f  "
                      "colcos=%.4f  offdiag_move=%.2e"
                      % (ds, seed, r["arm"], r["gate_mean"], r["own_segment_win_frac"],
                         r["offdiag_blocks_frozen_frac"], r["col_cos_mean_trained"],
                         r["offdiag_block_rel_move"]), flush=True)

    # aggregate
    print("\n=== aggregate over seeds ===")
    agg = {}
    for ds in datasets:
        for arm in ("fresh", "protected", "unmasked"):
            sel = [r for r in rows if r["ds"] == ds and r["arm"] == arm]
            if not sel:
                continue
            k = f"{ds}/{arm}"
            agg[k] = {kk: sum(r[kk] for r in sel) / len(sel)
                      for kk in ("gate_mean", "own_segment_win_frac",
                                "offdiag_blocks_frozen_frac",
                                "offdiag_block_rel_move", "col_cos_mean_trained",
                                "col_cos_min_trained", "col_cos_trained_vs_init",
                                "diag_block_rel_move")}
    hdr = ["gate_mean", "own_segment_win_frac", "offdiag_blocks_frozen_frac",
           "offdiag_block_rel_move", "col_cos_mean_trained", "col_cos_min_trained",
           "col_cos_trained_vs_init", "diag_block_rel_move"]
    print("%-22s" % "arm" + "".join("%16s" % h for h in hdr))
    for k, v in agg.items():
        print("%-22s" % k + "".join("%16.5g" % v[h] for h in hdr))

    with open(os.path.join(_here, args.out), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "agg": agg}, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
