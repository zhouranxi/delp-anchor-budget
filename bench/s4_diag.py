"""S4 diagnostic: does the trained model's OUTPUT actually depend on ctx?

Trains one full model (b in {0,10}, mnist, seed 0), then for test samples
measures:
  1. pred(ctx=cc) vs pred(ctx=other): fraction of samples whose argmax
     prediction CHANGES when the context one-hot is swapped;
  2. dend gate response: mean sigmoid(d) under own vs other ctx;
  3. dend.weight column similarity (own-task col vs other-task col).
If (1) is ~0, the rho-insensitivity is REAL (gating is context-degenerate),
not a noise-injection bug.

FIX 2026-10-27 (data-consistency audit): lambda is now actually applied to
the anchor penalty (`loss = loss + lam * pen`; the original omitted the
factor, so every diagnostic ran with an effectively lambda=1 penalty while
the record claimed lam=1000). nh is now explicit (default 256 = exp4's NH)
and both are reported in the output.
"""

import argparse, json, os, sys
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, _here)

import exp4_branch_protection as e4
import exp6_baselines as eb
from exp6_baselines import get_perms, eval_model, DEV, EPOCHS, BS, NTRAIN, LR
from exp4_branch_protection import permute
from branch_delp import (ADNet, BiasADNet, make_anchor, anchor_penalty,
                         flat_params)
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = e4.NH          # 256 -- the real default the dendritic arms use


def train_full(ds, seed, own_bias, nh=NH, lam=LAM):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, NTASK)
    if own_bias:
        model = BiasADNet(NTASK, NTASK, own_bias=own_bias,
                          bias_mode="fixed", nh=nh).to(DEV)
    else:
        model = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=nh).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors = []
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
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen   # FIX 2026-10-27: apply lambda
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(make_anchor(model, xs, phase, NTASK, "neuron"))
    return model, xte, yte, perms


def diag(ds, seed, own_bias, nh=NH):
    model, xte, yte, perms = train_full(ds, seed, own_bias, nh=nh)
    model.eval()

    flip_rates, gate_own, gate_other = [], [], []
    with torch.no_grad():
        for cc in range(NTASK):
            xt = permute(xte[:2000], perms[cc]).to(DEV)
            ctx_own = F.one_hot(torch.full((2000,), cc, device=DEV),
                                NTASK).float()
            out_own, _ = model(xt, ctx_own)
            pred_own = out_own.argmax(1)
            # try ALL other contexts
            for other in range(NTASK):
                if other == cc:
                    continue
                ctx_o = F.one_hot(torch.full((2000,), other, device=DEV),
                                  NTASK).float()
                out_o, _ = model(xt, ctx_o)
                pred_o = out_o.argmax(1)
                flip_rates.append((pred_o != pred_own).float().mean().item())
                # gate magnitude under each ctx
                t = model.ff(xt)
                u_own = model.dend(ctx_own).view(-1, model.nseg, model.nh)
                u_o = model.dend(ctx_o).view(-1, model.nseg, model.nh)
                d_own, _ = u_own.max(dim=1)
                d_o, _ = u_o.max(dim=1)
                gate_own.append(torch.sigmoid(d_own).mean().item())
                gate_other.append(torch.sigmoid(d_o).mean().item())

    # dend column similarity: task cc column vs task other column
    W = model.dend.weight.detach().view(model.nseg, model.nh, NTASK)
    sims = []
    for cc in range(NTASK):
        for other in range(NTASK):
            if other == cc:
                continue
            a, b = W[..., cc].flatten(), W[..., other].flatten()
            cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            sims.append(cos)

    import statistics as st
    return {
        "ds": ds, "seed": seed, "bias": own_bias,
        "pred_flip_rate_mean": st.mean(flip_rates),
        "pred_flip_rate_max": max(flip_rates),
        "gate_mean_own": st.mean(gate_own),
        "gate_mean_other": st.mean(gate_other),
        "dend_col_cosine_mean": st.mean(sims),
        "dend_col_cosine_min": min(sims),
        "nh": nh,
        "lam": LAM,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="mnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nh", type=int, default=NH)
    args = ap.parse_args()
    for b in (0.0, 10.0):
        r = diag(args.ds, args.seed, b, nh=args.nh)
        print(json.dumps(r, indent=1), flush=True)
