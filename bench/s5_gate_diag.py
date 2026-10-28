"""Diagnostic: which parts of dend.weight are ever updated, and what the
gate actually reads.

The paper claims (Sec. mechanism) that the dendritic context columns
"remain mutually orthogonal at their random initialization" and that the
context pathway is "unused as a modulator".  This script asks a sharper
question:

  protect_grads masks the gradient of W.view(nseg, nh, ntask) so that only
  the block [ctx_label, :, :] survives during the training of task
  ctx_label.  But the forward pass reads W @ ctx, i.e. the WHOLE column
  ctx_label -- all nseg blocks of it.  So parameters that are never
  updated still shape the gate.

  Concretely: W[s, :, t] with s != t is updated only during the training of
  task s (when the mask opens block s) -- but during task s the forward
  reads column s, not column t.  And during task t the mask opens block t
  only, so W[t, :, t] is updated but W[s, :, t] (s != t) is masked.  The
  masked entries receive zero gradient in EVERY phase.

  Therefore W[s, :, t] for s != t is frozen at its random initialization for
  the entire run, while still being multiplied into the gate for task t.

This script measures exactly that.
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from s4_diag import train_full, NTASK
from exp4_branch_protection import ADNet


def block_report(model, W0):
    W = model.dend.weight.detach()
    Wv = W.view(NTASK, model.nh, NTASK)
    W0v = W0.view(NTASK, model.nh, NTASK)
    moved = (Wv - W0v).norm(dim=1)          # (nseg, ntask) per-block L2 change
    mag0 = W0v.norm(dim=1)
    rel = (moved / mag0.clamp_min(1e-12))
    return Wv, moved, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="mnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # snapshot the initialization of the model that train_full will build
    if args.bias:
        from branch_delp import BiasADNet
        probe = BiasADNet(NTASK, NTASK, own_bias=args.bias, bias_mode="fixed")
    else:
        probe = ADNet(protected=True, nseg=NTASK, ntask=NTASK)
    W0 = probe.dend.weight.detach().clone()

    model, xte, yte, perms = train_full(args.ds, args.seed, args.bias)
    model.eval()
    Wv, moved, rel = block_report(model, W0)

    out = {"ds": args.ds, "seed": args.seed, "bias": args.bias}

    # --- 1. which (segment, task) blocks ever moved? ---
    offdiag = []
    diag = []
    for s in range(NTASK):
        for t in range(NTASK):
            (diag if s == t else offdiag).append(float(rel[s, t]))
    out["diag_block_rel_move_mean"] = sum(diag) / len(diag)
    out["offdiag_block_rel_move_mean"] = sum(offdiag) / len(offdiag)
    out["offdiag_block_rel_move_max"] = max(offdiag)
    out["offdiag_blocks_frozen_frac"] = sum(1 for x in offdiag if x < 1e-6) / len(offdiag)

    # --- 2. column geometry, as implemented vs the reviewer's object ---
    def colcos(M):
        s = []
        for c in range(NTASK):
            for o in range(NTASK):
                if o == c:
                    continue
                a, b = M[..., c].flatten(), M[..., o].flatten()
                s.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))))
        return s

    s_trained = colcos(Wv)
    s_init = colcos(W0v_of(W0))
    out["col_cos_mean_trained"] = sum(s_trained) / len(s_trained)
    out["col_cos_min_trained"] = min(s_trained)
    out["col_cos_mean_init"] = sum(s_init) / len(s_init)
    out["col_cos_min_init"] = min(s_init)

    # cosine between TRAINED column and INITIAL column (did the column move?)
    cos_move = []
    for c in range(NTASK):
        a = Wv[..., c].flatten()
        b = W0v_of(W0)[..., c].flatten()
        cos_move.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))))
    out["col_cos_trained_vs_init_mean"] = sum(cos_move) / len(cos_move)
    out["col_cos_trained_vs_init_min"] = min(cos_move)

    # --- 3. the reviewer's object: per-ROW T-dim vectors (what the review
    #        believed was being measured) ---
    rows = Wv.reshape(NTASK * model.nh, NTASK)
    rows0 = W0.reshape(NTASK * model.nh, NTASK)
    def rowcos(M, n=400):
        s = []
        sub = M[:n]
        for i in range(sub.size(0)):
            for j in range(i + 1, sub.size(0)):
                s.append(float(F.cosine_similarity(sub[i].unsqueeze(0), sub[j].unsqueeze(0))))
        return s
    r_tr = rowcos(rows)
    r_in = rowcos(rows0)
    out["rowcos_mean_trained"] = sum(r_tr) / len(r_tr)
    out["rowcos_min_trained"] = min(r_tr)
    out["rowcos_max_trained"] = max(r_tr)
    out["rowcos_mean_init"] = sum(r_in) / len(r_in)

    # --- 4. who wins the gate, and what is the gate value? ---
    wins = torch.zeros(NTASK, NTASK)      # wins[task, segment]
    gate_val = [[] for _ in range(NTASK)]
    with torch.no_grad():
        for t in range(NTASK):
            xt = torch.as_tensor(xte[:1000], dtype=torch.float32).to(model.ff.weight.device)
            xt = xt[:, perms[t]]
            ctx = F.one_hot(torch.full((xt.size(0),), t, device=xt.device),
                            NTASK).float()
            u = model.dend(ctx).view(-1, NTASK, model.nh)
            d, seg = u.max(dim=1)          # (batch, nh)
            for s in range(NTASK):
                wins[t, s] = float((seg == s).float().sum())
            gate_val[t].append(float(torch.sigmoid(d).mean()))
    out["win_matrix_task_by_segment"] = wins.tolist()
    tot = wins.sum()
    own_frac = []
    for t in range(NTASK):
        own_frac.append(float(wins[t, t] / wins[t].sum().clamp_min(1)))
    out["own_segment_win_frac_per_task"] = own_frac
    out["own_segment_win_frac_mean"] = sum(own_frac) / len(own_frac)
    out["gate_mean_per_task"] = [sum(v) / len(v) for v in gate_val]

    # --- 5. how much of the winning row's magnitude comes from the frozen
    #        off-diagonal blocks? ---
    diag_part, off_part = [], []
    for t in range(NTASK):
        d = Wv[t, :, t].norm()
        o = torch.cat([Wv[s, :, t] for s in range(NTASK) if s != t]).norm()
        diag_part.append(float(d))
        off_part.append(float(o))
    out["win_row_diag_norm_mean"] = sum(diag_part) / len(diag_part)
    out["win_row_offdiag_norm_mean"] = sum(off_part) / len(off_part)

    print(json.dumps(out, indent=1))
    with open(f"results_s5/gate_diag_{args.ds}_s{args.seed}_b{int(args.bias)}.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def W0v_of(W0):
    return W0.view(NTASK, -1, NTASK)


if __name__ == "__main__":
    main()
