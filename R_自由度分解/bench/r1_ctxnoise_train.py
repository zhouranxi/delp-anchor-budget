"""R1: TRAINING-time context noise -- does it corrupt usage stats / anchors?

Preregistration: bench/results_r1/R1_预注册.md (locked 2026-11-11 17:23 +0800,
before any R1 data).  Read that file before changing anything in here.

S4 (s4_ctxnoise.py) injected noise at EVAL only, with training ctx always
oracle.  R1 moves the noise into training.  There are THREE places ctx enters
training, checked line by line:
  1. forward            model(xb, ctx)          -> dend(ctx), and BiasADNet's
                                                  ctx.argmax(1) picks the
                                                  own-bias segment
  2. gradient mask      protect_grads(phase)    -> freezes the off-diagonal
                                                  blocks of dend.weight
  3. anchor construction make_anchor(..., task_id, ...) -> usage_stats /
                        fisher_raw build ctx from task_id and weight Fisher
                        by measured usage
main.tex:2664 says training-time errors corrupt usage statistics and anchors
"differently" from eval-time errors, so R1 separates the two chains:

  clean : all three use the true phase                     (shared control)
  fwd   : forward uses ctx_label; (2) and (3) stay true   (forward-only effect)
  full  : all three use ctx_label                         (+ usage/anchor chain)

Noise model (locked in the preregistration): per-PHASE mislabelling.  Each
phase independently draws ctx_label = phase w.p. 1-rho, or a uniformly random
OTHER task w.p. rho.  This is the engineering meaning of "a task's context /
label is wrong", and it is the only granularity at which protect_grads and
make_anchor (both take a single int) can be coherently corrupted.

Evaluation is ALWAYS oracle ctx, via eval_model, which returns the full
(phase, c) accuracy matrix.  Keeping that matrix is what makes `ell` available
-- S4 had the same matrix in memory but only wrote the final row, which is why
its preregistered Delta-ell was uncomputable (see S4 修订说明 §三).

Clean-arm equivalence with S4 is a HARD GATE (prereg §7.2): no extra torch RNG
calls are made on the clean path, so the random sequence is byte-identical to
s4_ctxnoise.run_noise at rho=0.  Label draws use an isolated random.Random.

Usage (server, GPU 2/3 shared cards):
  cd /mnt/mydisk/xinguan-dendritic/bench
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
    dendritic-lab/venv-gpu/bin/python r1_ctxnoise_train.py \
      --shard 0 --n-shards 2 --out ../results_r1/r1_grid.part0.json
"""

import argparse, json, os, random, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, _here)

import exp4_branch_protection as e4
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute
from branch_delp import (ADNet, BiasADNet, make_anchor, anchor_penalty,
                         flat_params)
from anchor_budget import load_data
from s4_ctxnoise import eval_noisy_final   # byte-identical eval-noise probe

LAM = 1000.0
NTASK = 10
SEEDS = list(range(10))
RHOS = [0.1, 0.2, 0.4]
BIASES = [0.0, 10.0]
MODES = ["clean", "fwd", "full"]
EVAL_RHOS = [0.0, 0.1, 0.2, 0.4]        # same grid as S4
NH = e4.NH                                # 256, the real ADNet default


def draw_label(phase, rho, rng, ntask):
    if rng.random() >= rho:
        return phase
    return (phase + 1 + rng.randrange(ntask - 1)) % ntask


def make_phase_labels(mode, rho, rng, ntask, rule):
    """Pre-draw the T per-phase labels BEFORE training.

    bernoulli  : the locked prereg rule (v1).  P(no mislabel) = (1-rho)^T.
    guaranteed : same draw, but if it yields zero mislabels, force ONE phase
                 (chosen by the same isolated RNG) to a random OTHER task, so
                 prereg §7.3 can be evaluated literally.  See
                 bench/results_r1/R1_修订说明.md (2026-11-11) for why v2 uses
                 this rule and for the estimand change it implies.

    The RNG call order of `bernoulli` is identical to v1's per-phase draws,
    so v1 remains bit-reproducible.  The clean path never touches this RNG.
    """
    if mode == "clean":
        return list(range(ntask))
    labels = [draw_label(p, rho, rng, ntask) for p in range(ntask)]
    if rule == "guaranteed" and all(l == p for p, l in enumerate(labels)):
        p = rng.randrange(ntask)
        labels[p] = (p + 1 + rng.randrange(ntask - 1)) % ntask
    return [int(x) for x in labels]


def run_one(ds, seed, own_bias, mode, rho, ntask=NTASK, lam=LAM, nh=NH,
            noise_rule="bernoulli"):
    """One training run.  The clean path is a literal reproduction of
    s4_ctxnoise.run_noise at rho=0 (same RNG order, same ops)."""
    torch.manual_seed(seed)
    # Isolated Python RNG: never touches the torch global generator, so the
    # clean arm stays bit-comparable with S4.
    rng = random.Random(100000 + seed * 97 + int(round(rho * 1000)) * 13
                        + (0 if own_bias == 0.0 else 1)
                        + MODES.index(mode) * 7919)

    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, ntask)

    if own_bias:
        model = BiasADNet(ntask, ntask, own_bias=own_bias,
                          bias_mode="fixed", nh=nh).to(DEV)
    else:
        model = ADNet(protected=True, nseg=ntask, ntask=ntask, nh=nh).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    phase_labels = make_phase_labels(mode, rho, rng, ntask, noise_rule)

    for phase in range(ntask):
        label = phase_labels[phase]
        mask_label = label if mode == "full" else phase
        anchor_label = label if mode == "full" else phase

        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), label, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen      # MUST carry lam (S4 §四)
                loss.backward()
                model.protect_grads(mask_label)
                opt.step()
        anchors.append(make_anchor(model, xs, anchor_label, ntask, "neuron"))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)

    # ---- metrics, all from the SAME oracle-ctx accuracy matrix ----
    ell = sum(acc[(c, c)] for c in range(ntask)) / ntask
    bwt, favg = finish(acc, ntask)

    # Bridge to S4: eval-time noise on the R1-trained model (final phase only).
    eval_favg = {}
    for er in EVAL_RHOS:
        if er == 0.0:
            final_acc = {k: v for k, v in acc.items() if k[0] == ntask - 1}
        else:
            final_acc = eval_noisy_final(model, xte, yte, perms, ntask,
                                         er, False, seed)
        eval_favg[str(er)] = round(
            sum(final_acc[(ntask - 1, c)] for c in range(ntask)) / ntask, 4)

    n_mis = sum(1 for p, l in enumerate(phase_labels) if l != p)
    return {
        "kind": "r1_train_ctxnoise",
        "dataset": ds, "seed": seed, "bias": own_bias,
        "mode": mode, "rho": rho, "ntask": ntask,
        "noise_rule": noise_rule,
        "lam": lam, "nh": nh,
        "ell": round(ell, 4), "avg": round(favg, 4), "bwt": round(bwt, 4),
        "final_avg_by_eval_rho": eval_favg,
        "phase_labels": phase_labels, "n_mislabeled": n_mis,
        "acc_matrix": {f"{p},{c}": round(v, 4)
                       for (p, c), v in sorted(acc.items())},
    }


def build_jobs(seeds, rhos, biases=None):
    b_list = BIASES if biases is None else biases
    jobs = []
    for ds in ("mnist", "fashion"):
        for b in b_list:
            for s in seeds:
                jobs.append((ds, b, s, "clean", 0.0))
                for mode in ("fwd", "full"):
                    for rho in rhos:
                        jobs.append((ds, b, s, mode, rho))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--rhos", type=str, default="0.1,0.2,0.4")
    ap.add_argument("--biases", type=str, default="0,10",
                    help="fixed own-bias levels; '3' alone runs the prereg "
                         "§3.2 conditional b=3 add-on without touching the "
                         "main grid's multiple-comparison family")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.08,
                    help="cap this process at this fraction of the visible "
                         "card (~2 GB on a 24 GB 3090); GPU 2/3 are shared "
                         "with llama-server")
    ap.add_argument("--noise-rule", choices=["bernoulli", "guaranteed"],
                    default="bernoulli",
                    help="bernoulli = locked prereg rule (v1); guaranteed = "
                         "v2 rerun, see results_r1/R1_修订说明.md")
    args = ap.parse_args()

    if torch.cuda.is_available():
        # With CUDA_VISIBLE_DEVICES=2 the visible device is index 0.
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=0)
        print(f"CUDA ok: {torch.cuda.get_device_name(0)}, cap "
              f"{args.mem_fraction:.2%}", flush=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]
    biases = [float(x) for x in args.biases.split(",") if x.strip()]
    jobs = build_jobs(seeds, rhos, biases)
    jobs = [j for i, j in enumerate(jobs)
            if i % args.n_shards == args.shard]

    print(f"R1 shard {args.shard}/{args.n_shards}: {len(jobs)} runs "
          f"(seeds={seeds}, rhos={rhos}, biases={biases}, DEV={DEV}, "
          f"lam={LAM}, nh={NH}, noise_rule={args.noise_rule})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    results, t0 = [], time.time()
    for i, (ds, b, s, mode, rho) in enumerate(jobs):
        r = run_one(ds, s, b, mode, rho, noise_rule=args.noise_rule)
        results.append(r)
        # Incremental checkpoint: a long shared-card run must not lose
        # everything if the server reboots or the card drops out.
        json.dump({"prereg": "bench/results_r1/R1_预注册.md",
                   "revision": "bench/results_r1/R1_修订说明.md",
                   "shard": args.shard, "n_shards": args.n_shards,
                   "seeds": seeds, "rhos": rhos, "biases": biases,
                   "lam": LAM, "nh": NH,
                   "ntask": NTASK, "noise_rule": args.noise_rule,
                   "results": results},
                  open(args.out, "w"), indent=1)
        print(f"[{i+1}/{len(jobs)}] ds={ds} b={b} s={s} {mode} rho={rho} "
              f"ell={r['ell']:.2f} avg={r['avg']:.2f} bwt={r['bwt']:.2f} "
              f"mis={r['n_mislabeled']} [{time.time()-t0:.0f}s]", flush=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
