# -*- coding: utf-8 -*-
"""r4_transformer.py -- R4: does the anchor-budget DIRECTION freedom
transfer to a Transformer+gated backbone?

Preregistration: bench/results_r4/R4_预注册.md (ratified 2026-11-12 09:02,
before any R4 data).  Three-stage flow, all in this one script:

  gprobe  -- G probe: base arm, lam=0, 2ds x 3 seeds; freezes G = E[sigmoid(d)]
             (E9 precedent: G_CONST=0.5749 was measured the same way)
  scan    -- lambda calibration: lam in {0,.1,1,10,100,1000} x {base,renorm}
             x 2ds x 5 seeds @ T=20;  lam* = argmax del ell(renorm-base);
             full curve is DISCLOSED (R3 lesson: single-point calibration
             can land on any point of a non-monotonic dose curve)
  main    -- 5 arms x 2ds x 20 seeds @ lam*, T=20 (default grid)

Everything inherits verbatim unless listed in the prereg:
  * task/optimiser protocol: exp6_baselines via r2_recency (Adam 1e-3,
    2 ep/task, BS 256, NTRAIN 10000, get_perms(seed,T), oracle eval)
  * intervention: r2_recency.recency_weights (IMPORTED, not reimplemented;
    prereg G2 forbids rewriting it)
  * gate: E9 isomorph (dend -> max_seg -> sigmoid; base kwta(t*sig(d),128),
    renorm kwta(t/G,128))
  * anchors: E9 per-parameter diagonal Fisher (50 draws, ctx0 one-hot,
    global mean normalisation), covering all nn.Linear + dend, excluding
    head and positional embedding (prereg D3)
  * NO gradient clipping (R2 MLP protocol), no dropout.

Usage (server GPU venv, shared cards 2/3 only):
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python r4_transformer.py \
      --mode gprobe --out results_r4/r4_gprobe.json
  CUDA_VISIBLE_DEVICES=2 ... --mode scan --g <G> --seeds 0,1,2,3,4 \
      --shard 0 --n-shards 2 --out results_r4/r4_scan.part0.json
  CUDA_VISIBLE_DEVICES=2 ... --mode main --lam <lam*> --g <G> \
      --shard 0 --n-shards 2 --out results_r4/r4_main.part0.json
"""
import argparse, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

from exp6_baselines import get_perms, eval_model, finish, DEV, EPOCHS, BS, NTRAIN, LR
from exp4_branch_protection import permute, kwta
from anchor_budget import load_data
from r2_recency import recency_weights     # prereg G2: verbatim reuse

# ---------------------------------------------------------------- constants
LAMS_SCAN = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
NFISHER = 50            # E9 draws
K_KEEP = 128            # 25% of 512 (E9)
NH = 512                # pooled feature width (E9)
D_MODEL, N_HEAD, N_LAYERS, FF = 256, 4, 2, 1024   # prereg D1 = A
SEQ, PIN = 28, 28       # 28 row-tokens of width 28
G_DEFAULT = 0.5         # gprobe placeholder; scan/main use frozen G

# arm -> (gate, alpha, reverse)
ARM_SPEC = {
    "base":         dict(gate="base",   alpha=0.0, reverse=False),
    "renorm":       dict(gate="renorm", alpha=0.0, reverse=False),
    "base_rec10":   dict(gate="base",   alpha=1.0, reverse=False),
    "renorm_rec10": dict(gate="renorm", alpha=1.0, reverse=False),
    "base_anti10":  dict(gate="base",   alpha=1.0, reverse=True),
}
DEFAULT_ARMS = list(ARM_SPEC)


# ---------------------------------------------------------------- model
class GatedTransformer(nn.Module):
    """Row-token Transformer -> pool -> 512-d feature -> dend gate -> head.

    Gate is E9's isomorph (e9_resnet.GatedResNet.forward); everything else
    is the minimal attention parameterisation fixed by prereg D1.
    """

    def __init__(self, ntask, arm="base", g=G_DEFAULT):
        super().__init__()
        self.input_proj = nn.Linear(PIN, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FF,
            activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.pool_proj = nn.Linear(D_MODEL, NH)
        self.dend = nn.Linear(ntask, ntask * NH)
        self.head = nn.Linear(NH, 10)
        self.nseg, self.nh, self.k, self.arm, self.g = ntask, NH, K_KEEP, arm, g
        self.gate_stats = []

    def features(self, x):
        b = x.size(0)
        s = self.input_proj(x.view(b, SEQ, PIN)) + self.pos
        s = self.enc(s)
        return self.pool_proj(s.mean(dim=1))          # (B, 512)

    def forward(self, x, ctx):
        t = self.features(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d = u.max(dim=1).values
        gsig = torch.sigmoid(d)
        if self.arm == "base":
            h = kwta(t * gsig, self.k)
        elif self.arm == "renorm":
            h = kwta(t / self.g, self.k)
        else:
            raise ValueError(self.arm)
        return self.head(h), None


# ---------------------------------------------------------------- anchors (E9 verbatim)
def anchor_keys(model):
    # prereg D3: all nn.Linear (+dend) covered; head and positional
    # embedding excluded.  `pos` is a bare Parameter (not a Linear).
    return [k for k, _ in model.named_parameters()
            if not k.startswith("head.") and not k.startswith("pos")]


def fisher_anchor(model, xfx, idx):
    model.eval()
    keys = anchor_keys(model)
    P = {k: v for k, v in dict(model.named_parameters()).items() if k in keys}
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    for i in range(NFISHER):
        xb = xfx[idx[i % len(idx)]].unsqueeze(0).to(DEV)
        ctx0 = F.one_hot(torch.zeros(1, dtype=torch.long, device=DEV),
                         model.nseg).float()   # one-hot: zeros would zero dend Fisher
        out = model(xb, ctx0)[0]
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            if v.grad is not None:
                fisher[k] += v.grad.detach() ** 2
    fisher = {k: f for k, f in fisher.items() if f.sum() > 0}
    n = sum(f.numel() for f in fisher.values())
    scale = sum(f.sum() for f in fisher.values()) / max(n, 1)
    for k in fisher:
        fisher[k] = fisher[k] / (scale + 1e-12)
    return fisher


def probe(model, xs, tid):
    with torch.no_grad():
        t = model.features(xs)
        ctx = F.one_hot(torch.full((xs.size(0),), tid, device=DEV),
                        model.nseg).float()
        u = model.dend(ctx).view(-1, model.nseg, model.nh)
        d = u.max(dim=1).values
        g = torch.sigmoid(d)
        return {"gate_mean": round(g.mean().item(), 6),
                "gate_std": round(g.std().item(), 6),
                "gate_q10": round(g.quantile(0.10).item(), 6),
                "gate_q90": round(g.quantile(0.90).item(), 6),
                "t_absmean": round(t.abs().mean().item(), 6)}


# ---------------------------------------------------------------- one run
def build_model(gate, ntask, seed, g):
    torch.manual_seed(seed)
    return GatedTransformer(ntask, arm=gate, g=g).to(DEV)


def run_one(ds, seed, ntask, arm, lam, g):
    spec = ARM_SPEC[arm]
    alpha = spec["alpha"]
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, ntask)
    model = build_model(spec["gate"], ntask, seed, g)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    phase_weights, weight_sums = [], []

    for phase in range(ntask):
        xs = permute(xtr[:NTRAIN], perms[phase])
        ys = ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        # weights are a function of the PHASE index (phase v holds v
        # historical anchors), NOT of len(anchors): lam=0 runs (gprobe)
        # build no anchors but the record must still satisfy G1's
        # sum(w)==v identity.  R2 could use len(anchors) only because
        # every R2 run has lam=1000>0.
        w = recency_weights(phase, alpha, ntask, spec["reverse"])
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
                if anchors and lam > 0:
                    P = dict(model.named_parameters())
                    pen = 0.0
                    for j, a in enumerate(anchors):
                        for k in a["fisher"]:            # anchored keys only
                            v = P[k]
                            pen = pen + w[j] * (a["fisher"][k]
                                                * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen              # MUST carry lam (G4)
                loss.backward()
                opt.step()
        if arm == "base":                # sigma(d) readout lives on the base gate
            model.gate_stats.append(probe(model, xs[:250].to(DEV), phase))
        if lam > 0:
            fx_idx = list(range(min(NFISHER * 2, xs.size(0))))
            fisher = fisher_anchor(model, xs, fx_idx)
            theta = {k: v.detach().clone()
                     for k, v in dict(model.named_parameters()).items()
                     if k in fisher}
            anchors.append({"theta": theta, "fisher": fisher})
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)

    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {
        "arm": arm, "gate": spec["gate"], "alpha": alpha,
        "tau": ntask / 4.0, "lam_eff": lam, "reverse": spec["reverse"],
        "ntask": ntask, "ds": ds, "seed": seed,
        "d_model": D_MODEL, "n_layers": N_LAYERS, "nhead": N_HEAD,
        "k_keep": K_KEEP, "g_const": g,
        "ell": round(ell, 4), "avg": round(favg, 4), "bwt": round(bwt, 4),
        "acc_matrix": [[round(acc[(p, c)], 4) for c in range(p + 1)]
                       for p in range(ntask)],
        "phase_weights": phase_weights, "weight_sums": weight_sums,
        "gate_stats": model.gate_stats,
    }


# ---------------------------------------------------------------- prereg gates
def gate_checks(rec):
    """G1/G2/G4: identity + runtime-assertion gates (prereg section 6).

    G3 (alpha==0 weights identity) is asserted on recency_weights itself in
    pilot_gates(); G6 (gate band) and G7 (scan-main consistency) are
    checked at analysis time over aggregated runs.
    """
    ok = True
    for pi, (ws, s) in enumerate(zip(rec["phase_weights"], rec["weight_sums"])):
        # phase pi holds pi anchors; sum(w) must equal that count (G1)
        if abs(s - pi) > 1e-6 or any(w <= 0 for w in ws):
            ok = False                                     # G1 conservation
    if rec["alpha"] != ARM_SPEC[rec["arm"]]["alpha"]:
        ok = False                                         # G2 dispatch
    if rec["arm"].endswith("anti10"):
        if not (rec["phase_weights"][-1][0] > rec["phase_weights"][-1][-1]):
            ok = False                                     # G2 mirror
    elif rec["alpha"] > 0:
        if not (rec["phase_weights"][-1][0] < rec["phase_weights"][-1][-1]):
            ok = False
    if rec["lam_eff"] is None:
        ok = False                                         # G4 lam carried
    return ok


def pilot_gates():
    """G3: alpha=0 returns exactly [1.0]*v (constructive identity check)."""
    for v in (0, 1, 5, 10, 20):
        w0 = recency_weights(v, 0.0, 20)
        assert w0 == ([1.0] * v if v else []), f"G3 FAIL alpha=0 at v={v}"
    w = recency_weights(20, 1.0, 20)
    assert abs(sum(w) - 20) < 1e-9 and all(x > 0 for x in w), "G3 FAIL conservation"
    wa = recency_weights(20, 1.0, 20, reverse=True)
    assert wa[0] > wa[-1], "G3 FAIL anti mirror"
    assert w[0] < w[-1], "G3 FAIL rec mirror"
    print("G3 pilot gates: PASS (alpha=0 identity, conservation, mirrors)")


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gprobe", "scan", "main"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--ntask", type=int, default=20)
    ap.add_argument("--lam", type=float, default=None,
                    help="main mode: frozen lam*; scan mode: ignored")
    ap.add_argument("--g", type=float, default=None,
                    help="frozen G from gprobe (required for scan/main)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.08)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=0)
    pilot_gates()                                          # G3, every launch

    dss = args.datasets.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.mode == "gprobe":
        jobs = [("base", ds, args.ntask, s, 0.0) for ds in dss for s in seeds]
        g = G_DEFAULT
    elif args.mode == "scan":
        assert args.g is not None, "scan requires frozen --g"
        block = []
        for lam_b in LAMS_SCAN:
            for arm in ("base", "renorm"):
                for ds in dss:
                    for s in seeds:
                        block.append((arm, ds, args.ntask, s, lam_b))
        jobs = [j for i, j in enumerate(block)
                if i % args.n_shards == args.shard]
        g = args.g
    else:
        assert args.g is not None and args.lam is not None, \
            "main requires frozen --g and --lam"
        jobs = [(arm, ds, args.ntask, s, args.lam)
                for arm in args.arms.split(",")
                for ds in dss for s in seeds]
        jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
        g = args.g

    print(f"R4 {args.mode} shard {args.shard}/{args.n_shards}: "
          f"{len(jobs)} runs (DEV={DEV}, g={g})", flush=True)

    meta = {"prereg": "bench/results_r4/R4_预注册.md", "mode": args.mode,
            "shard": args.shard, "n_shards": args.n_shards, "g": g,
            "lams_scan": LAMS_SCAN, "d_model": D_MODEL,
            "n_layers": N_LAYERS, "nhead": N_HEAD, "k_keep": K_KEEP}
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results, t0 = [], time.time()
    gmeans = []
    for i, (arm, ds, T, s, run_lam) in enumerate(jobs):
        r = run_one(ds, s, T, arm, run_lam, g)
        assert gate_checks(r), f"G1/G2/G4 FAIL: {arm} {ds} s{s}"
        if r["gate_stats"]:
            gmeans += [p["gate_mean"] for p in r["gate_stats"]]
        results.append(r)
        print(f"  [{i+1:3d}/{len(jobs)} {arm:<12s} {ds:<7s} lam={run_lam:<6g} "
              f"s{s:<2d}] ell={r['ell']:6.2f} avg={r['avg']:6.2f} "
              f"bwt={r['bwt']:7.2f}", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({**meta, "elapsed": round(time.time() - t0, 1),
                       "results": results}, f)
    if args.mode == "gprobe" and gmeans:
        import statistics as st
        G = round(st.mean(gmeans), 4)
        print(f"\n>>> frozen G = {G}  (E[sigmoid(d)] over "
              f"{len(gmeans)} phase-probes)", flush=True)
        meta["frozen_G"] = G
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({**meta, "results": results}, f)
    print(f"\nwrote {args.out} ({len(results)} runs, {time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
