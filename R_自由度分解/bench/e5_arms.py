# -*- coding: utf-8 -*-
"""e5_arms.py -- E5: damping renormalization & mortality-targeted bias.

E5 (主线 B · 机制驱动): from "deletion" to "management".
   F7/F10 show the bias's gain comes from saturating a never-used gate's
   constant damping to ~1 (soft gate deletion); hard deletion recovers 76%
   on MNIST.  Is there a better way that keeps information and only removes
   damping?

Six arms (A0-A5), strictly matched budget & parameter count, on the same
protocol as S1/S3 (T=10/20, lam=1000/5000, permuted MNIST/Fashion):

  A0 base              original network (ADNet, protected)
  A1 bias              constant own-segment bias (BiasADNet fixed mode)
  A2 pin               gate pinned to its measured constant 0.565
                       (h = kwta(t * 0.565))   -- F10 control
  A3 delete            segment gate deleted
                       (h = kwta(t * 1.0), i.e. no gate at all)
  A4 renormalize       explicit removal of the constant factor: keep the
                       relative ranking and context input
                       (h = kwta(t / 0.565))   -- rescale arm
  A5 mortality-targeted bias
                       apply bias ONLY to neurons meeting the mortality
                       criterion (post-kWTA never active / gate row == 0)

Key falsifiable prediction: A4/A5, if they keep ell while eliminating the
BWT cost (which A1 cannot, per F4/F14), give a "positive method with
boundary conditions".

Risk control: no-gate arm's anchor-weighting sensitivity is +/-3.9 (F10),
larger than the repair-deletion residual under test, so A4/A5 must be
compared at matched anchor weighting, with anchor weighting as a second
factor (2 levels) explicitly scanned.

Architectures: {MLP, GN-MLP, CNN} x {mnist, fashion} x T in {10,20} x n=10.

Usage (server CPU venv):
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
    python e5_arms.py --task-id 0 --n-shards 8 --seeds 0,1,2,3,4 \
      --datasets mnist,fashion --arch mlp --ntasks 10 --out e5_mlp_s0.json
Smoke:  python e5_arms.py --smoke
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR, flat_params)
from exp4_branch_protection import ADNet, permute, kwta
import branch_delp as _bd
from branch_delp import make_anchor, anchor_penalty
from anchor_budget import load_data

LAM = 1000.0
GATE_CONST = 0.565     # measured constant gate value (F10)

# ------------------------------------------------------------------ robust
# fisher_raw skips parameters whose gradient is None.  pin/delete/renorm
# arms compute h WITHOUT the gate factor, so dend weights (and, for GN,
# the GN layer) have no gradient and would crash `v.grad.detach()`.
# We patch branch_delp.fisher_raw in place (idempotent) to guard against
# this.  This is a module-level patch that affects all arches uniformly.

def _patched_fisher_raw(model, xs, task_id, ntask):
    import exp6_baselines as _eb
    model.eval()
    # CNN trunk: exp6_baselines.flat_params expects model.ff.weight/bias
    # (Linear); ConvADNet.ff is a ConvStack.  Route to cnn_delp.flat_params.
    if hasattr(model, "ff") and hasattr(model.ff, "c1"):
        import cnn_delp as _cnn
        P = _cnn.flat_params(model)
    else:
        P = flat_params(model)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    for i in range(_eb.NFISHER):
        xb = xs[i:i + 1].to(DEV)
        out, _ = model(xb, ctx1)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            if v.grad is not None:
                fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= _eb.NFISHER
    return fisher

_bd.fisher_raw = _patched_fisher_raw
# GN-MLP support: branch_delp.usage_stats computes t = model.ff(xb), which
# drops GNFFNet's GroupNorm.  Patch to a GN-aware version (delegates to the
# original for models without .gn).  make_anchor reads usage_stats through
# the module global, so the patch is visible there.
_orig_bd_usage_stats = _bd.usage_stats
def _bd_usage_stats_dispatch(model, xs, task_id, ntask):
    if hasattr(model, "gn"):
        return _gn_usage_stats(model, xs, task_id, ntask)
    return _orig_bd_usage_stats(model, xs, task_id, ntask)
_bd.usage_stats = _bd_usage_stats_dispatch

def _gn_usage_stats(model, xs, task_id, ntask):
    """usage_stats replica with t = model.gn(model.ff(xb)) for GN-MLP."""
    import exp6_baselines as _eb
    model.eval()
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    ns, nh = model.nseg, model.nh
    o = torch.zeros(nh, device=DEV)
    o_seg = torch.zeros(ns, nh, device=DEV)
    with torch.no_grad():
        for i in range(0, _eb.NUSAGE, 250):
            xb = xs[i:i + 250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.gn(model.ff(xb))
            u = model.dend(ctx).view(-1, ns, nh)
            d, seg = u.max(dim=1)
            h = kwta(t * torch.sigmoid(d))
            act = (h > 0).float()
            o += act.sum(0)
            for k in range(ns):
                o_seg[k] += (act * (seg == k).float()).sum(0)
    return o / _eb.NUSAGE, o_seg / _eb.NUSAGE

# make_anchor imports fisher_raw into its own namespace -- but since we
# call make_anchor through _bd.make_anchor, and make_anchor references
# the module-global fisher_raw, the patch is visible.
# CNN support: branch_delp.flat_params (from exp6_baselines) expects a
# Linear trunk.  Patch it to route ConvStack models to cnn_delp.flat_params,
# which produces anchor-compatible ff.* keys (fisher["ff.c4.weight"] etc.
# are handled inside cnn's own make_anchor path; here we only need the
# param serialisation for theta/penalty).
_orig_bd_flat = _bd.flat_params
def _bd_flat_params_dispatch(model):
    if hasattr(model, "ff") and hasattr(model.ff, "c1"):
        import cnn_delp as _cnn
        return _cnn.flat_params(model)
    return _orig_bd_flat(model)
_bd.flat_params = _bd_flat_params_dispatch
# usage_stats also touches the trunk only through model.ff(xb) (ConvStack
# accepts flat input) and model.dend -- dimension-agnostic, OK for CNN.
# make_anchor's neuron masking writes fisher["ff.w"]/["ff.b"]/["out.w"],
# keys that do NOT exist in cnn_delp.flat_params (it has ff.c1..ff.c4) and
# that mismatch the conv shapes.  Route CNN models to cnn's own
# make_anchor_cnn, which mirrors branch_delp.make_anchor("neuron") with
# conv-shaped keys and usage masking (fisher["ff.c4.weight"]).
_orig_bd_make_anchor = _bd.make_anchor
def _bd_make_anchor_dispatch(model, xs, task_id, ntask, gran):
    if hasattr(model, "ff") and hasattr(model.ff, "c1"):
        import cnn_delp as _cnn
        return _cnn.make_anchor_cnn(model, xs, task_id, ntask)
    return _orig_bd_make_anchor(model, xs, task_id, ntask, gran)
_bd.make_anchor = _bd_make_anchor_dispatch
ARMS = ["base", "bias", "pin", "delete", "renorm", "mtbias", "gscale"]


# --------------------------------------------------------------- model arms
class PinADNet(ADNet):
    """A2: gate pinned to the measured constant 0.565.

    h = kwta(t * 0.565).  The dendritic input is still computed (so the
    context pathway exists and is protected), but the sigmoid gate is
    replaced by the constant -- this is the F10 control.
    """
    def forward(self, x, ctx):
        t = self.ff(x)
        h = kwta(t * GATE_CONST)
        return self.out(h), None


class DeleteADNet(ADNet):
    """A3: segment gate deleted (no gate at all).

    h = kwta(t * 1.0).  The dendritic input is not used in the forward
    pass at all; the context pathway is architecturally absent.
    """
    def forward(self, x, ctx):
        t = self.ff(x)
        h = kwta(t)
        return self.out(h), None


class RenormADNet(ADNet):
    """A4: renormalize -- explicitly remove the constant damping factor.

    h = kwta(t / GATE_CONST).  Keeps the relative ranking and the context
    input (the gate is still computed for gradient purposes), but removes
    the constant 0.565 damping by dividing the trunk signal.
    """
    def forward(self, x, ctx):
        t = self.ff(x)
        h = kwta(t / GATE_CONST)
        return self.out(h), None


class GscaleADNet(ADNet):
    """A6 (E5c): keep the gate, rescale by the measured constant.

    h = kwta(t * sigmoid(d) / GATE_CONST).  Identical to base EXCEPT the
    overall 1/0.565 = 1.77x rescale of the gated signal.  Separates:
      gscale ≈ renorm  ->  renorm's gain = pure amplitude (un-damping),
                          gate removal is incidental;
      gscale ≈ base    ->  renorm's gain = gate removal itself.
    """
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d) / GATE_CONST)
        return self.out(h), seg_id


class MtBiasADNet(ADNet):
    """A5: mortality-targeted bias.

    Applies a constant bias to the OWN segment of neurons that meet the
    mortality criterion (post-kWTA never active / gate row == 0).  The
    mortality mask is computed from the usage statistics at anchor-creation
    time and fixed for the run.
    """
    def __init__(self, ntask, nseg, nh, bias=10.0, mort_mask=None):
        super().__init__(nh=nh, protected=True, nseg=nseg, ntask=ntask)
        self.bias = bias
        # mort_mask: (nh,) bool, True for dead neurons (fixed per run)
        self.mort_mask = mort_mask

    def set_mort_mask(self, mask):
        self.mort_mask = mask

    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.bias != 0.0 and self.mort_mask is not None:
            tid = ctx.argmax(1)                       # (batch,)
            bmask = torch.zeros_like(u)
            # apply bias to own segment ONLY on dead neurons
            bmask[:, tid] = self.bias * self.mort_mask.unsqueeze(0).float()
            u = u + bmask
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id


def build_model(arm, ntask, seed, arch, nh, bias=10.0):
    """Build the requested arm's model. arch: 'mlp'|'gnmlp'|'cnn'."""
    torch.manual_seed(seed)
    if arch == "mlp":
        if arm == "bias":
            from branch_delp import BiasADNet
            return BiasADNet(ntask, ntask, own_bias=bias, bias_mode="fixed",
                             nh=nh).to(DEV)
        if arm == "pin":
            return PinADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
        if arm == "delete":
            return DeleteADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
        if arm == "renorm":
            return RenormADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
        if arm == "gscale":
            return GscaleADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
        if arm == "mtbias":
            return MtBiasADNet(ntask, ntask, nh, bias=bias).to(DEV)
        return ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    elif arch == "gnmlp":
        import s5b_gnmlp as g
        if arm == "bias":
            return g.GNFFNet(ntask, ntask, nh=nh, own_bias=bias).to(DEV)
        if arm == "base":
            return g.GNFFNet(ntask, ntask, nh=nh, own_bias=0.0).to(DEV)
        base = g.GNFFNet(ntask, ntask, nh=nh, own_bias=0.0).to(DEV)
        if arm == "mtbias":
            # A5 on GN-MLP: mortality-targeted own-bias.  GNFFNet carries
            # own_bias/nseg/nh, so the CNN wrapper applies unchanged.
            # (Previously mtbias fell through to _attach_gate_variant's else
            #  branch and silently duplicated renorm -- fixed 2026-11-08.)
            base.own_bias = bias
            base.mort_mask = None
            _attach_mtbias_forward(base)
            return base
        # pin/delete/renorm/gscale: wrap the GN trunk with a modified gate factor
        _attach_gate_variant(base, arm, ntask)
        return base
    elif arch == "cnn":
        import cnn_delp as c
        base = c.ConvADNet(ntask, nseg=ntask, nh=nh)
        base.to(DEV)
        if arm in ("pin", "delete", "renorm", "gscale"):
            _attach_gate_variant(base, arm, ntask)
        # A1 bias / A5 mtbias: apply constant own-bias on the CNN trunk
        elif arm == "bias":
            base.own_bias = bias
            _attach_ownbias_forward(base)
        elif arm == "mtbias":
            base.own_bias = bias
            base.mort_mask = None
            _attach_mtbias_forward(base)
        return base
    raise ValueError(f"unknown arch {arch}")


def _trunk(model, x):
    """Trunk features with architecture-correct norm application.
    GN-MLP's GNFFNet computes t = self.gn(self.ff(x)); naive t = self.ff(x)
    in the wrappers below silently DROPPED the GroupNorm for the variant
    arms (2026-11-09 audit fix).  CNN's ConvStack has GN inside self.ff,
    plain MLP has no GN -- both unaffected."""
    t = model.ff(x)
    return model.gn(t) if hasattr(model, "gn") else t


def _attach_gate_variant(model, arm, ntask):
    """Attach a pin/delete/renorm forward to a GN-MLP or CNN model.

    These architectures compute t = trunk(x), u = dend(ctx), then
    h = kwta(t * factor).  pin/delete/renorm change ONLY the factor:
      pin    -> t * GATE_CONST     (gate pinned to measured constant)
      delete -> t                  (no gate)
      renorm -> t / GATE_CONST     (remove the constant damping)
    We monkey-patch forward; dend/kWTA/protect_grads are otherwise intact.
    """
    def patched(self, x, ctx):
        t = _trunk(self, x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg_id = u.max(dim=1)
        if arm == "pin":
            h = kwta(t * GATE_CONST)
        elif arm == "delete":
            h = kwta(t)
        elif arm == "gscale":
            h = kwta(t * torch.sigmoid(d) / GATE_CONST)
        else:
            h = kwta(t / GATE_CONST)
        return self.out(h), seg_id
    model.forward = patched.__get__(model, type(model))
    model._gate_variant = arm


def _attach_ownbias_forward(model):
    """CNN + constant own-segment bias (A1)."""
    def patched(self, x, ctx):
        t = _trunk(self, x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        tid = ctx.argmax(1)
        bias = torch.zeros_like(u)
        bias[torch.arange(u.size(0), device=u.device), tid] = self.own_bias
        u = u + bias
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id
    model.forward = patched.__get__(model, type(model))


def _attach_mtbias_forward(model):
    """CNN + mortality-targeted bias (A5). mort_mask set by run_one."""
    def patched(self, x, ctx):
        t = _trunk(self, x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        if self.own_bias != 0.0 and getattr(self, "mort_mask", None) is not None:
            tid = ctx.argmax(1)
            bmask = torch.zeros_like(u)
            bmask[:, tid] = self.own_bias * self.mort_mask.unsqueeze(0).float()
            u = u + bmask
        d, seg_id = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id
    model.forward = patched.__get__(model, type(model))
    if not hasattr(model, "mort_mask"):
        model.mort_mask = None


def usage_dead_mask(model, xs, task_id, ntask, n_samples=500):
    """Compute the mortality mask (post-kWTA never active) over usage samples."""
    model.eval()
    dev = model.dend.weight.device
    ctx1 = F.one_hot(torch.tensor([task_id], device=dev), ntask).float()
    o = torch.zeros(model.nh, device=dev)
    with torch.no_grad():
        for i in range(0, n_samples, 250):
            xb = xs[i:i + 250].to(dev)
            ctx = ctx1.expand(xb.size(0), -1)
            t = _trunk(model, xb)
            u = model.dend(ctx).view(-1, model.nseg, model.nh)
            d, seg = u.max(dim=1)
            h = kwta(t * torch.sigmoid(d))
            o += (h > 0).float().sum(0)
    return (o <= 0)     # (nh,) bool: True = dead (never active)


# --------------------------------------------------------------- one run
def run_one(seed, ds, ntask, arm, arch, lam=LAM, bias=10.0):
    xtr, ytr, xte, yte = load_data(ds)
    perms = get_perms(seed, ntask)
    nh = 256 if arch == "mlp" else (2000 if arch == "gnmlp" else 2048)
    model = build_model(arm, ntask, seed, arch, nh, bias=bias)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    mort_mask = None
    gate_stats = []

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
                    P = _bd_flat_params_dispatch(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + _bd.anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen
                loss.backward()
                if hasattr(model, "protect_grads"):
                    model.protect_grads(phase)
                opt.step()
        # A5: compute mortality mask at first anchor, then fix it
        if arm == "mtbias" and mort_mask is None:
            mort_mask = usage_dead_mask(model, xs, phase, ntask)
            if hasattr(model, "set_mort_mask"):
                model.set_mort_mask(mort_mask)
            else:
                model.mort_mask = mort_mask
        anchors.append(_bd.make_anchor(model, xs, phase, ntask, "neuron"))
        # E5b gate statistics on the base arm (light: 50 samples/phase):
        # E[sigmoid(d)], sd, E[|t|], and corr(gate, unit-usage) proxies that
        # test whether the 0.565 damping constant is plain-MLP-specific.
        if arm in ("base", "gscale") and arch in ("mlp", "gnmlp", "cnn"):
            model.eval()
            with torch.no_grad():
                xb = xs[:50].to(DEV)
                ctx1 = F.one_hot(torch.full((50,), phase, device=DEV),
                                 ntask).float()
                t = _trunk(model, xb)
                u = model.dend(ctx1).view(-1, model.nseg, model.nh)
                d, _ = u.max(dim=1)
                g = torch.sigmoid(d)
                rec_gate = {
                    "gate_mean": round(float(g.mean()), 4),
                    "gate_std": round(float(g.std()), 4),
                    "gate_q10": round(float(g.quantile(0.10)), 4),
                    "gate_q90": round(float(g.quantile(0.90)), 4),
                    "t_absmean": round(float(t.abs().mean()), 4),
                }
                gate_stats.append(rec_gate)
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)

    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    rec = {"ds": ds, "seed": seed, "ntask": ntask, "arm": arm, "arch": arch,
           "lam": lam, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
           "ell": round(ell, 2),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                          for p in range(ntask)]}
    if arm == "mtbias" and mort_mask is not None:
        rec["dead_frac"] = round(float(mort_mask.float().mean()), 4)
    if gate_stats:
        rec["gate_stats"] = gate_stats
    return rec


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ntasks", default="10")
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "gnmlp", "cnn"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--out", default="e5_result.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    datasets = args.datasets.split(",")
    arms = args.arms.split(",")
    ntasks = [int(x) for x in args.ntasks.split(",")]

    jobs = [(ds, s) for ds in datasets for s in seeds]
    total = len(jobs)
    shard = [j for i, j in enumerate(jobs) if i % args.n_shards == args.task_id]
    print(f"shard {args.task_id}/{args.n_shards}: {len(shard)} jobs of {total}",
          flush=True)

    results = []
    t0 = time.time()
    for ds, seed in shard:
        for ntask in ntasks:
            for arm in arms:
                if args.smoke and len(results) >= 2:
                    break
                r = run_one(seed, ds, ntask, arm, args.arch, lam=args.lam)
                results.append(r)
                print(f"  [{ds:<7} s{seed:<2} T{ntask:<3} {args.arch:<6} "
                      f"{arm:<8}] ell={r['ell']:6.2f} avg={r['final_avg']:6.2f} "
                      f"bwt={r['bwt']:6.2f}"
                      + (f" dead={r['dead_frac']:.3f}" if "dead_frac" in r else ""),
                      flush=True)
        if args.smoke:
            break

    elapsed = time.time() - t0
    out = {"results": results, "elapsed": round(elapsed, 1),
           "shard": args.task_id, "n_shards": args.n_shards,
           "config": {"seeds": seeds, "datasets": datasets, "arms": arms,
                      "ntasks": ntasks, "arch": args.arch, "lam": args.lam}}
    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out} ({len(results)} runs, {elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
