"""Is the constant gate value 0.565 a property of TRAINING, or a property of
the RANDOM INITIALIZATION?

The paper's mechanism section reports that the winning segment's gate value
is 0.565 under the correct context and 0.565 under every alternative, in
both the base and the repaired networks, and calls it "a constant of the
training run".

This script tests that attribution directly:

  (A) gate value of a FRESHLY INITIALIZED, untrained model
  (B) gate value of the trained model
  (C) analytic prediction from the initialization alone:
      E[ max over segments of a standard-normal row ] -> sigmoid

If (A) == (B), the constant is a property of the initialization and the
argmax, not of training, and the mechanism section has mis-attributed it.

Also computes the analytic prediction: the winning row of a column is
max_s W[s, h, c] over nseg near-independent blocks, so
E[gate] ~ sigmoid( E[max of nseg iid N(0, sigma_W^2)] ).
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from exp4_branch_protection import ADNet
from branch_delp import BiasADNet

NTASK = 10
NH = 256


def gate_profile(model, ntask=NTASK, n=2000, x=None):
    """Mean sigmoid(max-segment gate) per context, on a random-ish input."""
    model.eval()
    dev = model.dend.weight.device
    if x is None:
        x = torch.rand(n, 784, device=dev)
    out = []
    with torch.no_grad():
        for c in range(ntask):
            ctx = F.one_hot(torch.full((n,), c, device=dev), ntask).float()
            u = model.dend(ctx).view(-1, model.nseg, model.nh)
            d, seg = u.max(dim=1)
            out.append(float(torch.sigmoid(d).mean()))
    return out


def analytic_max_of_normals(k, sigma, n_samples=200000, seed=0):
    """E[max of k iid N(0, sigma)] via Monte Carlo (vectorised in chunks)."""
    g = torch.Generator().manual_seed(seed)
    tot, cnt = 0.0, 0
    chunk = 20000
    while cnt < n_samples:
        m = min(chunk, n_samples - cnt)
        z = torch.randn(m, k, generator=g) * sigma
        tot += float(z.max(dim=1).values.sum())
        cnt += m
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    fresh = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=NH)
    W = fresh.dend.weight.detach()
    sigma = float(W.std())
    bound = 1.0 / (NTASK ** 0.5)

    g_fresh = gate_profile(fresh)
    print("=== (A) freshly INITIALIZED model, no training ===")
    print("  per-context gate value: min=%.6f max=%.6f mean=%.6f"
          % (min(g_fresh), max(g_fresh), sum(g_fresh) / len(g_fresh)))

    # (C) analytic prediction from the initialization alone
    e_max = analytic_max_of_normals(NTASK, sigma)
    print("\n=== (C) analytic prediction from initialization ===")
    print("  init sigma = %.4f (kaiming bound 1/sqrt(fan_in)=%.4f)" % (sigma, bound))
    print("  E[max of %d iid N(0,sigma)] = %.4f" % (NTASK, e_max))
    print("  sigmoid(that) = %.6f" % (1 / (1 + pow(2.718281828, -e_max))))

    # how much does the trained row differ from init? load a trained model
    trained_path = os.path.join(_here, "results_s5", "gate_diag_mnist_s0_b0.json")
    if os.path.exists(trained_path):
        d = json.load(open(trained_path, encoding="utf-8"))
        print("\n=== (B) trained model (from s5_gate_diag) ===")
        print("  gate per task: min=%.6f max=%.6f"
              % (min(d["gate_mean_per_task"]), max(d["gate_mean_per_task"])))
        print("  trained column-vs-init cosine: mean=%.6f"
              % d["col_cos_trained_vs_init_mean"])
        print("  off-diagonal blocks frozen: %.1f%%"
              % (100 * d["offdiag_blocks_frozen_frac"]))
        print("  own segment wins: %.2f%% of neuron-slots"
              % (100 * d["own_segment_win_frac_mean"]))

    out = {
        "seed": args.seed,
        "init_gate_min": min(g_fresh),
        "init_gate_max": max(g_fresh),
        "init_gate_mean": sum(g_fresh) / len(g_fresh),
        "init_sigma": sigma,
        "analytic_Emax": e_max,
        "analytic_sigmoid": 1 / (1 + pow(2.718281828, -e_max)),
    }
    with open(f"results_s5/gate_analytic_s{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results_s5/gate_analytic_s%d.json" % args.seed)


if __name__ == "__main__":
    main()
