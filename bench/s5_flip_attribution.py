"""Are the mechanism section's three observations consequences of TRAINING,
or of the INITIALIZATION?

The paper's mechanism section reports, for fully trained networks:
  (i)   prediction flip rate under context swap: 14.8-30.1% (base)
  (ii)  gate value constant at 0.565 across contexts
  (iii) dendritic columns mutually orthogonal (mean cosine 0.003)

and reads all three as properties of a trained network.  This script
measures the SAME three quantities in three states:
  fresh      -- never trained
  protected  -- the paper's training (per-task gradient mask on)
  unmasked   -- identical training, gradient mask OFF (control)

If (i)-(iii) already hold in `fresh`, then the mechanism section has
attributed an architectural/initialization property to training.
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from exp6_baselines import get_perms, DEV
from exp4_branch_protection import permute
from branch_delp import ADNet
from anchor_budget import load_data
from s5_gate_attribution import train, NTASK, NH


def flip_rates(model, xte, perms, n=1500):
    model.eval()
    dev = model.dend.weight.device
    rates = []
    with torch.no_grad():
        for cc in range(NTASK):
            xt = xte[:n].to(dev)
            xt = xt[:, perms[cc]]
            ctx_own = F.one_hot(torch.full((n,), cc, device=dev), NTASK).float()
            out_own, _ = model(xt, ctx_own)
            pred_own = out_own.argmax(1)
            for other in range(NTASK):
                if other == cc:
                    continue
                ctx_o = F.one_hot(torch.full((n,), other, device=dev), NTASK).float()
                out_o, _ = model(xt, ctx_o)
                rates.append(float((out_o.argmax(1) != pred_own).float().mean()))
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mnist,fashion")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results_s5/flip_attribution.json")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets.split(","):
        for seed in [int(x) for x in args.seeds.split(",")]:
            xtr, ytr, xte, yte = load_data(ds)
            perms = get_perms(seed, NTASK)

            torch.manual_seed(seed)
            fresh = ADNet(protected=True, nseg=NTASK, ntask=NTASK, nh=NH).to(DEV)
            fr = flip_rates(fresh, xte, perms)
            rows.append({"ds": ds, "seed": seed, "arm": "fresh",
                         "flip_mean": sum(fr) / len(fr),
                         "flip_max": max(fr),
                         "flip_min": min(fr)})
            print("  %-8s s%d fresh      flip mean=%.4f max=%.4f"
                  % (ds, seed, rows[-1]["flip_mean"], rows[-1]["flip_max"]), flush=True)

            for masked in (True, False):
                model, W0, perms2, xte2 = train(ds, seed, 0.0, masked)
                fr = flip_rates(model, xte2, perms2)
                rows.append({"ds": ds, "seed": seed,
                             "arm": "protected" if masked else "unmasked",
                             "flip_mean": sum(fr) / len(fr),
                             "flip_max": max(fr),
                             "flip_min": min(fr)})
                print("  %-8s s%d %-10s flip mean=%.4f max=%.4f"
                      % (ds, seed, rows[-1]["arm"], rows[-1]["flip_mean"],
                         rows[-1]["flip_max"]), flush=True)

    agg = {}
    for ds in args.datasets.split(","):
        for arm in ("fresh", "protected", "unmasked"):
            sel = [r for r in rows if r["ds"] == ds and r["arm"] == arm]
            if sel:
                agg[f"{ds}/{arm}"] = {
                    "flip_mean": sum(r["flip_mean"] for r in sel) / len(sel),
                    "flip_max": sum(r["flip_max"] for r in sel) / len(sel)}
    print("\n=== aggregate ===")
    for k, v in agg.items():
        print("  %-22s flip_mean=%.4f  flip_max=%.4f" % (k, v["flip_mean"], v["flip_max"]))

    with open(os.path.join(_here, args.out), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "agg": agg}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
