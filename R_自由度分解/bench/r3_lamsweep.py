# -*- coding: utf-8 -*-
"""r3_lamsweep.py -- R3: conv lambda re-calibration sweep (prereg
bench/results_r3/R3_预注册.md).

Drives e5_arms.run_one VERBATIM (arch="cnn") over a fixed log-spaced lambda
axis; the ONLY new degree of freedom is lambda.  No file in the E5 chain is
modified.

    lambda in {0, 3.16, 31.6, 316, 1000, 3160}  x  {base, renorm}
    x {mnist, fashion} x T=10 x seeds 0..9  = 240 runs

Usage:
  CUDA_VISIBLE_DEVICES=2 python r3_lamsweep.py --task-id 0 --n-shards 2 \
      --out results_r3/r3_grid.part0.json
"""
import argparse, json, os, sys, time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import e5_arms as E5          # VERBATIM reuse (run_one carries lam already)

# --- Runtime device patch (R3_修订说明.md, 2026-11-12 00:37:30) ---------
# cnn_delp.make_anchor_cnn feeds a CPU batch to a model that may live on
# CUDA.  E5's conv grids only ever ran on CPU (run_e5_arch2.sh sets
# CUDA_VISIBLE_DEVICES=''), so the latent mismatch never fired; R3 runs on
# GPU and hit it in the pilot.  Fix WITHOUT touching the E5 files: wrap the
# anchor constructor and move the batch to the model's device.  Anchor
# semantics (Fisher, usage mask, dom stats) are byte-identical; the patch
# lives only in this process.
import cnn_delp as _cnn
_orig_make_anchor_cnn = _cnn.make_anchor_cnn


def _make_anchor_cnn_devfix(model, xs, task_id, ntask):
    xs = xs.to(E5.DEV)          # CPU batch -> model device (only change)
    return _orig_make_anchor_cnn(model, xs, task_id, ntask)


_cnn.make_anchor_cnn = _make_anchor_cnn_devfix
# ------------------------------------------------------------------------

LAMS = [0.0, 3.16, 31.6, 316.0, 1000.0, 3160.0]   # prereg §2 (log axis)
ARMS = ["base", "renorm"]                          # prereg §3
NTASK = 10
SEEDS = list(range(10))


def build_jobs():
    return [(lam, arm, ds, seed)
            for lam in LAMS
            for arm in ARMS
            for ds in ("mnist", "fashion")
            for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default="r3_result.json")
    args = ap.parse_args()

    jobs = build_jobs()
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.task_id]
    print(f"R3 shard {args.task_id}/{args.n_shards}: {len(jobs)} runs "
          f"(lams={LAMS}, arms={ARMS}, T={NTASK}, DEV={E5.DEV})", flush=True)

    results = []
    t0 = time.time()
    for i, (lam, arm, ds, seed) in enumerate(jobs):
        rec = E5.run_one(seed, ds, NTASK, arm, "cnn", lam=lam)
        rec["kind"] = "r3"
        results.append(rec)
        # G5 incremental checkpoint
        json.dump({"prereg": "bench/results_r3/R3_预注册.md",
                   "shard": args.task_id, "n_shards": args.n_shards,
                   "lams": LAMS, "arms": ARMS, "ntask": NTASK,
                   "results": results},
                  open(args.out, "w"), ensure_ascii=False)
        print(f"  [{i+1}/{len(jobs)}] lam={lam:<7g} {arm:7s} {ds:8s} s{seed} "
              f"ell={rec['ell']:.2f} bwt={rec['bwt']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"WROTE {args.out} {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
