# -*- coding: utf-8 -*-
"""analyze_r4_scan.py -- R4 lambda-calibration analysis (prereg section 2).

Reads bench/results_r4/r4_scan.part*.json, computes the dose curve
d_ell(renorm vs base) per lambda, picks lam* = argmax, and discloses the
FULL curve (R3 lesson).  Also runs prereg G6 (gate band) and prints the
per-dataset split.

Usage:  python analyze_r4_scan.py [--files "results_r4/r4_scan.part*.json"]
Writes: results_r4/r4_scan_analysis.json + prints a table.
"""
import argparse, glob, json, os, sys
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "bench") and
                              os.path.join(os.path.dirname(HERE), pat)
                              if not os.path.isabs(pat) else pat)):
        d = json.load(open(f, encoding="utf-8"))
        out += d["results"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=os.path.join(
        ROOT, "bench", "results_r4", "r4_scan.part*.json"))
    args = ap.parse_args()
    runs = load(args.files)
    lams = sorted({r["lam_eff"] for r in runs})
    dss = sorted({r["ds"] for r in runs})
    print(f"{len(runs)} runs over {len(lams)} lambdas x {dss}\n")

    idx = {(r["lam_eff"], r["arm"], r["ds"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})

    curve, rows = {}, []
    for lam in lams:
        for ds in dss + ["POOLED"]:
            dl, db = [], []
            for s in seeds:
                if ds == "POOLED":
                    src = [d for d in dss]
                else:
                    src = [ds]
                for d in src:
                    try:
                        dl.append(idx[(lam, "renorm", d, s)]["ell"]
                                  - idx[(lam, "base", d, s)]["ell"])
                        db.append(idx[(lam, "renorm", d, s)]["bwt"]
                                  - idx[(lam, "base", d, s)]["bwt"])
                    except KeyError:
                        pass
            if dl:
                curve[(lam, ds)] = st.mean(dl)
                rows.append((lam, ds, len(dl), st.mean(dl), st.mean(db)))

    print(f"{'lambda':>8} {'dataset':>8} {'n':>3} {'d_ell':>8} {'d_bwt':>8}")
    for lam, ds, n, dl, db in rows:
        print(f"{lam:>8g} {ds:>8} {n:>3} {dl:>+8.3f} {db:>+8.3f}")

    pooled = {lam: curve[(lam, "POOLED")] for lam in lams
              if (lam, "POOLED") in curve}
    lam_star = max(pooled, key=pooled.get)
    print(f"\nlam* = {lam_star:g}  (argmax pooled d_ell = "
          f"{pooled[lam_star]:+.3f})")
    if pooled[lam_star] <= 0:
        lam_star = min(pooled, key=lambda l: abs(pooled[l]))
        print("ALL d_ell <= 0 -> prereg branch: lam* = argmin |d_ell| = "
              f"{lam_star:g}; main candidate switches to base_rec10 "
              "(R2T C1 logic); record this in the revision notes.")

    # G6: gate band on base arms
    gm = [p["gate_mean"] for r in runs if r["arm"] == "base"
          for p in r["gate_stats"]]
    if gm:
        m, sd = st.mean(gm), st.pstdev(gm)
        print(f"G6 gate band: mean={m:.4f} sd={sd:.4f} -> "
              f"{'PASS' if 0.50 <= m <= 0.65 and sd < 0.05 else 'FAIL'}")

    out = {"lam_star": lam_star, "curve_pooled": pooled,
           "rows": [{"lam": l, "ds": d, "n": n, "d_ell": dl, "d_bwt": db}
                    for l, d, n, dl, db in rows]}
    dest = os.path.join(ROOT, "bench", "results_r4", "r4_scan_analysis.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
