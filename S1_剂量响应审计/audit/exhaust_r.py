#!/usr/bin/env python
"""Exhaustively search the aggregation that yields the paper's r values.

Paper: "per-run d_ell correlates with the mortality reduction (r = -0.49 and
-0.68)" for MNIST and Fashion respectively.

We try, for each dataset, every subset defined by
  (depth T in {10,20} or both) x (granularity in {neuron,branch} or both)
  x (dose in {1,3,10} or a single dose) x (statistic: pooled Pearson,
      mean of per-cell r, median of per-cell r, Spearman)
and report any that lands within 0.03 of the target.
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

RES = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results"
RES2 = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit/s1_dose_base"


def load():
    out = defaultdict(dict)
    for base in (RES, RES2):
        for pat in ("c2_*.json.part*", "c2t20_*.json.part*", "s3e2_t20_*.json",
                    "c2_*.json", "c2t20_*.json"):
            for f in sorted(glob.glob(os.path.join(base, pat))):
                fn = os.path.basename(f)
                if ".part" in fn and fn.endswith(".json"):
                    continue
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                rs = d.get("results", []) if isinstance(d, dict) else (
                    d if isinstance(d, list) else [])
                if not isinstance(rs, list):
                    continue
                for r in rs:
                    if not isinstance(r, dict) or r.get("error"):
                        continue
                    ob = r.get("own_bias")
                    if ob is None:
                        if "_b0" not in fn:
                            continue
                        ob = 0.0
                    else:
                        ob = float(ob)
                    key = (r.get("dataset"), r.get("ntask"), r.get("gran"), ob)
                    out[key].setdefault(r.get("seed"), r)
    return out


def deadf(rec):
    v = [d.get("dead_frac") for d in (rec.get("dom_per_phase") or [])
         if d.get("dead_frac") is not None]
    return st.mean(v) if v else None


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    data = load()
    for ds, target in (("mnist", -0.49), ("fashion", -0.68)):
        print(f"===== {ds}  target r = {target} =====")
        best = []
        Tsets = [(10,), (20,), (10, 20)]
        gsets = [("neuron",), ("branch",), ("neuron", "branch")]
        bsets = [(1.0,), (3.0,), (10.0,), (1.0, 3.0, 10.0)]
        for Ts in Tsets:
            for gs in gsets:
                for bs in bsets:
                    # pooled points
                    xs, ys, cells = [], [], []
                    for (d, T, g, b), arm in data.items():
                        if d != ds or T not in Ts or g not in gs or b not in bs:
                            continue
                        ref = data.get((d, T, g, 0.0), {})
                        cx, cy = [], []
                        for s in sorted(set(arm) & set(ref)):
                            d1, d0 = deadf(arm[s]), deadf(ref[s])
                            if d1 is None or d0 is None:
                                continue
                            cx.append(arm[s]["ell"] - ref[s]["ell"])
                            cy.append(d1 - d0)
                        if len(cx) >= 3:
                            cells.append((cx, cy))
                            xs.extend(cx)
                            ys.extend(cy)
                    if len(xs) < 3:
                        continue
                    r_pool = pearson(xs, ys)
                    r_sp = spearman(xs, ys)
                    r_mean = st.mean([pearson(cx, cy) for cx, cy in cells]) if cells else float("nan")
                    r_med = st.median([pearson(cx, cy) for cx, cy in cells]) if cells else float("nan")
                    desc = (f"T={Ts} g={gs} b={bs}")
                    for nm, r in (("pooled", r_pool), ("spearman", r_sp),
                                  ("mean-of-cell", r_mean), ("median-of-cell", r_med)):
                        if r == r and abs(abs(r) - abs(target)) < 0.06:
                            best.append((abs(abs(r) - abs(target)), desc, nm, r, len(xs)))
        if best:
            best.sort()
            for gap, desc, nm, r, n in best[:8]:
                print(f"  MATCH  {desc:44} {nm:14} r={r:+.3f}  n={n}")
        else:
            print("  no aggregation within 0.06 of the target")
        # print the extremes for orientation
        allr = []
        for Ts in Tsets:
            for gs in gsets:
                for bs in bsets:
                    xs, ys = [], []
                    for (d, T, g, b), arm in data.items():
                        if d != ds or T not in Ts or g not in gs or b not in bs:
                            continue
                        ref = data.get((d, T, g, 0.0), {})
                        for s in sorted(set(arm) & set(ref)):
                            d1, d0 = deadf(arm[s]), deadf(ref[s])
                            if d1 is None or d0 is None:
                                continue
                            xs.append(arm[s]["ell"] - ref[s]["ell"])
                            ys.append(d1 - d0)
                    if len(xs) >= 3:
                        allr.append((abs(pearson(xs, ys)), pearson(xs, ys), len(xs),
                                     f"T={Ts} g={gs} b={bs}"))
        allr.sort()
        print("  weakest |r| aggregations:")
        for a, r, n, d in allr[:5]:
            print(f"    {d:44} pooled r={r:+.3f} n={n}")
        print()


if __name__ == "__main__":
    main()
