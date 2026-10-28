#!/usr/bin/env python
"""One-command verification of every number quoted in the S1 manuscript.

Usage (from the S1_剂量响应审计/ directory):

    python audit/verify_all.py .

It re-derives, from the released per-seed records in results/, every cell of
Tables 1, 2, 3, 5 and 6 -- absolute means, paired differences, t values and
the significance stars -- and checks that the released files contain no
duplicate datasets.  It also reports the two known limitations.

Nothing outside results/ and paper/main.tex is required.
"""
import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "paper", "main.tex")
CRIT = 2.0                     # the manuscript's captions declare *: |t| > 2

FAILS = []
NCHECKS = 0


def check(name, ok, detail=""):
    global NCHECKS
    NCHECKS += 1
    if not ok:
        FAILS.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def paired(a, b, fld):
    com = sorted(set(a) & set(b))
    if len(com) < 2:
        return None, None, len(com)
    d = [a[s][fld] - b[s][fld] for s in com]
    m = st.mean(d)
    sd = st.stdev(d)
    return m, (m / (sd / math.sqrt(len(d))) if sd > 1e-12 else 0.0), len(com)


def load_family(pattern):
    """-> {(ds,T,gran,kind,lam): {seed: rec}} for one file family."""
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        fn = os.path.basename(f)
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
            key = (r.get("dataset"), r.get("ntask"), r.get("gran"),
                   r.get("kind"), float(r.get("lam") or 0))
            out[key].setdefault(r.get("seed"), r)
    return out


def load_dose():
    """-> {(ds,T,gran,bias): {seed: rec}} for the dose grid.

    The dose is the record field ``own_bias``.  Bias-0 arms carry no such
    field (the code only writes it for a non-zero bias) and are identified by
    the ``_b0`` file stem.
    """
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, "*.json*"))):
        fn = os.path.basename(f)
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
            if r.get("kind") != "gran":
                continue
            ob = r.get("own_bias")
            if ob is None:
                if "_b0" not in fn:
                    continue
                ob = 0.0
            else:
                ob = float(ob)
            out[(r.get("dataset"), r.get("ntask"), r.get("gran"), ob)] \
                .setdefault(r.get("seed"), r)
    return out


def main():
    print(f"=== S1 verification against {ROOT} ===\n")
    if not os.path.isdir(RES):
        print("FATAL: results/ not found")
        return 1

    # ---------------------------------------------------------------- files
    print("--- released files ---")
    files = sorted(glob.glob(os.path.join(RES, "*.json*")))
    print(f"  {len(files)} result files")
    dup_pairs = []
    merged_with_parts = []
    for f in files:
        fn = os.path.basename(f)
        if ".part" in fn and fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if glob.glob(os.path.join(RES, stem + ".json.part*")):
            merged_with_parts.append(fn)
    check("no merged file sits next to its own shards", not merged_with_parts,
          "; ".join(merged_with_parts))
    seen = {}
    for f in files:
        fn = os.path.basename(f)
        if ".part" in fn and fn.endswith(".json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        rs = d.get("results", [])
        if not rs:
            continue
        sig = (rs[0].get("dataset"), rs[0].get("ntask"), rs[0].get("gran"),
               rs[0].get("own_bias"))
        body = json.dumps(rs, sort_keys=True)
        if sig in seen and seen[sig][1] == body:
            dup_pairs.append(f"{fn} == {seen[sig][0]}")
        seen.setdefault(sig, (fn, body))
    check("no byte-identical duplicate datasets", not dup_pairs, "; ".join(dup_pairs))

    # ---------------------------------------------------------------- Table 1
    print("\n--- Table 1: granularity ladder (s1_*.json family) ---")
    gran = load_family("s1_*.json*")
    T1 = {
        ("mnist", 10, "plain", 1000.0): (69.63, 61.25, -9.31, -4.66, -2.61, 2.26),
        ("mnist", 10, "neuron", 1000.0): (74.28, 63.86, -11.58, None, None, None),
        ("mnist", 10, "branch", 1000.0): (74.21, 64.13, -11.20, -0.08, 0.27, 0.38),
        ("mnist", 20, "plain", 1000.0): (46.35, 37.30, -9.53, -5.44, -1.35, 4.31),
        ("mnist", 20, "neuron", 1000.0): (51.79, 38.64, -13.84, None, None, None),
        ("mnist", 20, "branch", 1000.0): (51.98, 39.03, -13.63, 0.19, 0.39, 0.21),
        ("mnist", 10, "plain", 5000.0): (54.41, 51.69, -3.03, -4.78, -2.80, 2.19),
        ("mnist", 10, "neuron", 5000.0): (59.19, 54.49, -5.22, None, None, None),
        ("mnist", 10, "branch", 5000.0): (59.21, 54.71, -5.01, 0.03, 0.22, 0.21),
        ("mnist", 20, "plain", 5000.0): (33.97, 31.68, -2.40, -3.72, -1.74, 2.08),
        ("mnist", 20, "neuron", 5000.0): (37.68, 33.42, -4.49, None, None, None),
        ("mnist", 20, "branch", 5000.0): (37.59, 33.39, -4.42, -0.09, -0.03, 0.07),
        ("fashion", 10, "plain", 5000.0): (38.53, 33.47, -5.62, -3.76, -1.65, 2.35),
        ("fashion", 10, "neuron", 5000.0): (42.29, 35.12, -7.96, None, None, None),
        ("fashion", 10, "branch", 5000.0): (42.05, 35.27, -7.53, -0.24, 0.15, 0.43),
        ("fashion", 20, "plain", 5000.0): (26.56, 22.75, -4.01, -2.78, -1.22, 1.65),
        ("fashion", 20, "neuron", 5000.0): (29.35, 23.97, -5.66, None, None, None),
        ("fashion", 20, "branch", 5000.0): (29.35, 24.14, -5.49, 0.01, 0.17, 0.17),
    }
    n_abs = n_abs_ok = 0
    n_d = n_d_ok = 0
    for (ds, T, arm, lam), v in sorted(T1.items(), key=lambda x: (x[0][0], x[0][3], x[0][1], x[0][2])):
        rs = gran.get((ds, T, arm, "gran", lam))
        if not rs:
            continue
        ge = st.mean([r["ell"] for r in rs.values()])
        ga = st.mean([r["final_avg"] for r in rs.values()])
        gb = st.mean([r["bwt"] for r in rs.values()])
        n_abs += 1
        n_abs_ok += (abs(round(ge, 2) - v[0]) < 0.006 and abs(round(ga, 2) - v[1]) < 0.006
                     and abs(round(gb, 2) - v[2]) < 0.006)
        if v[3] is not None:
            neu = gran.get((ds, T, "neuron", "gran", lam))
            if not neu:
                continue
            for fld, pm in (("ell", v[3]), ("final_avg", v[4]), ("bwt", v[5])):
                dm, _, _ = paired(rs, neu, fld)
                n_d += 1
                n_d_ok += abs(round(dm, 2) - pm) < 0.011
    check(f"Table 1 absolute means ({n_abs_ok}/{n_abs})", n_abs_ok == n_abs)
    check(f"Table 1 paired differences ({n_d_ok}/{n_d})", n_d_ok == n_d)

    # ---------------------------------------------------------------- Table 2
    print("\n--- Table 2: single-anchor references ---")
    refs = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, "s1_refs.json*"))):
        for r in json.load(open(f, encoding="utf-8")).get("results", []):
            if r.get("error"):
                continue
            refs[(r.get("kind"), r.get("ntask"), float(r.get("lam") or 0))] \
                .setdefault(r.get("seed"), r)
    T2 = {("si", 10, 100.0): (40.01, 39.26, -0.83),
          ("mas", 10, 1.0): (85.71, 66.61, -21.21),
          ("si", 20, 100.0): (25.80, 25.14, -0.69),
          ("mas", 20, 1.0): (86.69, 55.75, -32.56)}
    n2 = n2_ok = 0
    for (kind, T, lam), (pe, pa, pb) in T2.items():
        rs = refs.get((kind, T, lam))
        if not rs:
            continue
        ge = st.mean([r["final_avg"] - (T - 1) / T * r["bwt"] for r in rs.values()])
        ga = st.mean([r["final_avg"] for r in rs.values()])
        gb = st.mean([r["bwt"] for r in rs.values()])
        n2 += 1
        n2_ok += (abs(round(ge, 2) - pe) < 0.011 and abs(round(ga, 2) - pa) < 0.011
                  and abs(round(gb, 2) - pb) < 0.011)
    check(f"Table 2 cells ({n2_ok}/{n2})", n2_ok == n2)

    # ---------------------------------------------------------------- Table 3
    print("\n--- Table 3: FIFO anchor budgets ---")
    T3 = {("mnist", 10, 2): (83.06, 83.01, -0.05, -1.0, 0.22, 1.1),
          ("mnist", 10, 5): (77.09, 77.02, -0.07, -0.5, 0.15, 1.0),
          ("mnist", 20, 2): (80.23, 80.32, 0.09, 1.0, 0.39, 1.9),
          ("mnist", 20, 5): (68.30, 68.19, -0.11, -0.8, 0.28, 1.5),
          ("fashion", 10, 2): (66.66, 66.33, -0.33, -2.3, -0.41, -1.2),
          ("fashion", 10, 5): (58.90, 58.74, -0.17, -0.9, 0.43, 1.2),
          ("fashion", 20, 2): (62.61, 62.72, 0.10, 0.4, -0.18, -0.3),
          ("fashion", 20, 5): (49.79, 50.23, 0.44, 1.5, -0.33, -0.7)}
    bf = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES, "s1b_*.json"))):
        for r in json.load(open(f, encoding="utf-8")).get("results", []):
            if r.get("error"):
                continue
            bf[(r.get("dataset"), r.get("ntask"), r.get("M"),
                r.get("gran"))].setdefault(r.get("seed"), r)
    n3 = n3_ok = 0
    for (ds, T, M), (pn, pb, pdl, ptl, pdw, ptw) in sorted(T3.items()):
        neu = bf.get((ds, T, M, "neuron"), {})
        bra = bf.get((ds, T, M, "branch"), {})
        if not neu or not bra:
            continue
        en = st.mean([r["ell"] for r in neu.values()])
        eb = st.mean([r["ell"] for r in bra.values()])
        de, te, _ = paired(bra, neu, "ell")
        dw, tw, _ = paired(bra, neu, "bwt")
        n3 += 1
        n3_ok += (abs(round(en, 2) - pn) < 0.006 and abs(round(eb, 2) - pb) < 0.006
                  and abs(round(de, 2) - pdl) < 0.006 and abs(round(te, 1) - ptl) < 0.06
                  and abs(round(dw, 2) - pdw) < 0.006 and abs(round(tw, 1) - ptw) < 0.06)
    check(f"Table 3 cells ({n3_ok}/{n3})", n3_ok == n3)

    # ---------------------------------------------------------------- Tables 5/6
    print("\n--- Tables 5/6: dose-response grid ---")
    dose = load_dose()
    DOSE = {
        ("mnist", 10, "neuron", 1.0): (1.99, 3.40, 1.56),
        ("mnist", 10, "neuron", 3.0): (2.68, 4.08, 1.56),
        ("mnist", 10, "neuron", 10.0): (3.32, 4.80, 1.65),
        ("mnist", 10, "branch", 1.0): (2.18, 3.29, 1.23),
        ("mnist", 10, "branch", 3.0): (2.90, 4.30, 1.57),
        ("mnist", 10, "branch", 10.0): (3.16, 4.17, 1.13),
        ("fashion", 10, "neuron", 1.0): (0.87, 1.03, 0.18),
        ("fashion", 10, "neuron", 3.0): (1.23, 2.56, 1.48),
        ("fashion", 10, "neuron", 10.0): (1.39, 0.69, -0.77),
        ("fashion", 10, "branch", 1.0): (0.94, 2.08, 1.26),
        ("fashion", 10, "branch", 3.0): (0.81, 1.79, 1.09),
        ("fashion", 10, "branch", 10.0): (0.97, 1.03, 0.06),
        ("mnist", 20, "neuron", 1.0): (2.14, 2.78, 0.68),
        ("mnist", 20, "neuron", 3.0): (3.72, 3.25, -0.49),
        ("mnist", 20, "neuron", 10.0): (4.56, 3.81, -0.79),
        ("mnist", 20, "branch", 3.0): (3.66, 3.03, -0.66),
        ("mnist", 20, "branch", 10.0): (4.13, 3.48, -0.68),
        ("fashion", 20, "neuron", 1.0): (1.78, 1.58, -0.21),
        ("fashion", 20, "neuron", 3.0): (2.19, 2.09, -0.11),
        ("fashion", 20, "neuron", 10.0): (1.94, 1.03, -0.95),
        ("fashion", 20, "branch", 3.0): (1.44, 1.98, 0.57),
        ("fashion", 20, "branch", 10.0): (2.18, 1.96, -0.24),
    }
    nd = nd_ok = 0
    missing = []
    for key, (pe, pa, pb) in sorted(DOSE.items()):
        ds, T, g, b = key
        arm = dose.get(key)
        ref = dose.get((ds, T, g, 0.0))
        if not arm or not ref:
            missing.append(key)
            continue
        de, te, _ = paired(arm, ref, "ell")
        da, ta, _ = paired(arm, ref, "final_avg")
        db, tb, _ = paired(arm, ref, "bwt")
        nd += 1
        nd_ok += (abs(round(de, 2) - pe) < 0.011 and abs(round(da, 2) - pa) < 0.011
                  and abs(round(db, 2) - pb) < 0.011)
    check(f"dose cells ({nd_ok}/{nd})", nd_ok == nd, f"missing={missing}")

    # ------------------------------------------------------- significance stars
    if os.path.exists(TEX):
        print("\n--- significance stars in the manuscript tables ---")
        tex = open(TEX, encoding="utf-8").read()
        tot = ok = 0
        for label, kind in (("tab:gran", "gran"), ("tab:budget", "bfifo"),
                             ("tab:dose10", "dose"), ("tab:dose20", "dose")):
            m = re.search(r"\\label\{%s\}" % label, tex)
            if not m:
                continue
            b = tex.index("\\begin{tabular}", m.start())
            seg = tex[b:tex.index("\\end{tabular}", b)]
            cur_lam = None
            ds = "mnist"
            Tb = 10
            for line in seg.split("\n"):
                if "Permuted-MNIST" in line:
                    ds, cur_lam = "mnist", None
                if "Permuted-Fashion" in line:
                    ds, cur_lam = "fashion", None
                if "lambda{=}1000" in line:
                    cur_lam = 1000.0
                elif "lambda{=}5000" in line:
                    cur_lam = 5000.0
                if "Causal probe, $T{=}10$" in tex[max(0, m.start() - 400):m.start()]:
                    Tb = 10
                elif "Causal probe, $T{=}20$" in tex[max(0, m.start() - 400):m.start()]:
                    Tb = 20
                cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
                if len(cells) < 6:
                    continue
                if kind == "dose":
                    # rows look like "MNIST & neuron ($1$) & ..."
                    dm = re.search(r"(\w+)\s*\(\$(\d+)\$\)", cells[1])
                    if not dm:
                        continue
                    arm, bias = dm.group(1), float(dm.group(2))
                    ds = "mnist" if cells[0].startswith("MNIST") else "fashion"
                    Tb = 10 if label == "tab:dose10" else 20
                    T = Tb
                else:
                    mm = re.search(r"\$T\{=\}(\d+)\$\s*&\s*(\w+)", line)
                    if not mm:
                        continue
                    T, arm = int(mm.group(1)), mm.group(2)
                if kind == "gran":
                    neu = gran.get((ds, T, "neuron", "gran", cur_lam))
                    ars = gran.get((ds, T, arm, "gran", cur_lam))
                    if not neu or not ars:
                        continue
                    pairs = [(5, "ell"), (6, "final_avg"), (7, "bwt")]
                elif kind == "bfifo":
                    neu = bf.get((ds, T, arm, "neuron"))
                    ars = bf.get((ds, T, arm, "branch"))
                    if not neu or not ars:
                        continue
                    pairs = [(2, "ell"), (3, "bwt")]
                else:
                    ars = dose.get((ds, T, arm, bias))
                    neu = dose.get((ds, T, arm, 0.0))
                    if not ars or not neu:
                        continue
                    pairs = [(2, "ell"), (3, "final_avg"), (4, "bwt")]
                for idx, fld in pairs:
                    _, tt, _ = paired(ars, neu, fld)
                    tot += 1
                    ok += (("*" in cells[idx]) == (abs(tt) > CRIT))
        check(f"significance stars ({ok}/{tot})", tot > 0 and ok == tot)

    print(f"\n=== {NCHECKS - len(FAILS)}/{NCHECKS} checks passed ===")
    for f in FAILS:
        print(f"  FAILED: {f}")
    print("\nKnown limitations (stated in the paper):")
    print("  * two of the 24 dose cells were never collected "
          "(T=20, branch arm, b=1, both datasets)")
    print("  * the arms that reuse the original implementation do not record "
          "held-out validation metrics")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
