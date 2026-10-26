#!/usr/bin/env python
"""Recompute S1 Tables 5/6 from the released dose records.

The dose arms live in bench/results under the FILE-NAME scheme
  c2_<ds>_b{0,1,3,10}.json.part*        (T=10)
  c2t20_<ds>_b{3,10}.json.part*         (T=20)
  s3e2_t20_<ds>_b{1,3}.json              (T=20, neuron, S3 E2)
and the b=0 baseline in q8learned* / c2_*_b0.

The dose parameter is the record field ``own_bias`` (NOT ``bias``).
"""
import glob, json, math, os, statistics as st
from collections import defaultdict

RES = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results"
RES2 = "C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit/s1_dose_base"

PAPER = {
    ("mnist", 10, "neuron", 1):  (1.99, 3.40, 1.56),
    ("mnist", 10, "neuron", 3):  (2.68, 4.08, 1.56),
    ("mnist", 10, "neuron", 10): (3.32, 4.80, 1.65),
    ("mnist", 10, "branch", 10): (3.16, 4.17, 1.13),
    ("fashion", 10, "neuron", 1):  (0.87, 1.03, 0.18),
    ("fashion", 10, "neuron", 3):  (1.23, 2.56, 1.48),
    ("fashion", 10, "neuron", 10): (1.39, 0.69, -0.77),
    ("fashion", 10, "branch", 3):  (0.81, 1.79, 1.09),
    ("mnist", 20, "neuron", 1):  (2.14, 2.78, 0.68),
    ("mnist", 20, "neuron", 3):  (3.72, 3.25, -0.49),
    ("mnist", 20, "neuron", 10): (4.56, 3.81, -0.79),
    ("mnist", 20, "branch", 3):  (3.66, 3.03, -0.66),
    ("mnist", 20, "branch", 10): (4.13, 3.48, -0.68),
    ("fashion", 20, "neuron", 1):  (1.78, 1.58, -0.21),
    ("fashion", 20, "neuron", 3):  (2.19, 2.09, -0.11),
    ("fashion", 20, "neuron", 10): (1.94, 1.03, -0.95),
    ("fashion", 20, "branch", 3):  (1.44, 1.98, 0.57),
    ("fashion", 20, "branch", 10): (2.18, 1.96, -0.24),
}
# the 6 cells the paper does NOT report (the 24-cell grid minus 18)
EXTRA = [("mnist", 10, "branch", 1), ("mnist", 10, "branch", 3),
         ("fashion", 10, "branch", 1), ("fashion", 10, "branch", 10),
         ("mnist", 20, "branch", 1), ("fashion", 20, "branch", 1)]

TCRIT = 2.262      # df=9, two-sided 0.05


def load():
    """-> {(ds,T,gran,own_bias): {seed: rec}}

    The dose parameter is the record field ``own_bias``.  Records produced
    with b=0 do NOT carry the field at all (run_gran only writes it when the
    bias is non-zero), so the b=0 arms are identified by their file names
    (c2_*_b0 / c2t20_*_b0) rather than by the field.
    """
    out = defaultdict(dict)
    for base in (RES, RES2):
        for pat in ("c2_*.json.part*", "c2t20_*.json.part*", "s3e2_t20_*.json",
                    "c2_*.json", "c2t20_*.json"):
            for f in sorted(glob.glob(os.path.join(base, pat))):
                fn = os.path.basename(f)
                if ".part" in fn and fn.endswith(".json"):
                    continue           # avoid double counting merged + parts
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
                        # b=0 arms: identified by the file name
                        if "_b0" not in fn:
                            continue
                        ob = 0.0
                    else:
                        ob = float(ob)
                    key = (r.get("dataset"), r.get("ntask"), r.get("gran"), ob)
                    out[key].setdefault(r.get("seed"), r)
    return out


def paired(arm, ref, field):
    com = sorted(set(arm) & set(ref))
    if len(com) < 2:
        return None, None, len(com)
    d = [arm[s][field] - ref[s][field] for s in com]
    m = st.mean(d)
    sd = st.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 1e-12 else 0.0
    return m, t, len(d)


def own_win(rec):
    vals = []
    for d in rec.get("dom_per_phase") or []:
        h = d.get("dom_hist_active") or d.get("dom_hist")
        if not h or sum(h) == 0 or len(h) < 2:
            continue
        vals.append(h[0] / sum(h))
    return st.mean(vals) if vals else None


def deadf(rec):
    v = [d.get("dead_frac") for d in (rec.get("dom_per_phase") or [])
         if d.get("dead_frac") is not None]
    return st.mean(v) if v else None


def main():
    data = load()
    print("=== 载入的剂量格 ===")
    for k in sorted(data, key=lambda x: (str(x[0]), x[1] or 0, str(x[2]), x[3])):
        print(f"  {k[0]:8} T={str(k[1]):4} {str(k[2]):7} b={k[3]:5g} seeds={len(data[k])}")
    print()

    rows = []
    print("=== 论文 18 格：重算 vs 论文 ===")
    print(f"{'ds':8} {'T':>3} {'gran':7} {'b':>4} | {'Δell':>7} {'t':>6} | "
          f"{'Δavg':>7} {'t':>6} | {'ΔBWT':>7} {'t':>6} | {'own':>6} | 论文")
    for key in sorted(PAPER, key=lambda x: (x[0], x[1], x[2], x[3])):
        ds, T, gran, b = key
        arm = data.get(key, {})
        ref = data.get((ds, T, gran, 0.0), {})
        if not arm or not ref:
            print(f"{ds:8} {T:3} {gran:7} {b:4} | 缺 (arm={len(arm)} ref={len(ref)})")
            continue
        de, te, n = paired(arm, ref, "ell")
        da, ta, _ = paired(arm, ref, "final_avg")
        db, tb, _ = paired(arm, ref, "bwt")
        ow = [own_win(r) for r in arm.values()]
        ow = [x for x in ow if x is not None]
        o = st.mean(ow) if ow else None
        pe, pa, pb = PAPER[key]
        rows.append((key, de, te, da, ta, db, tb, n, o))
        print(f"{ds:8} {T:3} {gran:7} {b:4} | {de:+7.2f} {te:+6.1f} | {da:+7.2f} {ta:+6.1f} | "
              f"{db:+7.2f} {tb:+6.1f} | {(f'{o:.2f}' if o is not None else '-'):>6} | "
              f"({pe:+.2f},{pa:+.2f},{pb:+.2f})")

    print()
    print("=== 论文未报的 6 格（补齐 24 格网格）===")
    for key in EXTRA:
        ds, T, gran, b = key
        arm = data.get(key, {})
        ref = data.get((ds, T, gran, 0.0), {})
        if not arm or not ref:
            print(f"  {key} 缺 (arm={len(arm)} ref={len(ref)})")
            continue
        de, te, n = paired(arm, ref, "ell")
        da, ta, _ = paired(arm, ref, "final_avg")
        db, tb, _ = paired(arm, ref, "bwt")
        print(f"  {ds:8} T={T:3} {gran:7} b={b:4} | Δell={de:+7.2f} t={te:+6.1f} | "
              f"Δavg={da:+7.2f} t={ta:+6.1f} | ΔBWT={db:+7.2f} t={tb:+6.1f} | n={n}")

    print()
    n = len(rows)
    if n:
        sig_e = sum(1 for r in rows if abs(r[2]) >= TCRIT)
        sig_a = sum(1 for r in rows if abs(r[4]) >= TCRIT)
        sig_b = sum(1 for r in rows if abs(r[6]) >= TCRIT)
        both = sum(1 for r in rows if abs(r[2]) >= TCRIT and abs(r[4]) >= TCRIT)
        pos = sum(1 for r in rows if r[1] > 0)
        print(f"=== 重算汇总（论文的 {n} 格）===")
        print(f"  Δℓ>0          : {pos}/{n}")
        print(f"  Δℓ 显著       : {sig_e}/{n}   （论文声称 14/18）")
        print(f"  Δavg 显著     : {sig_a}/{n}   （论文声称 13/18 的双显著）")
        print(f"  ΔBWT 显著     : {sig_b}/{n}")
        print(f"  Δℓ&Δavg 双显著: {both}/{n}   （论文声称 13/18）")
        print(f"  符号检验 p    : 2^-{n} = {2.0**-n:.2e}")
        print()
        print("=== 逐格一致性 ===")
        allok = True
        for key, de, te, da, ta, db, tb, nn, o in rows:
            pe, pa, pb = PAPER[key]
            ok = (abs(de - pe) < 0.011 and abs(da - pa) < 0.011 and abs(db - pb) < 0.011)
            allok = allok and ok
            print(f"  {key[0]:8} T={key[1]:2} {key[2]:7} b={key[3]:3}: "
                  f"Δℓ {de:+6.2f}/{pe:+6.2f} Δavg {da:+6.2f}/{pa:+6.2f} "
                  f"ΔBWT {db:+6.2f}/{pb:+6.2f} {'OK' if ok else '<<<'}")
        print(f"\n全部吻合: {allok}")


if __name__ == "__main__":
    main()
