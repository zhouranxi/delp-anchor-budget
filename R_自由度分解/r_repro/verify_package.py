# -*- coding: utf-8 -*-
"""verify_package.py -- one-shot independent verification of the R1-R3 package.

  python r_repro/verify_package.py        # from the repository root

Recomputes every headline number DIRECTLY from the raw run JSONs (no reuse
of analyze_* intermediate stats) and compares against the values printed in
the four collection reports.  Exit code 0 iff all checks PASS.

Pure Python + statistics; no torch, no scipy, no GPU needed.
"""
import glob, json, math, os, sys
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = lambda *p: os.path.join(ROOT, "bench", *p)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f, encoding="utf-8"))
        out += d["results"] if isinstance(d, dict) and "results" in d else d
    return out


def paired(diffs):
    n = len(diffs)
    m, sd = st.mean(diffs), st.stdev(diffs)
    t = m / (sd / math.sqrt(n))
    p = t_two_tailed_p(t, n - 1)
    dz = m / sd if sd else float("inf")
    return {"mean": m, "sd": sd, "t": t, "p": p, "d_z": dz, "n": n}


def t_two_tailed_p(t, df):
    """Exact two-tailed p for Student t via the incomplete beta function
    (Numerical Recipes betai; x = df/(df+t^2) branch handled directly)."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    a, b = df / 2.0, df / 2.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    # I_x(a,b) for a=b: use the continued fraction on the correct branch
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            if abs(1.0 - d) < EPS:
                break
        return h

    def betai(a, b, x):
        # exp(lbeta + a ln x + b ln(1-x)) == x^a (1-x)^b / B(a,b)
        ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + a * math.log(x) + b * math.log(1.0 - x))
        bt = math.exp(ln_front)
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b

    # two-tailed: p = I_{df/(df+t^2)}(df/2, 1/2)  -- note b = 1/2 here
    a2, b2 = df / 2.0, 0.5
    ln_front = (math.lgamma(a2 + b2) - math.lgamma(a2) - math.lgamma(b2)
                + a2 * math.log(x) + b2 * math.log(1.0 - x))
    bt = math.exp(ln_front)
    if x < (a2 + 1.0) / (a2 + b2 + 2.0):
        p_one = bt * betacf(a2, b2, x) / a2
    else:
        p_one = 1.0 - bt * betacf(b2, a2, 1.0 - x) / b2
    return min(1.0, p_one)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
def verify_r1():
    print("\n=== R1: training-time context noise (authoritative v2) ===")
    v2 = load(B("results_r1", "r1_grid_v2.part*.json"))
    check("v2 run count == 280", len(v2) == 280, f"n={len(v2)}")

    idx = {(r["dataset"], float(r["bias"]), r["seed"], r["mode"],
            float(r["rho"])): r for r in v2}
    # clean arm exists only at rho=0 (shared control); noisy arms at each rho
    # compare against the rho=0 clean run of the same (ds, bias, seed).
    # 24 dAvg cells (mode in fwd/full x rho x bias x ds): Holm-min p >= 0.5
    ps = []
    for mode in ("fwd", "full"):
        for rho in (0.1, 0.2, 0.4):
            for bias in (0.0, 10.0):
                for ds in ("mnist", "fashion"):
                    d = [idx[(ds, bias, s, mode, rho)]["avg"]
                         - idx[(ds, bias, s, "clean", 0.0)]["avg"]
                         for s in range(10)]
                    ps.append(paired(d)["p"])
    adj = holm(ps)
    check("24 dAvg cells, Holm-min p >= 0.5 (report: 0.585)",
          min(adj) >= 0.5, f"min holm p={min(adj):.3f}")

    # H1 (rho=0.2): fwd-full per (ds, bias): |mean| < 0.5
    worst = 0.0
    for ds in ("mnist", "fashion"):
        for bias in (0.0, 10.0):
            d = [idx[(ds, bias, s, "full", 0.2)]["avg"]
                 - idx[(ds, bias, s, "fwd", 0.2)]["avg"] for s in range(10)]
            worst = max(worst, abs(st.mean(d)))
    check("H1 anchor chain |fwd-full| < 0.5 (report: max 0.422)",
          worst < 0.5, f"max |mean|={worst:.3f}")

    # survival B (report §4.2): at rho=0.1, (b=10)-(b=0) dEll vs clean gain,
    # retention ratio in [0.5, 1.2] (report: 93-112%)
    ratios = []
    for ds in ("mnist", "fashion"):
        clean_gain = st.mean(
            idx[(ds, 10.0, s, "clean", 0.0)]["ell"]
            - idx[(ds, 0.0, s, "clean", 0.0)]["ell"] for s in range(10))
        for mode in ("fwd", "full"):
            dl = st.mean(
                idx[(ds, 10.0, s, mode, 0.1)]["ell"]
                - idx[(ds, 0.0, s, mode, 0.1)]["ell"] for s in range(10))
            ratios.append(dl / clean_gain)
    check("survival B retention in [0.5, 1.2], all dEll>0 (report 93-112%)",
          all(0.5 <= r <= 1.2 for r in ratios),
          "ratios=" + ",".join(f"{r:.0%}" for r in ratios))

    # v1/v2 equivalence: 214 untouched runs bit-identical
    v1 = load(B("results_r1", "r1_grid.part*.json"))
    k1 = {(r["dataset"], float(r["bias"]), r["seed"], r["mode"],
           float(r["rho"])): r for r in v1}
    same = diff = 0
    for k, r2r in idx.items():
        o = k1.get(k)
        if not o:
            continue
        if r2r["mode"] != "clean" and (o["n_mislabeled"] == 0):
            continue  # forced run
        if all(abs(r2r[m] - o[m]) < 1e-9 for m in ("ell", "avg", "bwt")):
            same += 1
        else:
            diff += 1
    check("v1/v2 untouched runs bit-identical (214/214)",
          diff == 0 and same >= 214, f"same={same} diff={diff}")


def verify_r2():
    print("\n=== R2: anchor-budget direction on MLP ===")
    g = load(B("results_r2", "r2_grid.part*.json"))
    check("main grid == 240 runs", len(g) == 240, f"n={len(g)}")
    idx = {(r["arm"], r["ds"], r["ntask"], r["seed"]): r for r in g}

    # weight conservation across all runs
    bad = 0
    for r in g:
        for p, (w, s) in enumerate(zip(r["phase_weights"], r["weight_sums"])):
            if abs(s - p) > 1e-6 or any(x <= 0 for x in w) or len(w) != p:
                bad += 1
    check("G3 weight conservation (0 violations)", bad == 0, f"bad={bad}")

    # H1: renorm_rec10 four E7 thresholds
    def pooled(arm, ref, m, T):
        d = [idx[(arm, ds, T, s)][m] - idx[(ref, ds, T, s)][m]
             for ds in ("mnist", "fashion") for s in range(10)]
        return paired(d)

    c1e = pooled("renorm_rec10", "base", "ell", 20)
    c1b = pooled("renorm_rec10", "base", "bwt", 20)
    c3 = pooled("renorm_rec10", "renorm", "bwt", 20)
    c4e = pooled("renorm_rec10", "base", "ell", 10)
    c4b = pooled("renorm_rec10", "base", "bwt", 10)
    check("cond1 T20 Dell >= +4.0 (report +8.08)", c1e["mean"] >= 4.0,
          f"{c1e['mean']:+.3f}")
    check("cond2 T20 Dbwt >= -0.5 (report +0.37)", c1b["mean"] >= -0.5,
          f"{c1b['mean']:+.3f}")
    check("cond3 T20 Dbwt vs renorm >= +1.11 (report +1.52)",
          c3["mean"] >= 1.11, f"{c3['mean']:+.3f}")
    check("cond4 T10 Dell>=4.5 & Dbwt>=1.0 (report +5.84/+4.45)",
          c4e["mean"] >= 4.5 and c4b["mean"] >= 1.0,
          f"{c4e['mean']:+.2f}/{c4b['mean']:+.2f}")
    # Holm family (2 comparisons) == 6.42e-05
    c05 = pooled("renorm_rec05", "renorm", "bwt", 20)
    adj = holm([c3["p"], c05["p"]])
    check("Holm(rec10) ~ 6.42e-05 SIG", adj[0] < 0.05
          and abs(adj[0] - 6.419e-05) / 6.419e-05 < 0.05,
          f"holm={adj[0]:.3e}")

    # alpha=0 sentinel: 40 pairs bit-identical
    sen = load(B("results_r2", "r2_alpha0.part*.json"))
    sidx = {(r["arm"], r["ds"], r["ntask"], r["seed"]): r for r in sen}
    md = 0.0
    npairs = 0
    for k, r in sidx.items():
        if k[0] != "renorm_rec00":
            continue
        o = sidx[("renorm",) + k[1:]]
        md = max(md, max(abs(r[m] - o[m]) for m in ("ell", "avg", "bwt")))
        npairs += 1
    check("alpha=0 sentinel 40 pairs, max diff 0 (report 0.0)",
          npairs == 40 and md == 0.0, f"pairs={npairs} max={md}")

    # anti direction: 20/20 worse bwt than rec10
    anti = load(B("results_r2", "r2_anti.part*.json"))
    aidx = {(r["arm"], r["ds"], r["ntask"], r["seed"]): r for r in anti}
    neg = sum(1 for ds in ("mnist", "fashion") for s in range(10)
              if aidx[("renorm_anti10", ds, 20, s)]["bwt"]
              < idx[("renorm_rec10", ds, 20, s)]["bwt"])
    check("anti bwt worse 20/20 (report 20/20)", neg == 20, f"{neg}/20")


def verify_r2t():
    print("\n=== R2T: ResNet transfer (first round n=10 / extension n=20) ===")
    new1 = load(B("results_r2t", "r2t_grid.part*.json"))
    check("first-round new arms == 30", len(new1) == 30, f"n={len(new1)}")
    ext = load(B("results_r2t", "r2t_ext.part*.json"))
    ctrl_ext = load(B("results_r2t", "r2t_ctrl_ext.part*.json"))
    e9 = load(B("results_e9", "e9_grid_s*.json"))
    check("ext arms == 30, ctrl_ext == 20 (2 arms x 10 seeds), e9 reused == 60",
          len(ext) == 30 and len(ctrl_ext) == 20 and len(e9) == 60,
          f"{len(ext)}/{len(ctrl_ext)}/{len(e9)}")

    def stats(pool):
        idx = {(r["arm"], r["seed"]): r for r in pool}
        out = {}
        for cid, arm, ref in (("C1", "base_rec10", "base"),
                              ("C2", "renorm_rec10", "renorm")):
            for m in ("ell", "bwt"):
                d = [idx[(arm, s)][m] - idx[(ref, s)][m]
                     for s in sorted(k[1] for k in idx if k[0] == arm)]
                out[f"{cid}.{m}"] = paired(d)
                out[f"{cid}.{m}.pos"] = sum(1 for x in d if x > 0)
                out[f"{cid}.{m}.n"] = len(d)
        return out

    s1 = stats(new1 + e9)
    s2 = stats(new1 + ext + e9 + ctrl_ext)
    check("round1 C1.ell ~ +1.94 (the flip source)",
          abs(s1["C1.ell"]["mean"] - 1.94) < 0.05,
          f"{s1['C1.ell']['mean']:+.3f}")
    check("ext C1.ell ~ +0.72 p>0.3 (T3 verdict)",
          abs(s2["C1.ell"]["mean"] - 0.72) < 0.05 and s2["C1.ell"]["p"] > 0.3,
          f"{s2['C1.ell']['mean']:+.3f} p={s2['C1.ell']['p']:.3f}")
    check("ext C1 direction 11/9 (coin level)",
          (s2["C1.ell.pos"], s2["C1.ell.n"]) == (11, 20),
          f"{s2['C1.ell.pos']}/{s2['C1.ell.n']}")
    # C4 magnitude carried by renorm forward
    idx2 = {(r["arm"], r["seed"]): r
            for r in new1 + ext + e9 + ctrl_ext}
    c4 = paired([idx2[("base_rec10", s)]["ell"] - idx2[("renorm", s)]["ell"]
                 for s in range(20)])
    check("C4 base_rec10-renorm ~ +10.06 (20/0)",
          abs(c4["mean"] - 10.06) < 0.1
          and all(idx2[("base_rec10", s)]["ell"] > idx2[("renorm", s)]["ell"]
                  for s in range(20)),
          f"{c4['mean']:+.2f}")


def verify_r3():
    print("\n=== R3: conv lambda re-calibration ===")
    rs = load(B("results_r3", "r3_grid.part*.json"))
    check("grid == 240 runs", len(rs) == 240, f"n={len(rs)}")
    idx = {(float(r["lam"]), r["arm"], r["ds"], r["seed"]): r for r in rs}

    def gain(lam):
        d = [idx[(lam, "renorm", ds, s)]["ell"]
             - idx[(lam, "base", ds, s)]["ell"]
             for ds in ("mnist", "fashion") for s in range(10)]
        p = paired(d)
        return p, sum(1 for x in d if x > 0)

    table = {}
    for lam in (0.0, 3.16, 31.6, 316.0, 1000.0, 3160.0):
        table[lam] = gain(lam)
    # collapse point
    p1000 = table[1000.0][0]
    check("gain(1000) collapse ~ +0.19, p>0.5 (E5's single point)",
          abs(p1000["mean"] - 0.185) < 0.02 and p1000["p"] > 0.5,
          f"{p1000['mean']:+.3f} p={p1000['p']:.2f}")
    # significant positives at lam<=316 (Holm over the 6)
    adj = holm([table[l][0]["p"] for l in
                (0.0, 3.16, 31.6, 316.0, 1000.0, 3160.0)])
    lams = [0.0, 3.16, 31.6, 316.0, 1000.0, 3160.0]
    sig_low = sum(1 for l, a in zip(lams, adj) if l <= 316 and a < 0.05)
    check("lambda<=316 Holm SIG count == 4 (report 4/6)",
          sig_low == 4, f"sig_low={sig_low}")
    # best gain ~ +1.04 < M1 threshold +2
    best = max(table[l][0]["mean"] for l in lams)
    check("best gain in (1.0, 1.1) and < M1 +2.0 (intermediate verdict)",
          1.0 <= best <= 1.1, f"best={best:.3f}")
    # M4 dose: lam 3160 suppresses ell
    b0 = st.mean(idx[(3160.0, "base", ds, s)]["ell"]
                 for ds in ("mnist", "fashion") for s in range(10))
    b1k = st.mean(idx[(1000.0, "base", ds, s)]["ell"]
                  for ds in ("mnist", "fashion") for s in range(10))
    check("M4 dose: ell(base,3160) < ell(base,1000) by ~2 (report -2.07)",
          1.5 < (b1k - b0) < 2.6, f"diff={b1k - b0:.2f}")


def main():
    print("R-series package independent verification")
    print("(recomputed from raw JSONs; no analyzer intermediates)")
    for fn in (verify_r1, verify_r2, verify_r2t, verify_r3):
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} crashed", False, repr(e))
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n===== {n_pass}/{len(CHECKS)} PASS =====")
    if n_pass != len(CHECKS):
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  FAILED: {name} {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
