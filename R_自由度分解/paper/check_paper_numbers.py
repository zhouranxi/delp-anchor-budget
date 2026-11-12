# -*- coding: utf-8 -*-
"""check_paper_numbers.py -- scripted verification of EVERY count and
statistic quoted in the paper against the raw run records.

  python 新论文/check_paper_numbers.py    (from the repository root)

Extends r_repro/verify_package.py's headline checks to caption-level and
in-text counts (seed directions, added-seed splits, points paid, etc.).
Exit 0 iff all checks pass.  Born from writing-stage failure #7: two
caption numbers written from means instead of counts.
"""
import glob, json, math, os, sys
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "r_repro"))
from verify_package import paired, load, check, CHECKS  # reuse exact stats

PAPER = os.path.join(ROOT, "新论文", "paper", "main.tex")
if not os.path.exists(PAPER):  # Zenodo package layout (paper/ at package root)
    PAPER = os.path.join(ROOT, "paper", "main.tex")


def main():
    txt = open(PAPER, encoding="utf-8").read()

    # ---- fig_flip caption / R2T section counts ----
    new1 = load(os.path.join(ROOT, "bench", "results_r2t", "r2t_grid.part*.json"))
    ext = load(os.path.join(ROOT, "bench", "results_r2t", "r2t_ext.part*.json"))
    ctrl = load(os.path.join(ROOT, "bench", "results_r2t",
                             "r2t_ctrl_ext.part*.json"))
    e9 = load(os.path.join(ROOT, "bench", "results_e9", "e9_grid_s*.json"))

    def deltas(pool, seeds):
        idx = {(r["arm"], r["seed"]): r for r in pool}
        return [idx[("base_rec10", s)]["ell"] - idx[("base", s)]["ell"]
                for s in seeds]

    d1 = deltas(new1 + e9, range(10))
    d2 = deltas(new1 + ext + e9 + ctrl, range(20))
    r1_pos = sum(1 for x in d1 if x > 0)
    added = d2[10:]
    a_pos = sum(1 for x in added if x > 0)
    a_neg = len(added) - a_pos
    check("R2T first round: nine positive seeds (paper says nine)",
          r1_pos == 9 and "nine positive seeds" in txt, f"pos={r1_pos}")
    check("R2T added seeds: 2 positive / 8 negative (paper says $2$/$8$)",
          (a_pos, a_neg) == (2, 8) and "$2$ positive/$8$ negative" in txt,
          f"{a_pos}/{a_neg}")
    check("R2T added seeds prose '2 positive / 8 negative'",
          "split $2$ positive / $8$ negative" in txt)

    # ---- fig_frontier caption: anti pays ~3.2 points of final accuracy ----
    g = load(os.path.join(ROOT, "bench", "results_r2", "r2_grid.part*.json"))
    g += load(os.path.join(ROOT, "bench", "results_r2", "r2_anti.part*.json"))
    rec = st.mean(r["avg"] for r in g
                  if r["arm"] == "renorm_rec10" and r["ntask"] == 20)
    ant = st.mean(r["avg"] for r in g
                  if r["arm"] == "renorm_anti10" and r["ntask"] == 20)
    diff = rec - ant
    check("frontier caption: anti pays ~3.2 final-accuracy points",
          3.0 <= diff <= 3.3 and "$3.2$ points" in txt, f"diff={diff:.2f}")

    # ---- Table 1 split columns ----
    s2 = json.load(open(os.path.join(ROOT, "bench", "results_r2",
                                     "r2_stats.json"), encoding="utf-8"))
    splits = {}
    for arm in ("renorm_rec05", "renorm_rec10"):
        c = s2["cells"][f"{arm}|T20|vs_renorm|ell"]
        splits[arm] = (round(c["mnist"]["mean"], 2),
                       round(c["fashion"]["mean"], 2),
                       c["mnist"]["p"], c["fashion"]["p"])
    check("Table splits rec05: +0.10 / -0.58",
          splits["renorm_rec05"][0] == 0.10 and splits["renorm_rec05"][1] == -0.58,
          str(splits["renorm_rec05"][:2]))
    check("Table splits rec10: +1.10* / -0.22 with p=0.015/0.63",
          splits["renorm_rec10"][0] == 1.10 and splits["renorm_rec10"][1] == -0.22
          and abs(splits["renorm_rec10"][2] - 0.015) < 0.001
          and abs(splits["renorm_rec10"][3] - 0.63) < 0.01,
          str(splits["renorm_rec10"]))

    # ---- discipline section: ninth-failure disclosure present ----
    # (the disclosure legitimately QUOTES the old wrong numbers inside
    #  quotation marks; only flag them outside quotes)
    import re
    quoted = re.findall(r"``[^']*''", txt)
    outside = txt
    for q in quoted:
        outside = outside.replace(q, "")
    check("Discipline: ninth (caption) failure disclosed",
          "ninth, milder failure" in txt)
    check("old wrong strings gone (outside quotations)",
          "$3$ positive/$7$ negative" not in outside
          and "eight positive seeds" not in outside)

    # ---- nondeterminism paragraph ----
    check("R2T: cross-run noise paragraph present",
          "run-level noise can manufacture" in txt)

    # ---- shape-robustness defense ----
    check("R2: mirror-arm shape defense present",
          "survive its own mirror" in txt)
    # ---- run-count consistency (grill round 2, Q1) ----
    check("run count: no stale 1,570 anywhere",
          "1{,}570" not in txt and "1,570" not in txt and "1570" not in txt)
    # ---- protocol table (expansion pass b) ----
    # G values must match run records / prereg freeze
    r2 = json.load(open(ROOT + "/bench/results_r2/r2_grid.part0.json",
                        encoding="utf-8"))
    check("protocol: MLP/conv G=0.565 matches run records",
          r2.get("gate_const") == 0.565 and "$0.565$" in txt)
    check("protocol: ResNet G=0.5749 matches R2T prereg freeze",
          "0.5749" in open(ROOT + "/bench/results_r2t/R2T_预注册.md",
                           encoding="utf-8").read() and "0.5749" in txt)
    # optimizer constants vs code
    e9 = open(ROOT + "/bench/e9_resnet.py", encoding="utf-8").read()
    check("protocol: ResNet SGD row matches e9 constants",
          "EPOCHS, BS, LR, WD, MOM = 5, 128, 0.05, 5e-4, 0.9" in e9
          and "$128$ / $5$" in txt)
    exp4 = open(ROOT + "/bench/exp4_branch_protection.py",
                encoding="utf-8").read()
    check("protocol: MLP Adam row matches exp4 constants",
          "NTASKS, EPOCHS, BS, NTRAIN, NH = 5, 2, 256, 10000, 256" in exp4
          and "LR = 1e-3" in exp4 and "$256$ / $2$" in txt
          and "$10{,}000$" in txt)
    # R3 run arithmetic and MLP reference value
    check("protocol: R3 240-run arithmetic and MLP +5.48 reference",
          "240" in txt and "5.48" in txt
          and "5.48" in open(ROOT + "/bench/results_r3/R3_预注册.md",
                             encoding="utf-8").read())

    # ---- total run count rebuilt from raw JSON (grill round 3, Q1;
    #      extended with R4 when R4 entered the paper) ----
    def nrec(pattern):
        t = 0
        for f in sorted(glob.glob(ROOT + "/bench/" + pattern)):
            try:
                d = json.load(open(f, encoding="utf-8"))
                if isinstance(d, dict) and "results" in d:
                    t += len(d["results"])
            except Exception:
                pass
        return t

    ledger = {
        "R1 grid v2 (auth)":   nrec("results_r1/r1_grid_v2.part*.json"),
        "R1 b3 dose probe":    nrec("results_r1/r1_b3.part*.json"),
        "R1 pilot":            nrec("results_r1/r1_pilot.json"),
        "R2 grid":             nrec("results_r2/r2_grid.part*.json"),
        "R2 anti":             nrec("results_r2/r2_anti.part*.json"),
        "R2 alpha0 sentinel":  nrec("results_r2/r2_alpha0*.json"),
        "R2 e5gpu reference":  nrec("results_r2/r2_e5gpu_ref.part*.json"),
        "R2 pilot":            nrec("results_r2/r2_pilot.json"),
        "R2T grid":            nrec("results_r2t/r2t_grid*.json"),
        "R2T ext":             nrec("results_r2t/r2t_ext*.json"),
        "R2T ctrl ext":        nrec("results_r2t/r2t_ctrl_ext*.json"),
        "R2T pilot":           nrec("results_r2t/r2t_pilot.json"),
        "R3 grid":             nrec("results_r3/r3_grid.part*.json"),
        "R3 pilot":            nrec("results_r3/r3_pilot.json"),
        "R4 gprobe":           nrec("results_r4/r4_gprobe.json"),
        "R4 scan":             nrec("results_r4/r4_scan.part*.json"),
        "R4 main":             nrec("results_r4/r4_main.part*.json"),
    }
    total = sum(ledger.values())
    check("runs: ledger rebuilds 1,539 from raw JSON and text agrees",
          total == 1539 and "1{,}539" in txt
          and "1{,}213" not in txt)  # guard against silent drift

    # ================= R4 (Transformer) checks =================
    # raw main grid: rebuild arm means and the headline paired deltas
    r4 = load(os.path.join(ROOT, "bench", "results_r4",
                           "r4_main.part*.json"))
    arms = ("base", "renorm", "base_rec10", "renorm_rec10", "base_anti10")
    check("R4 main grid: 200 runs, five arms x 40 each",
          len(r4) == 200 and all(
              sum(1 for r in r4 if r["arm"] == a) == 40 for a in arms))

    def idx_of(arm, ds, seed):
        for r in r4:
            if r["arm"] == arm and r["ds"] == ds and r["seed"] == seed:
                return r
        return None

    # renorm - base, pooled over (ds, seed): +7.01, 40/40 positive
    dpool = []
    for ds in ("mnist", "fashion"):
        for s in range(20):
            a, b = idx_of("renorm", ds, s), idx_of("base", ds, s)
            if a and b:
                dpool.append(a["ell"] - b["ell"])
    pool_mean = st.mean(dpool)
    pool_pos = sum(1 for d in dpool if d > 0)
    check("R4 normalization transfers: renorm-base = +7.01, 40/40 seeds",
          round(pool_mean, 2) == 7.01 and pool_pos == 40 and len(dpool) == 40
          and "$+7.01$" in txt, f"mean={pool_mean:.3f} pos={pool_pos}")

    # frozen G and lam* from the calibration artifacts
    gp = json.load(open(os.path.join(ROOT, "bench", "results_r4",
                                     "r4_gprobe.json"), encoding="utf-8"))
    sc = json.load(open(os.path.join(ROOT, "bench", "results_r4",
                                     "r4_scan_analysis.json"),
                        encoding="utf-8"))
    check("R4 calibration: frozen G=0.5827 and lam*=100 in text and table",
          gp["frozen_G"] == 0.5827 and sc["lam_star"] == 100.0
          and "$0.5827$" in txt and "$100$ (sweep)" in txt
          and "$0.5827$ (probe)" in txt)

    # scan curve endpoints as quoted: +2.71 (lam=0), +7.10 (lam*=100),
    # +7.00 (lam=1000), monotone rise to the optimum
    cp = sc["curve_pooled"]
    check("R4 sweep: +2.71 / +7.10 at lam* / +7.00, argmax at lam*=100",
          round(cp["0.0"], 2) == 2.71 and round(cp["100.0"], 2) == 7.10
          and round(cp["1000.0"], 2) == 7.00
          and sc["lam_star"] == 100.0 and max(cp, key=cp.get) == "100.0"
          and "$+2.71$" in txt and "$+7.10$" in txt and "$+7.00$" in txt)

    # stats JSON agreement with the quoted comparison table
    ms = json.load(open(os.path.join(ROOT, "bench", "results_r4",
                                     "r4_main_stats.json"), encoding="utf-8"))
    c1e = ms["comparisons"]["C1"]["ell"]
    c2e = ms["comparisons"]["C2"]["ell"]
    c4e = ms["comparisons"]["C4"]["ell"]
    check("R4 verdict numbers: C1 +0.07 (p=.33), C2 +0.09 (p=.14) quoted",
          round(c1e["mean"], 2) == 0.07 and round(c1e["p"], 2) == 0.33
          and round(c2e["mean"], 2) == 0.09 and round(c2e["p"], 2) == 0.14
          and ms["levels"]["T3_fail"] and not ms["levels"]["T1_transfer"]
          and not ms["levels"]["T2_rescue"]
          and "reads \\textbf{T3}" in txt, 
          f"C1={c1e['mean']:.3f}/{c1e['p']:.3f} C2={c2e['mean']:.3f}/{c2e['p']:.3f}")
    check("R4 C4: -6.94 with 0/40 positive seeds, Holm family all ns",
          round(c4e["mean"], 2) == -6.94 and c4e["n_pos"] == 0
          and c4e["n_neg"] == 40
          and all(not v["significant"] for v in ms["holm_family"].values())
          and "$-6.94$" in txt and "$0/40$" in txt)

    # arm means table row values
    ex = ms["extras"]
    check("R4 arm means table row values (5 arms x ell/avg/bwt)",
          ex["base"]["ell"] == 54.68 and ex["base"]["avg"] == 24.0
          and ex["base"]["bwt"] == -32.29
          and ex["renorm"]["ell"] == 61.69 and ex["renorm"]["bwt"] == -39.69
          and ex["base_rec10"]["ell"] == 54.75
          and ex["renorm_rec10"]["ell"] == 61.77
          and ex["base_anti10"]["ell"] == 54.68
          and all(str(ex[a]["ell"]) in txt for a in arms))

    # discipline count and section header consistency
    check("R4: eight gate failures consistent across abstract/intro/table",
          "eight implementation-gate failures" in txt
          and "Eight times" in txt
          and "The Discipline: Eight Gate Failures" in txt
          and "eight preregistration-gate failures" in txt
          and "six preregistration-gate failures" not in txt)
    check("R4: weights recomputed as a function of the phase index (gate 7)",
          "function of the phase" in txt)
    check("R4 section present with calibrated-both-constants framing",
          "The Third Point on the Supply Map" in txt
          and "re-measured, not inherited" in txt)



    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n===== paper-number checks: {n_pass}/{len(CHECKS)} PASS =====")
    if n_pass != len(CHECKS):
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  FAILED: {name} {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
