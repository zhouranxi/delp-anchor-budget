# -*- coding: utf-8 -*-
"""analyze_r4_main.py -- R4 preregistration gate checker + criterion reader.

Pure Python (reuses R1's paired-t / Holm implementation via analyze_r2t-style
imports).  All gates are mathematical identities / runtime assertions only
(R4 prereg section 6); no bit-reproducibility clause (cuDNN, R2T lesson).

  python bench/analyze_r4_main.py \
      --files bench/results_r4/r4_main.part*.json \
      --scan bench/results_r4/r4_scan.part*.json \
      --out bench/results_r4/r4_main_stats.json

Differences from analyze_r2t.py (both preregistered):
  * pairing unit is (ds, seed); pooled n=40 is the primary inference, with
    per-dataset n=20 columns reported side-by-side (prereg section 4);
  * base/renorm controls are RUN IN THIS EXPERIMENT (not reused from E9);
  * G7 = calibration-vs-main consistency on overlapping seeds 0-4
    (mean |diff| <= 1.5 ell points, prereg section 6.7);
  * Holm family = C1.ell, C1.bwt, C2.ell, C2.bwt (4 tests).
"""
import argparse, glob, json, os, re, sys
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_r1 as A

ARMS = ["base", "renorm", "base_rec10", "renorm_rec10", "base_anti10"]
METRICS = ("ell", "avg", "bwt")
LAM_STAR = 100.0
G_FROZEN = 0.5827
NTASK = 20


def load(patterns):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            d = json.load(open(f, encoding="utf-8"))
            out += d["results"] if isinstance(d, dict) and "results" in d else d
    return out


def key(r):
    return (r["arm"], r["ds"], r["seed"])


def gate_check(runs, scan_runs, driver_path, verbose=True):
    g = {}

    # G1 weight conservation (identity, from logged phase_weights)
    bad_sum, bad_pos, bad_len = [], [], []
    for r in runs:
        for p, (w, s) in enumerate(zip(r["phase_weights"], r["weight_sums"])):
            if abs(s - p) > 1e-6:
                bad_sum.append((key(r), p, s))
            if any(x <= 0 for x in w):
                bad_pos.append((key(r), p))
            if len(w) != p:
                bad_len.append((key(r), p, len(w)))
    g["G1_weight_conservation"] = dict(
        ok=not (bad_sum or bad_pos or bad_len),
        bad_sum=len(bad_sum), bad_nonpositive=len(bad_pos),
        bad_length=len(bad_len))

    # G2 alpha dispatch + weight direction (identity).
    # rec arms: early < recent (recency upweights late phases);
    # anti arm: early > recent; alpha=0 arms: weights all ones (see G3).
    bad_alpha, bad_dir = [], []
    for r in runs:
        a, arm = r.get("alpha"), r["arm"]
        want_a = 0.0 if arm in ("base", "renorm") else 1.0
        if a != want_a:
            bad_alpha.append((key(r), a))
        if arm.endswith("anti10"):
            if not (r["phase_weights"][-1][0] > r["phase_weights"][-1][-1]):
                bad_dir.append((key(r), "anti-mirror"))
        elif arm.endswith("rec10"):
            if not (r["phase_weights"][-1][0] < r["phase_weights"][-1][-1]):
                bad_dir.append((key(r), "rec-mirror"))
    g["G2_alpha_dispatch"] = dict(ok=not (bad_alpha or bad_dir),
                                  bad_alpha=len(bad_alpha),
                                  bad_direction=len(bad_dir))

    # G3 alpha=0 equivalence (constructive, prereg 6.3):
    # (a) every alpha=0 run's logged weights are all ones;
    # (b) recency_weights(phase, alpha=0) returns exactly ones (verbatim
    #     import from r2_recency, no rewrite allowed).
    bad_a0_runs = []
    for r in runs:
        if r.get("alpha") == 0.0:
            for p, w in enumerate(r["phase_weights"]):
                if any(abs(x - 1.0) > 1e-12 for x in w) or len(w) != p:
                    bad_a0_runs.append(key(r))
                    break
    bad_a0_fn = []
    try:
        from r2_recency import recency_weights
        for ph in range(1, NTASK):
            w = recency_weights(ph, 0.0, NTASK, False)
            if len(w) != ph or any(abs(x - 1.0) > 1e-12 for x in w):
                bad_a0_fn.append(ph)
    except Exception as e:  # import failure itself is a gate failure
        bad_a0_fn.append("import: %s" % e)
    g["G3_alpha0_identity"] = dict(ok=not (bad_a0_runs or bad_a0_fn),
                                   bad_runs=len(bad_a0_runs),
                                   bad_fn_phases=len(bad_a0_fn))

    # G4 runtime constants (prereg 6.4).  Per-run recorded fields + driver
    # source constants (EPOCHS/BS/NTRAIN are module constants, not per-run
    # JSON fields; checking the source that produced these runs).
    bad_const = []
    for r in runs:
        if abs(r["lam_eff"] - LAM_STAR) > 1e-9 or r["d_model"] != 256 \
           or r["n_layers"] != 2 or r["k_keep"] != 128 \
           or abs(r["g_const"] - G_FROZEN) > 1e-9 or r["ntask"] != NTASK:
            bad_const.append(key(r))
    src_ok, src_note = True, ""
    try:
        src = open(driver_path, encoding="utf-8").read()
        # The driver inherits the R2 optimiser protocol VERBATIM via import
        # (prereg section 1: "task/optimiser protocol: exp6_baselines via
        # r2_recency"); asserting the imported module constants is the
        # faithful check of prereg 6.4 (revised: source-text regex -> module
        # import, criterion unchanged; see R4_修订说明.md note 5).
        if not re.search(r"from\s+exp6_baselines\s+import[^\n]*\bEPOCHS\b[^\n]*\bBS\b[^\n]*\bNTRAIN\b",
                         src):
            src_ok = False
            src_note += "driver does not import EPOCHS/BS/NTRAIN from exp6_baselines; "
        import exp6_baselines as EB
        for name, want in (("EPOCHS", 2), ("BS", 256), ("NTRAIN", 10000)):
            got = getattr(EB, name, None)
            if got != want:
                src_ok = False
                src_note += "%s=%r != %r; " % (name, got, want)
    except Exception as e:
        src_ok, src_note = False, str(e)
    g["G4_runtime_constants"] = dict(ok=not bad_const and src_ok,
                                     bad_runs=len(bad_const),
                                     source_ok=src_ok, source_note=src_note)

    # G5 completeness: metrics present, acc_matrix has ntask rows
    bad_fields = [key(r) for r in runs
                  if any(r.get(m) is None for m in METRICS)
                  or len(r["acc_matrix"]) != NTASK]
    g["G5_completeness"] = dict(ok=not bad_fields, n_bad=len(bad_fields))

    # G6 gate stats within E9/R2T band
    bad_gate = []
    for r in runs:
        for gs in r.get("gate_stats", []):
            if not (0.50 <= gs["gate_mean"] <= 0.65 and gs["gate_std"] < 0.05):
                bad_gate.append(key(r))
                break
    g["G6_gate_band"] = dict(ok=not bad_gate, n_bad=len(bad_gate),
                             window="[0.50,0.65], sd<0.05")

    # G7 calibration-vs-main consistency on overlapping seeds 0-4
    # (prereg 6.7): per (arm in base/renorm, ds), |mean_ell(main) -
    # mean_ell(scan lam=100)| <= 1.5
    scan_idx = {}
    for r in scan_runs:
        if r["arm"] in ("base", "renorm") and abs(r["lam_eff"] - LAM_STAR) < 1e-9 \
           and r["seed"] in range(5):
            scan_idx[(r["arm"], r["ds"], r["seed"])] = r
    diffs, detail = [], {}
    for arm in ("base", "renorm"):
        for ds in ("mnist", "fashion"):
            ms = [r["ell"] for r in runs
                  if r["arm"] == arm and r["ds"] == ds and r["seed"] in range(5)]
            ss = [scan_idx[k]["ell"] for k in
                  [(arm, ds, s) for s in range(5)] if k in scan_idx]
            if len(ms) == 5 and len(ss) == 5:
                d = st.mean(ms) - st.mean(ss)
                diffs.append(abs(d))
                detail["%s/%s" % (arm, ds)] = dict(main=round(st.mean(ms), 3),
                                                   scan=round(st.mean(ss), 3),
                                                   diff=round(d, 3))
            else:
                detail["%s/%s" % (arm, ds)] = dict(main_n=len(ms), scan_n=len(ss))
    g["G7_calib_consistency"] = dict(
        ok=bool(diffs) and max(diffs) <= 1.5,
        max_abs_diff=round(max(diffs), 3) if diffs else None,
        tol=1.5, detail=detail)

    g["all_ok"] = all(v["ok"] for v in g.values()
                      if isinstance(v, dict) and "ok" in v)
    if verbose:
        print("=== prereg section-6 hard implementation gates ===")
        for k, v in g.items():
            if isinstance(v, dict) and "ok" in v:
                print("  %-26s %s  %s" % (k, "PASS" if v["ok"] else "FAIL", v))
    return g


def paired_delta(idx, arm, ref, metric, filt=None):
    """Paired by (ds, seed).  filt: optional dict like {"ds": "mnist"}."""
    diffs = []
    keys = sorted(k for k in idx if k[0] == arm and k[0] == arm)
    for k in sorted(set(k for k in idx if k[0] == arm)):
        if filt and any(idx[k].get(f) != v for f, v in filt.items()):
            continue
        a, b = idx.get(k), idx.get((ref,) + k[1:])
        if a and b:
            diffs.append(a[metric] - b[metric])
    out = A.paired(diffs) if len(diffs) >= 2 else None
    if out:
        out["n_pos"] = sum(1 for d in diffs if d > 0)
        out["n_neg"] = sum(1 for d in diffs if d < 0)
    return out


def analyse(runs, scan_runs):
    idx = {key(r): r for r in runs}
    out = {"n_runs": len(runs), "comparisons": {}, "holm_family": {},
           "direction": {}, "per_dataset": {}, "levels": {}, "extras": {},
           "baseline_relative": {}}

    COMP = [("C1", "base_rec10", "base"),
            ("C2", "renorm_rec10", "renorm"),
            ("C3", "base_anti10", "base_rec10"),
            ("C4", "base_rec10", "renorm")]
    holm_inputs = []
    for cid, arm, ref in COMP:
        rec = {m: paired_delta(idx, arm, ref, m) for m in METRICS}
        out["comparisons"][cid] = {"arm": arm, "ref": ref, **rec}
        if cid in ("C1", "C2"):
            for m in ("ell", "bwt"):
                holm_inputs.append((cid + "." + m,
                                    rec[m]["p"] if rec[m] else 1.0))
    adj = A.holm([p for _, p in holm_inputs])
    for (cid_m, p), a in zip(holm_inputs, adj):
        cid, m = cid_m.split(".")
        rec = out["comparisons"][cid][m]
        out["holm_family"][cid_m] = {
            "raw_p": p, "holm_p": a, "significant": a < 0.05,
            "abs_d_z": round(abs(rec["d_z"]), 3) if rec else None}

    # per-dataset columns (prereg section 4: side-by-side, n=20 each)
    for ds in ("mnist", "fashion"):
        col = {}
        for cid, arm, ref in COMP:
            rec = {m: paired_delta(idx, arm, ref, m, {"ds": ds})
                   for m in ("ell", "bwt")}
            col[cid] = rec
        out["per_dataset"][ds] = col

    # direction consistency (pooled n=40)
    for cid in out["comparisons"]:
        d = out["comparisons"][cid]
        out["direction"][cid] = {
            m: {"pos": d[m]["n_pos"], "neg": d[m]["n_neg"],
                "all_same": d[m]["n_pos"] in (0, d[m]["n"])}
            for m in METRICS if d[m]}

    # three-level verdict (prereg section 3, written before data; thresholds
    # inherited verbatim from R2T section 5)
    c1, c2 = out["comparisons"]["C1"], out["comparisons"]["C2"]
    h = out["holm_family"]
    t1 = bool(c1["ell"] and c1["ell"]["mean"] > 0
              and c1["bwt"] and c1["bwt"]["mean"] > -1.0
              and h["C1.ell"]["significant"])
    t2 = bool(not t1 and c2["ell"] and h["C2.ell"]["significant"]
              and c2["bwt"] and c2["bwt"]["mean"] > -1.0)
    out["levels"] = {"T1_transfer": t1, "T2_rescue": t2,
                     "T3_fail": not (t1 or t2)}

    # raw arm means (pooled) + baseline-relative effect sizes
    for arm in ARMS:
        q = [idx[k] for k in idx if k[0] == arm]
        if q:
            out["extras"][arm] = {m: round(st.mean(r[m] for r in q), 2)
                                  for m in METRICS}
    if "base" in out["extras"] and "renorm" in out["extras"]:
        for cid, arm, ref in (("C1", "base_rec10", "base"),
                              ("C2", "renorm_rec10", "renorm")):
            e = out["comparisons"][cid]["ell"]
            if e:
                out["baseline_relative"][cid + ".ell"] = dict(
                    delta=round(e["mean"], 3),
                    baseline_ell=out["extras"][ref]["ell"],
                    rel_pct=round(100 * e["mean"] / out["extras"][ref]["ell"], 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="R4 main grid files (r4_main.part*.json)")
    ap.add_argument("--scan", nargs="+", required=True,
                    help="R4 calibration scan files (r4_scan.part*.json), for G7")
    ap.add_argument("--driver", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "r4_transformer.py"))
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-gate-fail", action="store_true")
    a = ap.parse_args()

    runs = load(a.files)
    scan_runs = load(a.scan)
    n_expect = 200
    if len(runs) != n_expect:
        print("WARNING: expected %d runs, got %d" % (n_expect, len(runs)),
              file=sys.stderr)

    gates = gate_check(runs, scan_runs, a.driver)
    if not gates["all_ok"] and not a.allow_gate_fail:
        print("ABORT: prereg section-6 gate failed -> void data, diagnose, "
              "rerun. Do NOT relax the criterion.", file=sys.stderr)
        sys.exit(2)

    stats = analyse(runs, scan_runs)
    stats["gates"] = gates

    print("\n=== comparisons (paired by (ds,seed), pooled n=40) ===")
    for cid, v in stats["comparisons"].items():
        e, b = v["ell"], v["bwt"]
        print("  %s %s vs %s: Dell=%s (p=%.3g)  Dbwt=%s (p=%.3g)" % (
            cid, v["arm"], v["ref"],
            ("%+.2f" % e["mean"]) if e else "NA", e["p"] if e else 1.0,
            ("%+.2f" % b["mean"]) if b else "NA", b["p"] if b else 1.0))
    print("\n=== per-dataset columns (n=20 each) ===")
    for ds, col in stats["per_dataset"].items():
        print("  [%s]" % ds)
        for cid in ("C1", "C2"):
            v = col[cid]
            print("    %s: Dell=%s (p=%.3g)  Dbwt=%s (p=%.3g)" % (
                cid,
                ("%+.2f" % v["ell"]["mean"]) if v["ell"] else "NA",
                v["ell"]["p"] if v["ell"] else 1.0,
                ("%+.2f" % v["bwt"]["mean"]) if v["bwt"] else "NA",
                v["bwt"]["p"] if v["bwt"] else 1.0))
    print("\n=== Holm family (C1/C2 x ell/bwt, 4 tests) ===")
    for k, v in stats["holm_family"].items():
        print("  %s: raw=%.3g holm=%.3g %s |d_z|=%s" % (
            k, v["raw_p"], v["holm_p"],
            "SIG" if v["significant"] else "ns", v["abs_d_z"]))
    print("\n=== direction consistency (pooled n=40) ===")
    for cid, v in stats["direction"].items():
        print("  %s: %s" % (cid, "  ".join(
            "%s +%d/-%d" % (m, d["pos"], d["neg"]) for m, d in v.items())))
    print("\n=== three-level verdict (thresholds inherited from R2T) ===")
    print(" ", stats["levels"])
    print("\n=== baseline-relative effect sizes ===")
    for k, v in stats["baseline_relative"].items():
        print("  %s: delta=%+.2f on baseline ell=%.2f -> %+.2f%%" % (
            k, v["delta"], v["baseline_ell"], v["rel_pct"]))
    print("\n=== arm means (pooled, 40 runs each) ===")
    for arm, v in stats["extras"].items():
        print("  %-12s ell=%6.2f avg=%6.2f bwt=%+7.2f" % (
            arm, v["ell"], v["avg"], v["bwt"]))

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
