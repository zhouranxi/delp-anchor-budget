"""Non-parametric significance testing for the S2 manuscript (ASC submission).

Motivation: Applied Soft Computing reviewers routinely ask for non-parametric
tests (Wilcoxon / Friedman) rather than t-tests alone, because "our number is
2 points higher" without a distribution-free significance test is treated as
noise. This script recomputes, straight from the raw per-seed records:

  * Shapiro-Wilk on the paired deltas (is the t-test's normality assumption met?)
  * Wilcoxon signed-rank on the paired deltas (the distribution-free test)
  * the paired t-test, for comparison

Run from the package root:  python audit/audit_nonparametric.py
Nothing is hard-coded; every number comes from results/anchor_budget_all.json.
"""
import json
import os
import statistics as st

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "anchor_budget_all.json")

with open(RESULTS, encoding="utf-8") as f:
    ALL = json.load(f)["results"]

LAM = 5000.0


def get(policy, M, ntask, kind="delp", dataset="mnist", lam=LAM, files=None):
    """Collect records for one (policy, M) cell, keyed by seed.

    The unbounded arm is stored with an arbitrary M tag in some files, so it
    is matched on policy + seed only (M is ignored for 'unbounded').
    """
    out = {}
    for r in (files if files is not None else ALL):
        if r.get("kind") != kind or "error" in r:
            continue
        if r.get("policy") != policy or r.get("ntask") != ntask:
            continue
        if r.get("dataset", "mnist") != dataset:
            continue
        if str(r.get("lam", LAM)) != str(lam):
            continue
        if policy != "unbounded" and r.get("M") != M:
            continue
        out[r["seed"]] = r
    return out


def load_all():
    """Load every results/*.json* record into one pool.

    The CIFAR runs live in their own files and use the dataset label
    'cifar100' (not 'cifar'), and they carry a different lam, so a single
    file is not enough to reproduce the CIFAR policy grid. Several files use
    the '.json.partN' suffix, so match on '.json*'.
    """
    pool = list(ALL)
    rdir = os.path.join(ROOT, "results")
    seen = set()
    for fn in sorted(os.listdir(rdir)):
        if ".json" not in fn or fn == "anchor_budget_all.json":
            continue
        path = os.path.join(rdir, fn)
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        rs = d.get("results", d) if isinstance(d, dict) else d
        if isinstance(rs, list):
            pool.extend(rs)
    return pool


ALL = load_all()


def fmt_p(p):
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.4f}"


def paired_report(arm, M, ntask, label):
    ub = get("unbounded", None, ntask)
    ar = get(arm, M, ntask)
    seeds = sorted(set(ub) & set(ar))
    if len(seeds) < 3:
        print(f"{label}: SKIP (only {len(seeds)} paired seeds)")
        return None
    da = [ar[s]["final_avg"] - ub[s]["final_avg"] for s in seeds]
    db = [ar[s]["bwt"] - ub[s]["bwt"] for s in seeds]
    print(f"\n{label}   (n = {len(seeds)} paired seeds)")
    for name, dv in (("final accuracy", da), ("BWT", db)):
        W, p = stats.wilcoxon(dv, alternative="two-sided", method="auto")
        sw, swp = stats.shapiro(dv)
        t = stats.ttest_rel([ar[s]["final_avg"] for s in seeds],
                            [ub[s]["final_avg"] for s in seeds]) \
            if name.startswith("final") else \
            stats.ttest_rel([ar[s]["bwt"] for s in seeds],
                            [ub[s]["bwt"] for s in seeds])
        sign = "all > 0" if all(x > 0 for x in dv) else (
            "all < 0" if all(x < 0 for x in dv) else "mixed")
        print(f"  {name:14s} mean Δ = {st.mean(dv):+7.2f}  sd = {st.stdev(dv):5.2f}"
              f"  ({sign})")
        print(f"  {'':14s} Wilcoxon signed-rank: W = {W:.1f}, {fmt_p(p)}")
        print(f"  {'':14s} paired t-test:        t = {t.statistic:.2f}, "
              f"{fmt_p(t.pvalue)}")
        print(f"  {'':14s} Shapiro-Wilk on Δ:    W = {sw:.3f}, {fmt_p(swp)}"
              f"  (normality {'OK' if swp > 0.05 else 'VIOLATED'})")
    return da, db


if __name__ == "__main__":
    print("=" * 78)
    print("Non-parametric significance audit -- recomputed from raw records")
    print("=" * 78)
    for M, nt, lab in [
        (2, 10, "FIFO M=2 vs unbounded, Permuted-MNIST T=10"),
        (5, 10, "FIFO M=5 vs unbounded, Permuted-MNIST T=10"),
        (5, 20, "FIFO M=5 vs unbounded, Permuted-MNIST T=20"),
    ]:
        paired_report("fifo", M, nt, lab)

    print("\n" + "=" * 78)
    print("CIFAR policy grid, T=20 (the 'five of six' claim)")
    print("=" * 78)
    # CIFAR uses dataset label 'cifar100' and its own lambda; find them
    cif = [r for r in ALL if r.get("dataset") == "cifar100" and "error" not in r]
    if not cif:
        print("  no CIFAR records found in results/")
    else:
        lams = sorted(set(r.get("lam") for r in cif))
        print(f"  CIFAR records: {len(cif)}   lambda values present: {lams}")
        for nt in (10, 20):
            c10 = [r for r in cif if r.get("ntask") == nt]
            if not c10:
                continue
            lam_c = max(set(r.get("lam") for r in c10),
                        key=lambda L: sum(1 for r in c10 if r.get("lam") == L))
            print(f"\n  --- T={nt}, lambda={lam_c} ---")
            ub = get("unbounded", None, nt, dataset="cifar100", lam=lam_c)
            pols = ["fifo", "agew", "random", "redun", "agew2", "merge"]
            rows = []
            for pol in pols:
                for M in (2, 5):
                    ar = get(pol, M, nt, dataset="cifar100", lam=lam_c)
                    seeds = sorted(set(ub) & set(ar))
                    if len(seeds) < 3:
                        continue
                    da = [ar[s]["final_avg"] - ub[s]["final_avg"] for s in seeds]
                    W, p = stats.wilcoxon(da, alternative="two-sided", method="auto")
                    rows.append((pol, M, len(seeds), st.mean(da), W, p))
            if not rows:
                print("    (no paired policy records)")
                continue
            for pol, M, n, md, W, p in rows:
                flag = "*" if (p < 0.05 and md > 0) else " "
                print(f"   {flag} {pol:7s} M={M}  n={n:2d}  mean Δ = {md:+6.2f}  "
                      f"Wilcoxon W = {W:5.1f}  {fmt_p(p)}")
            print("    (* = significantly beats unbounded at α=0.05)")

    # Friedman omnibus across policies (MNIST, where all policies have n=10)
    print("\n" + "=" * 78)
    print("Friedman omnibus test across policies, Permuted-MNIST T=20, M=5")
    print("=" * 78)
    pols = ["fifo", "agew", "random", "redun", "agew2", "merge"]
    seeds = None
    series = []
    for pol in pols:
        ar = get(pol, 5, 20)
        s = set(ar) & set(get("unbounded", None, 20))
        seeds = s if seeds is None else (seeds & s)
        series.append((pol, ar))
    seeds = sorted(seeds)
    if len(seeds) >= 4:
        cols = [[ar[s]["final_avg"] for s in seeds] for _, ar in series]
        try:
            stat, p = stats.friedmanchisquare(*cols)
            print(f"  n = {len(seeds)} paired seeds across {len(pols)} policies")
            print(f"  Friedman chi2 = {stat:.2f}, {fmt_p(p)}")
        except Exception as e:
            print(f"  Friedman failed: {e}")
    else:
        print(f"  only {len(seeds)} common paired seeds -- skipped")
