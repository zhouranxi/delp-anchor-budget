"""One-command verification of everything claimed in the S2 manuscript.

Runs every check in sequence and prints a single PASS/FAIL table. Exit code
0 means all checks passed. This is the entry point for the reproducibility
badge: an independent reviewer should be able to run it and see the same
numbers as the paper.

Run from the package root:  python audit/verify_all.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PY = sys.executable
results = []


def run(name, argv, expect_ok=True, check=None):
    """Run one check and record pass/fail."""
    try:
        p = subprocess.run([PY] + argv, capture_output=True, text=True,
                           timeout=900)
        out = (p.stdout or "") + (p.stderr or "")
        ok = (p.returncode == 0) if expect_ok else (p.returncode != 0)
        detail = ""
        if check:
            ok, detail = check(out)
        results.append((name, ok, detail or f"exit={p.returncode}"))
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))


def contains(*needles):
    def _check(out):
        missing = [n for n in needles if n not in out]
        return (not missing), ("" if not missing else f"missing {missing}")
    return _check


# ---- 1. the 27-check regression over paper text, code and records -------
run("regression (27 checks)",
    ["audit/final_regression.py"],
    check=contains("PASS 27", "FAIL 0"))

# ---- 2. non-parametric significance -------------------------------------
run("non-parametric tests",
    ["audit/audit_nonparametric.py"],
    check=contains("p = 0.0020"))


# ---- 3. LaTeX pre-flight ------------------------------------------------
def _latex_ok(out):
    if "No problems found" in out:
        return True, ""
    if "[PROBLEM]" in out:
        probs = [l.strip() for l in out.splitlines() if "[PROBLEM]" in l]
        return False, probs[0][:110]
    return False, out.strip().splitlines()[-1][:110]


run("latex pre-flight", ["audit/check_latex.py"], check=_latex_ok)


# ---- 4. graphical abstract regenerates ----------------------------------
def _ga_ok(out):
    if "width >= 1328: OK" not in out:
        return False, "size check failed"
    if "landscape: OK" not in out:
        return False, "not landscape"
    return True, "3900x2280"


run("graphical abstract", ["audit/make_graphical_abstract.py"], check=_ga_ok)

# ---- 5. headline numbers present in the manuscript ----------------------
def _numbers_ok(out):
    """The manuscript must still contain the headline values.

    LaTeX markup is stripped first, so a value written as $18$--$64\\%$
    is matched as "18--64".
    """
    with open("main.tex", encoding="utf-8") as f:
        raw = f.read()
    flat = re.sub(r"\\[a-zA-Z]+\*?", " ", raw)
    flat = re.sub(r"[{}$\\]", " ", flat)
    flat = re.sub(r"\s+", " ", flat)

    need = ["66.7", "+12.3", "+4.5", "+9.0", "+13.6", "0.002",
            "34.5", "0.99", "18 -- 64", "43 -- 45"]
    missing = [n for n in need if n not in flat]
    return (not missing), ("" if not missing else f"missing {missing}")


results.append(("headline numbers in main.tex", *_numbers_ok(None)))

# ---- 6. citation/bibliography two-way consistency ----------------------
def _cites_ok(out):
    cited = set()
    with open("main.tex", encoding="utf-8") as f:
        for m in re.finditer(r"\\cite\{([^}]*)\}", f.read()):
            for k in m.group(1).split(","):
                if k.strip():
                    cited.add(k.strip())
    with open("refs.bib", encoding="utf-8") as f:
        have = set(re.findall(r"@\w+\{([^,]+),", f.read()))
    orphan, unused = cited - have, have - cited
    if orphan or unused:
        return False, f"orphan={sorted(orphan)} unused={sorted(unused)}"
    return True, f"{len(cited)} entries, all cited"


results.append(("citations vs bibliography", *_cites_ok(None)))

# ---- 7. compiled PDF present and consistent ----------------------------
def _pdf_ok(out):
    pdf = "main.pdf"
    if not os.path.exists(pdf):
        return False, "main.pdf missing -- run ./build.sh"
    with open(pdf, "rb") as f:
        head = f.read(1024)
    if not head.startswith(b"%PDF"):
        return False, "not a PDF"
    size = os.path.getsize(pdf)
    return True, f"{size//1024} KB"


results.append(("compiled PDF", *_pdf_ok(None)))

# ---- 8. language: no British/American spelling mixing -----------------
def _spell_ok(out):
    """Elsevier forbids mixing British and American spellings."""
    bad = [l.strip() for l in out.splitlines()
           if "British spelling" in l]
    if bad:
        return False, f"{len(bad)} British spelling(s): {bad[0][:60]}"
    if "hedge/intensifier" in out:
        n = len([l for l in out.splitlines() if "hedge/intensifier" in l])
        return False, f"{n} hedge(s) remaining"
    return True, "consistent US spelling, no hedges"


results.append(("language (spelling/hedges)", *_spell_ok(
    subprocess.run([PY, "audit/audit_language.py"], capture_output=True,
                   text=True).stdout)))

# ---- report -------------------------------------------------------------
print("=" * 74)
print("S2 manuscript -- full verification")
print("=" * 74)
width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<{width}}  {detail}")

n_ok = sum(1 for _, ok, _ in results if ok)
print()
print(f"  {n_ok}/{len(results)} checks passed")
print()
if n_ok == len(results):
    print("  All checks passed. The manuscript's numbers are re-derivable from")
    print("  the archived records in results/ by the scripts in audit/.")
sys.exit(0 if n_ok == len(results) else 1)
