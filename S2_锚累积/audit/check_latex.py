"""Pre-compile sanity checks for the S2 manuscript.

Catches the LaTeX errors that would otherwise only surface as compiler
warnings or a failed build:

  1. \\ref{...} pointing at a label that does not exist   -> renders as ??
  2. \\label{...} defined twice
  3. \\cite{...} with no matching entry in refs.bib       -> undefined citation
  4. refs.bib entries never cited in the text             -> silently dropped by BibTeX
  5. tabular column spec vs. the actual number of cells in each row
  6. unbalanced braces / $ math delimiters
  7. leftover journal names from a previous target

Run from the package root:  python audit/check_latex.py
Exit code 0 = all clean.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "main.tex")
BIB = os.path.join(ROOT, "refs.bib")

with open(TEX, encoding="utf-8") as f:
    tex = f.read()
with open(BIB, encoding="utf-8") as f:
    bib = f.read()

problems = []
notes = []


def add(msg):
    problems.append(msg)


# ---- strip comments (unescaped % only) ---------------------------
def strip_comments(s):
    out = []
    for line in s.split("\n"):
        res, esc, in_m = [], False, False
        for ch in line:
            if esc:
                res.append(ch)
                esc = False
                continue
            if ch == "\\":
                res.append(ch)
                esc = True
                continue
            if ch == "%":
                break
            res.append(ch)
        out.append("".join(res))
    return "\n".join(out)


body = strip_comments(tex)

# ---- 1 & 2: labels vs refs ---------------------------------------
labels = re.findall(r"\\label\{([^}]*)\}", body)
refs = re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]*)\}", body)

dupes = {x for x in labels if labels.count(x) > 1}
if dupes:
    add(f"duplicate \\label: {sorted(dupes)}")

missing = sorted(set(refs) - set(labels))
if missing:
    add(f"\\ref to undefined label (renders as ??): {missing}")
else:
    notes.append(f"labels/refs consistent: {len(set(labels))} labels, "
                 f"{len(set(refs))} refs")

unused_labels = sorted(set(labels) - set(refs))
if unused_labels:
    notes.append(f"labels never referenced (harmless): {unused_labels}")

# ---- 3 & 4: citations vs bib ------------------------------------
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k:
            cited.add(k)

bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))

orphan = sorted(cited - bibkeys)
if orphan:
    add(f"\\cite with no refs.bib entry (undefined citation): {orphan}")

unused = sorted(bibkeys - cited)
if unused:
    add(f"refs.bib entries never cited -- BibTeX will SILENTLY DROP them: "
        f"{unused}  (fix: cite them in the text, or delete them)")
else:
    notes.append(f"citations/bib consistent: {len(cited)} entries, all cited")

# ---- 5: tabular geometry -----------------------------------------
bad_tab = 0
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                     body, re.S):
    spec, tbody = m.group(1), m.group(2)
    ncol = len(re.findall(r"[lcr]", spec))
    for r in tbody.split("\\\\"):
        if "&" not in r:
            continue
        if re.search(r"toprule|midrule|bottomrule|cmidrule|multicolumn",
                     r):
            continue
        cells = len(r.split("&"))
        if cells != ncol:
            line = body[:m.start()].count("\n") + 1
            add(f"tabular at ~line {line}: spec {spec!r} implies {ncol} cols "
                f"but row has {cells}: {r.strip()[:60]!r}")
            bad_tab += 1
if not bad_tab:
    notes.append("all tabular column specs match their rows")

# ---- 6: brace / math balance -------------------------------------
stripped = re.sub(r"\\[a-zA-Z]+", "", body)
if stripped.count("{") != stripped.count("}"):
    add(f"brace imbalance: {stripped.count('{')} open vs "
        f"{stripped.count('}')} close")
if stripped.count("$") % 2 != 0:
    add(f"odd number of $ math delimiters: {stripped.count('$')}")

# ---- 6b: environment begin/end pairing --------------------------
import collections
opens = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", body))
closes = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", body))
for env in sorted(set(list(opens) + list(closes))):
    if opens[env] != closes[env]:
        add(f"environment {env!r} unbalanced: "
            f"{opens[env]} \\begin vs {closes[env]} \\end")

# ---- 6c: unescaped % inside a line (not a full-line comment) -----
# A '%' that starts a comment is fine. A '%' appearing mid-line after
# non-comment text must be escaped as \% or it swallows the rest of the line.
for i, raw in enumerate(body.split("\n"), 1):
    s = raw.strip()
    if not s or s.startswith("%"):
        continue
    # find a % that is not preceded by a backslash
    for m in re.finditer(r"(?<!\\)%", raw):
        add(f"line {i}: unescaped '%' will comment out the rest of the line: "
            f"{raw.strip()[:70]!r}")
        break

# ---- 7: stale journal names --------------------------------------
for name in ["Applied Soft Computing", "Neurocomputing", "Neural Networks",
             "Information Sciences", "Engineering Applications of Artificial"]:
    hits = [i + 1 for i, l in enumerate(tex.split("\n")) if name in l]
    if hits:
        # allowed: a comment explaining the choice, or the decision history
        for h in hits:
            line = tex.split("\n")[h - 1]
            if line.strip().startswith("%"):
                continue
            if "\\journal" in line:
                continue
            if "decision" in line.lower() or "NEUCOM" in line:
                continue
            add(f"stale journal name {name!r} at line {h}: {line.strip()[:80]!r}")

# ---- report -------------------------------------------------------
print("=" * 72)
print("LaTeX pre-compile checks")
print("=" * 72)
for n in notes:
    print(f"  [info] {n}")
print()
if problems:
    for p in problems:
        print(f"  [PROBLEM] {p}")
    print(f"\n{len(problems)} problem(s) found.")
    sys.exit(1)
print("No problems found. Ready to compile.")
sys.exit(0)
