"""Academic-English lint for the S2 manuscript.

Elsevier states that poor language is one of the leading causes of desk
rejection -- often ahead of novelty. This script surfaces the usual
non-native-English patterns in the *prose* of main.tex (LaTeX commands,
math, and the bibliography are excluded) so they can be fixed by hand.

Run from the package root:  python audit/audit_language.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "main.tex")

with open(TEX, encoding="utf-8") as f:
    raw = f.read()

# ---- strip comments, math, and LaTeX -----------------------------
text = re.sub(r"(?<!\\)%.*", "", raw)
text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", text, flags=re.S)
text = re.sub(r"\\begin\{align\}.*?\\end\{align\}", " ", text, flags=re.S)
text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
text = re.sub(r"\$[^$]*\$", " ", text)
text = re.sub(r"\\label\{[^}]*\}", " ", text)
text = re.sub(r"\\ref\{[^}]*\}", "REF", text)
text = re.sub(r"\\cite[a-z]*\{[^}]*\}", "CITE", text)
text = re.sub(r"\\footnote\{[^}]*\}", " ", text)
text = re.sub(r"\\texttt\{[^}]*\}", "CODE", text)
text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
text = re.sub(r"[{}$&_^~]", " ", text)
text = re.sub(r"\\\\", " ", text)

# sentence-ish chunks with line numbers preserved
lines = raw.split("\n")
issues = {}

LONG_SENT = 45          # words

# British vs American spelling: Elsevier accepts either but forbids mixing.
# The manuscript is consistently American, so any British variant is an error.
SPELLING = {
    "behaviour": "behavior", "behavioural": "behavioral",
    "behaviourally": "behaviorally", "behaviouralism": "behavioralism",
    "colour": "color", "colours": "colors", "coloured": "colored",
    "colouring": "coloring", "colourful": "colorful",
    "favour": "favor", "favours": "favors", "favoured": "favored",
    "favouring": "favoring", "favourable": "favorable",
    "modelling": "modeling", "modelled": "modeled", "labelling": "labeling",
    "labelled": "labeled", "centre": "center", "centres": "centers",
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "generalise": "generalize", "generalised": "generalized",
    "normalise": "normalize", "normalised": "normalized",
    "recognise": "recognize", "recognised": "recognized",
    "emphasise": "emphasize", "emphasised": "emphasized",
    "summarise": "summarize", "summarised": "summarized",
    "optimise": "optimize", "optimised": "optimized",
    "characterise": "characterize", "characterised": "characterized",
    "minimise": "minimize", "maximise": "maximize",
    "utilise": "utilize", "utilised": "utilized",
    "parameterise": "parameterize", "parameterised": "parameterized",
    "organisation": "organization", "organisations": "organizations",
    "initialise": "initialize", "initialised": "initialized",
}

# Hedges / intensifiers that the writing standards call out as unnecessary.
# "rather than" is excluded: it is a legitimate contrastive construction
# ("X rather than Y"), not a hedge.
HEDGES = [
    r"\bclearly\b", r"\bobviously\b", r"\bevidently\b",
    r"\bvery\s+\w+", r"\bquite\b", r"\bextremely\b",
    r"\bhighly\s+(suggest|indicate|likely)\b",
]

PHRASES = [
    # (pattern, label, suggestion)
    (r"\bin order to\b", "in order to", "to"),
    (r"\bdue to the fact that\b", "due to the fact that", "because"),
    (r"\bin spite of the fact that\b", "in spite of the fact that", "although"),
    (r"\bit is worth noting that\b", "it is worth noting that",
     "notably, / (drop)"),
    (r"\bit should be noted that\b", "it should be noted that", "(drop)"),
    (r"\bvery\s+(important|significant|clear|useful|large|small|good|bad)\b",
     "very + adj", "stronger adjective"),
    (r"\bquite\s+(a|an)?\b", "quite", "(delete)"),
    (r"\bkind of\b|\bsort of\b", "kind of / sort of", "(delete)"),
    (r"\ba lot of\b", "a lot of", "many / much"),
    (r"\bcan be seen from\b", "can be seen from", "shows / follows from"),
    (r"\bwe would like to\b", "we would like to", "we / (drop)"),
    (r"\bit is important to note that\b", "it is important to note that",
     "(drop or rephrase)"),
    (r"\bin the case of\b", "in the case of", "for / when"),
    (r"\bfor the purpose of\b", "for the purpose of", "to / for"),
    (r"\bwith respect to\b", "with respect to", "for / concerning"),
    (r"\bon the basis of\b", "on the basis of", "from / using"),
    (r"\bmake a comparison between\b", "make a comparison between", "compare"),
    (r"\bcarry out\b", "carry out", "run / perform"),
    (r"\bgive rise to\b", "give rise to", "produce / cause"),
    (r"\btake into consideration\b", "take into consideration", "consider"),
    (r"\bin the process of\b", "in the process of", "while / when"),
    (r"\bplay a role in\b", "play a role in", "affect / matter for"),
    (r"\bis able to\b", "is able to", "can"),
    (r"\bare able to\b", "are able to", "can"),
    (r"\bhas the ability to\b", "has the ability to", "can"),
    (r"\bthere is a possibility that\b", "there is a possibility that", "(drop)"),
    (r"\bwe found that that\b", "we found that that", "we found that"),
    (r"\bthe the\b", "the the", "the"),
    (r"\bof of\b", "of of", "of"),
    (r"\band and\b", "and and", "and"),
    (r"\bthat that\b", "that that", "that"),
    (r"\bis is\b", "is is", "is"),
    (r"\bin in\b", "in in", "in"),
    (r"\bdata is\b", "data is", "data are (or keep, but be consistent)"),
]

# ---- scan ---------------------------------------------------------
for i, line in enumerate(lines, 1):
    stripped = re.sub(r"(?<!\\)%.*", "", line)
    if not stripped.strip():
        continue
    # skip pure-command lines
    if stripped.strip().startswith("\\") and " " not in stripped.strip()[:40]:
        continue

    for pat, label, sug in PHRASES:
        for m in re.finditer(pat, stripped, re.I):
            issues.setdefault(label, []).append((i, sug, stripped.strip()[:95]))

    # repeated words
    for m in re.finditer(r"\b(\w+)\s+\1\b", stripped, re.I):
        w = m.group(1)
        if w.lower() not in ("that", "had", "the"):
            issues.setdefault(f"repeated word: {w}", []).append(
                (i, "delete one", stripped.strip()[:95]))

    # British/American spelling mixing (Elsevier forbids mixing)
    for brit, amer in SPELLING.items():
        for m in re.finditer(rf"\b{brit}\b", stripped, re.I):
            issues.setdefault(f"British spelling: {brit}", []).append(
                (i, f"use American '{amer}'", stripped.strip()[:95]))

    # unnecessary hedges / intensifiers
    for pat in HEDGES:
        for m in re.finditer(pat, stripped, re.I):
            issues.setdefault(f"hedge/intensifier: {m.group(0).strip()}", []).append(
                (i, "delete or strengthen", stripped.strip()[:95]))

# ---- long sentences (operate on the flattened prose) --------------
flat = re.sub(r"\s+", " ", text)
sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", flat)
for s in sentences:
    w = len(re.findall(r"[A-Za-z][A-Za-z'-]*", s))
    if w > LONG_SENT:
        issues.setdefault(f"long sentence (>{LONG_SENT} words)", []).append(
            (0, f"{w} words -- split it", s[:95]))

# ---- report -------------------------------------------------------
print("=" * 74)
print("Academic-English lint  (prose only; LaTeX/math/bib excluded)")
print("=" * 74)
if not issues:
    print("  No patterns flagged.")
else:
    total = 0
    for label in sorted(issues, key=lambda k: -len(issues[k])):
        rows = issues[label]
        total += len(rows)
        sug = rows[0][1]
        print(f"\n  [{len(rows):2d}] {label}  ->  {sug}")
        shown = 0
        for ln, _, ctx in rows:
            if ln == 0 or shown >= 4:
                continue
            print(f"        L{ln}: {ctx}")
            shown += 1
    print(f"\n  total flags: {total}")
print()
print("Note: line 0 = flattened prose (line number not attributable).")
