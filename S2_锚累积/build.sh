#!/usr/bin/env bash
# Build the S2 manuscript (Pattern Recognition submission).
#
# Uses the TinyTeX already present on this machine (C:/Users/Lance/TinyTeX),
# which ships elsarticle.cls and every package main.tex needs.
#
#   ./build.sh          normal build (pdflatex -> bibtex -> pdflatex x2)
#   ./build.sh clean    remove all build artefacts first
#
# Artefacts produced: main.pdf, main.aux, main.bbl, main.blg, main.log

set -euo pipefail

TINYTEX_BIN="/c/Users/Lance/TinyTeX/bin/windows"
export PATH="$TINYTEX_BIN:$PATH"

cd "$(dirname "$0")"

if [ "${1:-}" = "clean" ]; then
    rm -f main.aux main.bbl main.blg main.log main.out main.pdf
    echo "cleaned"
fi

# 0) pre-flight: catch LaTeX errors before burning 4 compile passes
echo "==> pre-flight checks"
python audit/check_latex.py

# 1) first pdflatex: writes main.aux (citation keys) and main.bbl is stale
echo "==> pdflatex (pass 1)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /dev/null

# 2) bibtex: resolve citations against refs.bib
echo "==> bibtex"
bibtex main

# 3 & 4) re-run pdflatex twice: resolve \ref/\cite, then stabilise page breaks
echo "==> pdflatex (pass 2)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /dev/null
echo "==> pdflatex (pass 3)"
pdflatex -interaction=nonstopmode -file-line-error main.tex > /dev/null

# 5) report
echo
echo "==> build report"
python audit/check_latex.py | tail -8
echo
grep -E "Output written on main\.pdf" main.log || true
echo "bibitems in main.bbl : $(grep -c '^\\bibitem' main.bbl)"
echo "bibcites in main.aux : $(grep -c 'bibcite' main.aux)"
echo "undefined refs/cites : $(grep -cE 'Citation .* undefined|Reference .* undefined' main.log || true)"
echo
echo "main.pdf -> $(pwd)/main.pdf"
