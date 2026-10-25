# -*- coding: utf-8 -*-
"""One-shot S1 reproduction driver.

Re-derives every paper artifact that S1 (Branch-DeLP / dose-response audit)
reports, from the committed raw records under bench/results/. This is the
"reproduce the analysis + figures" chain, NOT retraining — the trained
records are the committed data; retraining (bench/branch_delp.py,
signed_drift.py, gating_evolution.py) needs a GPU box and is launched
separately (see README.md §3).

Run from the repo root:
    python s1_repro/run_s1_repro.py

It calls, in order:
  1. bench/analyze_s1.py          -> bench/results/s1_paired_stats.json
  2. paper_s1/make_figs.py        -> paper_s1/figs/fig_{evolution,deadfrac,artifact,dose}.pdf

Exit code 0 only if both steps succeed and their outputs exist.
"""
import os, sys, subprocess, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

def run(rel, tag):
    script = os.path.join(ROOT, rel)
    print(f"\n===== [{tag}] {rel} =====")
    r = subprocess.run([PY, script], cwd=ROOT)
    if r.returncode != 0:
        print(f"[{tag}] FAILED (rc={r.returncode})")
        sys.exit(1)

def check(path, tag):
    if not os.path.exists(path):
        print(f"[{tag}] MISSING OUTPUT: {path}")
        sys.exit(1)
    print(f"[{tag}] OK  {path}")

if __name__ == "__main__":
    run("bench/analyze_s1.py", "analyze_s1")
    check(os.path.join(ROOT, "bench", "results", "s1_paired_stats.json"), "analyze_s1")

    run("paper_s1/make_figs.py", "make_figs")
    for f in ("fig_evolution", "fig_deadfrac", "fig_artifact", "fig_dose"):
        check(os.path.join(ROOT, "paper_s1", "figs", f + ".pdf"), "make_figs")

    print("\nS1 analysis + figures reproduced. See s1_repro/manifest.txt for the data map.")
