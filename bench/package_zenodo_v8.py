# -*- coding: utf-8 -*-
"""Build the Zenodo v8 package from the current repository state.

v8 differs from v7 in three ways:

  1. The manuscript now carries the A5 mortality scan (160 server runs) as
     Table `tab:scanmort`, plus the n=5 mechanism-attribution numbers and the
     n=10 signed-drift statistics.  The per-phase t values in the text were
     re-derived on a second machine because the n=3 values were not
     reproducible (the sample sd of three numbers amplifies float noise).
  2. The supplementary material reads its S1 table from the n=10 server run
     and its S6 table from the n=5 server run, so the supplement and the
     main text quote the same runs.
  3. The code deposit gains the scripts that produced those numbers and the
     protocol-consistency checker, plus the raw server result files.

Every file is copied from the repository; nothing is regenerated here.
"""
import hashlib
import json
import os
import shutil
import sys

REPO = r'C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\dab_t\delp-anchor-budget-main'
PKG = os.environ.get('ZENODO_PKG', r'C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\_zenodo_v8')
AUDIT = r'C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\_audit_s3'

PAPER = os.path.join(REPO, 'S123_合并稿', 'paper')
TMLR = os.path.join(REPO, 'S123_合并稿', 'paper_tmlr')
BENCH = os.path.join(REPO, 'bench')
RESULTS = os.path.join(BENCH, 'results_s5')
SERVER = os.path.join(RESULTS, 'server')

# (destination relative path, source absolute path)
PLAN = [
    # ---------------- paper/ ----------------
    ('paper/main.tex', os.path.join(PAPER, 'main.tex')),
    ('paper/main.pdf', os.path.join(PAPER, 'main.pdf')),
    ('paper/main.bbl', os.path.join(PAPER, 'main.bbl')),
    ('paper/supplementary.tex', os.path.join(PAPER, 'supplementary.tex')),
    ('paper/supplementary.pdf', os.path.join(PAPER, 'supplementary.pdf')),
    ('paper/refs.bib', os.path.join(PAPER, 'refs.bib')),
    ('paper/stats_recomputed.json', os.path.join(PAPER, 'stats_recomputed.json')),
    ('paper/stats_recomputed.md', os.path.join(PAPER, 'stats_recomputed.md')),
    ('paper/s5_analysis.json', os.path.join(PAPER, 's5_analysis.json')),
    ('paper/s5_analysis.md', os.path.join(PAPER, 's5_analysis.md')),
    ('paper/scan_mortality.json', os.path.join(PAPER, 'scan_mortality.json')),
    ('paper/scan_mortality.md', os.path.join(PAPER, 'scan_mortality.md')),
    ('paper/metric_sensitivity.json', os.path.join(PAPER, 'metric_sensitivity.json')),
    ('paper/tost_results.json', os.path.join(PAPER, 'tost_results.json')),
    ('paper/zenodo_metadata.json', os.path.join(PAPER, 'zenodo_metadata.json')),

    # ---------------- paper/figs ----------------
    ('paper/figs/fig_headroom.pdf', os.path.join(PAPER, 'figs', 'fig_headroom.pdf')),
    ('paper/figs/fig_mortality.pdf', os.path.join(PAPER, 'figs', 'fig_mortality.pdf')),
    ('paper/figs/fig_tradeoff.pdf', os.path.join(PAPER, 'figs', 'fig_tradeoff.pdf')),
    ('paper/figs/fig_dose.pdf', os.path.join(PAPER, 'figs', 'fig_dose.pdf')),
    ('paper/figs/fig_dose3.pdf', os.path.join(PAPER, 'figs', 'fig_dose3.pdf')),
    ('paper/figs/fig_evolution.pdf', os.path.join(PAPER, 'figs', 'fig_evolution.pdf')),
    ('paper/figs/fig_artifact.pdf', os.path.join(PAPER, 'figs', 'fig_artifact.pdf')),
    ('paper/figs/fig_deadfrac.pdf', os.path.join(PAPER, 'figs', 'fig_deadfrac.pdf')),

    # ---------------- tmlr/ ----------------
    ('tmlr/main.tex', os.path.join(TMLR, 'main.tex')),
    ('tmlr/main.pdf', os.path.join(TMLR, 'main.pdf')),
    ('tmlr/refs.bib', os.path.join(TMLR, 'refs.bib')),
    ('tmlr/tmlr.bst', os.path.join(TMLR, 'tmlr.bst')),
    ('tmlr/tmlr.sty', os.path.join(TMLR, 'tmlr.sty')),

    # ---------------- tmlr/figs ----------------
    ('tmlr/figs/fig_headroom.pdf', os.path.join(TMLR, 'figs', 'fig_headroom.pdf')),
    ('tmlr/figs/fig_mortality.pdf', os.path.join(TMLR, 'figs', 'fig_mortality.pdf')),
    ('tmlr/figs/fig_tradeoff.pdf', os.path.join(TMLR, 'figs', 'fig_tradeoff.pdf')),
    ('tmlr/figs/fig_dose.pdf', os.path.join(TMLR, 'figs', 'fig_dose.pdf')),
    ('tmlr/figs/fig_dose3.pdf', os.path.join(TMLR, 'figs', 'fig_dose3.pdf')),
    ('tmlr/figs/fig_evolution.pdf', os.path.join(TMLR, 'figs', 'fig_evolution.pdf')),
    ('tmlr/figs/fig_artifact.pdf', os.path.join(TMLR, 'figs', 'fig_artifact.pdf')),
    ('tmlr/figs/fig_deadfrac.pdf', os.path.join(TMLR, 'figs', 'fig_deadfrac.pdf')),

    # ---------------- data/ (as deposited in v7) ----------------
    ('data/flip_attribution.json', os.path.join(RESULTS, 'flip_attribution.json')),
    ('data/gate_analytic_s0.json', os.path.join(RESULTS, 'gate_analytic_s0.json')),
    ('data/gate_attribution.json', os.path.join(RESULTS, 'gate_attribution.json')),
    ('data/gate_diag_mnist_s0_b0.json', os.path.join(RESULTS, 'gate_diag_mnist_s0_b0.json')),
    ('data/s5_anchor.json', os.path.join(RESULTS, 's5_anchor.json')),
    ('data/s5_baselines.json', os.path.join(RESULTS, 's5_baselines.json')),
    ('data/s5_oracle.json', os.path.join(RESULTS, 's5_oracle.json')),
    ('data/s5_spec.json', os.path.join(RESULTS, 's5_spec.json')),
    ('data/s5_taskrel.json', os.path.join(RESULTS, 's5_taskrel.json')),
    ('data/signeddrift_recheck.json', os.path.join(RESULTS, 'signeddrift_recheck.json')),

    # ---------------- data/ (new in v8: the independent second-machine run) ----
    ('data/server_gate_attribution_s5.json', os.path.join(SERVER, 'server_gate_attribution_s5.json')),
    ('data/server_flip_attribution_s5.json', os.path.join(SERVER, 'server_flip_attribution_s5.json')),
    ('data/server_signeddrift_n10.json', os.path.join(SERVER, 'server_signeddrift_n10.json')),
    ('data/server_scan_mortality.json', os.path.join(SERVER, 'server_scan_mortality.json')),
    ('data/server_s4ctx.json', os.path.join(SERVER, 'server_s4ctx.json')),

    # ---------------- code/ ----------------
    ('code/s4_ablation_gate.py', os.path.join(BENCH, 's4_ablation_gate.py')),
    ('code/s4_ctxnoise.py', os.path.join(BENCH, 's4_ctxnoise.py')),
    ('code/s4_diag.py', os.path.join(BENCH, 's4_diag.py')),
    ('code/signed_drift.py', os.path.join(BENCH, 'signed_drift.py')),
    ('code/check_protocol_consistency.py', os.path.join(BENCH, 'check_protocol_consistency.py')),
    ('code/s5_scan_mortality.py', os.path.join(BENCH, 's5_scan_mortality.py')),
    ('code/compare_server_cpu.py', os.path.join(BENCH, 'compare_server_cpu.py')),
    ('code/server_rerun.sh', os.path.join(BENCH, 'server_rerun.sh')),
    ('code/exp6_baselines.py', os.path.join(BENCH, 'exp6_baselines.py')),
    ('code/exp4_branch_protection.py', os.path.join(BENCH, 'exp4_branch_protection.py')),
]


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def main():
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG)

    missing = []
    copied = 0
    total_bytes = 0
    for rel, src in PLAN:
        if not os.path.exists(src):
            missing.append((rel, src))
            continue
        dst = os.path.join(PKG, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        total_bytes += os.path.getsize(dst)

    print(f'package root : {PKG}')
    print(f'copied       : {copied}/{len(PLAN)} files, {total_bytes/1024:.0f} KiB')
    if missing:
        print(f'\nMISSING ({len(missing)}):')
        for rel, src in missing:
            print(f'  {rel}\n     <- {src}')
        sys.exit(1)

    # The metadata file must carry the v8 version and the new DOI reference.
    meta_path = os.path.join(PKG, 'paper', 'zenodo_metadata.json')
    meta = json.load(open(meta_path, encoding='utf-8'))
    print(f'\nmetadata version (pre-edit): {meta.get("version")}')
    print(f'metadata title: {meta.get("title", "")[:70]}')
    print(f'metadata keywords: {len(meta.get("keywords", []))}')


if __name__ == '__main__':
    main()
