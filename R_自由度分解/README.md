# Reproduction package

**One Degree of Freedom Moves the Frontier, the Other Four Do Not:
A Preregistered Factorization of an Active-Dendrite Repair, Its
Architecture-Bound Supply, and Five Failed Escapes**

Version 2: adds experiment R4 (Transformer backbone) to the package and
updates the manuscript; results of version 1 are unchanged.

## Verify every number in the paper from raw records

```
python r_repro/verify_package.py       # 25 checks: headline statistics
python paper/check_paper_numbers.py    # 27 checks: seed-direction counts,
                                       #   per-dataset splits, figure-caption
                                       #   numbers, protocol-table constants,
                                       #   the 1,539-run ledger, and the R4
                                       #   verdict numbers
```

Both run from this directory (or any directory preserving the layout) and
need only the Python standard library. Exit code 0 means every check passed.

## Rebuild the figures

```
python paper/make_figs.py    # requires matplotlib; writes paper/figs/
```

## Rebuild the PDF

```
cd paper && xelatex main && bibtex main && xelatex main && xelatex main
```

## Layout

- `paper/` -- manuscript (LaTeX source, compiled PDF, figures, bibliography),
  both verifiers, and the figure-generation script.
- `bench/` -- the local modules used by the R1-R4 runners, plus the
  raw per-run JSON records of all six experiments (`results_r1`,
  `results_r2`, `results_r2t`, `results_r3`, `results_r4`) and the reused
  audit controls (`results_e9`). Preregistrations with dated amendments and
  collection reports sit alongside the records they govern.
- `r_repro/` -- the package-level verifier, its manifest, and a README.

## Run the experiments

Each runner in `bench/` reproduces its full preregistered grid with default
arguments (PyTorch required); all support `--shard i --n-shards k` for
parallel sharding. `r2_recency.py` and `r4_transformer.py` require an
explicit `--out` path. Collected shards are concatenated as the
corresponding `.part*.json` records under `bench/results_*/`.

```
python bench/r1_ctxnoise_train.py      # R1: context corruption, 280 runs
python bench/r2_recency.py --out out/  # R2: anchor-budget direction, 440 runs
python bench/r2t_resnet.py             # R2T: ResNet-18+GN transfer, 80 runs
python bench/r3_lamsweep.py            # R3: lambda dose curve, 240 runs
# R4 (three modes; lambda-sweep and main expect the frozen G=0.5827):
python bench/r4_transformer.py --mode gprobe --out out/gprobe.json
python bench/r4_transformer.py --mode scan  --g 0.5827 --out out/scan.json
python bench/r4_transformer.py --mode main  --g 0.5827 --lam 100 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --out out/main.json
python bench/analyze_r4_scan.py        # lam* = argmax pooled delta-ell
python bench/analyze_r4_main.py --files results_r4/r4_main.part*.json --scan results_r4/r4_scan.part*.json --out r4_main_stats.json
```

Gate arithmetic, Holm families, and trigger clauses are as fixed in the
preregistrations; deviations are logged in the dated amendment notes,
preregistrations unmodified.
