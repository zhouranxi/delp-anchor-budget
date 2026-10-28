# Reproducing the results

Everything in the manuscript is re-derived from the raw per-seed records in
`results/` by the scripts in `audit/`. No number in the paper is hand-typed.

## Quick start

```bash
pip install -r requirements.txt
python audit/verify_all.py
```

`verify_all.py` runs every check and prints a single PASS/FAIL summary.
Expected result: **all checks PASS**.

## What each script does

| Script | What it verifies | Expected |
|---|---|---|
| `audit/final_regression.py` | 27 consistency checks over paper text, code and raw records | **27 PASS / 0 FAIL** |
| `audit/audit_nonparametric.py` | Recomputes Wilcoxon signed-rank, Shapiro-Wilk and Friedman tests on the paired deltas | Wilcoxon `p = 0.0020` on all four headline comparisons |
| `audit/check_latex.py` | LaTeX pre-flight: labels/refs, citation↔bibliography, table column specs, environment pairing, brace and `$` balance | "No problems found" |
| `audit/make_graphical_abstract.py` | Regenerates `graphical_abstract.png`; every plotted value is recomputed from `results/anchor_budget_all.json` | 3900×2280 px, landscape |
| `audit/rename_figures.py` | Creates the `Figure_1..Figure_5` upload copies | 10 files |

`audit/REVISION_CHECKLIST.md` is the edit log from the previous review round,
including the two conclusions the audit itself had to retract.

## Independent-environment check (already performed)

The verification scripts have been run in a **separate Python 3.12.9 conda
environment** (`open_manus`, with its own numpy/scipy/matplotlib versions and
`torch 2.6.0+cpu`), not only in the environment the manuscript was written in:

```
[1] regression          -> PASS 27 / FAIL 0
[2] non-parametric      -> Wilcoxon W = 0.0, p = 0.0020 (all four headline comparisons)
[3] latex pre-flight    -> No problems found
```

The numbers therefore do not depend on the particular interpreter or package
build used to produce them.

A `Dockerfile` is included for a fully pinned environment:

```bash
docker build -t delp-anchor-budget .
docker run --rm delp-anchor-budget
```

(The container runs the full verification as its entry point, so a failing
check fails the build. It requires a working Docker engine; the equivalent
native check above needs nothing beyond Python.)

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `pillow` — for the analysis and figure scripts
- `torch` (CPU is sufficient) — only needed to re-run the *experiments*;
  the verification scripts above do not need it
- A LaTeX distribution with `elsarticle` for `build.sh` (TeX Live / MiKTeX / TinyTeX)

## Reproducing the experiments themselves

The experiment scripts live in `code/`. They are deterministic given a seed
(permutation matrices are drawn once per seed and shared across arms).

```bash
python code/anchor_budget.py          # Permuted-MNIST / Fashion-MNIST
python code/anchor_budget_cifar.py    # Split-CIFAR-100
python code/ab_ewc_arms.py            # plain EWC and online EWC arms
python code/make_figs.py              # Figures 1-4
python code/make_fig5_cifar.py        # Figure 5
```

Runtime is dominated by the CIFAR runs (ResNet-18, 20 tasks, 10 seeds).
The analysis and audit scripts re-derive every published number from the
archived records in seconds, without re-running training.

## Data and code availability

- Code and complete per-run records: https://github.com/zhouranxi/delp-anchor-budget
- This manuscript and its full reproducibility package: doi:10.5281/zenodo.23668523
- Precursor audit study: doi:10.5281/zenodo.23602281
- Companion segment-specialization audit: doi:10.5281/zenodo.23637973
- Underlying dataset and code release: doi:10.5281/zenodo.23639562

## Provenance of the records

`results/` holds the raw JSON records: per-run `acc_matrix` (per-phase
accuracy), `bwt`, `ell` (the learning term), `n_anchors_final`, and — where
instrumented — `survivor_history` (the anchor-stack survivor trace) and
`n_merges`. `results/*.json.part*` are the sharded outputs of the original
parallel runs and are concatenated by the audit scripts; they are kept in
their original sharded form so that the provenance of each shard is visible.
