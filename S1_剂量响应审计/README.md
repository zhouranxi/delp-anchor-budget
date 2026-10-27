# S1 — Segment Specialization in Active-Dendrite Continual Learning

Audit of an apparent dendritic segment–task specialization in active-dendrite continual
learning, the dead-neuron artifact that fakes it, and a causal repair.

This directory is **self-contained**: every number quoted in the paper can be re-derived
from the files in `results/` alone, without touching the rest of the repository.

```
S1_剂量响应审计/
├── README.md              ← this file
├── paper/                  the manuscript (main.tex, compiled main.pdf), figures, refs
├── results/                every per-seed record behind every table and figure
├── code/                   the experiment scripts (branch_delp.py and its modules)
├── audit/                  scripts that re-derive the paper's numbers from results/
├── FILE_INDEX.json         (in results/) the file-name map and the rename history
└── S1_*_报告.md, S1_*_预注册.md   collection reports and pre-registrations
```

## One-command verification

```bash
python audit/test_standalone.py .          # from inside this directory
```

It checks that

* Table 1 (granularity ladder) reproduces,
* Table 2 (single-anchor references) reproduces,
* Table 3 (FIFO budget) has its source records,
* all 22 dose cells of Tables 5/6 reproduce **including every significance star**,
* no result file is a duplicate of another.

Expected output: `5 passed, 0 failed` (7 checks in the current version).

## Where each number comes from

| Paper item | Source records |
|---|---|
| Table 1 (granularity ladder, λ=1000 MNIST / 5000 Fashion) | `results/s1_l1000_t10.json`, `s1_l1000_t20.json`, `s1_mnist.json`, `s1_fashion.json` |
| Table 2 (SI / MAS references) | `results/s1_refs.json` (`kind` = `si` / `mas`) |
| Table 3 (FIFO budget) | `results/s1b_{mnist,fashion}_t{10,20}.json` (`kind` = `bfifo`) |
| Tables 5/6 (dose–response), Fig. 3 | `results/c2_<ds>_b{0,1,3,10}.json.part*` (T=10) and `results/c2t20_<ds>_b{0,3,10}.json.part*` plus `results/c2t20_<ds>_b1_neuron.json` (T=20) |
| Fig. 1 (evolution), Fig. 2 (dead fraction) | `results/gevo_{mnist_t10,mnist_t20,fashion_t10}.json`, `results/untrained_probe.json` |
| §compensation (signed drift) | `results/signeddrift.json` |

### Result-file naming convention

```
c2_<dataset>_b<bias>.json.part<0..3>      T=10 dose arms (bias 0/1/3/10)
c2t20_<dataset>_b<bias>.json.part<0..3>   T=20 dose arms (bias 0/3/10)
c2t20_<dataset>_b1_neuron.json           T=20 neuron arm at bias 1 (single file:
                                          no shards were produced for bias 1)
```

The dose is the record field **`own_bias`**. The bias-0 baseline arms do not carry the
field at all (the code only writes it for a non-zero bias), so they are identified by the
`_b0` file stem. The granularity-ladder and reference arms are not dose runs and carry no
`own_bias` field either — they are keyed by `(dataset, T, granularity, kind, lambda)`.

`results/FILE_INDEX.json` records a 2026-10-27 clean-up: the merged `c2_<ds>_b{1,3}.json`
files held the **T=20 neuron** arms while their same-stem `.part*` shards held the **T=10**
arms, and the `s3e2_t20_*` copies were byte-identical duplicates. The `b=1` files were
renamed to `c2t20_<ds>_b1_neuron.json`; the `b=3` files were removed because their records
are fully contained in the `c2t20_<ds>_b3.json.part0..3` shards. No record was lost.

## Re-running the experiments

```bash
python code/branch_delp.py --task-id 0 --n-tasks 1 \
    --granularities neuron,branch,plain --ntasks 10,20 --lams 1000 \
    --seeds 0,1,2,3,4,5,6,7,8,9 --out results/s1_mnist.json
```

The dose grid is the same script with `--own-bias <b> --bias-mode fixed`
(`--bias-mode none` for the bias-0 baseline).

Environment: PyTorch + torchvision, one CUDA GPU; `requirements.txt` at the repository
root pins the versions used. A single T=10 run takes ~12 s on an RTX 3090; the full grid
is 24 cells × 10 seeds.

## Known limitations (as stated in the paper)

* Two cells of the 24-cell dose grid were never collected: `T=20`, branch arm, `b=1`
  (for both datasets). The paper reports 22 of the 24 cells and says so explicitly.
* The arms that reuse the original implementation do not record held-out validation
  metrics, so their λ values are carried over from the original pipeline rather than
  re-selected on validation; the corrected re-implementation arms do select on validation.
  This asymmetry is stated in the paper rather than hidden.
