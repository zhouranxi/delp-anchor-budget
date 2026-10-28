# delp-anchor-budget

**The Anchor Accumulation Problem: Bounded Anchor Sets Recover Plasticity in
Usage-Localized Elastic Weight Consolidation**

Code and raw per-seed records for the S2 study. Companion repository to the
DeLP audit ([`zhouranxi/delp-audit`](https://github.com/zhouranxi/delp-audit),
preprint DOI [10.5281/zenodo.23685788](https://doi.org/10.5281/zenodo.23685788)).

## Papers in this repository

Each paper ships as a **self-contained folder** (paper sources + code + raw
per-run records + preregistration/collection reports), independently
verifiable:

| Folder | Paper | Status |
|---|---|---|
| `S1_剂量响应审计/` | *Segment Specialization in Active-Dendrite Continual Learning: An Audit, an Artifact, and a Causal Repair* — dead-neuron measurement artifact, anti-specialization, one-parameter dose–response repair | Preprint [doi:10.5281/zenodo.23685865](https://doi.org/10.5281/zenodo.23685865); submitted to Neural Networks |
| `S2_锚累积/` | *The Anchor Accumulation Problem* — bounded anchor sets, eviction criteria, the soft-bound law ℓ ≈ 84 − 11·ln h | Preprint [doi:10.5281/zenodo.23685906](https://doi.org/10.5281/zenodo.23685906) — revised after the Neurocomputing decision (protocol description corrected, all headline numbers re-derived; see `S2_锚累积/rejection_fix/`) |
| `S3_修复边界/` | *Repair Headroom, Not Dying Gates* — where the S1 repair transfers (conv backbone: no) and what tracks the gain (baseline strength, not mortality) | Preprint [doi:10.5281/zenodo.23685385](https://doi.org/10.5281/zenodo.23685385) — includes the 2×2 width × normalization scan (320 runs) separating the GroupNorm effect from the width effect |
| `S4_gate_bypass/` | *The Repair Saturates the Gate* — context noise as a degeneracy diagnostic; four-arm control (base / repaired / constant-gate / no-gate, 80 runs) | Preprint [doi:10.5281/zenodo.23686689](https://doi.org/10.5281/zenodo.23686689) — **conclusion reversed after the audit**: the repair does not bypass the gate, it saturates it (σ(10+u) ≈ 1) |
| **`S123_合并稿/`** | **What an Active-Dendrite Continual Learner Actually Learns: A Dead-Neuron Artifact, a Granularity Result That Did Not Survive Its Own Controls, and a Repair That Is a Damping Factor** — the combined manuscript of S1 + S3 + S4, with the controls added after the first review | **Current submission version** — preprint [doi:10.5281/zenodo.23729840](https://doi.org/10.5281/zenodo.23729840). Submitted to TMLR. 74 pp Elsevier format + 31 pp TMLR format + 7 pp supplementary, 11 tables, 8 figures. All statistics Holm-corrected with CIs and effect sizes; equivalence testing (TOST) replaces reliance on failure to reject. Supersedes the three separate records above, which are retained as the individually-published versions of record and now cross-link here |

### Why the three were merged

S1 reported the own-segment bias as a *causal repair*; S3 showed it does not
transfer to a convolutional backbone; S4 showed it works by saturating the
dendritic gate rather than by restoring routing. Those three statements
cannot all be true at once, and publishing them apart invites exactly the
cross-reading a reviewer will do. The merged manuscript states them in one
narrative — hypothesis, artifact, apparent repair, boundary failure,
mechanism re-qualification — and rewrites the title, abstract, and
conclusion so that the third result is presented as an apparent success that
the fourth and fifth results overturn, rather than as a contradiction.

The three original folders are kept unchanged so that the DOIs already
issued for them continue to resolve to the documents they were issued for.

### Controls added after the first review

The merged manuscript contains a section reporting five controls that a
review of the first version identified as missing rather than as errors,
and the earlier sections, the abstract and the conclusion are edited to
reflect what they showed:

1. **Specificity of the intervention** (`bench/s5_bias_specificity.py`, 100 runs).
   A bias of `b=10` on the own segment, on a random segment, on every segment,
   and a pure constant rescaling of the trunk. The bias acts on whichever
   segment the network is using; `own` is not privileged, and the rescaling
   arm does better than any bias arm.
2. **Positive control for the instrument**
   (`bench/s5_oracle_control.py`, 60 runs). With task-dominant gating forced
   by construction, the dominant-segment instrument reads 1.000 where the
   base network reads 0.055, so the paper's negative readings are real.
3. **Construction diversity of the granularity result**
   (`bench/s5_anchor_variants.py`, 80 runs). A segment-resolved anchor with
   refreshed per-segment references does not reproduce the null, so the
   general claim that sub-neuron granularity is worthless is withdrawn.
4. **Contemporary protection methods** (`bench/s5_modern_baselines.py`,
   40 runs). Dark Experience Replay reaches 86.4 against DeLP's 64.0 on
   Permuted-MNIST and 75.3 against 42.5 on Permuted-Fashion, with markedly
   less forgetting in both.
5. **Metric sensitivity and equivalence testing**
   (`_audit_s3/compute_metric_sensitivity.py`, `_audit_s3/compute_tost.py`).
   The robustness null survives a sensitivity analysis, and the comparisons
   the text describes as showing no difference are now accompanied by TOST
   equivalence tests rather than by failure to reject.

The task-relevance probe was also re-run (`bench/s5_task_relevance.py`,
40 runs), because the original run's script and per-seed records were not
in the artifact; the reproduced numbers replace the earlier ones in the
text.

6. **Attribution of the mechanism measurements**
   (`bench/s5_gate_attribution.py`, `bench/s5_flip_attribution.py`,
   `bench/s5_gate_analytic.py`). The winning-segment gate value and the
   own-segment win rate were reported as properties of a *trained* network.
   They are in fact properties of the *initialization*: an untrained
   network already reads 0.5641 where the trained one reads 0.5646. The
   cause is the per-task gradient mask, which is applied to blocks while
   the forward pass reads whole columns, so 90% of the gate matrix
   receives no gradient in any phase (measured relative displacement
   exactly 0.0 in all 90 off-diagonal blocks, every seed). Removing the
   mask frees those blocks but still produces no task-segment
   specialization, so the negative result stands and the mechanism
   attribution is corrected. The raw per-seed records for all five controls are in
`bench/results_s5/`.


### Archival records (all six, cross-linked)

| Record | Version DOI | Concept DOI (always latest) |
|---|---|---|
| Where Do the Gains Come From? (precursor audit) | [10.5281/zenodo.23685788](https://doi.org/10.5281/zenodo.23685788) | 10.5281/zenodo.23572226 |
| S1 — Segment Specialization | [10.5281/zenodo.23685865](https://doi.org/10.5281/zenodo.23685865) | 10.5281/zenodo.23637972 |
| S2 — The Anchor Accumulation Problem | [10.5281/zenodo.23685906](https://doi.org/10.5281/zenodo.23685906) | 10.5281/zenodo.23668522 |
| S3 — Repair Headroom, Not Dying Gates | [10.5281/zenodo.23685385](https://doi.org/10.5281/zenodo.23685385) | 10.5281/zenodo.23685384 |
| S4 — The Repair Saturates the Gate | [10.5281/zenodo.23686689](https://doi.org/10.5281/zenodo.23686689) | 10.5281/zenodo.23685276 |
| Data & code (software) | [10.5281/zenodo.23685822](https://doi.org/10.5281/zenodo.23685822) | 10.5281/zenodo.23639561 |

> **S3/S4 记录已作废的数字**：审计发现两处缺陷——弹性锚惩罚漏乘系数 λ，以及
> no-gate 消融臂用了 2000 隐层而树突臂用 256（参数差 7.8×）。两篇实验已重跑，
> 被取代的旧文件在数据包内 `bench/results/DEPRECATED_README.md` 中逐个点名。

The descriptions below refer to the S2 study (this repository's original
scope).

DeLP appends one elastic anchor set (posterior mean + Fisher diagonal) per
task, so the protection penalty grows with sequence length. The audit showed
the cost lands in the *learning term* of the retention decomposition
`ā_T = ℓ + (T−1)/T · BWT`. This study treats the anchor stack itself as the
object of study: bound it at `M` entries and ask (Q1) what is recovered and at
what stability cost, (Q2) whether the budget is a knob distinct from λ, and
(Q3) which eviction criterion survives.

## Headline results

| Benchmark | Comparison | Δ final acc | Paired t |
|---|---|---|---|
| Permuted-MNIST T=10 | fifo M=5 vs unbounded (n=10 seeds) | **+4.51 ± 0.67** | t(9)=21.4 |
| Permuted-MNIST T=20 | fifo M=5 vs unbounded (n=10 seeds) | **+8.98 ± 1.49** | t(9)=19.1 |
| Permuted-MNIST T=10 | fifo M=2 vs unbounded (n=10 paired) | **+12.3** | t(9)=39.6 |
| Fashion-MNIST T=10  | fifo M=2 vs unbounded (n=10 paired) | +13.6 | t(9)=18.3 |
| Fashion-MNIST T=10  | fifo M=5 vs unbounded (n=10 paired) | +2.9 | t(9)=11.5 |
| Split-CIFAR-100 T=20 | five of six bounded policies vs unbounded (n=10; merge −0.5 is the exception) | up to **+5.0** (fifo) | t(9)=7.1 |

Mass-keyed eviction (redun) collapses via *grandfathering* — it freezes the
first M anchors and evicts every later anchor at birth (BWT −34.5 at T=20);
age-first criteria are immune. Fisher-cosine *merge* compresses the stack to
M entries while the accuracy trajectory stays indistinguishable from
unbounded (the penalty *value* drops one-sidedly, but its gradient is
unchanged — merging shifts only a parameter-independent constant), separating
the memory axis from the plasticity axis.

**Single-anchor references (R7, n=10, same protocol & λ grid).** Plain EWC
(keep only the previous task's anchor) posts the highest final accuracy of
any arm on Permuted-MNIST (68.3–71.1 at T=10, 56.4–59.3 at T=20) by
maximizing the learning term (ℓ 80.7–85.3) — at 3–5× the forgetting of
unbounded DeLP (BWT −15.8/−25.7).

**Soft bound beats hard bound, and they obey one law (R8, n=10).**
Sweeping online EWC's discount γ ∈ {0.5, 0.9, 1}: γ=0.5 overtakes
DeLP-fifo at *every* (λ, T) grid cell (paired +4.7…+7.9, all |t(9)|≥10.4)
at matched BWT — an honest negative result for our own repair. The
mechanism: discounting is itself a *soft* bound (mass converges to a
horizon ≈ 1/(1−γ) tasks). Across all five bounded-mass arms the learning
term is log-linear in the protected-mass horizon h:

| arm | h | ℓ (T=20, λ=1000) |
|---|---|---|
| plain EWC | 1 | 83.7 |
| online γ=0.5 | 2 | 76.1 |
| DeLP-fifo M=5 | 5 | 68.3 |
| online γ=0.9 | 10 | 56.6 |
| unbounded | 20 | 51.8 |

ℓ ≈ 84 − 11·ln h, **R² = 0.99** (`analyze_soft_hard_bound.py`). Robustness:
the fit holds per-seed (every seed individually R² ≥ 0.98, slope −10.9±0.6),
replicates at T=10 (R² = 0.998 on the four-arm fit excluding γ=0.9), and
the ordering is monotone at every grid λ
(mean-fit R² 0.99/0.94/0.80 as λ tightens). Bounding
accumulated anchor mass — not the eviction mechanism — is the active
ingredient; the explicit window's residual is an auditable, discrete,
policy-selectable protection set.

**Merge, measured (R8 instrumentation, 40 runs).** Merge removes exactly
the inter-anchor cross terms: the stack penalty *value* drops one-sidedly
(per-run maxima 43–45% at M=2, worst 64%; 18–34% at M=5; 77% of individual
merges change that value) while the penalty *gradient* is unchanged —
merging shifts only a parameter-independent constant, which has zero effect
on SGD — and the accuracy trajectory remains
indistinguishable from unbounded.

**Budget dose–response (R8, T=20, λ=5000, n=10).** M = 2,3,4,5,6,8,10 →
accuracy 52.4, 48.9, 45.0, 42.4, 40.2, 36.5, 35.1 (BWT −24.5 → −5.2);
every M significantly above unbounded at the same λ=5000 (33.4; paired
t(9) 36.0 → 9.1).

**No-protection reference (R7, Split-CIFAR-100, n=10).** λ=0 (plain SGD)
reaches 27.0±1.7 (T=10) and 36.9±3.4 (T=20) — above every DeLP arm.
Under task-incremental evaluation with a fixed masked head, trunk drift is
cheap and protection at any λ is a net tax: the budget effect is real within
the protected family, but protection itself must be checked against a
no-protection reference before the mechanism is credited.

## Repository layout

```
bench/
  anchor_budget.py        bounded-anchor main: 7 eviction policies,
                          Permuted-MNIST / Fashion-MNIST, survivor_history
  anchor_budget_cifar.py  Split-CIFAR-100 variant: ResNet-18 + GroupNorm,
                          masked-CE task-incremental heads, backbone-only
                          anchors with globally normalized Fisher, grad clip
  ab_ewc_arms.py          plain-EWC / online-EWC (Chaudhry 2018, γ=1) arms on
                          the identical protocol (reviewer-response round R7)
  analyze_s2r6.py         paired analysis of the 260-record CIFAR grid
  make_figs.py            Figs 1–4 (Pareto, age profiles, criteria, stacks)
  make_fig5_cifar.py      Fig 5 (CIFAR stack traces + decomposition check)
  launch_s2r6.sh          4-GPU launch for the CIFAR grid
  launch_r7.sh            4-GPU launch for the R7 arms (EWC / CIFAR λ=0 /
                          Fashion n=10 extension)
  results/                raw per-seed JSON records (all runs, all seeds)
                          -- key committed inputs to the S2 figure/analysis
                          --   pipeline (S1 records co-live here, prefixed
                          --      c2_*/q8*/gevo_*/signeddrift/s1_*):
                          --   anchor_budget_all.json  (Fig 1-3 main data)
                          --   s2r4_d1_local.json      (Fig 4 stack traces:
                          --                              T=20 M=2 seed0-2,
                          --                              fifo/redun/agew2/merge)
                          --   s2r6_b_t*.json.part*    (Fig 5 CIFAR stack traces)
figs/                     paper figures (PDF + PNG)
```

## Shared base modules (vendored)

`exp4_branch_protection.py` and `exp6_baselines.py` are the shared
architecture + protocol base that every S1/S2 training script imports
(`ADNet`, `permute`, `kwta`, and the constants `SEEDS/EPOCHS/BS/NTRAIN/NH/
LR/DEV` plus `get_perms/flat_params/eval_model/finish/fisher_usage`). They
are **not** developed in this repository: they are vendored, byte-identical,
from the frozen DeLP-audit base

- upstream: `zhouranxi/delp-audit` @ `dendritic-lab/`,
  commit `ea83e26185aedf98471eff9c129e9b98dcaf6da6` (main),
  preprint Zenodo doi:10.5281/zenodo.23685788.

Both files carry a `VENDORED ... do not edit here` header pinning that
commit. If the audit is ever re-frozen, bump the two headers and re-sync
the copies here — do not diverge them.

## Protocols

**Permuted-MNIST / Fashion-MNIST.** 784–256 FC trunk, K=T dendritic segments,
k-WTA (k=64), linear head; Adam lr 1e-3, batch 256, 2 epochs / 10k samples per
task; λ_DeLP = 5000; paired arms share seeds 0–9; permutations drawn once per
seed and shared by all arms. Full details: `anchor_budget.py` docstring.

**Split-CIFAR-100.** ResNet-18 (32×32 variant) with GroupNorm (BN running
stats are buffers that drift under later tasks and are excluded by design).
The 100 classes are split into T contiguous slices of 100/T classes each
(10 classes/task at T=10, 5 classes/task at T=20; the code partitions a
random permutation of the 100 classes, generator seed 100+seed, into T
blocks); masked cross-entropy over each task's own 100/T classes
(task-incremental), so
head rows of past tasks are protected by construction and anchors cover the
backbone only; Fisher rescaled to unit global mean, λ = 0.1 (calibrated on
unbounded DeLP only), grad clip 5; SGD lr 0.05, 5 epochs per task; T ∈ {10,20},
n = 10 seeds. Absolute accuracies are task-incremental and not comparable to
class-incremental literature; all claims are within-family paired comparisons.

## Reproducing

```bash
# MNIST policy grid (single GPU; shard with --task-id/--n-tasks)
python bench/anchor_budget.py --task-id 0 --n-tasks 1 \
    --policies unbounded,fifo,random,redun,merge,agew,agew2 \
    --budgets 5 --lams 5000 --ntasks 10,20 --seeds 0,1,2 \
    --out anchor_budget_result.json

# Split-CIFAR-100 grid (per GPU lane)
bash bench/launch_s2r6.sh 0 0 0.1

# Paired analysis + figures (expects bench/results/*.json)
python bench/analyze_s2r6.py
python bench/make_figs.py && python bench/make_fig5_cifar.py
```

Raw records: `bench/results/`. Every JSON record carries `bwt`, `final_avg`,
`ell`, `n_anchors_final`, the full per-phase `acc_matrix`, and (MNIST/CIFAR
main grid) the anchor-stack `survivor_history` that underlies the trace
figures. The CIFAR decomposition check is exact on all 260 records
(max deviation 5×10⁻³, i.e. two-decimal rounding).

Fig 4's stack-trace input `s2r4_d1_local.json` (Permuted-MNIST, T=20,
M=2, the four bounded policies × seeds 0–2, each carrying a 20-phase
`survivor_history`) is committed under `bench/results/` alongside the
other records — it is a genuine server-side run artifact, not a scratch
placeholder, so a clean clone reproduces Figs 1–5 without staging
anything. S1's co-living records (`c2_*`, `q8*`, `gevo_*`,
`signeddrift.json`, `s1_*`, `diag_*`) are inert to the S2 pipeline: the
S2 figure/analysis scripts read only the fixed S2 files above, so
adding S1 records here cannot perturb S2 figures.

## Citation

```bibtex
@misc{zhou2026anchorbudget,
  author       = {Zhou, Ranxi},
  title        = {The Anchor Accumulation Problem: Bounded Anchor Sets
                  Recover Plasticity in Usage-Localized Elastic Weight
                  Consolidation},
  howpublished = {https://github.com/zhouranxi/delp-anchor-budget},
  year         = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
