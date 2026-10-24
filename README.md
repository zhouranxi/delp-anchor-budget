# delp-anchor-budget

**The Anchor Accumulation Problem: Bounded Anchor Sets Recover Plasticity in
Usage-Localized Elastic Weight Consolidation**

Code and raw per-seed records for the S2 study. Companion repository to the
DeLP audit ([`zhouranxi/delp-audit`](https://github.com/zhouranxi/delp-audit),
preprint DOI [10.5281/zenodo.23572226](https://doi.org/10.5281/zenodo.23572226)).

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
| Permuted-MNIST T=10 | fifo M=2 vs unbounded | +13.0 | — |
| Fashion-MNIST T=10  | fifo M=2 vs unbounded (n=10 paired) | +13.6 | t(9)=18.3 |
| Fashion-MNIST T=10  | fifo M=5 vs unbounded (n=10 paired) | +2.9 | t(9)=11.5 |
| Split-CIFAR-100 T=20 | every bounded policy vs unbounded (n=10) | up to **+5.0** (fifo) | t(9)=7.1 |

Mass-keyed eviction (redun) collapses via *grandfathering* — it freezes the
first M anchors and evicts every later anchor at birth (BWT −34.5 at T=20);
age-first criteria are immune. Fisher-cosine *merge* compresses the stack to
M entries with a bit-near-identical trajectory, separating the memory axis
from the plasticity axis.

**Single-anchor references (R7, n=10, same protocol & λ grid).** Plain EWC
(keep only the previous task's anchor) posts the highest final accuracy of
any arm on Permuted-MNIST (68.3–71.1 at T=10, 56.4–59.4 at T=20) by
maximizing the learning term (ℓ 81–85) — at 3–5× the forgetting of
unbounded DeLP (BWT −15.8/−25.7). Online EWC (Chaudhry 2018, γ=1) at its
best λ=1000 reaches 67.5/46.0: statistically indistinguishable from
DeLP-fifo at the same λ at T=10 (67.4, t(9)=−0.3), behind it at T=20
(fifo +3.5, t(9)=7.0). Neither scheme can accumulate the stack problem —
that is the point of the comparison.

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
figs/                     paper figures (PDF + PNG)
```

## Protocols

**Permuted-MNIST / Fashion-MNIST.** 784–256 FC trunk, K=T dendritic segments,
k-WTA (k=64), linear head; Adam lr 1e-3, batch 256, 2 epochs / 10k samples per
task; λ_DeLP = 5000; paired arms share seeds 0–9; permutations drawn once per
seed and shared by all arms. Full details: `anchor_budget.py` docstring.

**Split-CIFAR-100.** ResNet-18 (32×32 variant) with GroupNorm (BN running
stats are buffers that drift under later tasks and are excluded by design);
masked cross-entropy over each task's own 10 classes (task-incremental), so
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
