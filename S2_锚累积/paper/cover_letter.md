# Cover Letter — Neurocomputing submission

**Manuscript title:** The Anchor Accumulation Problem: Bounded Anchor Sets Recover Plasticity in Usage-Localized Elastic Weight Consolidation

**Article type:** Research paper
**Corresponding author:** Ranxi Zhou (Hunan University of Chinese Medicine, Changsha, China)
**Preprint:** doi:10.5281/zenodo.23572226 (precursor audit study)

---

Dear Editor,

We are pleased to submit the above manuscript for consideration in *Neurocomputing*.

Usage-localized elastic weight consolidation (DeLP) protects dendritic segments with per-neuron elastic anchors accumulated over the task sequence. In a precursor audit study (archived as a Zenodo preprint), we showed that usage weighting survives a parameter-budget cap while other design choices add little. The present manuscript identifies and studies the cost that *any* such stack-building scheme pays as the task sequence grows: anchors from early tasks persist at full weight and act as an elastic drag on later learning. We call this the **anchor accumulation problem** and study it directly by bounding the anchor stack at M entries.

The manuscript makes the following contributions:

1. **A decomposition-based diagnosis.** Writing average retention as a_T = ℓ + (T−1)/T·BWT localizes the cost of long sequences not in the architecture but in the *learning term* ℓ. The identity is verified to rounding precision on every experimental record across three benchmarks (~650 paired runs).

2. **Bounded anchor sets as a repair, with explicit references that bound the claim.** A FIFO budget of M = 2 recovers +4.5 and +9.0 points of final accuracy over unbounded DeLP at T = 10 and T = 20 (n = 10 paired seeds, t(9) > 19), monotone across a seven-point dose–response (M = 2…10), and the effect replicates on Fashion-MNIST (n = 10, +13.6, t(9) = 18.3). Rather than stopping at within-family comparisons, we also run the schemes that *cannot* accumulate the problem — plain EWC and online EWC (Chaudhry et al., 2018) over a γ-sweep — and a no-protection baseline on Split-CIFAR-100. We report an honest negative result for our own repair: with a tuned discount (γ = 0.5), online EWC overtakes the budgeted scheme at every grid cell at matched forgetting. The explanation is mechanism-level and, we believe, the paper's most transferable finding: discounting is itself a *soft* bound (mass converges to a horizon ≈ 1/(1−γ)), and across soft and hard bounds the learning term obeys one log-linear law in the protected-mass horizon (R² = 0.99). Bounding accumulated anchor mass — not the eviction mechanism — is the active ingredient.

3. **A mechanistic study of eviction criteria.** Trace instrumentation of anchor stacks (survivor histories) shows that mass-keyed criteria inherit a *grandfathering trap* — eviction-time quality equals staleness — and freeze on the oldest anchors, bit-for-bit identically across seeds and across domains. Age is the only sound eviction key. Penalty instrumentation of merging makes the formerly qualitative "loss-free" claim quantitative: the penalty drops one-sidedly (per-run maxima 18–64%) exactly as the inter-anchor cross terms are removed, while the accuracy trajectory stays indistinguishable from unbounded.

4. **A control-axis separation, restated by measurement.** The budget and the elastic weight λ are distinct knobs: both trade protection for plasticity, but the budget controls the *age structure* of forgetting (a profile the eviction trace predicts and the measurements confirm) while λ controls its *amount*. What the explicit window retains relative to a discounted posterior is an auditable, discrete, policy-selectable protection set — not superior accuracy on this benchmark.

We believe this work fits the scope of *Neurocomputing*'s coverage of learning systems and continual learning. Because the manuscript reports single-anchor and no-protection references alongside the budgeted mechanism — and a γ-sweep that turns our own repair's loss into the mechanism-level law above — practitioners get not only a repair but also the boundary conditions telling them when protection — and therefore bounding — is worth deploying at all.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration by any other journal. A precursor audit study by the same author is archived as a Zenodo preprint (doi:10.5281/zenodo.23572226) and is cited and clearly distinguished from the present work; all experiments, analysis, and text in this manuscript are new. The author declares no competing interests. All code and complete per-run result records are publicly available in the project repository linked in the manuscript.

Sincerely,
Ranxi Zhou
Hunan University of Chinese Medicine
300 Xueshi Avenue, Hanpu, Yuelu District, Changsha 410208, Hunan, China
Email: zhouranxi@foxmail.com
