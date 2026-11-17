# Cover Letter — Information Sciences

**Manuscript title:** The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft–Hard Protection Law, and the Auditable Cost of Regularized Continual Learning

**Article type:** Original research work
**Corresponding author:** Ranxi Zhou (Hunan University of Chinese Medicine, Changsha, China)
**Preprint / reproducibility package:** doi:10.5281/zenodo.23668522

---

Dear Editors,

I am pleased to submit the above manuscript for consideration in *Information Sciences*.

**What the paper is about.** Regularized continual-learning schemes maintain a growing store of protection state — one anchor set per task — that is consulted at every update. From an information- and knowledge-management perspective this state *is* a knowledge base: entries are appended, never curated, never bounded, never audited. The paper makes that knowledge base the object of study. It identifies the anchor accumulation problem (full-weight stale anchors taxing later learning), repairs it with bounded anchor sets, shows that eviction design is governed by one age-over-mass constraint, and establishes a log-linear law (R² = 0.99) under which online EWC's soft discount and the explicit hard window become interchangeable points on one curve of learning term versus protected-mass horizon.

**Submission history disclosure.** Earlier versions of this work were declined by two other Elsevier journals, both without external review: *Pattern Recognition* (manuscript PR-D-26-16069, Area Editor, 10 November 2026) and *Neurocomputing* (manuscript NEUCOM-D-26-24422, 26 October 2026). The *Pattern Recognition* editor's comments are quoted verbatim (one word restored in brackets where the journal's HTML rendering dropped it):

> This paper presents a theoretical and empirical study on Usage-Localized Elastic Weight Consolidation, a continual learning method. Though detailed analysis and ablation studies are presented, the work did not show substantial benefits to the PR community: EWC has been shown low performance in comparison with later CL methods, and this paper does [not] compare with recent CL methods on various datasets. Such study does not provide valuable reference to the community, thus it is rejected without external review.

The present version responds substantively on both counts:

1. **On "does not compare with recent CL methods": the manuscript does compare, and now says so visibly.** Section 2 (Related work) benchmarks against recent methods spanning the three protection families — SI (Zenke et al., 2017), MAS (Aljundi et al., 2018), EWC and online EWC (Kirkpatrick et al., 2017; Chaudhry et al., 2018), and copyback-AL / learned-anchor variants (F2L, Hassan et al., 2025; DLC-PA, Sun et al., 2026; LoRaC, Ling et al., 2026) — and the experimental sections compare directly against SI and tuned online EWC on the same protocol. The comparison is per-seed paired and statistically reported (paired t and Wilcoxon signed-rank).
2. **On "did not show substantial benefits": the strongest form of this concern is now a headline result.** On Split-CIFAR-100 under task-incremental evaluation, a no-protection baseline (plain SGD) beats every protection arm, including the best budgeted one (+2.2 points over fifo M = 2 at T = 10; +3.7 over unbounded; both Wilcoxon-significant). The manuscript states up front that bounding is necessary but not sufficient, that protection must justify its cost against a no-protection reference, and that the framework's value is diagnosis and auditability — it *quantifies exactly when the simpler method wins, and why*. Where benefits do exist they are characterized precisely: under the anchor-drag regime (Permuted-MNIST, Fashion-MNIST, long sequences), bounded anchor sets recover +4.5 / +13.0 points over unbounded consolidation, and a log-linear soft–hard protection law (R² = 0.99) unifies online EWC's discount with explicit windows.
3. **The mechanism-level law is foregrounded** in its own section, elevating the contribution from a case study of one architecture to a statement about all bounded-mass protection schemes.
4. **The knowledge-centric framing** runs through the introduction and discussion: the anchor stack as an uncurated knowledge base, the bounded family as the restoration of named entries, maintenance operations, and auditability. Each anchor carries an origin tag, the survivor history records each maintenance event, and the protected set becomes a discrete, named, inspectable object rather than an implicit scalar — the property this journal's *Information and Knowledge* area treats as first-class.
5. **Tightened presentation:** the statistics rationale, hyperparameters, the three full numerical grids, and the complete per-task accuracy matrices were moved to a separate supplementary PDF, keeping the main text at 31 pages with 5 figures and 3 tables.

**Why this journal.** *Information Sciences* emphasizes a balanced coverage of both theory and practice, and this manuscript contributes to both. The theory is a decomposition and a law: a retention identity that localizes the cost of protection in the learning term, and a log-linear relation (ℓ ≈ 84 − 11 ln h, R² = 0.99) that puts a scalar discount and an explicit window on one curve. The practice is a set of design rules and an audit trail: bounded anchor sets, an age-first eviction criterion, and origin-tagged survivor histories that make the protection set readable from the stack trace. The finding that a no-protection reference can beat an entire protected family is reported as a first-class contribution rather than suppressed, and we regard that transparency as the substance of the audit framework rather than a caveat on it.

**Declarations.** The manuscript is original and has not been published previously. Neither of the two earlier submissions described above is under consideration; both have been formally declined. The author declares no competing interests and no funding. Code, per-seed records, and the audit scripts that re-derive every quoted number are publicly available (GitHub: zhouranxi/delp-anchor-budget; Zenodo concept DOI: 10.5281/zenodo.23668522), and the dataset link is provided in the submission system as the journal's Option C research-data policy requires. A declaration of AI-assisted language editing is included in the manuscript. A complete CRediT author statement, a declaration of competing interests, and the author vita are submitted as separate files.

Sincerely,
Ranxi Zhou
Hunan University of Chinese Medicine
zhouranxi@foxmail.com
