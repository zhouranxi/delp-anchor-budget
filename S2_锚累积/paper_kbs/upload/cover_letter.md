# Cover Letter — Knowledge-Based Systems

**Manuscript title:** The Anchor Accumulation Problem: Bounded Anchor Sets, a Soft–Hard Protection Law, and the Auditable Cost of Regularized Continual Learning

**Article type:** Original high-quality research paper
**Corresponding author:** Ranxi Zhou (Hunan University of Chinese Medicine, Changsha, China)
**Preprint / reproducibility package:** doi:10.5281/zenodo.23668522

---

Dear Editors,

I am pleased to submit the above manuscript for consideration in *Knowledge-Based Systems*.

**What the paper is about.** Regularized continual-learning schemes maintain a growing store of protection state — one anchor set per task — that is consulted at every update. From a knowledge-based-systems perspective, this state *is* a knowledge base: a growing, structured store of what has been learned, consulted at every update, and trusted in proportion to its age. Yet this knowledge base is rarely treated as a first-class design object — it is appended to, never curated, bounded, or audited. The paper makes it the object of study. It identifies the anchor accumulation problem (full-weight stale anchors taxing later learning), repairs it with bounded anchor sets, shows that eviction design is governed by one age-over-mass constraint, and establishes a log-linear law (R² = 0.99) under which online EWC's soft discount and the explicit hard window become interchangeable points on one curve of learning term versus protected-mass horizon.

**Submission history disclosure.** Earlier versions of this work were declined by two other Elsevier journals, both without external review: *Pattern Recognition* (manuscript PR-D-26-16069, Area Editor, 10 November 2026) and *Neurocomputing* (manuscript NEUCOM-D-26-24422, 26 October 2026). The *Pattern Recognition* editor's comments are quoted verbatim (one word restored in brackets where the journal's HTML rendering dropped it):

> This paper presents a theoretical and empirical study on Usage-Localized Elastic Weight Consolidation, a continual learning method. Though detailed analysis and ablation studies are presented, the work did not show substantial benefits to the PR community: EWC has been shown low performance in comparison with later CL methods, and this paper does [not] compare with recent CL methods on various datasets. Such study does not provide valuable reference to the community, thus it is rejected without external review.

The present version responds substantively on both counts:

1. **On "does not compare with recent CL methods": the manuscript does compare, and now says so visibly.** Section 2 (Related work) benchmarks against recent methods spanning the three protection families — SI (Zenke et al., 2017), MAS (Aljundi et al., 2018), EWC and online EWC (Kirkpatrick et al., 2017; Chaudhry et al., 2018), and copyback-AL / learned-anchor variants (F2L, Hassan et al., 2025; DLC-PA, Sun et al., 2026; LoRaC, Ling et al., 2026) — and the experimental sections compare directly against SI and tuned online EWC on the same protocol. The comparison is per-seed paired and statistically reported (paired t and Wilcoxon signed-rank).
2. **On "did not show substantial benefits": the strongest form of this concern is now a headline result.** On Split-CIFAR-100 under task-incremental evaluation, a no-protection baseline (plain SGD) beats every protection arm, including the best budgeted one (+2.2 points over fifo M = 2 at T = 10; +3.7 over unbounded; both Wilcoxon-significant). The manuscript states up front that bounding is necessary but not sufficient, that protection must justify its cost against a no-protection reference, and that the framework's value is diagnosis and auditability — it *quantifies exactly when the simpler method wins, and why*. Where benefits do exist they are characterized precisely: under the anchor-drag regime (Permuted-MNIST, Fashion-MNIST, long sequences), bounded anchor sets recover +4.5 / +13.0 points over unbounded consolidation, and a log-linear soft–hard protection law (R² = 0.99) unifies online EWC's discount with explicit windows.
3. **The mechanism-level law is foregrounded** in its own section, elevating the contribution from a case study of one architecture to a statement about all bounded-mass protection schemes.
4. **A knowledge-based framing** runs through the introduction and discussion: the anchor stack as an uncurated knowledge base, the bounded family as the restoration of named entries, maintenance operations, and auditability. This framing matches the scope of *Knowledge-Based Systems*: the journal covers "knowledge-based and other artificial intelligence techniques-based systems", and lists "knowledge presentation and engineering" among its leading topics, alongside "machine learning theory, methodology and algorithms" and "data science theory, methodologies and techniques". The paper sits at that intersection — it contributes a machine-learning methodology and an auditable knowledge-representation scheme rather than an application study.
5. **Tightened presentation:** the statistics rationale, hyperparameters, the three full numerical grids, two auxiliary figures, and the complete per-task accuracy matrices were moved to a separate supplementary PDF.

**Why this journal.** The findings — an eviction design space with one governing constraint, a soft–hard protection law, and a protection-must-justify-itself boundary — are stated as reusable knowledge for designers of learning systems. The audit framework (origin-tagged anchors, survivor traces, recomputable statistics) is released in full, so that the protection set of any regularized continual-learning pipeline can be inspected rather than trusted. The negative result is not a caveat on the contribution but the reason the framework is worth having: it tells a designer exactly when the simpler method wins, and why.

**Declarations.** The manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests and no funding. Code, per-seed records, and the scripts that re-derive every quoted number are publicly available (GitHub: zhouranxi/delp-anchor-budget; Zenodo concept DOI: 10.5281/zenodo.23668522), and the dataset link is provided in the submission system as the journal's Option C research-data policy requires. A declaration of AI-assisted language editing is included in the manuscript. A complete CRediT author statement and a declaration of competing interests are submitted as separate files.

Sincerely,
Ranxi Zhou
Hunan University of Chinese Medicine
zhouranxi@foxmail.com
