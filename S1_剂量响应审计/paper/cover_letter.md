# Cover Letter

**Journal:** Neural Networks (Elsevier)
**Manuscript type:** Full-length research article
**Title:** Dead neurons, anti-specialization, and a one-parameter repair: auditing and causally restoring segment--task specialization in active-dendrite networks

Dear Editors,

We are pleased to submit our manuscript "Dead neurons, anti-specialization, and a one-parameter repair: auditing and causally restoring segment--task specialization in active-dendrite networks" for consideration in Neural Networks.

Active-dendrite networks route each neuron's computation through competing dendritic segments selected by task context, and a growing literature interprets the resulting gating statistics as evidence of segment--task specialization. Our manuscript audits that assumption end-to-end in this architecture family and reaches three conclusions:

1. **A measurement artifact.** Under production sampling budgets, 56--63% of neurons are never activated, and the standard argmax-over-all-neurons instrumentation attributes every dead neuron to segment 0 — fabricating an apparent "segment-0 dominance" of ~60% that vanishes entirely under active-only filtering. Any study that measures segment routing with naive argmax is at risk of the same artifact.

2. **Anti-specialization, not specialization.** With the artifact controlled, the true routing picture is the opposite of the literature's narrative: each phase's own training pushes the own-segment win rate from chance to *below* chance, and anchor Fisher mass is too diffuse across segments (top-1 share 11–18%) for any segment-level protection scheme to partition meaningfully.

3. **A causal repair and its mechanism.** Because the missing specialization is load-bearing, restoring it causally improves the retention–plasticity trade-off: a one-parameter constant bias on the own-segment dendritic input raises the learning term in every cell of a {dataset}×{depth}×{granularity} grid (Δℓ > 0 in 20/20 dose cells, significant in 18/20, n=10 paired), with a mild, honestly reported forgetting trade-off at depth. Signed-drift instrumentation shows the network never counteracts the bias at the parameter level — the response saturates for a scale reason, not an adversarial one — and a trainable-bias probe shows the network left to itself under-doses the repair by an order of magnitude.

We believe this work fits Neural Networks' scope on three counts: it (i) delivers a reusable methodological warning for the active-dendrite literature, (ii) provides a mechanistic, measurement-backed account of why segment structure does not specialize in this family, and (iii) demonstrates that a minimal, interpretable intervention suffices to shift the retention–plasticity balance — a result of direct interest to continual-learning and neuromorphic readership of the journal.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no conflicts of interest. All code, per-run records, and analysis scripts are publicly available at https://github.com/zhouranxi/delp-anchor-budget.

Thank you for your consideration.

Sincerely,
The Authors
