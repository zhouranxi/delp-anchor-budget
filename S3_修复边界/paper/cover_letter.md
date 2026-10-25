Dear Editors,

Please consider our manuscript "Repair Headroom, Not Dying Gates:
When an Own-Segment Bias Helps Continual Learning — and When It
Silently Reverses" for publication in Neural Networks.

The paper is the companion to a dose–response audit of dendritic
context-gated protection (submitted separately). That audit found,
on bare-MLP active-dendrite networks, that segment–task
specialization fails to develop, that the standard
dominant-segment diagnostic is contaminated by dead neurons, and
that a constant own-segment bias causally repairs the learning term.
The natural question — the one this paper answers — is whether any
of that transfers beyond the bare-MLP family.

We test transfer with two pre-registered configurations: a
convolutional active-dendrite network (160 runs) and a decisive
same-architecture control with a single GroupNorm layer inserted
into the MLP (80 runs). Both fail their pre-registered criteria, and
the pattern of failure is the contribution:

- The repair does not transfer: 4 of 12 conv dose cells significant
  (criterion required 15/20), and final accuracy significantly
  reverses on strong-feature backbones.
- The dead-neuron artifact disappears entirely — conv+GroupNorm
  mortality is exactly zero — establishing that the artifact is
  architecture-bound, not instrument-bound.
- The tempting "dying-gate" explanation is rejected by the control:
  GroupNorm raises mortality while collapsing the repair gain, so
  mortality and gain decouple.
- What tracks the gain is baseline weakness (repair headroom); we
  report both the qualitative regularity and the failure of its
  quantitative version, with the counterexample identified.

We believe this fits Neural Networks' interest in rigorous negative
results with mechanistic content: every claim is pre-registered,
paired by seed, and all cells — including failed ones — are
reported. Each paper in this series ships with a self-contained
verification folder (source, code, data, pre-registrations), and
this one is no exception.

The work is original and not under consideration elsewhere. The
authors declare no competing interests.

Sincerely,
Ranxi Zhou
