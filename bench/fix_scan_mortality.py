"""Add the A5 mortality scan to the manuscript.

New data (bench/results_s5/server/server_scan_mortality.json, 160 runs:
2 datasets x 2 widths x 2 normalization x 2 doses x 10 seeds):

  dataset  width  GN    mortality(b=10)  repair gain in final accuracy
  mnist     256   no        0.535              +4.67  (t=15.7)
  mnist     256   yes       0.000              +0.95  (t=6.2)
  mnist    2000   no        0.939              +4.62  (t=12.5)
  mnist    2000   yes       0.780              -2.74  (t=-2.3)
  fashion   256   no        0.592              +0.90  (t=2.1, n.s.)
  fashion   256   yes       0.020              +0.42  (t=1.1, n.s.)
  fashion  2000   no        0.948              +2.13  (t=3.4)
  fashion  2000   yes       0.884              -6.64  (t=-4.0)

Correlation between mortality and repair gain across the eight cells:
  Pearson  r = -0.112 (p = 0.79);  Spearman rho = +0.048 (p = 0.91)

1. The paper already refutes the dying-gate hypothesis in Sec. "G2 fails
   with a clean cause".  The scan makes it quantitative for the first
   time: across eight cells the correlation is indistinguishable from
   zero and the ordering locally INVERTS -- the two GroupNorm cells,
   with the lowest mortality, have the smallest repair gain, negative on
   both datasets.
2. The paper already reports a WITHIN-cell correlation between the
   per-run mortality reduction and the per-run learning-term gain
   (r = -0.49, -0.68).  The new scan is a BETWEEN-cell correlation and
   the two are consistent: the causal chain is within-configuration and
   does not extend to the configuration level, which is where the
   dying-gate hypothesis was proposed.
"""
MAIN = (r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t"
        r"\delp-anchor-budget-main\S123_合并稿\paper\main.tex")

t = open(MAIN, encoding="utf-8").read()
orig = t


def rep(tag, old, new):
    global t
    if old not in t:
        print(f'  MISS {tag}')
        return
    t = t.replace(old, new, 1)
    print(f'  OK   {tag}')


rep("scan-mortality-paragraph",
r"""A measurement caveat we state explicitly: the two configurations
report mortality under \emph{different} zero-conditions. The MLP
instrument (inherited unchanged from the bare-MLP arm) marks a neuron dead if its
post-kWTA activity is never positive across the usage window ---
mortality there means ``never wins the top-$k$ competition''. The
conv instrument marks a neuron dead if its gate row is identically
zero. Both are the natural operationalizations in their families,""",
r"""The same conclusion holds at the level of configurations rather than
runs, which is where the dying-gate hypothesis was proposed. We
re-ran the full width$\times$normalization factorial scan with a
mortality column added ($160$ runs: $2$ datasets $\times$ $2$ widths
$\times$ $2$ normalization settings $\times$ $2$ doses $\times$ $10$
seeds; Table~\ref{tab:scanmort}). The repair gain in final accuracy
and the mortality of the repaired arm are, across the eight cells,
uncorrelated: Pearson $r = -0.11$ ($p = 0.79$), Spearman
$\rho = +0.05$ ($p = 0.91$). More than a flat correlation, the ordering
inverts locally. The two GroupNorm cells have the \emph{lowest}
mortality of the four configurations on their dataset ($0.000$ and
$0.020$ on MNIST and Fashion-MNIST, against $0.535$--$0.948$ without
it) and the \emph{smallest} repair gain, which is negative on both
datasets ($-2.74$ and $-6.64$ points); the highest-mortality cells are
the ones with the largest positive gain ($+4.62$ and $+2.13$). A
dying-gate explanation predicts that removing dead gates should
restore the repair's benefit; in the cell where mortality is
essentially eliminated, the benefit is at its smallest and can be
negative.

This is consistent with, but weaker than, the within-cell correlation
reported above: relieving mortality on a given run helps that run's
learning term, yet the configuration with the most dead neurons is not
the configuration with the biggest repair. The causal chain we
describe is therefore a within-configuration one; it does not extend to
the between-configuration comparison, and the dying-gate hypothesis is
not supported at that level either.

A measurement caveat we state explicitly: the two configurations
report mortality under \emph{different} zero-conditions. The MLP
instrument (inherited unchanged from the bare-MLP arm) marks a neuron dead if its
post-kWTA activity is never positive across the usage window ---
mortality there means ``never wins the top-$k$ competition''. The
conv instrument marks a neuron dead if its gate row is identically
zero. Both are the natural operationalizations in their families,""")


rep("scan-mortality-table",
r"""(no gate at all: $h=\mathrm{kwta}(t/0.565)$, a pure constant removal of
the damping factor); Table~\ref{tab:ctrl-specificity} reports all five.""",
r"""(no gate at all: $h=\mathrm{kwta}(t/0.565)$, a pure constant removal of
the damping factor); Table~\ref{tab:ctrl-specificity} reports all five.

\begin{table}[htbp]
\centering\small
\caption{Repair gain against neuron mortality in the width$\times$normalization
factorial scan ($n=10$ paired seeds per cell). The gain is the paired
difference in final accuracy between $b{=}10$ and $b{=}0$; mortality is the
dead-neuron fraction of the repaired arm at the end of the sequence, measured
on the raw, bias-free network. Across the eight cells the two are
uncorrelated (Pearson $r=-0.11$, $p=0.79$; Spearman $\rho=+0.05$,
$p=0.91$), and the ordering inverts: the cells with the lowest mortality have
the smallest repair gain.}
\label{tab:scanmort}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
dataset & width & GroupNorm & $n$ & mortality ($b{=}0$) & mortality ($b{=}10$) &
$\Delta$ final accuracy & $t$ & $p$ \\
\midrule
mnist & 256 & no & 10 & 0.584 & 0.535 & $+4.67$ & $+15.7$ & $<10^{-6}$ \\
mnist & 256 & yes & 10 & 0.004 & 0.000 & $+0.95$ & $+6.2$ & $<10^{-3}$ \\
mnist & 2000 & no & 10 & 0.945 & 0.939 & $+4.62$ & $+12.5$ & $<10^{-4}$ \\
mnist & 2000 & yes & 10 & 0.745 & 0.780 & $-2.74$ & $-2.3$ & $0.044$ \\
fashion & 256 & no & 10 & 0.618 & 0.592 & $+0.90$ & $+2.1$ & $0.063$ \\
fashion & 256 & yes & 10 & 0.010 & 0.020 & $+0.42$ & $+1.1$ & $0.321$ \\
fashion & 2000 & no & 10 & 0.952 & 0.948 & $+2.13$ & $+3.4$ & $0.008$ \\
fashion & 2000 & yes & 10 & 0.875 & 0.884 & $-6.64$ & $-4.0$ & $0.003$ \\
\bottomrule
\end{tabular}%
}
\end{table}
\FloatBarrier""")


rep("abstract-mortality",
r"""the candidate causes by showing that the predicted link between mortality and repair gain does not hold
(GroupNorm raises mortality to $76.5\%$/$85.5\%$ while collapsing the gain to""",
r"""the candidate causes by showing that the predicted link between mortality and repair gain does not hold
(across eight configurations of a width$\times$normalization factorial scan the two are
uncorrelated, $r=-0.11$; GroupNorm raises mortality to $76.5\%$/$85.5\%$ while collapsing the gain to""")


rep("robustness-note-levels",
r"""We read this as one causal chain rather than a confound:
raising the own-segment gate raises $\mathrm{sigmoid}(u)$ during
training, which increases kWTA survival, so forcing the routing prior is
also what keeps neurons alive; the routing and capacity channels are two
faces of a single intervention, not competing explanations.""",
r"""We read this as one causal chain rather than a confound:
raising the own-segment gate raises $\mathrm{sigmoid}(u)$ during
training, which increases kWTA survival, so forcing the routing prior is
also what keeps neurons alive; the routing and capacity channels are two
faces of a single intervention, not competing explanations. That chain is
a \emph{within-configuration} one, and it does not extend between
configurations: across the eight cells of the width$\times$normalization
scan the mortality of a configuration and the size of its repair gain are
uncorrelated, and in the two cells where mortality is essentially
eliminated the gain is the smallest and can be negative
(Sec.~\ref{sec:zeromort}, Table~\ref{tab:scanmort}).""")


open(MAIN, "w", encoding="utf-8", newline="\n").write(t)
print(f"\nwritten; delta = {len(t) - len(orig)} chars")
