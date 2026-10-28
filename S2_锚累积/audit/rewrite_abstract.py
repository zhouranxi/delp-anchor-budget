"""Rewrite the S2 abstract to fit Applied Soft Computing's 250-word limit.

Replaces the \\begin{abstract}...\\end{abstract} block in main.tex with a
compressed version, then verifies the word count with a strict counter that
treats LaTeX math symbols as words (an upper bound on what Elsevier counts).

Run from the package root:  python audit/rewrite_abstract.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "main.tex")

NEW = r"""\begin{abstract}
Usage-localized elastic weight consolidation (\delp{}) protects dendritic
segments with per-neuron elastic anchors accumulated over tasks.
Decomposing retention as
$\bar a_T = \ell + \frac{T-1}{T}\,\mathrm{BWT}$ localizes the cost of long
sequences in the \emph{learning term} $\ell$, not the architecture:/nanchors persist at full weight as an elastic drag on later learning. We call
this the \emph{anchor accumulation problem} and bound the stack at $M$
entries. A FIFO budget recovers $+4.5$ and $+9.0$ points of final accuracy
over unbounded \delp{} at $T{=}10$ and $T{=}20$ ($M{=}5$; $n{=}10$ paired
seeds; Wilcoxon $p=0.002$); at $M{=}2$, $66.7\%$ ($+12.3$ points). The budget
is a distinct knob from the elastic weight $\lambda$: it controls the
\emph{age structure} of forgetting, $\lambda$ its \emph{amount}. Mass-keyed
eviction criteria fail --- they grandfather stale anchors and evict anchors
at birth, losing $-34.5$ BWT --- whereas age-first criteria are immune.
Merging removes redundant inter-anchor cross terms: the penalty \emph{value}
drops one-sidedly by up to $64\%$ (gradient unchanged) while accuracy stays
indistinguishable from unbounded. The effect replicates on Fashion-MNIST and
Split-CIFAR-100, where at $T{=}20$, $M{=}2$, five of six bounded policies
beat unbounded (up to $+5.0$). A tuned online-EWC discount overtakes the
budget at every grid cell: discounting is itself a soft bound, and soft and
hard bounds obey one log-linear law in the protected-mass horizon
($R^2=0.99$) --- so bounding anchor mass, not eviction, is the active
ingredient. A no-protection baseline stays above every arm on
Split-CIFAR-100; the residual is an auditable protection set.
\end{abstract}"""


def count_words(tex_block):
    """Strict word count: LaTeX control sequences -> their argument, then
    every alphanumeric run counts (so bare math symbols count as words)."""
    a = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r" \1 ", tex_block)
    a = re.sub(r"\\[a-zA-Z]+", " ", a)
    a = re.sub(r"[{}$]", " ", a)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", a))


if __name__ == "__main__":
    with open(TEX, encoding="utf-8") as f:
        src = f.read()

    old = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", src, re.S)
    if not old:
        raise SystemExit("no abstract block found")

    w_old = count_words(old.group(0))
    w_new = count_words(NEW)
    print(f"old abstract: {w_old} words")
    print(f"new abstract: {w_new} words  (limit 250) -> "
          f"{'OK' if w_new <= 250 else 'OVER by ' + str(w_new - 250)}")

    if w_new <= 250:
        src = src[:old.start()] + NEW + src[old.end():]
        with open(TEX, "w", encoding="utf-8") as f:
            f.write(src)
        print("main.tex updated.")
    else:
        print("NOT written -- still over the limit; tighten NEW first.")
