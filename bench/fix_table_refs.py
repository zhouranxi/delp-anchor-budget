"""Add the two missing cross-references.

`tab:ablation` (the S4 four-arm control) and `tab:ctrl-specificity` (the
A1 five-arm specificity control) are the two tables the review most needs to
look at, and neither is referred to from the text.  A reviewer who has to
guess which claim a table supports reads it as decoration.

Both tables sit immediately after the paragraph that describes their
design, so the reference belongs at the end of that paragraph.
"""
MAIN = (r"C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\dab_t"
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


# --- tab:ablation ---
rep("tab-ablation-ref",
r"""under an explicit deprecation notice.

\begin{table}[t]
\centering
\caption{Four-arm control experiment""",
r"""under an explicit deprecation notice (Table~\ref{tab:ablation}).

\begin{table}[t]
\centering
\caption{Four-arm control experiment""")


# --- tab:ctrl-specificity ---
rep("tab-ctrl-specificity-ref",
r"""(no gate at all: $h=\mathrm{kwta}(t/0.565)$, a pure constant removal of
the damping factor).

\begin{table}[t]
\centering\small
\caption{Where the own-segment bias acts.""",
r"""(no gate at all: $h=\mathrm{kwta}(t/0.565)$, a pure constant removal of
the damping factor); Table~\ref{tab:ctrl-specificity} reports all five.

\begin{table}[t]
\centering\small
\caption{Where the own-segment bias acts.""")


open(MAIN, "w", encoding="utf-8", newline="\n").write(t)
print(f"\nwritten; delta = {len(t) - len(orig)} chars")
