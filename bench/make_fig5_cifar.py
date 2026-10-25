# -*- coding: utf-8 -*-
"""Fig 5: CIFAR stack traces (T=20, M=2, seed 0, 6 policies) + decomposition check."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "paper_s2", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

def _find(fname):
    """bench/ (server scratch) -> bench/results/ (committed archive) -> CWD."""
    for cand in (os.path.join(HERE, fname),
                 os.path.join(HERE, "results", fname),
                 fname):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{fname} not found in {HERE}, {HERE}/results or CWD")

recs = []
for f in ["s2r6_b_t20.json.part0", "s2r6_b_t20.json.part1",
          "s2r6_b_t10.json.part0", "s2r6_b_t10.json.part1",
          "s2r6_b_t10u.json.part0", "s2r6_b_t10u.json.part1"]:
    recs += json.load(open(_find(f)))["results"]

# ---- fig 5: stack traces, T=20 M=2 seed 0, all six bounded policies ----
pols = ["fifo", "redun", "agew", "agew2", "merge", "random"]
fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.4), sharey=True, sharex=True)
for ax, pol in zip(axes.ravel(), pols):
    rec = [r for r in recs if r["ntask"] == 20 and r["policy"] == pol
           and r["M"] == 2 and r["seed"] == 0][0]
    sh = rec["survivor_history"]
    G = np.zeros((20, 20))
    for k, snap in enumerate(sh):
        for o in snap:
            G[o, k] = 1.0
    ax.imshow(G, aspect="auto", interpolation="nearest", cmap="Blues",
              vmin=0, vmax=3)
    ax.set_title(pol, fontsize=9)
    ax.set_xticks([0, 9, 19])
    ax.set_yticks([0, 9, 19])
for ax in axes[1]:
    ax.set_xlabel("phase")
for ax in axes[:, 0]:
    ax.set_ylabel("anchor origin task")
fig.subplots_adjust(hspace=0.32, wspace=0.10)
fig.savefig(os.path.join(OUT, "fig5_cifar_stacktrace.pdf"))
fig.savefig(os.path.join(OUT, "fig5_cifar_stacktrace.png"))
plt.close(fig)
print("wrote fig5_cifar_stacktrace")

# ---- decomposition check: a_T = ell + (T-1)/T * BWT  (CIFAR, all records) ----
worst = 0.0
for r in recs:
    T = r["ntask"]
    full = r["acc_matrix"]
    row = full[-1]
    ell = sum(full[i][i] for i in range(T)) / T
    aT = sum(row) / T
    rhs = ell + (T - 1) / T * r["bwt"]
    worst = max(worst, abs(aT - rhs))
print(f"decomposition |a_T - (ell + (T-1)/T*BWT)| max over {len(recs)} records: {worst:.2e}")
