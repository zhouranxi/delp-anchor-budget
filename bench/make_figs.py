"""Generate paper figures for S2 (bounded anchor sets).

Inputs : anchor_budget_all.json  (168 records, all rounds)
         s2r4_d1_local.json      (survivor_history traces, T=20 M=2)
Outputs: ../paper_s2/figs/*.pdf (+ .png previews)
"""
import json
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "paper_s2", "figs")
os.makedirs(OUT, exist_ok=True)

C = {"unbounded": "#666666", "fifo": "#0072B2", "merge": "#009E73",
     "redun": "#D55E00", "random": "#CC79A7", "agew": "#E69F00",
     "agew2": "#56B4E9", "si": "#000000"}
MARK = {"unbounded": "*", "fifo": "o", "merge": "s", "redun": "^",
        "random": "D", "agew2": "P", "agew": "v"}
LABEL = {"unbounded": "unbounded (DeLP)", "fifo": "fifo", "merge": "merge",
         "redun": "redun", "random": "random", "agew2": "agew2", "agew": "agew"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

allr = json.load(open(os.path.join(HERE, "anchor_budget_all.json")))["results"]
tr = json.load(open(os.path.join(HERE, "s2r4_d1_local.json")))


def arms(ds="mnist", lam=5000.0):
    """key -> list of records"""
    agg = {}
    for r in allr:
        if r["kind"] != "delp" or "error" in r:
            continue
        if r.get("dataset", "mnist") != ds or r.get("lam", 5000.0) != lam:
            continue
        agg.setdefault((r["policy"], r["M"], r["ntask"]), []).append(r)
    return agg


def mstat(rs, k):
    v = [r[k] for r in rs]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: Pareto
def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    MK_M = {2: "o", 5: "s", 10: "^"}        # marker encodes budget M
    handles = {}
    for ax, nt in zip(axes, (10, 20)):
        A = {}
        for r in allr:
            if r["kind"] != "delp" or "error" in r or r["ntask"] != nt:
                continue
            if r.get("dataset", "mnist") != "mnist" or r.get("lam", 5000.0) != 5000.0:
                continue
            A.setdefault((r["policy"], r["M"]), []).append(r)
        if nt == 10:
            si = [r for r in allr if r["kind"] == "si" and "error" not in r]
            if si:
                m, _ = mstat(si, "final_avg")
                ax.axhline(m, color=C["si"], ls="--", lw=0.8, alpha=0.6)
                ax.text(0.98, m + 0.6, "SI (corrected)", fontsize=6.5,
                        color=C["si"], ha="right",
                        transform=ax.get_yaxis_transform())
        for (pol, M), rs in sorted(A.items()):
            mb, sb = mstat(rs, "bwt")
            ma, sa = mstat(rs, "final_avg")
            h = ax.errorbar(-mb, ma, xerr=sb, yerr=sa,
                            fmt=MK_M[M], color=C[pol], ms=5, capsize=2,
                            lw=0.8, alpha=0.9, elinewidth=0.8)
            if pol == "unbounded":
                handles[pol] = h
        ax.set_xlabel(r"forgetting $=-\mathrm{BWT}$ (points)")
        ax.set_ylabel("final avg. accuracy (%)")
        ax.set_title(f"Permuted-MNIST, T={nt}")
    # figure-level legend: color = policy, marker = budget
    from matplotlib.lines import Line2D
    order = ["unbounded", "fifo", "merge", "agew2", "agew", "redun", "random"]
    items = [Line2D([], [], color=C[p], marker="o", ms=5, lw=0.8,
                    label=LABEL[p]) for p in order if p != "unbounded"]
    items.insert(0, Line2D([], [], color=C["unbounded"], marker="*", ms=8,
                           lw=0.8, label=LABEL["unbounded"]))
    fig.legend(handles=items, loc="lower center", ncol=4, frameon=False,
               columnspacing=1.0, handletextpad=0.3, borderaxespad=0.15,
               bbox_to_anchor=(0.5, -0.16))
    fig.text(0.5, -0.245, "marker encodes budget: \u25CB M=2    \u25A1 M=5    \u25B3 M=10",
             ha="center", fontsize=7.5)
    fig.subplots_adjust(bottom=0.30)
    save(fig, "fig1_pareto")


# ------------------------------------------------------- fig 2: age structure
def fig_ageprofile():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    series = [
        ("unbounded", 5, 5000.0, "unbounded, $\\lambda{=}5000$", C["unbounded"], ":"),
        ("unbounded", 5, 1000.0, "unbounded, $\\lambda{=}1000$", "#999999", "-."),
        ("fifo", 2, 5000.0, "fifo, $M{=}2$, $\\lambda{=}5000$", C["fifo"], "-"),
        ("fifo", 5, 5000.0, "fifo, $M{=}5$, $\\lambda{=}5000$", "#7FBFDF", "-"),
    ]
    nt = 10
    for pol, M, lam, lab, col, ls in series:
        rs = [r for r in allr if r["kind"] == "delp" and "error" not in r
              and r.get("policy") == pol and r.get("M") == M
              and r.get("lam", 5000.0) == lam and r["ntask"] == nt
              and r.get("dataset", "mnist") == "mnist"]
        prof = [st.mean([r["acc_matrix"][-1][c] for r in rs]) for c in range(nt)]
        ax.plot(range(nt), prof, ls, color=col, marker="o", ms=3.5,
                lw=1.2, label=lab)
    ax.set_xlabel("task index (age: 0 = oldest)")
    ax.set_ylabel("final-phase accuracy (%)")
    ax.set_title("Budget re-allocates forgetting; $\\lambda$ scales it")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(30, 90)
    save(fig, "fig2_ageprofile")


# --------------------------------------------- fig 3: criteria + lambda robust
def fig_criteria():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    # (a) grandfathering profile, M=2 T=10
    ax = axes[0]
    nt = 10
    for pol, lab, col in (("fifo", "fifo ($M{=}2$)", C["fifo"]),
                          ("agew2", "agew2 ($M{=}2$)", C["agew2"]),
                          ("redun", "redun ($M{=}2$)", C["redun"])):
        rs = [r for r in allr if r["kind"] == "delp" and "error" not in r
              and r.get("policy") == pol and r.get("M") == 2
              and r["ntask"] == nt and r.get("lam", 5000.0) == 5000.0
              and r.get("dataset", "mnist") == "mnist" and r["seed"] in (0, 1, 2)]
        prof = [st.mean([r["acc_matrix"][-1][c] for r in rs]) for c in range(nt)]
        ax.plot(range(nt), prof, marker="o", ms=3.5, lw=1.2, color=col, label=lab)
    ax.set_xlabel("task index")
    ax.set_ylabel("final-phase accuracy (%)")
    ax.set_title("(a) eviction criterion, $M{=}2$, T=10")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, columnspacing=0.8, handletextpad=0.3)
    # (b) lambda robustness, T=20
    ax = axes[1]
    nt = 20
    for pol, lab, col in (("fifo", "fifo ($M{=}5$)", C["fifo"]),
                          ("unbounded", "unbounded", C["unbounded"])):
        xs, ys, es = [], [], []
        for lam in (1000.0, 5000.0, 20000.0):
            rs = [r for r in allr if r["kind"] == "delp" and "error" not in r
                  and r.get("policy") == pol and r.get("M") == 5
                  and r.get("lam", 5000.0) == lam and r["ntask"] == nt
                  and r.get("dataset", "mnist") == "mnist"]
            m, s = mstat(rs, "final_avg")
            xs.append(lam); ys.append(m); es.append(s)
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, ms=4.5,
                    capsize=2, lw=1.2, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1000.0, 5000.0, 20000.0])
    ax.set_xticklabels(["1000", "5000", "20000"])
    ax.set_xlabel(r"elastic weight $\lambda$")
    ax.set_ylabel("final avg. accuracy (%)")
    ax.set_title("(b) $\\lambda$ robustness, T=20")
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.32)
    save(fig, "fig3_criteria")


# ------------------------------------------------ fig 4: stack composition
def fig_stacktrace():
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.3), sharey=True)
    for ax, pol in zip(axes, ("fifo", "redun", "agew2", "merge")):
        rec = [r for r in tr if r["policy"] == pol and r["seed"] == 0][0]
        sh = rec["survivor_history"]
        G = np.zeros((20, 20))              # [origin, phase]
        for k, snap in enumerate(sh):
            for o in snap:
                G[o, k] = 1.0
        ax.imshow(G, aspect="auto", interpolation="nearest", cmap="Blues",
                  vmin=0, vmax=3)
        ax.set_title(pol, fontsize=9)
        ax.set_xlabel("phase")
        if pol == "fifo":
            ax.set_ylabel("anchor origin task")
        ax.set_xticks([0, 9, 19])
        ax.set_yticks([0, 9, 19])
    save(fig, "fig4_stacktrace")


fig_pareto()
fig_ageprofile()
fig_criteria()
fig_stacktrace()
print("all figures ->", os.path.abspath(OUT))
