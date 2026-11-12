# -*- coding: utf-8 -*-
"""make_figs.py -- three figures for the new paper, from raw run JSONs.

  python 新论文/make_figs.py     (from the repository root)

Outputs into 新论文/paper/figs/:
  fig_frontier.pdf   R2: gain-retention frontier at T=20 (five arms + anti)
  fig_flip.pdf       R2T: first round vs extension, C1 per-seed deltas
  fig_lambda.pdf     R3: gain(lambda) dose curve with Holm significance
"""
import glob, json, math, os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "新论文", "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f, encoding="utf-8"))
        out += d["results"] if isinstance(d, dict) and "results" in d else d
    return out


# ---------------------------------------------------------------- fig 1
def fig_frontier():
    g = load(os.path.join(ROOT, "bench", "results_r2",
                          "r2_grid.part*.json")) + \
        load(os.path.join(ROOT, "bench", "results_r2", "r2_anti.part*.json"))
    T = 20
    mean = lambda q, m: st.mean(r[m] for r in q)
    arms = [("base", "baseline (gate as-is)", "o", "#7f7f7f"),
            ("renorm", "repair (gate removed, $1/G$ rescale)", "s", "#1f77b4"),
            ("renorm_lamup", "repair, $\\lambda{\\times}3.13$", "^", "#ff7f0e"),
            ("renorm_rec05", "+ budget $\\to$ recent ($\\alpha{=}0.5$)",
             "D", "#2ca02c"),
            ("renorm_rec10", "+ budget $\\to$ recent ($\\alpha{=}1$)",
             "*", "#d62728"),
            ("renorm_anti10", "+ budget $\\to$ old ($\\alpha{=}1$)",
             "v", "#9467bd")]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for arm, lab, mk, c in arms:
        q = [r for r in g if r["arm"] == arm and r["ntask"] == T]
        x = -mean(q, "bwt")          # retention (higher = better)
        y = mean(q, "avg")           # final average
        s = 130 if mk == "*" else 55
        ax.scatter(x, y, marker=mk, s=s, c=c, label=lab, zorder=3,
                   edgecolors="white", linewidths=0.5)
    ax.set_xlabel("retention  $-\\mathrm{BWT}$ (points, higher = better)")
    ax.set_ylabel("final average accuracy $\\bar{a}$ (points)")
    ax.set_title("R2: the gain--retention frontier at $T{=}20$ (pooled, $n{=}20$)")
    # annotate the two frontier extremes
    q = [r for r in g if r["arm"] == "renorm_rec10" and r["ntask"] == T]
    ax.annotate("frontier moves\n(here)", xy=(-mean(q, "bwt"), mean(q, "avg")),
                xytext=(8, -22), textcoords="offset points", fontsize=7.5,
                color="#d62728",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7))
    q = [r for r in g if r["arm"] == "renorm_lamup" and r["ntask"] == T]
    ax.annotate("$\\lambda$ alone slides\nalong this direction",
                xy=(-mean(q, "bwt"), mean(q, "avg")),
                xytext=(-88, -30), textcoords="offset points", fontsize=7.5,
                color="#ff7f0e",
                arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=0.7))
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(FIGS, "fig_frontier.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_flip():
    new1 = load(os.path.join(ROOT, "bench", "results_r2t", "r2t_grid.part*.json"))
    ext = load(os.path.join(ROOT, "bench", "results_r2t", "r2t_ext.part*.json"))
    ctrl_ext = load(os.path.join(ROOT, "bench", "results_r2t",
                                 "r2t_ctrl_ext.part*.json"))
    e9 = load(os.path.join(ROOT, "bench", "results_e9", "e9_grid_s*.json"))

    def seeds_deltas(pool, seeds):
        idx = {(r["arm"], r["seed"]): r for r in pool}
        return [idx[("base_rec10", s)]["ell"] - idx[("base", s)]["ell"]
                for s in seeds]

    d1 = seeds_deltas(new1 + e9, range(10))
    d2 = seeds_deltas(new1 + ext + e9 + ctrl_ext, range(20))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6),
                             sharey=True, width_ratios=[1, 1.6])
    for ax, d, title in zip(axes, (d1, d2),
                            ("first round, $n{=}10$", "extension, $n{=}20$")):
        xs = range(1, len(d) + 1)
        cols = ["#2ca02c" if x > 0 else "#d62728" for x in d]
        ax.bar(xs, d, color=cols, width=0.72, zorder=3)
        m = st.mean(d)
        ax.axhline(m, color="black", lw=1.1, ls="--", zorder=4)
        ax.axhline(0, color="black", lw=0.6, zorder=2)
        ax.set_title(title + (f"  (mean ${m:+.2f}$)"), fontsize=8.5)
        ax.set_xlabel("seed")
    axes[0].set_ylabel("$\\Delta\\ell$ (base\\_rec10 $-$ base)")
    axes[0].set_xticks(range(1, 11))
    axes[1].set_xticks([1, 5, 10, 15, 20])
    fig.suptitle("R2T: the extension clause overturns an apparent transfer "
                 "(green = positive)", fontsize=9, y=1.04)
    fig.savefig(os.path.join(FIGS, "fig_flip.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_lambda():
    rs = load(os.path.join(ROOT, "bench", "results_r3", "r3_grid.part*.json"))
    idx = {(float(r["lam"]), r["arm"], r["ds"], r["seed"]): r for r in rs}
    LAMS = [0.0, 3.16, 31.6, 316.0, 1000.0, 3160.0]
    means, sigs = [], []

    def paired(d):
        n = len(d)
        m, sd = st.mean(d), st.stdev(d)
        t = m / (sd / math.sqrt(n))
        return m, t

    raw_ps = []
    for lam in LAMS:
        d = [idx[(lam, "renorm", ds, s)]["ell"] - idx[(lam, "base", ds, s)]["ell"]
             for ds in ("mnist", "fashion") for s in range(10)]
        m, t = paired(d)
        means.append(m)
        # p via t-dist approximation good enough for plotting significance
        raw_ps.append(t)  # keep t; Holm with normal approx below
    # Holm via normal-approx p (plot-only; exact values in the stats JSON)
    from math import erfc, sqrt
    ps = [erfc(abs(t) / sqrt(2)) for t in raw_ps]
    order = sorted(range(6), key=lambda i: ps[i])
    adj = [0.0] * 6
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (6 - r) * ps[i])
        adj[i] = min(1.0, run)
    sigs = [a < 0.05 for a in adj]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    xs = list(range(6))
    labels = ["0", "3.2", "32", "316", "1000", "3160"]
    cols = ["#2ca02c" if s else "#b0b0b0" for s in sigs]
    ax.axhline(0, color="black", lw=0.6)
    bars = ax.bar(xs, means, color=cols, width=0.66, zorder=3)
    for x, m, s in zip(xs, means, sigs):
        ax.text(x, m + (0.03 if m >= 0 else -0.07), f"{m:+.2f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("$\\lambda$ (protection strength; the companion audit used "
                  "$1000$)")
    ax.set_ylabel("$\\Delta\\ell$ (repair $-$ baseline)")
    ax.set_title("R3: gate effect vs.\\ $\\lambda$ on the conv trunk\n"
                 "(green = Holm-significant; error of the single audited "
                 "point in gray band)", fontsize=8.5)
    # highlight the audited point
    ax.axvspan(3.6, 4.4, color="#d9d9d9", alpha=0.55, zorder=1)
    ax.text(4, max(means) * 0.92, "audited\n$\\lambda$", ha="center",
            fontsize=7.5, color="#555555")
    fig.savefig(os.path.join(FIGS, "fig_lambda.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_frontier()
    fig_flip()
    fig_lambda()
    for f in sorted(os.listdir(FIGS)):
        print("wrote", f)
