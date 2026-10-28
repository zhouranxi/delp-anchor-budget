"""Graphical abstract for the S2 manuscript (Applied Soft Computing submission).

Every number plotted here is recomputed from the raw per-seed records in
results/anchor_budget_all.json -- nothing is hard-coded. Run from the package
root:  python audit/make_graphical_abstract.py

Output: graphical_abstract.png  (>= 1328 px wide, landscape, 13 cm readable)
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "anchor_budget_all.json")
OUT = os.path.join(ROOT, "graphical_abstract.png")

LAM = 5000.0

C = {
    "unbounded": "#444441",
    "fifo":      "#1D9E75",
    "merge":     "#888780",
    "agew2":     "#85B7EB",
    "agew":      "#BA7517",
    "redun":     "#993C1D",
    "random":    "#ED93B1",
}
LABEL = {
    "unbounded": "unbounded (DeLP)",
    "fifo":      "FIFO",
    "merge":     "merge",
    "agew2":     "age-weighted 2",
    "agew":      "age-weighted",
    "redun":     "redundancy",
    "random":    "random",
}
MK_M = {2: "o", 5: "s", 10: "^"}

# ----------------------------------------------------------------- load data
with open(RESULTS, encoding="utf-8") as f:
    allr = json.load(f)["results"]


def mstat(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def collect(nt):
    A = {}
    for r in allr:
        if r["kind"] != "delp" or "error" in r or r["ntask"] != nt:
            continue
        if r.get("dataset", "mnist") != "mnist" or r.get("lam", LAM) != LAM:
            continue
        A.setdefault((r["policy"], r["M"]), []).append(r)
    return A


A10, A20 = collect(10), collect(20)

# unbounded references
unb = {}
for nt in (10, 20):
    rs = [r for r in allr
          if r["kind"] == "delp" and r["policy"] == "unbounded"
          and r["ntask"] == nt and r.get("dataset", "mnist") == "mnist"
          and r.get("lam", LAM) == LAM and "error" not in r]
    unb[nt] = mstat(rs, "final_avg")

# SI reference (T=10 only, as in fig1)
si_acc = None
si_rs = [r for r in allr if r["kind"] == "si" and "error" not in r]
if si_rs:
    si_acc, _ = mstat(si_rs, "final_avg")

# headline: FIFO M=2 at T=10
fifo2 = A10.get(("fifo", 2), [])
fifo2_acc = mstat(fifo2, "final_avg") if fifo2 else None
fifo2_bwt = mstat(fifo2, "bwt") if fifo2 else None

# ------------------------------------------------------- figure setup (13 cm)
DPI = 300
W_CM, H_CM = 13.0, 7.6
fig = plt.figure(figsize=(W_CM, H_CM), dpi=DPI)
fig.patch.set_facecolor("white")

# left column: mechanism schematic
axL = fig.add_axes([0.045, 0.235, 0.38, 0.60])
axL.set_xlim(0, 10)
axL.set_ylim(0, 10)
axL.axis("off")

# right column: the pareto plot
axR = fig.add_axes([0.475, 0.335, 0.475, 0.50])

# ---- title
fig.text(0.5, 0.975,
         "Bounding the anchor stack recovers plasticity in elastic weight consolidation",
         ha="center", va="top", fontsize=12.0, fontweight="bold", color="#1B1B1B")
fig.text(0.5, 0.917,
         "Anchor accumulation problem  ·  Permuted-MNIST  ·  DeLP with per-neuron elastic anchors",
         ha="center", va="top", fontsize=8.8, color="#55534F")


def rbox(ax, x, y, w, h, text, fc, ec, tc="#1B1B1B", fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22",
                                linewidth=0.9, facecolor=fc, edgecolor=ec,
                                mutation_scale=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color="#6B6862", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, linewidth=lw,
                                 color=color, zorder=1,
                                 shrinkA=1.0, shrinkB=1.0))


# --- 1. anchor stack grows with tasks
rbox(axL, 0.35, 8.05, 4.5, 1.55,
     "Anchor stack grows with tasks\none anchor set appended per task",
     "#F1EFE8", "#B4B2A9", fs=8.0)
for i, t in enumerate(["task 1", "task 2", "task 3", "...", "task T"]):
    x = 0.62 + i * 0.85
    rbox(axL, x, 7.58, 0.72, 0.40, t, "#FFFFFF", "#B4B2A9", fs=6.0)
arrow(axL, 2.6, 7.45, 2.6, 6.95)

# --- 2. the problem: early anchors act as elastic drag
rbox(axL, 0.35, 5.30, 4.5, 1.40,
     "Anchor accumulation problem\nearly anchors persist at full weight →\nelastic drag on later learning",
     "#FCEBEB", "#E24B4A", fs=7.8)
arrow(axL, 2.6, 5.17, 2.6, 4.67, color="#A32D2D")

# --- 3. the fix: bounded window
rbox(axL, 0.35, 2.90, 4.5, 1.40,
     "Bounded anchor set: keep M anchors\n(FIFO eviction) → early drag removed",
     "#E1F5EE", "#1D9E75", fs=7.8, bold=True)
arrow(axL, 2.6, 2.77, 2.6, 2.27, color="#0F6E56")

# --- 4. the knob distinction
rbox(axL, 0.35, 0.20, 2.15, 1.90,
     "Budget M\n→ age structure\nof forgetting",
     "#EEEDFE", "#7F77DD", fs=7.4)
rbox(axL, 2.70, 0.20, 2.15, 1.90,
     "Weight λ\n→ amount\nof forgetting",
     "#EEEDFE", "#7F77DD", fs=7.4)

# ------------------------------------------------------ right: pareto panel
for ax, nt in zip([axR], [10]):
    MK = MK_M
    for (pol, M), rs in sorted(A10.items()):
        mb, sb = mstat(rs, "bwt")
        ma, sa = mstat(rs, "final_avg")
        ax.errorbar(-mb, ma, xerr=sb, yerr=sa, fmt=MK[M], color=C[pol],
                    ms=6.5, capsize=2.5, lw=0.9, alpha=0.92,
                    elinewidth=0.9, zorder=3)
    if si_acc is not None:
        ax.axhline(si_acc, color="#8A8880", ls="--", lw=0.9, alpha=0.75, zorder=1)
        ax.text(0.985, si_acc + 0.7, "SI (corrected)", fontsize=7.0,
                color="#6B6862", ha="right",
                transform=ax.get_yaxis_transform())

    # highlight the unbounded point and the FIFO M=2 point
    if unb[10][0] is not None:
        ub = [r for r in allr
              if r["kind"] == "delp" and r["policy"] == "unbounded"
              and r["ntask"] == 10 and r.get("dataset", "mnist") == "mnist"
              and r.get("lam", LAM) == LAM and "error" not in r]
        if ub:
            mub, sub = mstat(ub, "bwt")
            aub, saub = mstat(ub, "final_avg")
            ax.plot(-mub, aub, marker="*", ms=17, color=C["unbounded"],
                    markeredgecolor="white", markeredgewidth=0.9, zorder=5)

    ax.set_xlabel("forgetting  =  −BWT  (points)", fontsize=9.0, labelpad=3)
    ax.set_ylabel("final avg. accuracy (%)", fontsize=9.0, labelpad=3)
    ax.set_title("Permuted-MNIST,  T = 10  (λ = 5000)", fontsize=9.8,
                 pad=6, color="#1B1B1B")
    ax.tick_params(labelsize=8.2, colors="#4A4844")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#B4B2A9")
    ax.grid(True, lw=0.5, color="#E6E4DE", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    # headroom so the panel title and the topmost points never clip
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.10)

# annotate the headline point
if fifo2_acc and fifo2_bwt:
    axR.annotate(f"FIFO, M=2:  {fifo2_acc[0]:.1f}%\n(+{fifo2_acc[0]-unb[10][0]:.1f} vs unbounded)",
                 xy=(-fifo2_bwt[0], fifo2_acc[0]),
                 xytext=(0.30, 0.955), textcoords="axes fraction",
                 fontsize=8.0, color="#0F6E56", fontweight="bold",
                 ha="left", va="top", linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.32", fc="#E1F5EE",
                           ec="#1D9E75", lw=0.8),
                 arrowprops=dict(arrowstyle="-", color="#1D9E75", lw=0.9,
                                 connectionstyle="arc3,rad=0.12"))

# legend
order = ["unbounded", "fifo", "merge", "agew2", "agew", "redun", "random"]
items = [Line2D([], [], color=C[p], marker="o", ms=6, lw=0.9,
                label=LABEL[p]) for p in order if p != "unbounded"]
items.insert(0, Line2D([], [], color=C["unbounded"], marker="*", ms=11,
                       lw=0.9, label=LABEL["unbounded"]))
leg = axR.legend(handles=items, loc="upper center", ncol=4, frameon=False,
                 fontsize=7.4, columnspacing=0.9, handletextpad=0.3,
                 handlelength=1.2, borderaxespad=0.1,
                 bbox_to_anchor=(0.5, -0.155))
for t in leg.get_texts():
    t.set_color("#3A3835")
fig.text(0.712, 0.212,
         "marker encodes budget:   ○ M = 2      □ M = 5      △ M = 10",
         ha="center", va="top", fontsize=7.6, color="#55534F")

# --------------------------------------------------------- bottom takeaway
fig.text(0.045, 0.145,
         "Bounding accumulated anchor mass — not the eviction rule — is the active ingredient:",
         ha="left", va="top", fontsize=9.2, color="#1B1B1B")
fig.text(0.045, 0.088,
         "soft (discount) and hard (window) bounds fall on one log-linear law in the protected-mass horizon.",
         ha="left", va="top", fontsize=8.5, color="#55534F")

fig.savefig(OUT, dpi=DPI, facecolor="white", bbox_inches=None)

# --------------------------------------------------------------- verify size
from PIL import Image
with Image.open(OUT) as im:
    w, h = im.size
print(f"written: {OUT}")
print(f"pixels: {w} x {h}   (need width >= 1328: {'OK' if w >= 1328 else 'FAIL'})")
print(f"aspect: {w/h:.3f} (landscape: {'OK' if w > h else 'FAIL'})")
print(f"at 13 cm wide -> {w/ (13/2.54):.0f} px/inch effective")
