# -*- coding: utf-8 -*-
"""S2 审计第8轮（收尾）：87/85、38.5 spread、agew2栈主张、merge per-T均值、Table2 n=10 值、T10拟合点集."""
import json, math, statistics as st
from pathlib import Path
from collections import Counter

R2 = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")
RB = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results")

def load_dir(base, parts):
    recs = []
    for p in sorted(Path(base).glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

mn = load_dir(R2, "anchor_budget_all.json")

# ---------- 1. 87/85%（T=20 M=2 fifo 栈内任务18,19 终相精度）----------
print("== 1. fifo M=2 栈内保护（论文: 87/85% final accuracy at M=2）==")
for T in (10, 20):
    rs = [r for r in mn if r["ntask"]==T and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==2]
    if rs:
        in_stack = [r["acc_matrix"][-1][j] for r in rs for j in range(T-2, T)]
        oldest = st.mean([r["acc_matrix"][-1][0] for r in rs])
        t18 = st.mean([r["acc_matrix"][-1][T-2] for r in rs]); t19 = st.mean([r["acc_matrix"][-1][T-1] for r in rs])
        print(f"  T={T}: 栈内两任务 final={t18:.1f}/{t19:.1f}  最老任务={oldest:.1f} (n={len(rs)})")

# ---------- 2. 38.5 spread ----------
print("\n== 2. 38.5-point spread 归属 ==")
for pol, lam, M in (("unbounded",1000.0,None),("unbounded",5000.0,None),("fifo",5000.0,2),("fifo",1000.0,5),("fifo",5000.0,5)):
    for T in (10, 20):
        rs = [r for r in mn if r["ntask"]==T and r["policy"]==pol and r.get("lam")==lam and (M is None or r["M"]==M)]
        if rs:
            prof = [st.mean([r["acc_matrix"][-1][j] for r in rs]) for j in range(T)]
            print(f"  {pol:9} λ={lam:.0f} M={M} T={T}: spread={max(prof)-min(prof):.1f}  首={prof[0]:.1f} 尾={prof[-1]:.1f}")

# ---------- 3. merge per-(T,M) max_rel_drift 均值 ----------
print("\n== 3. merge per-(T,M) 均值（论文: 43-45%@M=2 worst 64%, 18-34%@M=5）==")
mg = load_dir(RB, "r8_merge.json")
for T in (10, 20):
    for M in (2, 5):
        rs = [r for r in mg if r["ntask"]==T and r["M"]==M]
        if rs:
            mx = [r["max_rel_drift"] for r in rs]
            print(f"  T={T} M={M}: mean(max_rel_drift)={st.mean(mx)*100:.1f}%  range=[{min(mx)*100:.0f}%,{max(mx)*100:.0f}%] (n={len(rs)})")

# ---------- 4. Table 2 改 n=10 后的完整值 ----------
print("\n== 4. Table 2 候选修正值（unbounded/fifo 用 n=10）==")
for T in (10, 20):
    for pol in ("unbounded", "fifo"):
        rs = [r for r in mn if r["ntask"]==T and r["policy"]==pol and r.get("lam")==5000.0 and r["M"]==5]
        b = st.mean([r["bwt"] for r in rs]); bs = st.stdev([r["bwt"] for r in rs])
        a = st.mean([r["final_avg"] for r in rs]); e = st.mean([r["ell"] for r in rs])
        print(f"  T={T} {pol:10} n=10: BWT={b:+.1f}±{bs:.1f} acc={a:.1f} ℓ={e:.1f}")

# ---------- 5. T=10 log-linear 4点拟合（γ.9 vs unbounded 哪个进点集）----------
print("\n== 5. T=10 拟合的两种 4 点集 ==")
ewc_all = load_dir(R2, "r7_ewc*.json*")
g5 = load_dir(RB, "r8_gamma05.json"); g9 = load_dir(RB, "r8_gamma09.json")
def ell_of(rs): return st.mean([r["ell"] for r in rs])
pl = ell_of([r for r in ewc_all if r.get("mode")=="plain" and r["ntask"]==10 and r.get("lam")==1000.0])
g5_ = ell_of([r for r in g5 if r["ntask"]==10 and r.get("lam")==1000.0])
g9_ = ell_of([r for r in g9 if r["ntask"]==10 and r.get("lam")==1000.0])
fi = ell_of([r for r in mn if r["ntask"]==10 and r["policy"]=="fifo" and r.get("lam")==1000.0 and r["M"]==5])
un = ell_of([r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==1000.0])
print(f"  ℓ: plain={pl:.2f} γ.5={g5_:.2f} window={fi:.2f} γ.9={g9_:.2f} unbounded={un:.2f}")
def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sl = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    ic = my - sl*mx
    r2 = 1 - sum((y-(sl*x+ic))**2 for x,y in zip(xs,ys))/sum((y-my)**2 for y in ys)
    return sl, ic, r2
import math as m
xs4 = [m.log(h) for h in (1,2,5,10)]
print(f"  4点含γ.9: {fit(xs4, [pl,g5_,fi,g9_])}")
print(f"  4点含unb: {fit(xs4, [pl,g5_,fi,un])}")
xs5 = [m.log(h) for h in (1,2,5,10,20)]
print(f"  5点全含:  {fit(xs5, [pl,g5_,fi,g9_,un])}")

# ---------- 6. agew2 T=20 M=2 论文原文主张 vs 数据 ----------
print("\n== 6. agew2 栈（数据: (0,19)×6, 其他各1）==")
cifar = load_dir(R2, "s2r6_b_t20*.json*")
ag2 = [r for r in cifar if r.get("dataset")=="cifar100" and r["policy"]=="agew2" and r["M"]==2]
for r in ag2:
    print(f"  seed {r['seed']}: final stack={tuple(r['survivor_history'][-1])}")
