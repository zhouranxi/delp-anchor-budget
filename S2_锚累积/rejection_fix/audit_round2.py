# -*- coding: utf-8 -*-
"""S2 审计第2轮：MNIST 主表、Fashion、EWC、λ=0 子集、附录矩阵."""
import json, math, statistics as st, re
from pathlib import Path

BASE = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")
VENV = None
def load(parts):
    recs = []
    for p in sorted(BASE.glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

def mean_sd(xs): return st.mean(xs), st.stdev(xs)
def paired_t(deltas):
    m = st.mean(deltas); s = st.stdev(deltas); n = len(deltas)
    return m / (s / math.sqrt(n))

# ---------- 1. λ=0 的 20 条到底是什么 ----------
print("== 1. λ=0 20 条记录的构成 ==")
lam0 = load("r7_cifar_lam0_t*.json*")
from collections import Counter
c = Counter((r["ntask"], r.get("policy"), r.get("M"), r.get("lam")) for r in lam0)
for k, v in sorted(c.items()):
    print(f"  ntask={k[0]:2} policy={k[1]:10} M={k[2]} lam={k[3]}: {v} 条")
# M=2 子集的 sd
for nt in (10, 20):
    for M in sorted(set(r.get("M") for r in lam0)):
        rs = [r for r in lam0 if r["ntask"] == nt and r.get("M") == M]
        if rs:
            m, s = mean_sd([r["final_avg"] for r in rs])
            print(f"  λ=0 T={nt} M={M} 子集: {m:.2f}±{s:.2f} (n={len(rs)})  论文: T10 27.0±1.7 / T20 36.9±3.4")
# 种子是否重复
for nt in (10, 20):
    for M in sorted(set(r.get("M") for r in lam0)):
        seeds = [r["seed"] for r in lam0 if r["ntask"] == nt and r.get("M") == M]
        dup = len(seeds) - len(set(seeds))
        print(f"  λ=0 T={nt} M={M}: seeds={sorted(seeds)} 重复={dup}")

# ---------- 2. T=20 unbounded ell/BWT ----------
print("\n== 2. T=20 unbounded ell/BWT（论文: ℓ 31.4, BWT -1.7）==")
cifar = load("s2r6_b_t10*.json*") + load("s2r6_b_t20*.json*")
cifar = [r for r in cifar if r.get("dataset") == "cifar100"]
unb20 = [r for r in cifar if r["ntask"] == 20 and r["policy"] == "unbounded"]
print(f"  T=20 unbounded: ℓ={st.mean([r['ell'] for r in unb20]):.2f} BWT={st.mean([r['bwt'] for r in unb20]):.2f}")

# agew M=5 T=20 精确值
ag = [r for r in cifar if r["ntask"] == 20 and r["policy"] == "agew" and r["M"] == 5]
m, s = mean_sd([r["final_avg"] for r in ag])
print(f"  T=20 agew M=5 精确均值: {m:.3f}±{s:.3f} (论文 31.6±3.7)")

# ---------- 3. Fashion ----------
print("\n== 3. Fashion（论文: unbounded?, fifo M=2 48.7±2.5, +13.6, t(9)=18.3, 摘要 +13.6 t(9)=18.3）==")
fash = load("r7_fashion*.json*")
cf = Counter((r["ntask"], r["policy"], r["M"]) for r in fash)
for k, v in sorted(cf.items()):
    print(f"  ntask={k[0]} policy={k[1]:10} M={k[2]}: {v} 条")
fu = {r["seed"]: r for r in fash if r["policy"] == "unbounded"}
ff = {r["seed"]: r for r in fash if r["policy"] == "fifo" and r["M"] == 2}
if fu and ff:
    m, s = mean_sd([r["final_avg"] for r in ff.values()])
    mu, su = mean_sd([r["final_avg"] for r in fu.values()])
    seeds = sorted(set(fu) & set(ff))
    deltas = [ff[sd]["final_avg"] - fu[sd]["final_avg"] for sd in seeds]
    print(f"  fashion unbounded: {mu:.2f}±{su:.2f}  fifo M=2: {m:.2f}±{s:.2f}")
    print(f"  paired Δ={st.mean(deltas):+.2f} t({len(deltas)-1})={paired_t(deltas):.1f}  (论文 +13.6, t=18.3)")

# ---------- 4. EWC 臂 ----------
print("\n== 4. EWC（论文: plain EWC ℓ 3–5×遗忘; online EWC γ sweep 覆盖 budget）==")
ewc = load("r7_ewc*.json*")
ce = Counter((r.get("kind"), r.get("ntask"), r.get("policy"), r.get("M"), r.get("lam"), r.get("dataset", "mnist")) for r in ewc)
for k, v in sorted(ce.items(), key=lambda x: str(x[0])):
    print(f"  {k}: {v} 条")
# 按 kind 打 ell/bwt
for kind in sorted(set(r.get("kind") for r in ewc)):
    rs = [r for r in ewc if r.get("kind") == kind]
    print(f"  {kind}: n={len(rs)} final={st.mean([r['final_avg'] for r in rs]):.2f} ℓ={st.mean([r['ell'] for r in rs]):.2f} BWT={st.mean([r['bwt'] for r in rs]):.2f}")

# ---------- 5. MNIST 主实验 ----------
print("\n== 5. MNIST 主实验（anchor_budget_all.json）==")
mn = load("anchor_budget_all.json")
cm = Counter((r["ntask"], r["policy"], r["M"]) for r in mn)
print(f"  总记录 {len(mn)}")
# 论文头排数字: M=2 T=10 68.2 (+13.0 over unbounded 55.2/55.3); BWT collapse -34.5 (mass-first at T=10)
for pol in ("unbounded", "fifo"):
    for M in (None, 2):
        if pol == "unbounded" and M: continue
        rs = [r for r in mn if r["ntask"] == 10 and r["policy"] == pol and (M is None or r["M"] == M)]
        if rs:
            m, s = mean_sd([r["final_avg"] for r in rs])
            print(f"  MNIST T=10 {pol:10} M={M}: {m:.2f}±{s:.2f} (n={len(rs)})")
# redun BWT collapse -34.5?
for pol in ("redun", "agew"):
    rs = [r for r in mn if r["ntask"] == 10 and r["policy"] == pol]
    if rs:
        for M in sorted(set(r["M"] for r in rs)):
            rM = [r for r in rs if r["M"] == M]
            print(f"  MNIST T=10 {pol:10} M={M}: final={st.mean([r['final_avg'] for r in rM]):.2f} BWT={st.mean([r['bwt'] for r in rM]):.2f} (n={len(rM)})")
