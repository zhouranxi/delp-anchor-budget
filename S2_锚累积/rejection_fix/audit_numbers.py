# -*- coding: utf-8 -*-
"""S2 审计：把论文 Table 6 / no-protection / ell / BWT / t 值全部从原始 JSON 重算对账."""
import json, math, statistics as st
from pathlib import Path

BASE = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")

def load(parts):
    """合并 part0/part1 为记录列表."""
    recs = []
    for p in sorted(BASE.glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

def mean_sd(xs):
    return st.mean(xs), st.stdev(xs)

def paired_t(deltas):
    """deltas: list of (x_i - y_i); returns t(9)."""
    m = st.mean(deltas); s = st.stdev(deltas); n = len(deltas)
    return m / (s / math.sqrt(n))

# ---------- 1. CIFAR bounded (s2r6_b) ----------
cifar = load("s2r6_b_t10*.json*") + load("s2r6_b_t20*.json*")
cifar = [r for r in cifar if r.get("dataset") == "cifar100"]
print(f"== CIFAR bounded records: {len(cifar)}")

def sel(ntask, policy, M):
    return [r for r in cifar if r["ntask"] == ntask and r["policy"] == policy and r["M"] == M]

def by_seed(rs):
    return {r["seed"]: r for r in rs}

unb10 = by_seed([r for r in cifar if r["ntask"] == 10 and r["policy"] == "unbounded"])
unb20 = by_seed([r for r in cifar if r["ntask"] == 20 and r["policy"] == "unbounded"])
print(f"unbounded seeds T=10: {sorted(unb10)}  T=20: {sorted(unb20)}")

print("\n== Table 6 对账 (论文: unbounded 23.3±2.6 / 29.8±4.2) ==")
for tag, ub in (("T=10", unb10), ("T=20", unb20)):
    m, s = mean_sd([r["final_avg"] for r in ub.values()])
    print(f"unbounded {tag}: {m:.1f}±{s:.1f}  (n={len(ub)})")

paper = {  # (ntask, policy, M): (mean, sd, delta)
    (10,"fifo",2):(24.8,2.8,+1.5),(10,"fifo",5):(23.5,2.6,+0.2),
    (10,"random",2):(24.7,2.4,+1.5),(10,"random",5):(22.4,2.7,-0.9),
    (10,"redun",2):(24.0,2.5,+0.7),(10,"redun",5):(23.6,2.0,+0.3),
    (10,"merge",2):(22.3,1.7,-1.0),(10,"merge",5):(21.8,3.2,-1.5),
    (10,"agew",2):(24.0,3.8,+0.7),(10,"agew",5):(22.7,3.4,-0.6),
    (10,"agew2",2):(23.7,2.6,+0.4),(10,"agew2",5):(22.3,3.3,-1.0),
    (20,"fifo",2):(34.8,4.8,+5.0),(20,"fifo",5):(32.2,4.3,+2.4),
    (20,"random",2):(33.4,3.9,+3.6),(20,"random",5):(32.2,4.7,+2.4),
    (20,"redun",2):(33.2,4.7,+3.4),(20,"redun",5):(32.9,3.3,+3.1),
    (20,"merge",2):(30.8,5.2,+1.0),(20,"merge",5):(29.3,3.4,-0.5),
    (20,"agew",2):(34.0,4.3,+4.2),(20,"agew",5):(31.6,3.7,+1.8),
    (20,"agew2",2):(32.5,3.7,+2.7),(20,"agew2",5):(31.1,3.4,+1.3),
}
print("\n== 各格 mean±sd 与 Δ 对账 ==")
bad = 0
for (nt, pol, M), (pm, ps, pd) in sorted(paper.items()):
    rs = sel(nt, pol, M)
    if not rs:
        print(f"MISSING: T={nt} {pol} M={M}"); bad += 1; continue
    ub = unb10 if nt == 10 else unb20
    m, s = mean_sd([r["final_avg"] for r in rs])
    seeds = sorted(set(r["seed"] for r in rs) & set(ub))
    deltas = [by_seed(rs)[sd]["final_avg"] - ub[sd]["final_avg"] for sd in seeds]
    dm = st.mean(deltas) if deltas else float("nan")
    flag = "" if (abs(m-pm) <= 0.051 and abs(dm-pd) <= 0.051) else "  <<< MISMATCH"
    if flag: bad += 1
    print(f"T={nt:2} {pol:7} M={M}: got {m:5.1f}±{s:4.1f} Δ={dm:+5.2f} (n={len(rs)}, paired n={len(deltas)}) | paper {pm}±{ps} Δ={pd:+.1f}{flag}")

# ---------- 2. t 值 ----------
print("\n== t 值对账 (论文: fifo +5.0 t=7.1, agew +4.2 t=7.3, random +3.6 t=9.2, redun +3.4 t=4.0, agew2 +1.3) ==")
for pol, M, pt in (("fifo",2,7.1),("agew",2,7.3),("random",2,9.2),("redun",2,4.0),("agew2",2,None)):
    rs = sel(20, pol, M)
    seeds = sorted(set(r["seed"] for r in rs) & set(unb20))
    deltas = [by_seed(rs)[sd]["final_avg"] - unb20[sd]["final_avg"] for sd in seeds]
    t = paired_t(deltas)
    print(f"T=20 {pol:7} M={M}: Δ={st.mean(deltas):+.2f} t({len(deltas)-1})={t:.1f}  (paper t={pt})")

# T=10 时论文说 +1.5 t≈2
for pol in ("fifo","random"):
    rs = sel(10, pol, 2)
    seeds = sorted(set(r["seed"] for r in rs) & set(unb10))
    deltas = [by_seed(rs)[sd]["final_avg"] - unb10[sd]["final_avg"] for sd in seeds]
    print(f"T=10 {pol:7} M=2: Δ={st.mean(deltas):+.2f} t={paired_t(deltas):.1f}  (paper ≈+1.5, t≈2)")

# ---------- 3. ell / BWT ----------
print("\n== ell/BWT 对账 (论文: fifo M=2 raises ℓ 31.4→36.8, BWT -1.7→-2.1) ==")
for tag, ub in (("T=10 unbounded", unb10),):
    print(f"{tag}: ℓ={st.mean([r['ell'] for r in ub.values()]):.1f} BWT={st.mean([r['bwt'] for r in ub.values()]):.1f}")
rs = sel(20, "fifo", 2)
print(f"T=20 fifo M=2:     ℓ={st.mean([r['ell'] for r in rs]):.1f} BWT={st.mean([r['bwt'] for r in rs]):.1f}")
# 论文这句 "fifo M=2 raises ℓ from 31.4 to 36.8 while BWT moves -1.7 to -2.1" —— 核对它到底指 T=20 还是 T=10
rs10 = sel(10, "fifo", 2)
print(f"T=10 fifo M=2:     ℓ={st.mean([r['ell'] for r in rs10]):.1f} BWT={st.mean([r['bwt'] for r in rs10]):.1f}")

# ---------- 4. no-protection (λ=0) ----------
print("\n== λ=0 对账 (论文: 27.0±1.7 T=10, 36.9±3.4 T=20; fifo M=2 +2.1 t=1.5 n.s.; ℓ 46.1/64.3; BWT -21.3/-28.9) ==")
lam0 = load("r7_cifar_lam0_t10*.json*") + load("r7_cifar_lam0_t20*.json*")
print(f"λ=0 records: {len(lam0)}, kinds: {set(r.get('kind') for r in lam0)}")
for nt in (10, 20):
    rs = [r for r in lam0 if r.get("ntask") == nt]
    if rs:
        m, s = mean_sd([r["final_avg"] for r in rs])
        ell = st.mean([r["ell"] for r in rs]); bwt = st.mean([r["bwt"] for r in rs])
        print(f"λ=0 T={nt}: {m:.1f}±{s:.1f} (n={len(rs)}) ℓ={ell:.1f} BWT={bwt:.1f}")
    else:
        print(f"λ=0 T={nt}: NO RECORDS")
# fifo M=2 vs λ=0 的 paired diff
for nt, ub0 in ((10, None), (20, None)):
    rs0 = by_seed([r for r in lam0 if r.get("ntask") == nt])
    rsf = by_seed(sel(nt, "fifo", 2))
    seeds = sorted(set(rs0) & set(rsf))
    if seeds:
        deltas = [rs0[sd]["final_avg"] - rsf[sd]["final_avg"] for sd in seeds]
        print(f"λ=0 - fifo M=2 (T={nt}): Δ={st.mean(deltas):+.2f} t({len(deltas)-1})={paired_t(deltas):.1f}  (paper T=20: +2.1, t=1.5)")

# ---------- 5. 记录总数 ----------
print(f"\n== 记录总数核对（论文说 'over all 260 CIFAR records'）==")
print(f"bounded CIFAR: {len(cifar)}, λ=0: {len(lam0)}, total: {len(cifar)+len(lam0)}")

# ---------- 6. Fashion ----------
print("\n== Fashion 对账 (论文: 48.7±2.5 M=2, +13.6, t(9)=18.3) ==")
fash = load("r7_fashion*.json*")
print(f"fashion records: {len(fash)}, sample keys:", sorted(fash[0].keys()) if fash else "none")
