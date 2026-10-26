# -*- coding: utf-8 -*-
"""S2 审计第5轮：摘要68.2归属、Table2 n=3全表、tab:paired t值、γ每λ对比、merge penalty、CIFAR栈轨迹."""
import json, math, statistics as st
from pathlib import Path
from collections import Counter, defaultdict

R2 = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")
RB = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results")

def load_dir(base, parts):
    recs = []
    for p in sorted(Path(base).glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

def mean_sd(xs): return st.mean(xs), st.stdev(xs)
def paired_t(deltas):
    m = st.mean(deltas); s = st.stdev(deltas); n = len(deltas)
    return m / (s / math.sqrt(n))

mn = load_dir(R2, "anchor_budget_all.json")

# ---------- 1. 摘要 68.2 / +13.0 归属 ----------
print("== 1. 摘要 68.2/+13.0 归属 ==")
for pol in ("fifo", "agew2", "redun", "agew", "merge", "random"):
    rs = [r for r in mn if r["ntask"]==10 and r["policy"]==pol and r.get("lam")==5000 and r["M"]==2]
    if not rs: continue
    by3 = [r for r in rs if r["seed"] in (0,1,2)]
    m3, s3 = mean_sd([r["final_avg"] for r in by3]) if len(by3)>1 else (None,None)
    mall, sall = mean_sd([r["final_avg"] for r in rs])
    print(f"  T=10 {pol:7} M=2: n=3 {m3:.2f} | n={len(rs)} {mall:.2f}±{sall:.2f}")
u_all = [r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==5000]
u3 = [r for r in u_all if r["seed"] in (0,1,2)]
mu3 = st.mean([r["final_avg"] for r in u3]); mu, su = mean_sd([r["final_avg"] for r in u_all])
print(f"  unbounded: n=3 {mu3:.2f} | n=10 {mu:.2f}±{su:.2f}")
for pol in ("agew2", "fifo"):
    rs = {r["seed"]: r for r in mn if r["ntask"]==10 and r["policy"]==pol and r.get("lam")==5000 and r["M"]==2}
    u = {r["seed"]: r for r in u_all}
    common = sorted(set(rs) & set(u))
    d = [rs[s]["final_avg"] - u[s]["final_avg"] for s in common]
    d3 = [rs[s]["final_avg"] - u[s]["final_avg"] for s in common if s in (0,1,2)]
    print(f"  {pol} M=2 − unbounded: n={len(common)} Δ={st.mean(d):+.2f} | n=3 Δ={st.mean(d3):+.2f}")

# ---------- 2. Table 2 (tab:grid, M=5, n=3) 全对账 ----------
print("\n== 2. Table 2 (M=5, n=3) 对账: 论文 unbounded -5.5/55.3/60.2, fifo -6.6/59.7/65.6, random -9.1/60.2/68.4, redun -18.3/53.6/70.1, agew2 -8.4/60.8/68.3, merge -5.5/55.3/60.3 ==")
paper_grid = {  # (T, arm): (BWT, acc, ell)
    (10,"unbounded"): (-5.5,55.3,60.2), (10,"fifo"): (-6.6,59.7,65.6),
    (10,"random"): (-9.1,60.2,68.4), (10,"redun"): (-18.3,53.6,70.1),
    (10,"agew"): (-18.3,53.6,70.1), (10,"agew2"): (-8.4,60.8,68.3),
    (10,"merge"): (-5.5,55.3,60.3),
    (20,"unbounded"): (-4.5,33.6,37.9), (20,"fifo"): (-11.7,42.4,53.5),
    (20,"random"): (-13.5,42.0,None), (20,"redun"): (-30.2,36.3,None),
    (20,"agew"): (-30.2,36.3,None), (20,"agew2"): (-16.5,42.3,None),
    (20,"merge"): (-4.5,33.6,None),
}
for (T, arm), (pb, pa, pe) in paper_grid.items():
    rs = [r for r in mn if r["ntask"]==T and r["policy"]==arm and r.get("lam")==5000 and r["M"]==5]
    rs3 = [r for r in rs if r["seed"] in (0,1,2)]
    use = rs3 if len(rs3) >= 3 else rs
    b = st.mean([r["bwt"] for r in use]); a = st.mean([r["final_avg"] for r in use])
    e = st.mean([r["ell"] for r in use])
    pe_s = f"/{pe}" if pe else ""
    ok = "✓" if (abs(b-pb)<0.06 and abs(a-pa)<0.06 and (pe is None or abs(e-pe)<0.06)) else "<<< CHECK"
    print(f"  T={T} {arm:10}: got BWT={b:+6.2f} acc={a:5.2f} ℓ={e:5.2f} (n={len(use)}) | paper {pb}/{pa}{pe_s} {ok}")

# ---------- 3. tab:paired t 值 ----------
print("\n== 3. tab:paired (fifo M=5 vs unbounded, n=10) ==")
for T, pd_, pt_ in ((10, 4.51, 21.4), (20, 8.98, 19.1)):
    u = {r["seed"]: r for r in mn if r["ntask"]==T and r["policy"]=="unbounded" and r.get("lam")==5000}
    f = {r["seed"]: r for r in mn if r["ntask"]==T and r["policy"]=="fifo" and r.get("lam")==5000 and r["M"]==5}
    common = sorted(set(u) & set(f))
    d = [f[s]["final_avg"] - u[s]["final_avg"] for s in common]
    db = [f[s]["bwt"] - u[s]["bwt"] for s in common]
    m, s_ = mean_sd(d)
    print(f"  fifo T={T}: Δ={m:+.2f}±{s_:.2f} t={paired_t(d):.1f} ΔBWT={st.mean(db):+.2f}±{st.stdev(db):.2f} | paper {pd_}±?, t={pt_}")

# ---------- 4. γ 每格对比（"overtakes at every grid cell"）----------
print("\n== 4. online EWC γ 每 λ 格 vs fifo M=2/M=5 ==")
fifo = {}
for T in (10, 20):
    for M in (2, 5):
        rs = [r for r in mn if r["ntask"]==T and r["policy"]=="fifo" and r.get("lam")==5000 and r["M"]==M]
        fifo[(T,M)] = st.mean([r["final_avg"] for r in rs])
for gam, fn in (("0.5","r8_gamma05.json"), ("0.9","r8_gamma09.json")):
    rs = load_dir(RB, fn)
    for T in (10, 20):
        for lam in (1000.0, 5000.0, 20000.0):
            sub = [r for r in rs if r["ntask"]==T and r.get("lam")==lam]
            if sub:
                m = st.mean([r["final_avg"] for r in sub])
                f2, f5 = fifo[(T,2)], fifo[(T,5)]
                print(f"  γ={gam} T={T} λ={lam:.0f}: online={m:5.2f} vs fifoM2={f2:5.2f}({'WIN' if m>f2 else 'lose'}) fifoM5={f5:5.2f}({'WIN' if m>f5 else 'lose'})")
# γ=1 (r7_ewc)
ewc = [r for r in load_dir(R2, "r7_ewc*.json*") if r.get("kind")=="ewc"]
on1 = [r for r in ewc if r.get("mode")=="online" or r.get("policy")=="online"]
print(f"  r7_ewc online(γ=1) records: {len(on1)}")
for T in (10, 20):
    for lam in (1000.0, 5000.0, 20000.0):
        sub = [r for r in on1 if r["ntask"]==T and r.get("lam")==lam]
        if sub:
            m = st.mean([r["final_avg"] for r in sub])
            f2, f5 = fifo[(T,2)], fifo[(T,5)]
            print(f"  γ=1   T={T} λ={lam:.0f}: online={m:5.2f} vs fifoM2={f2:5.2f}({'WIN' if m>f2 else 'lose'}) fifoM5={f5:5.2f}({'WIN' if m>f5 else 'lose'})")

# ---------- 5. merge penalty 统计 ----------
print("\n== 5. merge penalty（论文: 43-45%@M=2 worst 64%, 18-34%@M=5, 77% move; T-M merges/run）==")
mg = load_dir(RB, "r8_merge.json")
for M in (2, 5):
    rs = [r for r in mg if r["M"]==M]
    maxes = [r["max_rel_drift"] for r in rs if "max_rel_drift" in r]
    all_drifts = []
    for r in rs:
        all_drifts.extend(r.get("merge_drifts", []))
    nm = [r.get("n_merges") for r in rs]
    if maxes:
        print(f"  M={M}: runs={len(rs)} max_rel_drift 范围 [{min(maxes):.3f}, {max(maxes):.3f}] | merge数与T-M一致: {all(abs(n-(r['ntask']-M))==0 for n,r in zip(nm,rs))}")
        moved = sum(1 for d in all_drifts if abs(d) > 1e-12)
        print(f"    总 merge 事件 {len(all_drifts)}, 非零 {moved} ({100*moved/len(all_drifts):.0f}%)")

# ---------- 6. CIFAR 栈轨迹（"all ten seeds agree"）----------
print("\n== 6. CIFAR 栈轨迹 ==")
cifar = load_dir(R2, "s2r6_b_t10*.json*") + load_dir(R2, "s2r6_b_t20*.json*")
cifar = [r for r in cifar if r.get("dataset")=="cifar100"]
for pol in ("fifo","redun","agew","agew2"):
    rs = [r for r in cifar if r["ntask"]==20 and r["policy"]==pol and r["M"]==2]
    finals = Counter(tuple(r["survivor_history"][-1]) for r in rs)
    print(f"  T=20 M=2 {pol:6}: 最终栈分布 {dict(finals)} (n={len(rs)})")
