 # -*- coding: utf-8 -*-
"""S2 审计第4轮：MNIST 主表全数字对账 + γ sweep + merge penalty + 附录矩阵."""
import json, math, statistics as st
from pathlib import Path
from collections import Counter, defaultdict

R2 = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")
RB = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results")

def load_dir(base, parts):
    recs = []
    for p in sorted(base.glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

def mean_sd(xs): return st.mean(xs), st.stdev(xs)
def paired_t(deltas):
    m = st.mean(deltas); s = st.stdev(deltas); n = len(deltas)
    return m / (s / math.sqrt(n))

# ---------- 1. MNIST 主表 ----------
print("== 1. MNIST 主表 (anchor_budget_all.json) 全构成 ==")
mn = load_dir(R2, "anchor_budget_all.json")
for k, v in sorted(Counter((r["ntask"], r["policy"], r["M"], r.get("lam")) for r in mn).items()):
    print(f"  T={k[0]:2} {k[1]:10} M={str(k[2]):4} λ={str(k[3]):7}: {v} 条")

print("\n== 论文头排数字对账（λ=5000, T=10）==")
# 论文: unbounded 55.2→60.2? 68.2 @M=2 (+13.0 over unbounded 55.2/55.3)
for pol, M in (("unbounded", 2), ("unbounded", 5), ("fifo", 2), ("fifo", 5), ("fifo", None)):
    rs = [r for r in mn if r["ntask"]==10 and r["policy"]==pol and r.get("lam")==5000
          and (M is None or r["M"]==M)]
    if M is None:
        rs = [r for r in rs if r["policy"]=="fifo"]
    by_seed = {}
    for r in rs: by_seed.setdefault(r["seed"], r)  # M 惰性去重
    if by_seed:
        m, s = mean_sd([r["final_avg"] for r in by_seed.values()])
        print(f"  T=10 {pol:10} M={str(M):4} λ=5000: {m:.2f}±{s:.2f} n={len(by_seed)}")
# 配对 fifo M=2 - unbounded
u = {r["seed"]: r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==5000}
f2 = {r["seed"]: r for r in mn if r["ntask"]==10 and r["policy"]=="fifo" and r.get("lam")==5000 and r["M"]==2}
common = sorted(set(u) & set(f2))
if common:
    d = [f2[s]["final_avg"] - u[s]["final_avg"] for s in common]
    print(f"  paired fifo M=2 − unbounded: Δ={st.mean(d):+.2f} t({len(d)-1})={paired_t(d):.1f}  论文: +13.0")

# BWT collapse -34.5（redun/agew, T=20?）
print("\n== BWT collapse（论文: -34.5）==")
for pol in ("redun", "agew", "agew2", "random", "merge"):
    for nt in (10, 20):
        rs = [r for r in mn if r["ntask"]==nt and r["policy"]==pol and r.get("lam")==5000]
        for M in sorted(set(r["M"] for r in rs)):
            rM = [r for r in rs if r["M"]==M]
            b = st.mean([r["bwt"] for r in rM])
            print(f"  T={nt:2} {pol:7} M={M:2}: BWT={b:+7.2f} final={st.mean([r['final_avg'] for r in rM]):6.2f} (n={len(rM)})")

# ---------- 2. γ sweep（r8_gamma05/09）----------
print("\n== 2. γ sweep（论文: γ∈{0.5,0.9,1}，tuned discount 覆盖 budget）==")
for fn in ("r8_gamma05.json", "r8_gamma09.json"):
    rs = load_dir(RB, fn)
    if rs:
        for k, v in sorted(Counter((r.get("kind"), r.get("mode", r.get("policy")), r["ntask"]) for r in rs).items()):
            sub = [r for r in rs if (r.get("kind"), r.get("mode", r.get("policy")), r["ntask"]) == k]
            m, s = mean_sd([r["final_avg"] for r in sub])
            ell = st.mean([r["ell"] for r in sub]); bwt = st.mean([r["bwt"] for r in sub])
            print(f"  {fn} {k}: n={v} final={m:.2f}±{s:.2f} ℓ={ell:.2f} BWT={bwt:+.2f}")
    else:
        print(f"  {fn}: 空")

# ---------- 3. merge penalty（r8_merge）----------
print("\n== 3. merge penalty 插桩 ==")
mg = load_dir(RB, "r8_merge.json")
print(f"r8_merge: {len(mg)} 条")
if mg:
    print("keys:", sorted(mg[0].keys()))
    print(json.dumps(mg[0])[:600])

# ---------- 4. 附录矩阵 ----------
print("\n== 4. 附录完整矩阵（appx_mat_t10/t20.tex 里的数字）==")
for fn in ("S2_锚累积/paper/appx_mat_t10.tex", "S2_锚累积/paper/appx_mat_t20.tex"):
    p = R2.parent / "paper" / Path(fn).name
    txt = p.read_text(encoding="utf-8", errors="ignore")
    print(f"--- {p.name}: {len(txt)} chars, 前200: {txt[:200]!r}")

# ---------- 5. Fashion BWT/M=5 数字 ----------
print("\n== 5. Fashion fifo M=5（论文正文引用?）==")
recs = load_dir(RB, "s2r3_c2*.json.part*") + load_dir(RB, "r7_fashion.json.part*")
fash = [r for r in recs if r.get("dataset") == "fashion"]
u5 = {r["seed"]: r for r in fash if r["policy"]=="unbounded" and r["M"]==5}
ff5 = {r["seed"]: r for r in fash if r["policy"]=="fifo" and r["M"]==5}
m, s = mean_sd([r["final_avg"] for r in ff5.values()])
print(f"  fifo M=5: {m:.2f}±{s:.2f} (n={len(ff5)})")

# ---------- 6. 椭圆: T=20 MNIST unbounded/fifo 主数字 ----------
print("\n== 6. T=20 MNIST 主数字（论文: 33.6/37.9?）==")
for pol in ("unbounded", "fifo"):
    for M in (2, 5, None):
        rs = [r for r in mn if r["ntask"]==20 and r["policy"]==pol and r.get("lam")==5000
              and (M is None or r["M"]==M)]
        by_seed = {}
        for r in rs: by_seed.setdefault(r["seed"], r)
        if by_seed:
            m, s = mean_sd([r["final_avg"] for r in by_seed.values()])
            ell = st.mean([r["ell"] for r in by_seed.values()]); bwt = st.mean([r["bwt"] for r in by_seed.values()])
            print(f"  T=20 {pol:10} M={str(M):4}: {m:.2f}±{s:.2f} ℓ={ell:.2f} BWT={bwt:+.2f} n={len(by_seed)}")
