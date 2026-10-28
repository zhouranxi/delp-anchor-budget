# -*- coding: utf-8 -*-
"""S2 审计第7轮：λ0重复检查、27.7 evicted、drop-age约定、r8_mgrid剂量、T10拟合、3-5x、Table2 T20来源、年龄剖面数字、SI."""
import json, math, statistics as st, itertools
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

def paired_t(d):
    m = st.mean(d); s = st.stdev(d)
    return m / (s / math.sqrt(len(d)))

# ---------- 1. λ=0 重复是否 bit-identical ----------
print("== 1. λ=0 CIFAR 同 seed 两条记录是否一致 ==")
lam0 = load_dir(R2, "r7_cifar_lam0_t10*.json*") + load_dir(R2, "r7_cifar_lam0_t20*.json*")
for T in (10, 20):
    rs = [r for r in lam0 if r["ntask"] == T]
    by_seed = {}
    for r in rs: by_seed.setdefault(r["seed"], []).append(r)
    ident = sum(1 for s, v in by_seed.items() if len(v) == 2 and v[0]["final_avg"] == v[1]["final_avg"] and v[0]["acc_matrix"] == v[1]["acc_matrix"])
    print(f"  T={T}: {len(by_seed)} seeds, bit-identical 对数 {ident}/{len(by_seed)}")
    # 去重后 n=10 的 mean/sd
    uniq = [v[0] for v in by_seed.values()]
    m, s = st.mean([r["final_avg"] for r in uniq]), st.stdev([r["final_avg"] for r in uniq])
    print(f"  T={T} 去重 n={len(uniq)}: {m:.2f}±{s:.2f}   论文: 27.0±1.7 / 36.9±3.4")
    # 若两条是独立 run，sd over 20：
    m20, s20 = st.mean([r["final_avg"] for r in rs]), st.stdev([r["final_avg"] for r in rs])
    print(f"  T={T} 全 20 条: {m20:.2f}±{s20:.2f}")

# ---------- 2. 27.7 evicted oldest ----------
print("\n== 2. fifo λ=1000 T=20 M=5: evicted drop 的各种口径 ==")
mn = load_dir(R2, "anchor_budget_all.json")
fifo1 = [r for r in mn if r["ntask"]==20 and r["policy"]=="fifo" and r.get("lam")==1000.0 and r["M"]==5]
per = []
for r in fifo1:
    T = r["ntask"]; A = r["acc_matrix"]
    drops = [max(row[j] for row in A[j:]) - A[-1][j] for j in range(T)]
    per.append(drops)
mean_drops = [st.mean([d[j] for d in per]) for j in range(20)]
print(f"  逐任务平均 drop: {[round(x,1) for x in mean_drops]}")
for k in (3, 5, 10, 15):
    print(f"  最老 {k} 任务平均 drop: {st.mean(mean_drops[:k]):.1f}")
print(f"  全部 evicted(前15): {st.mean(mean_drops[:15]):.1f}  栈内(后5): {st.mean(mean_drops[15:]):.1f}")
# λ=5000 也试
fifo5 = [r for r in mn if r["ntask"]==20 and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==5]
per5 = []
for r in fifo5:
    T = r["ntask"]; A = r["acc_matrix"]
    per5.append([max(row[j] for row in A[j:]) - A[-1][j] for j in range(T)])
md5 = [st.mean([d[j] for d in per5]) for j in range(20)]
print(f"  λ=5000: evicted前15={st.mean(md5[:15]):.1f} 栈内={st.mean(md5[15:]):.1f} 最老5={st.mean(md5[:5]):.1f}")

# ---------- 3. drop-age 相关（task-index 约定 + 含最后任务）----------
print("\n== 3. drop-age 相关：论文 −0.94(plain) −0.79(fifo) −0.70(online) −0.43(unbounded) ==")
ewc_all = load_dir(R2, "r7_ewc*.json*")
unb1 = [r for r in mn if r["ntask"]==20 and r["policy"]=="unbounded" and r.get("lam")==1000.0]
plain1 = [r for r in ewc_all if r.get("mode")=="plain" and r["ntask"]==20 and r.get("lam")==1000.0]
onl1 = [r for r in ewc_all if r.get("mode")=="online" and r["ntask"]==20 and r.get("lam")==1000.0]

def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den

def drop_age(rs, include_last, index_convention):
    out = []
    for r in rs:
        T = r["ntask"]; A = r["acc_matrix"]
        rng = range(T) if include_last else range(T-1)
        drops = [max(row[j] for row in A[j:]) - A[-1][j] for j in rng]
        # index_convention: 'taskidx' = 任务下标(0=最老), 'age' = T-1-j
        ages = [j if index_convention=='taskidx' else T-1-j for j in rng]
        out.append(corr(ages, drops))
    return st.mean(out)

for inc in (False, True):
    for conv in ('taskidx', 'age'):
        row = []
        for tag, rs in (("plain",plain1),("fifo",fifo1),("online",onl1),("unb",unb1)):
            row.append(f"{tag}={drop_age(rs, inc, conv):+.2f}")
        print(f"  include_last={inc} conv={conv}: {' '.join(row)}")

# ---------- 4. r8_mgrid 剂量响应 ----------
print("\n== 4. r8_mgrid 剂量响应（论文: M∈{2,3,4,5,6,8,10} T=20 λ=5000）==")
mg = load_dir(RB, "r8_mgrid.json")
print(f"  r8_mgrid: {len(mg)} 条")
cm = Counter((r["ntask"], r["policy"], r["M"], r.get("lam")) for r in mg)
for k, v in sorted(cm.items(), key=lambda x: str(x)):
    sub = [r for r in mg if (r["ntask"], r["policy"], r["M"], r.get("lam"))==k]
    print(f"  T={k[0]} {k[1]} M={k[2]} λ={k[3]}: n={v} acc={st.mean([r['final_avg'] for r in sub]):.2f} ℓ={st.mean([r['ell'] for r in sub]):.2f}")

# ---------- 5. T=10 log-linear 拟合（论文: R²=1.00）----------
print("\n== 5. T=10 拟合 ==")
g5 = load_dir(RB, "r8_gamma05.json"); g9 = load_dir(RB, "r8_gamma09.json")
pl10 = [r for r in ewc_all if r.get("mode")=="plain" and r["ntask"]==10 and r.get("lam")==1000.0]
o5_10 = [r for r in g5 if r["ntask"]==10 and r.get("lam")==1000.0]
o9_10 = [r for r in g9 if r["ntask"]==10 and r.get("lam")==1000.0]
fi10 = [r for r in mn if r["ntask"]==10 and r["policy"]=="fifo" and r.get("lam")==1000.0 and r["M"]==5]
un10 = [r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==1000.0]
ells = [st.mean([r["ell"] for r in rs]) for rs in (pl10, o5_10, fi10, o9_10, un10)]
xs = [math.log(h) for h in (1,2,5,10,20)]
mx, my = st.mean(xs), st.mean(ells)
slope = sum((x-mx)*(y-my) for x,y in zip(xs,ells))/sum((x-mx)**2 for x in xs)
inter = my - slope*mx
r2 = 1 - sum((y-(slope*x+inter))**2 for x,y in zip(xs,ells))/sum((y-my)**2 for y in ells)
print(f"  T=10 λ=1000: ℓ={[round(e,1) for e in ells]}")
print(f"  拟合 slope={slope:.1f} intercept={inter:.1f} R²={r2:.3f}  论文: R²=1.00")

# ---------- 6. plain EWC 3-5× ----------
print("\n== 6. plain EWC ℓ vs BWT 倍率 ==")
for T in (10, 20):
    for lam in (1000.0, 5000.0, 20000.0):
        pl = [r for r in ewc_all if r.get("mode")=="plain" and r["ntask"]==T and r.get("lam")==lam]
        if pl:
            ell = st.mean([r["ell"] for r in pl]); bwt = st.mean([r["bwt"] for r in pl])
            print(f"  plain T={T} λ={lam:.0f}: ℓ={ell:.1f} BWT={bwt:.1f} |ℓ/BWT|={abs(ell/bwt):.1f}x")

# ---------- 7. Table 2 T=20 unbounded 33.6 猎取 ----------
print("\n== 7. Table 2 T=20 unbounded (−4.5/33.6/37.9) 种子子集搜索 ==")
u20 = [r for r in mn if r["ntask"]==20 and r["policy"]=="unbounded" and r.get("lam")==5000.0]
by_seed = {r["seed"]: r for r in u20}
found = []
for combo in itertools.combinations(sorted(by_seed), 3):
    sub = [by_seed[s] for s in combo]
    b, a, e = st.mean([r["bwt"] for r in sub]), st.mean([r["final_avg"] for r in sub]), st.mean([r["ell"] for r in sub])
    if abs(a - 33.6) < 0.05 and abs(b + 4.5) < 0.05:
        found.append((combo, round(b,2), round(a,2), round(e,2)))
print(f"  匹配 (−4.5, 33.6) 的 3 种子组合: {found[:8]} (共 {len(found)})")
print(f"  全部 10 种子单察: ", {s: (round(by_seed[s]['final_avg'],1), round(by_seed[s]['bwt'],1)) for s in sorted(by_seed)})
f20 = [r for r in mn if r["ntask"]==20 and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==5]
byf = {r["seed"]: r for r in f20}
found2 = []
for combo in itertools.combinations(sorted(byf), 3):
    sub = [byf[s] for s in combo]
    b, a = st.mean([r["bwt"] for r in sub]), st.mean([r["final_avg"] for r in sub])
    if abs(a - 42.4) < 0.05 and abs(b + 11.7) < 0.05:
        found2.append(combo)
print(f"  fifo T=20 匹配 (−11.7, 42.4) 的 3 种子组合数: {len(found2)}; n=10 实测: BWT={st.mean([r['bwt'] for r in f20]):.2f} acc={st.mean([r['final_avg'] for r in f20]):.2f}")

# ---------- 8. 年龄剖面数字（38.5 spread, 87/85, gradient）----------
print("\n== 8. 论文年龄剖面数字 ==")
# fifo M=2 T=10 λ=5000 栈内任务 8,9 final acc
fm2 = [r for r in mn if r["ntask"]==10 and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==2]
acc89 = st.mean([r["acc_matrix"][-1][8] for r in fm2]); acc9 = st.mean([r["acc_matrix"][-1][9] for r in fm2])
print(f"  fifo M=2 T=10: 任务8 final={acc89:.0f}% 任务9 final={acc9:.0f}%  (论文: 87/85%)")
# unbounded λ=1000 T=10 剖面 spread
un10l = [r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==1000.0]
prof = [st.mean([r["acc_matrix"][-1][j] for r in un10l]) for j in range(10)]
print(f"  unbounded λ=1000 T=10 终相剖面: {[round(x,1) for x in prof]}  spread={max(prof)-min(prof):.1f}")
# fifo M=2 T=10 λ=5000 剖面
prof2 = [st.mean([r["acc_matrix"][-1][j] for r in fm2]) for j in range(10)]
print(f"  fifo M=2 λ=5000 T=10 剖面: {[round(x,1) for x in prof2]}  spread={max(prof2)-min(prof2):.1f}")
# 38.5: 哪个臂? 试 unbounded λ=5000 T=10
un10l5 = [r for r in mn if r["ntask"]==10 and r["policy"]=="unbounded" and r.get("lam")==5000.0]
prof3 = [st.mean([r["acc_matrix"][-1][j] for r in un10l5]) for j in range(10)]
print(f"  unbounded λ=5000 T=10 剖面: {[round(x,1) for x in prof3]}  spread={max(prof3)-min(prof3):.1f}")

# ---------- 9. SI 记录在哪引用 ----------
print("\n== 9. SI 基线 (40.1/39.0/42.0) ==")
print("  需 grep 论文 'SI\\|synaptic'")

# ---------- 10. T=5 引用来源 ----------
print("\n== 10. T=5 claim（ℓ 比 T=5 低 5.4）==")
print("  本仓库无 ntask=5 记录; S1 论文可能为源 → 需查 S1 tex")
