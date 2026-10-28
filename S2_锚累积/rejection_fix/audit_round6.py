# -*- coding: utf-8 -*-
"""S2 审计第6轮：γ=0.5 同λ配对、ell γ链、log-linear拟合、drop-age相关、8.6/27.7、λ=1000对比、T=5数据、Table2 T=20行归属."""
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

def paired_t(d):
    m = st.mean(d); s = st.stdev(d)
    return m / (s / math.sqrt(len(d)))

mn = load_dir(R2, "anchor_budget_all.json")
ewc_all = load_dir(R2, "r7_ewc*.json*")
g5 = load_dir(RB, "r8_gamma05.json")
g9 = load_dir(RB, "r8_gamma09.json")

def fifo_m5(T, lam):
    return {r["seed"]: r for r in mn if r["ntask"]==T and r["policy"]=="fifo" and r.get("lam")==lam and r["M"]==5}

def online(recs, T, lam):
    return {r["seed"]: r for r in recs if r["ntask"]==T and r.get("lam")==lam
            and r.get("mode") == "online"}

# ---------- 1. γ=0.5 vs fifo M=5 同λ 配对（论文 L420-422: +4.7..+7.9, |t(9)|≥10.4, BWT -19.5 vs -19.7）----------
print("== 1. γ=0.5 同λ配对 vs fifo M=5 ==")
for T in (10, 20):
    for lam in (1000.0, 5000.0, 20000.0):
        f = fifo_m5(T, lam); o = online(g5, T, lam)
        common = sorted(set(f) & set(o))
        d = [o[s]["final_avg"] - f[s]["final_avg"] for s in common]
        print(f"  T={T} λ={lam:.0f}: paired Δ={st.mean(d):+.2f} t({len(d)-1})={paired_t(d):.1f} n={len(d)}")

# λ=1000 T=20 BWT: online γ0.5 -19.5 vs fifo -19.7?
o = online(g5, 20, 1000.0); f = fifo_m5(20, 1000.0)
print(f"  BWT: online γ.5={st.mean([r['bwt'] for r in o.values()]):.1f} fifoM5={st.mean([r['bwt'] for r in f.values()]):.1f}")

# ---------- 2. ell γ链（论文: 51.1 → 56.6 → 76.1, T=20 λ=1000）----------
print("\n== 2. ell γ链 (T=20, λ=1000) ==")
plain = [r for r in ewc_all if r.get("mode")=="plain" and r["ntask"]==20 and r.get("lam")==1000.0]
o1 = online(ewc_all, 20, 1000.0)
o9 = online(g9, 20, 1000.0)
o5 = online(g5, 20, 1000.0)
for tag, rs in (("plain(γ=1?)", plain), ("online γ=1", list(o1.values())), ("online γ=0.9", list(o9.values())), ("online γ=0.5", list(o5.values()))):
    if rs:
        print(f"  {tag}: n={len(rs)} ℓ={st.mean([r['ell'] for r in rs]):.2f} BWT={st.mean([r['bwt'] for r in rs]):+.2f} acc={st.mean([r['final_avg'] for r in rs]):.2f}")

# λ=1000 两侧: plain 67.5/46.0? fifo λ=1000 67.4?
print("\n== 3. λ=1000 对比（论文: online 67.5/46.0; fifo λ1000 67.4; T=20 fifo +3.5 t=7.0）==")
for T in (10, 20):
    f = fifo_m5(T, 1000.0); o = online(ewc_all, T, 1000.0)
    mf = st.mean([r["final_avg"] for r in f.values()]); mo = st.mean([r["final_avg"] for r in o.values()])
    common = sorted(set(f) & set(o))
    d = [f[s]["final_avg"] - o[s]["final_avg"] for s in common]
    print(f"  T={T}: fifo λ1000={mf:.2f} online λ1000={mo:.2f} paired fifo−online Δ={st.mean(d):+.2f} t({len(d)-1})={paired_t(d):.1f}")
pl = [r for r in ewc_all if r.get("mode")=="plain"]
for T in (10, 20):
    sub = [r for r in pl if r["ntask"]==T and r.get("lam")==1000.0]
    if sub:
        print(f"  plain λ1000 T={T}: acc={st.mean([r['final_avg'] for r in sub]):.2f}")

# ---------- 4. log-linear 拟合 ℓ ≈ 84 − 11 ln h (λ=1000, T=20) ----------
print("\n== 4. log-linear 拟合（h=1 plain, 2 γ.5, 5 window, 10 γ.9, 20 unbounded）==")
unb1 = {r["seed"]: r for r in mn if r["ntask"]==20 and r["policy"]=="unbounded" and r.get("lam")==1000.0}
fifo1 = fifo_m5(20, 1000.0)
ells = {}
ells["plain h=1"] = st.mean([r["ell"] for r in plain])
ells["γ.5 h=2"] = st.mean([r["ell"] for r in o5.values()])
ells["window h=5"] = st.mean([r["ell"] for r in fifo1.values()])
ells["γ.9 h=10"] = st.mean([r["ell"] for r in o9.values()])
ells["unbounded h=20"] = st.mean([r["ell"] for r in unb1.values()])
for k, v in ells.items(): print(f"  {k}: ℓ={v:.2f}")
import math as m
xs = [m.log(h) for h in (1,2,5,10,20)]
ys = [ells[k] for k in ("plain h=1","γ.5 h=2","window h=5","γ.9 h=10","unbounded h=20")]
n = len(xs)
mx, my = st.mean(xs), st.mean(ys)
slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
inter = my - slope*mx
ss_res = sum((y-(slope*x+inter))**2 for x,y in zip(xs,ys))
ss_tot = sum((y-my)**2 for y in ys)
r2 = 1 - ss_res/ss_tot
print(f"  拟合: ℓ ≈ {inter:.1f} − {abs(slope):.1f} ln h, R²={r2:.3f}   论文: 84 − 11, R²=0.99")

# per-seed R²
print("  per-seed:")
pr2 = []
for seed in range(10):
    try:
        e5 = st.mean([r["ell"] for r in o5.values() if r["seed"]==seed])
        e9 = st.mean([r["ell"] for r in o9.values() if r["seed"]==seed])
        ep = st.mean([r["ell"] for r in plain if r["seed"]==seed])
        ef = st.mean([r["ell"] for r in fifo1.values() if r["seed"]==seed])
        eu = st.mean([r["ell"] for r in unb1.values() if r["seed"]==seed])
        ys_ = [ep, e5, ef, e9, eu]
        my_ = st.mean(ys_)
        sl = sum((x-mx)*(y-my_) for x,y in zip(xs,ys_)) / sum((x-mx)**2 for x in xs)
        ic = my_ - sl*mx
        r_ = 1 - sum((y-(sl*x+ic))**2 for x,y in zip(xs,ys_))/sum((y-my_)**2 for y in ys_)
        pr2.append((seed, sl, r_))
    except Exception as ex:
        print(f"    seed {seed}: 缺数据 {ex}")
for seed, sl, r_ in pr2:
    print(f"    seed {seed}: slope={sl:+.1f} R²={r_:.3f}")
if pr2:
    sls = [s for _,s,_ in pr2]
    print(f"    slope mean={st.mean(sls):.1f}±{st.stdev(sls):.1f}  论文: −10.9±0.6, per-seed R²≥0.98")

# ---------- 5. drop-age 相关（T=20 λ=1000: −0.94 plain, −0.79 fifo, −0.70 online, −0.43 unbounded）----------
print("\n== 5. drop-age 相关 ==")
def drop_age_corr(rs):
    # drop = peak acc per task − final acc; age = T − task index
    corrs = []
    for r in rs:
        T = r["ntask"]; A = r["acc_matrix"]
        drops = [max(row[j] for row in A[j:]) - A[-1][j] for j in range(T-1)]
        ages = list(range(T-1, 0, -1))  # task0 oldest
        # pearson
        n_ = len(drops)
        mx_, my_ = st.mean(ages), st.mean(drops)
        num = sum((a-mx_)*(d-my_) for a,d in zip(ages,drops))
        den = m.sqrt(sum((a-mx_)**2 for a in ages) * sum((d-my_)**2 for d in drops))
        corrs.append(num/den)
    return st.mean(corrs)
for tag, rs in (("plain", plain), ("fifo", list(fifo1.values())), ("online γ=1", list(o1.values())), ("unbounded", list(unb1.values()))):
    if rs:
        print(f"  {tag}: mean drop-age ρ={drop_age_corr(rs):+.2f}")

# ---------- 6. 8.6 recent vs 27.7 evicted oldest（fifo T=20?）----------
print("\n== 6. 8.6/27.7（论文: trace 预测 8.6 recent vs 27.7 evicted oldest; fifo M=5 T=20 λ=1000?）==")
# fifo M=5 T=20: 栈=最近5任务{15..19}; recent=在栈任务, evicted=任务0-14
per_seed = []
for r in fifo1.values():
    T = r["ntask"]; A = r["acc_matrix"]
    drops = [max(row[j] for row in A[j:]) - A[-1][j] for j in range(T)]
    in_stack = drops[T-5:]; evicted = drops[:T-5]
    per_seed.append((st.mean(in_stack), st.mean(evicted)))
if per_seed:
    a = st.mean([x for x,_ in per_seed]); b = st.mean([y for _,y in per_seed])
    print(f"  fifo λ=1000 T=20 M=5: recent(栈内5) drop={a:.1f}, evicted(前15) drop={b:.1f}")

# ---------- 7. Table 2 T=20 行: n=3 vs n=10 ----------
print("\n== 7. Table 2 T=20 行归属 ==")
for pol in ("unbounded", "fifo"):
    rs = [r for r in mn if r["ntask"]==20 and r["policy"]==pol and r.get("lam")==5000 and r["M"]==5]
    r3 = [r for r in rs if r["seed"] in (0,1,2)]
    for tag, use in ((f"n=3 seeds={sorted(r['seed'] for r in r3)}", r3), (f"n=10", rs)):
        print(f"  {pol:10} {tag}: BWT={st.mean([r['bwt'] for r in use]):+.2f} acc={st.mean([r['final_avg'] for r in use]):.2f} ℓ={st.mean([r['ell'] for r in use]):.2f}")
    print(f"    论文: unbounded −4.5/33.6/37.9, fifo −11.7/42.4/53.5")

# ---------- 8. T=5 数据存在吗（"ℓ sits 5.4 below its T=5 value"）----------
print("\n== 8. T=5 数据 ==")
t5 = [r for r in mn if r.get("ntask") == 5]
print(f"  anchor_budget_all 里 ntask=5: {len(t5)} 条")
odd = [r for r in mn if r.get("policy") not in ("unbounded","fifo","random","redun","agew","agew2","merge")]
print(f"  policy 异常记录: {len(odd)} 条; ntask 分布: {Counter(r.get('ntask') for r in odd)}")
for r in odd[:3]: print("   ", {k: r.get(k) for k in ('kind','ntask','policy','lam','final_avg','ell')})

# ---------- 9. ℓ(T=5)? 查全库 ntask=5 ----------
allf = list(R2.glob("*.json*")) + list(RB.glob("*.json*"))
found5 = 0
for p in allf:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    recs = d["results"] if isinstance(d, dict) and "results" in d else (d if isinstance(d, list) else [])
    for r in recs:
        if isinstance(r, dict) and r.get("ntask") == 5:
            found5 += 1
            if found5 <= 5: print(f"   {p.name}: {r.get('kind')}/{r.get('policy')}/lam={r.get('lam')} ℓ={r.get('ell')}")
print(f"  全库 ntask=5 记录: {found5} 条")
