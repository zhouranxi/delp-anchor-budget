# -*- coding: utf-8 -*-
"""S2 审计第3轮：Fashion 合并 c2+r7、r8 gamma sweep、r8 merge penalty、MNIST 主表核对."""
import json, math, statistics as st
from pathlib import Path
from collections import Counter

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

# ---------- 1. Fashion 合并 ----------
print("== 1. Fashion 全量（c2_fashion 0-2 + r7_fashion 3-9 = 10 种子？）==")
fash_all = load_dir(RB, "c2_fashion_b*.json.part*") + load_dir(R2, "r7_fashion*.json*")
fash = [r for r in fash_all if r.get("dataset") == "fashion" or "fashion" in str(r)]
# 有些记录可能没 dataset 字段，直接按文件来源标注
f_c2 = load_dir(RB, "c2_fashion_b*.json.part*")
f_r7 = load_dir(R2, "r7_fashion*.json*")
print(f"c2_fashion records: {len(f_c2)}  r7_fashion: {len(f_r7)}")
cf = Counter((r["ntask"], r["policy"], r["M"], r.get("lam")) for r in f_c2)
print("c2 构成:", dict(cf))
cf7 = Counter((r["ntask"], r["policy"], r["M"], r.get("lam")) for r in f_r7)
print("r7 构成:", dict(cf7))
# 合并 T=10 λ=5000 fifo/unbounded M=2
for pol, M, pm, ps, pd_, pt in (("unbounded", 2, None, None, None, None),
                                 ("fifo", 2, 48.7, 2.5, 13.6, 18.3),
                                 ("unbounded", 5, None, None, None, None),
                                 ("fifo", 5, None, None, None, None)):
    rs = [r for r in f_c2 + f_r7 if r["ntask"] == 10 and r["policy"] == pol
          and r["M"] == M and r.get("lam") == 5000]
    seeds = sorted(set(r["seed"] for r in rs))
    # 去重（同 seed 取一条）
    by_seed = {r["seed"]: r for r in rs}
    if by_seed:
        m, s = mean_sd([r["final_avg"] for r in by_seed.values()])
        print(f"fashion T=10 {pol:10} M={M} λ=5000: {m:.2f}±{s:.2f} seeds={seeds} n={len(by_seed)}"
              + (f"  | 论文 {pm}±{ps}" if pm else ""))

# paired fifo-unbounded M=2
fu = {r["seed"]: r for r in f_c2 + f_r7 if r["ntask"] == 10 and r["policy"] == "unbounded" and r["M"] == 2 and r.get("lam") == 5000}
ff = {r["seed"]: r for r in f_c2 + f_r7 if r["ntask"] == 10 and r["policy"] == "fifo" and r["M"] == 2 and r.get("lam") == 5000}
seeds = sorted(set(fu) & set(ff))
if seeds:
    deltas = [ff[sd]["final_avg"] - fu[sd]["final_avg"] for sd in seeds]
    print(f"paired Δ={st.mean(deltas):+.2f} t({len(deltas)-1})={paired_t(deltas):.1f}  (论文 +13.6, t(9)=18.3)")

# ---------- 2. γ sweep ----------
print("\n== 2. online EWC γ sweep（r8_gamma05/09 + r7_ewc γ=1）==")
g5 = load_dir(RB, "r8_gamma05.json")
g9 = load_dir(RB, "r8_gamma09.json")
ewc1 = [r for r in load_dir(R2, "r7_ewc*.json*") if r.get("kind") == "ewc"]
print(f"gamma05: {len(g5)}  gamma09: {len(g9)}  r7_ewc(γ=1): {len(ewc1)}")
for tag, rs in (("γ=0.5", g5), ("γ=0.9", g9), ("γ=1 (r7_ewc)", ewc1)):
    if not rs: print(f"{tag}: 无记录"); continue
    ck = Counter((r.get("kind"), r.get("mode", r.get("policy")), r["ntask"], r.get("lam")) for r in rs)
    print(f"{tag} 构成样例: {list(ck.items())[:6]}")

# ---------- 3. r8 merge penalty ----------
print("\n== 3. r8_merge penalty 插桩（论文: 40 runs, 43-45% @M=2 worst 64%, 18-34% @M=5, 77% move）==")
mg = load_dir(RB, "r8_merge.json")
print(f"r8_merge records: {len(mg)}")
if mg:
    print("keys:", sorted(mg[0].keys()))
    print("sample:", json.dumps(mg[0])[:500])
# r8_mgrid / r8_m2topup
mg2 = load_dir(RB, "r8_mgrid.json")
print(f"r8_mgrid: {len(mg2)}")
if mg2: print("sample:", json.dumps(mg2[0])[:300])

# ---------- 4. MNIST 主表（anchor_budget_all λ=5000?）----------
print("\n== 4. MNIST 主表（λ 分组重算）==")
mn = load_dir(R2, "anchor_budget_all.json")
for lam in sorted(set(r.get("lam") for r in mn)):
    sub = [r for r in mn if r.get("lam") == lam and r["ntask"] == 10]
    csub = Counter((r["policy"], r["M"]) for r in sub)
    print(f"λ={lam}: {dict(list(csub.items())[:8])}")
