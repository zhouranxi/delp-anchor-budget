# -*- coding: utf-8 -*-
"""S2 审计第9轮（终局）：附录矩阵逐数、Fig2 λ族、SI 引用、Fig stack CIFAR、fig:criteria bit-identical、pre-reg对照."""
import json, math, statistics as st, re
from pathlib import Path
from collections import Counter

R2 = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/results")
RB = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-10-08-27/dab_t/delp-anchor-budget-main/bench/results")
RB = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/bench/results")
PAPER = Path(r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/dab_t/delp-anchor-budget-main/S2_锚累积/paper")

def load_dir(base, parts):
    recs = []
    for p in sorted(Path(base).glob(parts)):
        d = json.load(open(p))
        recs.extend(d["results"] if isinstance(d, dict) and "results" in d else d)
    return recs

mn = load_dir(R2, "anchor_budget_all.json")

# ---------- 1. 附录矩阵逐数核对 ----------
print("== 1. 附录矩阵（appx_mat_t10/t20.tex）vs 数据 ==")
for T, fname in ((10, "appx_mat_t10.tex"), (20, "appx_mat_t20.tex")):
    txt = (PAPER / fname).read_text(encoding="utf-8")
    rows = re.findall(r"^(\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", txt, re.M)
    n_mismatch = 0
    for task_s, u, f, ra, re_, me in rows:
        task = int(task_s) - 1
        vals = {"unbounded": float(u), "fifo": float(f), "random": float(ra), "redun": float(re_), "merge": float(me)}
        for pol, pv in vals.items():
            rs = [r for r in mn if r["ntask"]==T and r["policy"]==pol and r.get("lam")==5000.0 and r["M"]==5]
            got = st.mean([r["acc_matrix"][-1][task] for r in rs if r["seed"] in (0,1,2)])
            if abs(got - pv) > 0.55:
                n_mismatch += 1
                print(f"  T={T} 任务{task_s} {pol}: paper={pv} got={got:.1f}  <<<")
    print(f"  T={T}: {len(rows)} 行核对完, 不匹配 {n_mismatch} 处")

# ---------- 2. SI 引用 ----------
print("\n== 2. SI 基线引用 ==")
si = [r for r in mn if r.get("kind") == "si"]
if si:
    print(f"  SI 记录: {[(r['ntask'], round(r['final_avg'],1)) for r in si]}")
grep_si = PAPER.joinpath("main.tex").read_text(encoding="utf-8")
for m in re.finditer(r"SI[^.\n]{0,120}", grep_si):
    print("  paper:", m.group(0)[:130])

# ---------- 3. Fig stack CIFAR: merge grandfather-protects partially ----------
print("\n== 3. CIFAR merge 栈 ==")
cifar = [r for r in load_dir(R2, "s2r6_b_t20*.json*") if r.get("dataset")=="cifar100"]
mg = [r for r in cifar if r["policy"]=="merge" and r["M"]==2]
for r in mg[:5]:
    print(f"  seed {r['seed']}: stack={tuple(r['survivor_history'][-1])}")
c10 = Counter(tuple(r["survivor_history"][-1]) for r in mg)
print(f"  分布: {dict(c10)}")

# ---------- 4. fig:criteria bit-identical（agew == redun）----------
print("\n== 4. agew ≡ redun bit-identical ==")
for T in (10, 20):
    for M in (2, 5):
        rw = [r for r in mn if r["ntask"]==T and r["policy"]=="redun" and r.get("lam")==5000.0 and r["M"]==M]
        ag = [r for r in mn if r["ntask"]==T and r["policy"]=="agew" and r.get("lam")==5000.0 and r["M"]==M]
        pairs = 0; ident = 0
        by_seed_r = {r["seed"]: r for r in rw}; by_seed_a = {r["seed"]: r for r in ag}
        for s in set(by_seed_r) & set(by_seed_a):
            pairs += 1
            if by_seed_r[s]["acc_matrix"] == by_seed_a[s]["acc_matrix"]:
                ident += 1
        print(f"  T={T} M={M}: {ident}/{pairs} bit-identical")

# ---------- 5. dose-response 正文数字（M sweep 42→52?）----------
print("\n== 5. 剂量响应 r8_mgrid + 已有 M=2,5 ==")
mgrid = load_dir(RB, "r8_mgrid.json")
points = {}
for M in (2,3,4,5,6,8,10):
    rs = [r for r in mgrid if r["ntask"]==20 and r.get("lam")==5000.0 and r["M"]==M]
    if not rs:
        rs = [r for r in mn if r["ntask"]==20 and r["policy"]=="fifo" and r.get("lam")==5000.0 and r["M"]==M]
    if rs:
        m, s = st.mean([r["final_avg"] for r in rs]), st.stdev([r["final_avg"] for r in rs])
        ell = st.mean([r["ell"] for r in rs]); bwt = st.mean([r["bwt"] for r in rs])
        points[M] = (m, s, ell, bwt, len(rs))
        print(f"  M={M:2}: acc={m:.1f}±{s:.1f} ℓ={ell:.1f} BWT={bwt:+.1f} n={len(rs)}")

# 正文引用: "accuracy rises monotonically from 33.6 (M=unbounded=T) to 52.1 at M=2"? 找原文
txt = PAPER.joinpath("main.tex").read_text(encoding="utf-8")
for m in re.finditer(r"[^.]*dose[^.]*\.", txt):
    print("  [dose text]", m.group(0)[:200])
for m in re.finditer(r"monotone[^.]*\.", txt):
    print("  [monotone]", m.group(0)[:200])
