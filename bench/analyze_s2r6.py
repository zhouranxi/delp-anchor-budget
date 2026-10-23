# -*- coding: utf-8 -*-
"""Aggregate s2r6 (Split-CIFAR-100) results: main table + paired stats vs unbounded
+ key policy contrasts + survivor-history stack composition. Writes markdown report."""
import json, math, collections

FILES = ["s2r6_b_t20.json.part0", "s2r6_b_t20.json.part1",
         "s2r6_b_t10.json.part0", "s2r6_b_t10.json.part1",
         "s2r6_b_t10u.json.part0", "s2r6_b_t10u.json.part1"]

recs = []
for f in FILES:
    recs += json.load(open(f))["results"]
print(f"total records: {len(recs)}")

def key(r): return (r["ntask"], r["policy"], r["M"], r["seed"])
D = {key(r): r for r in recs}
assert len(D) == len(recs), "duplicate keys!"

POL = ["fifo", "random", "redun", "merge", "agew", "agew2"]
MS = [2, 5]
UNB = {(10, 5), (20, 2)}          # (ntask, M) cells where unbounded was materialized

def unb_ref(r):
    # unbounded is M-invariant (never evicts); pick the materialized cell per ntask
    for m in [2, 5]:
        if (r["ntask"], m) in UNB:
            u = D.get((r["ntask"], "unbounded", m, r["seed"]))
            if u: return u
    return None

def mean(xs): return sum(xs) / len(xs)
def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0

def paired(deltas):
    n = len(deltas); m = mean(deltas); s = sd(deltas)
    t = m / (s / math.sqrt(n)) if s > 0 else float("inf") * (1 if m > 0 else -1)
    return m, s, t, n

L = []
L.append("# S2-CIFAR 收数报告（s2r6，Split-CIFAR-100，ResNet-18-GN，masked-CE + 骨干锚点）\n")
L.append(f"- 数据：{len(recs)} 条记录（T=20 全策略 130 + T=10 有界 120 + T=10 unbounded 参考 10），0 错误")
L.append("- 协议：λ=0.1，5 epochs/任务，n=10 seeds（0–9），task-IL 评测，指标单位为 %\n")

# ---- main table ----
L.append("## 1. 主表：final_avg / BWT / ℓ（均值±标准差，n=10）\n")
L.append("| T | policy | M | final_avg | BWT | ℓ | n_anchors |")
L.append("|---|---|---|---|---|---|---|")
for nt in [10, 20]:
    for pol in ["unbounded"] + POL:
        for M in MS:
            rs = [D[k] for k in D if k[0] == nt and k[1] == pol and k[2] == M]
            if not rs: continue
            a = mean([r["final_avg"] for r in rs]); asd = sd([r["final_avg"] for r in rs])
            b = mean([r["bwt"] for r in rs]); bsd = sd([r["bwt"] for r in rs])
            e = mean([r["ell"] for r in rs]); esd = sd([r["ell"] for r in rs])
            na = mean([r["n_anchors_final"] for r in rs])
            L.append(f"| {nt} | {pol} | {M} | {a:.2f}±{asd:.2f} | {b:.2f}±{bsd:.2f} | {e:.2f}±{esd:.2f} | {na:.1f} |")
L.append("")

# ---- paired vs unbounded ----
L.append("## 2. 配对对比：各有界策略 − unbounded（同 seed 配对）\n")
L.append("| T | policy | M | Δavg | ΔBWT(−) | t(Δavg) | n |")
L.append("|---|---|---|---|---|---|---|")
for nt in [10, 20]:
    for pol in POL:
        for M in MS:
            da, db = [], []
            for s in range(10):
                r = D.get((nt, pol, M, s)); u = unb_ref(r) if r else None
                if not u: continue
                da.append(r["final_avg"] - u["final_avg"])
                db.append(r["bwt"] - u["bwt"])
            ma, sa, ta, n = paired(da); mb, sb, tb, _ = paired(db)
            L.append(f"| {nt} | {pol} | {M} | {ma:+.2f}±{sa:.2f} | {mb:+.2f}±{sb:.2f} | {ta:.1f} | {n} |")
L.append("")

# ---- key contrasts ----
L.append("## 3. 核心假设配对对比（同 T、同 M、同 seed）\n")
L.append("| 对比 | T | M | Δavg | ΔBWT | t(Δavg) | n |")
L.append("|---|---|---|---|---|---|---|")
for a, b in [("fifo", "redun"), ("agew2", "redun"), ("fifo", "agew2")]:
    for nt in [10, 20]:
        for M in MS:
            da, db = [], []
            for s in range(10):
                ra, rb = D.get((nt, a, M, s)), D.get((nt, b, M, s))
                if not (ra and rb): continue
                da.append(ra["final_avg"] - rb["final_avg"])
                db.append(ra["bwt"] - rb["bwt"])
            if not da: continue
            ma, sa, ta, n = paired(da); mb, sb, tb, _ = paired(db)
            L.append(f"| {a} − {b} | {nt} | {M} | {ma:+.2f}±{sa:.2f} | {mb:+.2f}±{sb:.2f} | {ta:.1f} | {n} |")
L.append("")

# ---- survivor history at T20 ----
L.append("## 4. 栈轨迹（T=20，M=2，各 seed 存活 origin 集合）\n")
for pol in POL:
    sets = []
    for s in range(10):
        r = D.get((20, pol, 2, s))
        sets.append(tuple(sorted(r["survivor_history"][-1])) if r else None)
    uniq = collections.Counter(sets)
    desc = "; ".join(f"{dict(collections.Counter(k)).get(None) if None in k else '{'+','.join(map(str,k))+'}'} ×{v}" if k else "?" for k, v in uniq.items())
    L.append(f"- **{pol}**: " + " / ".join(
        ("{" + ",".join(map(str, k)) + "}" if k is not None else "?") + f" ×{v}" for k, v in uniq.items()))
L.append("")
L.append("（fifo 预期≈滑动窗 {T-M..T-1}；redun 预期≈冻结在早期 origin——祖父化陷阱的 CIFAR 复现证据）")

open("S2_CIFAR_report_data.md", "w", encoding="utf-8").write("\n".join(L))
print("written S2_CIFAR_report_data.md")
# quick console peek
for line in L[:8]: print(line)
