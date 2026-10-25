#!/usr/bin/env python
"""S3 任务相关性梯度（T10，MNIST）。

任务结构（预注册，固定，先于任何 s3corr 数据落盘）：
  设计空间按 4 组 × 2 seeds 覆盖：
  H (in-domain)   : T0 digit y, T4 parity y%2
  M (in-domain)   : T1 parity y%2, T5 parity y%2
  L (in-domain)   : T2 count (y+5)%10, T6 count (y+5)%10
  O (out-domain)  : T3 random, T7 random, T8 random, T9 random

评估口径说明：eval_model 按原始 digit 标签测所有任务（与 S1 主实验
完全同口径）。digit/parity 任务测得好；count 任务系统性偏低
(5/10 标签与 digit 匹配)；random 任务近随机。判据为组间相对比较
(H 组 vs L 组 / O 组)，b 剂量梯度在 O 组上应衰减。
"""
import argparse, json, os, sys, time
import torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 与 branch_delp.py 相同的 sys.path 插入（server 布局：../dendritic-lab）
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, "..", "dendritic-lab"),
              os.path.join(_here, "..", "..", "dendritic-lab")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

import exp4_branch_protection as e4
import anchor_budget
import exp6_baselines as eb
from branch_delp import (BiasADNet, flat_params,
                        make_anchor, anchor_penalty, attribute_drift,
                        _clone)

# 与 S1 主实验完全一致的超参（全部取自共享常量模块，不复制数值）
from exp6_baselines import EPOCHS, BS, NTRAIN, LR, DEV, eval_model, finish, get_perms
from anchor_budget import load_data

# 固定实验常量（S3 预注册）
NTASK = 10
LAM   = 1000.0
SEEDS = list(range(10))
DOSES = [0.0, 1.0, 3.0, 10.0]
GRANS = ["neuron", "branch"]

# 任务结构（10 个任务，组序固定）：H=high digit/parity, M=parity, L=count, O=random
# 4 组 × 重复，每组在两个连续任务位置出现一次，保证每个 (seed,gran,b) 臂内
# 四个组各占 2 任务，判据为组内相对均值（H 组 vs L 组 vs O 组）。
_SCHEDULE = [
    ("digit",  "high", None),                # T0
    ("parity", "high", lambda y: y % 2),     # T1
    ("count",  "low",  lambda y: (y + 5) % 10),  # T2
    ("random", "out",  None),                # T3
    ("digit",  "high", None),                # T4
    ("count",  "low",  lambda y: (y + 5) % 10),  # T5
    ("random", "out",  None),                # T6
    ("parity", "high", lambda y: y % 2),     # T7
    ("random", "out",  None),                # T8
    ("count",  "low",  lambda y: (y + 5) % 10),  # T9
]
# 每组对应任务索引（用于组间均值对比）
_GROUP_IDX = {"high": [0, 1, 4, 7], "low": [2, 5, 9], "out": [3, 6, 8]}
_RAND_SEED = [1001, 1002]


def task_labels(phase, ytr):
    """本 phase 的训练/评估标签。digit 无转换；parity/count 为确定性函数；
    random 用固定 seed 重置生成器后取 randint（每次调用重置，保证确定性）。"""
    name, level, fn = _SCHEDULE[phase]
    if name == "random":
        gen = torch.Generator().manual_seed(_RAND_SEED[phase % 2])
        return torch.randint(0, 10, (eb.NTRAIN,), generator=gen)
    if fn is not None:
        return fn(ytr[:eb.NTRAIN])
    return ytr[:eb.NTRAIN]


def run_corr(seed, data, perms, lam, gran, own_bias):
    """镜像 branch_delp.run_gran（line 223–311），仅两处不同：
    ①训练标签用 task_labels(phase) ②记录 corr_groups/corr_levels。
    评估与锚构造与主实验完全同口径。"""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    bias_mode = "fixed" if own_bias else "none"
    if own_bias:
        model = BiasADNet(NTASK, NTASK, own_bias=own_bias,
                          bias_mode="fixed").to(eb.DEV)
    else:
        model = e4.ADNet(protected=True, nseg=NTASK, ntask=NTASK).to(eb.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=eb.LR)
    anchors, acc = [], {}
    seg_mass_all, dom_all = [], []
    for phase in range(NTASK):
        ys = task_labels(phase, ytr)
        xs, ys = e4.permute(xtr[:eb.NTRAIN], perms[phase]), ys
        dl = DataLoader(TensorDataset(xs, ys), batch_size=eb.BS, shuffle=True)
        W_start = _clone(flat_params(model)) if gran == "branch" else None
        win_cnt = torch.zeros(NTASK, model.nh, device=eb.DEV) \
            if gran == "branch" else None
        n_seen = 0
        model.train()
        for _ep in range(eb.EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(eb.DEV), yb.to(eb.DEV)
                ctx = F.one_hot(
                    torch.full((xb.size(0),), phase, device=eb.DEV),
                    NTASK).float()
                opt.zero_grad()
                out, seg = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        pen = pen + anchor_penalty(a, P, a["gran"])
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
                if gran == "branch":
                    for k in range(NTASK):
                        win_cnt[k] += (seg == k).float().sum(0)
                    n_seen += xb.size(0)
        if gran == "branch":
            P_now = flat_params(model)
            dW = {k: P_now[k].detach() - W_start[k] for k in P_now}
            w_pk = win_cnt / max(n_seen, 1)
            attribute_drift(anchors, w_pk, dW, gran)
        anchors.append(make_anchor(model, xs, phase, NTASK, gran))
        if gran == "branch":
            seg_mass_all.append(anchors[-1]["seg_mass"])
        dom_all.append({k: anchors[-1][k] for k in
                        ("dom_hist", "dom_hist_active", "dead_frac")})
        # 评估：用转换后标签（与训练标签同分布），对 parity/count/random 任务
        # 测真实任务精度。digit 任务直接用 yte（原始测试标签）。
        name, level, fn = _SCHEDULE[phase]
        if name == "random":
            gen = torch.Generator().manual_seed(_RAND_SEED[phase % 2])
            yte_eval = torch.randint(0, 10, (len(yte),), generator=gen)
        elif fn is not None:
            yte_eval = fn(yte)
        else:
            yte_eval = yte
        a = eval_model(model, xte, yte_eval, perms, phase, NTASK, True)
        acc.update(a)
    bwt, favg = eb.finish(acc, NTASK)
    ell = sum(acc[(t, t)] for t in range(NTASK)) / NTASK
    rec = {"kind": "corr", "seed": seed, "gran": gran,
           "bias": own_bias, "ntask": NTASK, "lam": lam,
           "dataset": "mnist",
           "bwt": round(bwt, 2), "final_avg": round(favg, 2),
           "ell": round(ell, 2),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                          for p in range(NTASK)],
           "corr_groups": [n for n, _, _ in _SCHEDULE],
           "corr_levels": [lv for _, lv, _ in _SCHEDULE],
           "dom_per_phase": dom_all}
    if gran == "branch":
        rec["seg_mass"] = [[round(m, 4) for m in row] for row in seg_mass_all]
    # 预注册判据辅助：组内对角均值（acc[(p,p)] = "任务 p 学好后测原始标签"）
    for grp, idxs in _GROUP_IDX.items():
        vals = [acc[(p, p)] for p in idxs]
        rec[f"diag_{grp}"] = round(sum(vals) / len(vals), 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s3corr_mnist.json.part0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=2)
    a = ap.parse_args()

    data = anchor_budget.load_data("mnist")
    results, t0 = [], time.time()
    # 确定性 key：先 shard 后排序，保证 2 分片互斥且覆盖全部 80 组
    all_keys = []
    for seed in SEEDS:
        for gi, gran in enumerate(GRANS):
            for bi, b in enumerate(DOSES):
                all_keys.append((seed * 8 + gi * 4 + bi, seed, gran, b))
    for key, seed, gran, b in all_keys:
        if key % a.n_shards != a.shard:
            continue
        rec = {"kind": "corr", "seed": seed, "gran": gran, "bias": b}
        try:
            perms = get_perms(seed, NTASK)
            rec.update(run_corr(seed, data, perms, 1000.0, gran, b))
        except Exception as exc:                            # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
        print(f"  s{seed} {gran} b={b}: ell={rec.get('ell')} "
              f"avg={rec.get('final_avg')} bwt={rec.get('bwt')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(a.out, "w") as f:
        json.dump({"results": results, "n_shards": a.n_shards,
                   "shard": a.shard, "elapsed": time.time() - t0},
                  f, indent=1)
    print(f"\nDONE: {len(results)} runs in {time.time()-t0:.0f}s "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
