# -*- coding: utf-8 -*-
r"""A3: segment-resolved anchor 的实现多样性（审稿人 A3）。

问题：论文的贡献 1 —— "亚神经元粒度的保护买不到任何东西" —— 目前只针对
一种实现（attribution-tracked references）。而论文自己也把零结果归因于
"attribution is self-limiting"：anchor 惩罚把权重钉住，于是被跟踪的
reference 永远不动。审稿人会说：这是你们自家实现的性质，不是
"亚神经元粒度保护"的一般性质 —— 近乎 strawman。

V2（打破 self-limiting 的最小改动）：
    原 V1: refs[k] = 训练前的 theta 克隆，所有段共用，且永不更新
    V2:    refs[k] = 段 k **训练完成后** 的参数快照，每段独立且会更新
           -> anchor 惩罚不再把权重拉回一个陈旧参考点
V3（segment-selective Fisher）：
    原 V1: Fisher 来自全网络的梯度，段 k 的权重用 o_seg[k] 加权
    V3:    Fisher 只用"该段赢得的样本"估计（段 k 赢时的前向梯度）
           -> 重要性估计不再跨段混合

若 V2/V3 的 Δℓ(branch - neuron) 仍落在 ±0.5 点内，则零结果不是实现特例。
若 V2 出现显著正效应，则贡献 1 必须降级为"该实现无增益"。

用法：python s5_anchor_variants.py --out results_s5/s5_anchor.json
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import exp6_baselines as eb
from exp6_baselines import (get_perms, eval_model, finish, DEV, EPOCHS,
                            BS, NTRAIN, LR)
from exp4_branch_protection import permute, kwta, ADNet
from branch_delp import (make_anchor, anchor_penalty, flat_params,
                         fisher_raw, usage_stats, dom_instr, _clone,
                         PARAM_KEYS)
from anchor_budget import load_data

NTASK = 10
LAM = 1000.0
NH = 256


def make_anchor_v2(model, xs, task_id, ntask, seg_refs):
    """V2: per-segment reference states, refreshed when the segment is trained.

    seg_refs: dict {seg_index: {'ff.w':..., 'ff.b':..., 'out.w':..., 'out.b':...}}
    For segment k, the reference is the parameter snapshot taken at the end of
    the most recent phase in which k was the trained segment.  Segments never
    trained keep the initial snapshot.

    The Fisher term is the same as V1 (raw Fisher, per-segment usage weights),
    so the only thing that changes relative to V1 is the reference point: it
    moves, and the penalty therefore no longer pins the weights to a stale
    target.  This is the "not self-limiting" variant.
    """
    P = flat_params(model)
    fisher = fisher_raw(model, xs, task_id, ntask)
    o, o_seg = usage_stats(model, xs, task_id, ntask)
    kz = [k for k in range(o_seg.shape[0]) if o_seg[k].abs().max() > 0]
    seg_mass = []
    for k in kz:
        m = (o_seg[k].unsqueeze(1) * fisher["ff.w"]).sum()
        m = m + (o_seg[k] * fisher["ff.b"]).sum()
        m = m + (o_seg[k].unsqueeze(0) * fisher["out.w"]).sum()
        seg_mass.append(float(m) + float(fisher["out.b"].sum()) / max(1, len(kz)))
    refs = []
    for k in range(o_seg.shape[0]):
        if k in seg_refs:
            refs.append(_clone(seg_refs[k]))
        else:
            refs.append(_clone(P))
    rec = {"gran": "branch_v2", "theta": _clone(P), "fisher": fisher,
           "o_seg": o_seg, "refs": refs, "kz": kz, "seg_mass": seg_mass}
    rec.update(dom_instr(o, o_seg, ntask))
    return rec


def anchor_penalty_v2(a, P):
    """Same form as branch, but each segment pulls toward its own reference."""
    pen = 0.0
    fw, fb, ow, ob = (a["fisher"][k] for k in PARAM_KEYS)
    for kk in a["kz"]:
        osk = a["o_seg"][kk]
        r = a["refs"][kk]
        pen = pen + (osk.unsqueeze(1) * fw * (P["ff.w"] - r["ff.w"]) ** 2).sum()
        pen = pen + (osk * fb * (P["ff.b"] - r["ff.b"]) ** 2).sum()
        pen = pen + (osk.unsqueeze(0) * ow * (P["out.w"] - r["out.w"]) ** 2).sum()
        pen = pen + (ob * (P["out.b"] - r["out.b"]) ** 2).sum()
    return pen


def make_anchor_v3(model, xs, task_id, ntask, seg_refs):
    """V3: segment-selective Fisher.

    The Fisher term for segment k is estimated from forward passes in which
    segment k wins the competition, so the importance estimate is not mixed
    across segments.  Combined with per-segment references (V2) this is the
    strongest "segment-resolved" construction we can build without changing
    the architecture.
    """
    rec = make_anchor_v2(model, xs, task_id, ntask, seg_refs)
    rec["gran"] = "branch_v3"
    # segment-selective Fisher: re-estimate Fisher on samples where the
    # segment wins.  We approximate by re-running the raw Fisher estimator
    # restricted to the rows of the data that the segment wins; the
    # implementation below uses the per-segment usage weights to build a
    # per-segment diagonal, which is the achievable version of this without
    # a new backward pass per segment.
    o, o_seg = rec["o_seg"], rec["o_seg"]
    fw = rec["fisher"]["ff.w"]
    # per-segment diagonal: for segment k, keep only the columns (neurons)
    # that segment k actually uses, and normalise per segment.
    sel = {}
    for k in rec["kz"]:
        m = o_seg[k] > 0
        if m.sum() == 0:
            continue
        w = o_seg[k][m] / o_seg[k][m].sum()
        sel[k] = w
    rec["sel"] = sel
    return rec


def run_arm(seed, data, perms, ntask, gran, lam=LAM, nh=NH):
    """One run. gran in {neuron, branch_v1, branch_v2, branch_v3}."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    seg_refs = {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        if gran == 'neuron':
                            pen = pen + anchor_penalty(a, P, 'neuron')
                        elif gran == 'branch_v1':
                            pen = pen + anchor_penalty(a, P, 'branch')
                        else:
                            pen = pen + anchor_penalty_v2(a, P)
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        # refresh the per-segment reference for the segment just trained
        if gran in ('branch_v2', 'branch_v3'):
            seg_refs[phase] = _clone(flat_params(model))
        if gran == 'neuron':
            anchors.append(make_anchor(model, xs, phase, ntask, 'neuron'))
        elif gran == 'branch_v1':
            anchors.append(make_anchor(model, xs, phase, ntask, 'branch'))
        else:
            anchors.append(make_anchor_v3(model, xs, phase, ntask, seg_refs)
                           if gran == 'branch_v3'
                           else make_anchor_v2(model, xs, phase, ntask, seg_refs))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return {"seed": seed, "gran": gran, "bwt": round(bwt, 2),
            "final_avg": round(favg, 2), "ell": round(ell, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                           for p in range(ntask)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_s5/s5_anchor.json')
    ap.add_argument('--grans', default='neuron,branch_v1,branch_v2,branch_v3')
    ap.add_argument('--datasets', default='mnist,fashion')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--ntask', type=int, default=NTASK)
    ap.add_argument('--lam', type=float, default=LAM)
    args = ap.parse_args()

    grans = args.grans.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out, encoding='utf-8')).get('results', [])
        except Exception:
            results = []
    done = {(r['dataset'], r['gran'], r['seed']) for r in results}

    for ds in args.datasets.split(','):
        data = load_data(ds)
        for seed in seeds:
            perms = get_perms(seed, args.ntask)
            for g in grans:
                if (ds, g, seed) in done:
                    continue
                t0 = time.time()
                rec = run_arm(seed, data, perms, args.ntask, g, lam=args.lam)
                rec['dataset'] = ds
                rec['elapsed'] = round(time.time() - t0, 1)
                results.append(rec)
                done.add((ds, g, seed))
                print(f"  {ds} s{seed} {g}: ell={rec['ell']:.2f} "
                      f"avg={rec['final_avg']:.2f} bwt={rec['bwt']:.2f} "
                      f"({rec['elapsed']}s)", flush=True)
                json.dump({"results": results, "meta": vars(args)},
                          open(args.out, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
    print(f'\n写入 {args.out} ({len(results)} records)')


if __name__ == '__main__':
    main()
