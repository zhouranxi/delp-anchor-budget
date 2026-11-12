"""e9_resnet.py -- E9: Split-CIFAR-100 external validity for the E5 gate story.

Reuses the calibrated S2 Split-CIFAR-100 protocol (anchor_budget_cifar.py):
ResNet-18 + GN(32,C), task-IL masked softmax, global-normalized diagonal
Fisher anchors on the backbone, lam=0.1, SGD(0.05, 0.9, 5e-4), 5 epochs,
clip 5.0.  ON TOP of that we insert the minimal dendritic gate between
avgpool and fc (E5 semantics):

    t   = bb(x)                       # (B, 512) pooled features
    u   = dend(ctx).view(-1, nseg, 512)
    d   = u.max(1).values
    h   = kwta(t * sigmoid(d), k=128) # 25% keep-rate as in E5 (64/256)
    out = head(h)

Arms:
    plain   head(t)                     no gate, no kwta  (= S2 model)
    base    kwta(t * sigmoid(d))
    pin     kwta(t * G)
    delete  kwta(t)
    renorm  kwta(t / G)
    gscale  kwta(t * sigmoid(d) / G)

G is the pilot-measured E[sigmoid(d)] (frozen for the grid).  The gate
module (dend) and the head are TRAINED per task; anchors cover backbone +
dend, exclude head (masked softmax keeps head rows from drifting, so the
head gains nothing from pinning but would inflate Fisher scale).

Usage:
  # smoke (~2 min on GPU)
  CUDA_VISIBLE_DEVICES=0 python e9_resnet.py --smoke --out /tmp/e9_smoke.json
  # pilot: 3 seeds, measures G and gate stats
  CUDA_VISIBLE_DEVICES=0 python e9_resnet.py --arms base --seeds 0,1,2 --ntasks 10 \
      --out results_e9/e9_pilot.json
  # grid
  CUDA_VISIBLE_DEVICES=0 python e9_resnet.py --task-id 0 --n-shards 3 \
      --arms plain,base,pin,delete,renorm,gscale --seeds 0,1,2,3,4,5,6,7,8,9 \
      --ntasks 10 --out results_e9/e9_grid_s0.json
"""
import argparse, json, statistics as st, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
EPOCHS, BS, LR, WD, MOM = 5, 128, 0.05, 5e-4, 0.9
NFISHER = 50
CLIP_NORM = 5.0
LAM = 0.1
K_KEEP = 128          # 25% of 512, matches E5's 64/256 ratio
NH = 512              # pooled feature width
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ["plain", "base", "pin", "delete", "renorm", "gscale"]
G_CONST = 0.5749      # E9 pilot-measured E[sigmoid(d)] on ResNet-18+GN (3 seeds × 10 phases)


def kwta(h, k=K_KEEP):
    thr = h.topk(min(k, h.size(1)), dim=1).values[:, -1:]
    return h * (h >= thr).float()


# ---------------------------------------------------------------- data
def load_cifar100():
    from torchvision import datasets, transforms
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    d_tr = datasets.CIFAR100("./data", train=True, download=True, transform=tf_tr)
    d_te = datasets.CIFAR100("./data", train=False, download=True, transform=tf_te)
    d_fx = datasets.CIFAR100("./data", train=True, download=True, transform=tf_te)
    ytr = torch.tensor(d_tr.targets)
    yte = torch.tensor(d_te.targets)
    xte = torch.stack([d_te[i][0] for i in range(len(d_te))])
    xfx = torch.stack([d_fx[i][0] for i in range(len(d_fx))])
    return d_tr, (ytr, yte, xte, xfx)


# ---------------------------------------------------------------- model
def _bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name,
                    nn.GroupNorm(min(32, child.num_features), child.num_features))
        else:
            _bn_to_gn(child)


class GatedResNet(nn.Module):
    """ResNet-18+GN backbone -> dendritic gate -> head.  nseg = ntask."""

    def __init__(self, ntask, arm="base", g=G_CONST, nh=NH, k=K_KEEP):
        super().__init__()
        from torchvision.models import resnet18
        bb = resnet18(num_classes=100)
        bb.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        bb.maxpool = nn.Identity()
        _bn_to_gn(bb)
        self.bb = bb                     # keeps fc for state_dict compat; unused
        self.fc = nn.Linear(nh, 100)
        self.dend = nn.Linear(ntask, ntask * nh)
        self.nseg, self.nh, self.k, self.arm, self.g = ntask, nh, k, arm, g
        self.gate_stats = []             # per-phase [gate_mean,std,q10,q90,t_absmean]

    def features(self, x):
        return self.bb.avgpool(self.bb.layer4(
            self.bb.layer3(self.bb.layer2(self.bb.layer1(
                self.bb.relu(self.bb.bn1(self.bb.conv1(x)))))))).flatten(1)

    def forward(self, x, ctx):
        t = self.features(x)
        if self.arm == "plain":
            return self.fc(t)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d = u.max(dim=1).values
        gsig = torch.sigmoid(d)
        if self.arm == "base":
            h = kwta(t * gsig, self.k)
        elif self.arm == "pin":
            h = kwta(t * self.g, self.k)
        elif self.arm == "delete":
            h = kwta(t, self.k)
        elif self.arm == "renorm":
            h = kwta(t / self.g, self.k)
        elif self.arm == "gscale":
            h = kwta(t * gsig / self.g, self.k)
        else:
            raise ValueError(self.arm)
        return self.fc(h)


# ---------------------------------------------------------------- anchors
def anchor_keys(model):
    return [k for k, _ in model.named_parameters()
            if not k.startswith("fc.") and not k.startswith("bb.fc.")]


def fisher_anchor(model, xfx, idx):
    model.eval()
    keys = anchor_keys(model)
    P = {k: v for k, v in dict(model.named_parameters()).items() if k in keys}
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    for i in range(NFISHER):
        xb = xfx[idx[i % len(idx)]].unsqueeze(0).to(DEV)
        ctx0 = F.one_hot(torch.zeros(1, dtype=torch.long, device=DEV),
                         model.nseg).float()          # one-hot, not zeros:
        out = model(xb, ctx0)                         # zeros would zero dend.weight Fisher
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            if v.grad is not None:          # plain arm: dend gets no grad
                fisher[k] += v.grad.detach() ** 2
    # drop params that never received gradient (dead in this arm's graph)
    fisher = {k: f for k, f in fisher.items() if f.sum() > 0}
    n = sum(f.numel() for f in fisher.values())
    scale = sum(f.sum() for f in fisher.values()) / max(n, 1)
    for k in fisher:
        fisher[k] = fisher[k] / (scale + 1e-12)
    return fisher


def eval_tasks(model, task_cls, yte, xte, phase):
    model.eval()
    acc = {}
    with torch.no_grad():
        for cc in range(phase + 1):
            cls = task_cls[cc]
            idx = torch.tensor([i for i, y in enumerate(yte.tolist()) if y in cls])
            xt = xte[idx].to(DEV)
            yt = yte[idx].to(DEV)
            lut = torch.tensor(cls, device=DEV)
            correct = 0
            for i in range(0, xt.size(0), 512):
                ctx = F.one_hot(torch.full((xt[i:i+512].size(0),), cc, device=DEV),
                                model.nseg).float()
                out = model(xt[i:i + 512], ctx)[:, lut]
                pred = lut[out.argmax(1)]
                correct += (pred == yt[i:i + 512]).sum().item()
            acc[(phase, cc)] = correct / xt.size(0) * 100
    return acc


def probe(model, xs, tid):
    with torch.no_grad():
        t = model.features(xs)
        ctx = F.one_hot(torch.full((xs.size(0),), tid, device=DEV), model.nseg).float()
        u = model.dend(ctx).view(-1, model.nseg, model.nh)
        d = u.max(dim=1).values
        g = torch.sigmoid(d)
        return {"gate_mean": g.mean().item(), "gate_std": g.std().item(),
                "gate_q10": g.quantile(0.10).item(), "gate_q90": g.quantile(0.90).item(),
                "t_absmean": t.abs().mean().item()}


# ---------------------------------------------------------------- one run
def run_one(seed, ntask, arm, d_tr, aux, lam=LAM, epochs=EPOCHS):
    ytr, yte, xte, xfx = aux
    g = torch.Generator().manual_seed(100 + seed)
    order = torch.randperm(100, generator=g).tolist()
    cpt = 100 // ntask
    task_cls = [order[t * cpt:(t + 1) * cpt] for t in range(ntask)]
    torch.manual_seed(seed)
    model = GatedResNet(ntask, arm=arm).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    anchors, acc = [], {}
    t0 = time.time()
    for phase in range(ntask):
        cls = task_cls[phase]
        lut = torch.tensor(cls, device=DEV)
        idx = [i for i, y in enumerate(ytr.tolist()) if y in cls]
        dl = DataLoader(Subset(d_tr, idx), batch_size=BS, shuffle=True,
                        num_workers=2, pin_memory=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                local = torch.zeros_like(yb)
                for ci, c in enumerate(cls):
                    local[yb == c] = ci
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
                opt.zero_grad()
                out = model(xb, ctx)[:, lut]
                loss = F.cross_entropy(out, local)
                if anchors and lam > 0:
                    P = dict(model.named_parameters())
                    pen = 0.0
                    for a in anchors:
                        for k in a["fisher"]:          # only anchored keys
                            v = P[k]
                            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
        if arm != "plain":
            xs_probe = torch.stack([d_tr[i][0]
                                    for i in idx[:250]]).to(DEV)
            model.gate_stats.append(probe(model, xs_probe, phase))
        if lam > 0:                     # plain/no-anchor arms skip Fisher cost
            fx_idx = idx[:NFISHER * 2]
            fisher = fisher_anchor(model, xfx, fx_idx)
            theta = {k: v.detach().clone()
                     for k, v in dict(model.named_parameters()).items() if k in fisher}
            anchors.append({"theta": theta, "fisher": fisher})
        acc.update(eval_tasks(model, task_cls, yte, xte, phase))
        print(f"  s{seed} {arm} T{phase}: t{phase}={acc[(phase, phase)]:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    bwt = sum(acc[(ntask - 1, c)] - acc[(c, c)] for c in range(ntask - 1)) / (ntask - 1)
    favg = sum(acc[(ntask - 1, c)] for c in range(ntask)) / ntask
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    rec = {"kind": "delp", "ntask": ntask, "seed": seed, "arm": arm, "lam": lam,
           "dataset": "cifar100", "backbone": "resnet18-gn", "protocol": "task-IL",
           "bwt": round(bwt, 2), "final_avg": round(favg, 2), "ell": round(ell, 2),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                          for p in range(ntask)],
           "gate_stats": model.gate_stats}
    return rec


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="e9_result.json")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ntasks", type=int, default=10)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    d_tr, aux = load_cifar100()
    if args.smoke:
        recs = [run_one(0, 10, "base", d_tr, aux, epochs=1),
                run_one(0, 10, "plain", d_tr, aux, lam=0.0, epochs=1)]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        arms = args.arms.split(",")
        jobs = [(a, s) for a in arms for s in seeds]
        jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.task_id]
        recs = []
        for arm, seed in jobs:
            recs.append(run_one(seed, args.ntasks, arm, d_tr, aux,
                                lam=args.lam, epochs=args.epochs))
    json.dump({"results": recs}, open(args.out, "w"))
    print("WROTE", args.out, len(recs), "runs")


if __name__ == "__main__":
    main()
