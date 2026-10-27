# -*- coding: utf-8 -*-
"""把 S3/S4 的 refs.bib 引用整理成 cite-hallucination-check 的输入 JSON。"""
import json, os

OUT = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit_cites'
os.makedirs(OUT, exist_ok=True)

ALL_COMP = ["title", "authors", "publisher", "date", "url", "doi"]

def mk(raw, title, authors, publisher, date, url, doi, claim='', note=''):
    return {
        "id": 0, "raw": raw, "claim": claim,
        "triage": "verifiable", "famous": False, "broken_url_only": False,
        "components": {k: "unknown" for k in ALL_COMP},
        "best_source": "", "note": note,
        "_seed": {"title": title, "authors": authors, "publisher": publisher,
                  "date": date, "url": url, "doi": doi},
    }

cits = []

# ---------- S3 refs.bib ----------
cits.append(mk(
    "Iyer, Abhiram and Grewal, Karan and Velu, Akash and Souza, Lucas Oliveira and Forest, Jeremy and Ahmad, Subutai. Avoiding catastrophe: Active dendrites enable multi-task learning in dynamic environments. Frontiers in Neurorobotics 16 (2022). doi:10.3389/fnbot.2022.846219",
    "Avoiding catastrophe: Active dendrites enable multi-task learning in dynamic environments",
    "Abhiram Iyer; Karan Grewal; Akash Velu; Lucas Oliveira Souza; Jeremy Forest; Subutai Ahmad",
    "Frontiers in Neurorobotics", "2022", "", "10.3389/fnbot.2022.846219",
    claim="S3 背景：ADNet 架构来源；S4 架构描述与保护梯度设计来源"))

cits.append(mk(
    "Boahen, Kwabena. Dendrocentric learning for synthetic intelligence. Nature 612, 43-50 (2022)",
    "Dendrocentric learning for synthetic intelligence",
    "Kwabena Boahen", "Nature", "2022", "", "",
    claim="S3/S4：树突计算文献支撑 'artifact mechanism should exist wherever segments and hard routing coexist'"))

cits.append(mk(
    "Poirazi, Panayiota and Papoutsi, Athanasia. Illuminating dendritic function with computational models. Nature Reviews Neuroscience 21, 303-321 (2020)",
    "Illuminating dendritic function with computational models",
    "Panayiota Poirazi; Athanasia Papoutsi", "Nature Reviews Neuroscience", "2020", "", "",
    claim="S3/S4：树突计算文献支撑"))

cits.append(mk(
    "Masse, Nicolas Y. and Grant, Gregory D. and Freedman, David J.. Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. PNAS 115(44), E10467-E10475 (2018)",
    "Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization",
    "Nicolas Y. Masse; Gregory D. Grant; David J. Freedman",
    "Proceedings of the National Academy of Sciences", "2018", "", "",
    claim="S3：protected-gradient 与 context-gating 设计来源"))

cits.append(mk(
    "Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and others. Overcoming catastrophic forgetting in neural networks. PNAS 114(13), 3521-3526 (2017)",
    "Overcoming catastrophic forgetting in neural networks",
    "James Kirkpatrick; Razvan Pascanu; Neil Rabinowitz; et al.",
    "Proceedings of the National Academy of Sciences", "2017", "", "",
    claim="S3/S4：EWC 来源"))

cits.append(mk(
    "Parisi, German I. and Kemker, Ronald and Part, Jose L. and Kanan, Christopher and Wermter, Stefan. Continual lifelong learning with neural networks: A review. Neural Networks 113, 54-71 (2019)",
    "Continual lifelong learning with neural networks: A review",
    "German I. Parisi; Ronald Kemker; Jose L. Part; Christopher Kanan; Stefan Wermter",
    "Neural Networks", "2019", "", "",
    claim="S3：Permuted-MNIST / Permuted-Fashion 数据集来源"))

cits.append(mk(
    "Zhou, Ranxi. Segment Specialization in Active-Dendrite Continual Learning: An Audit, an Artifact, and a Causal Repair. Preprint, Zenodo, version 1, doi:10.5281/zenodo.23637973 (2026)",
    "Segment Specialization in Active-Dendrite Continual Learning: An Audit, an Artifact, and a Causal Repair",
    "Ranxi Zhou", "Zenodo", "2026", "https://doi.org/10.5281/zenodo.23637973",
    "10.5281/zenodo.23637973",
    claim="S3/S4：S1 伴生论文（本次已修订并重新发布为 version 2）"))

# ---------- S4 refs.bib ----------
cits.append(mk(
    "Zhou, Ranxi. Repair Headroom, Not Dying Gates: When an Own-Segment Bias Helps Continual Learning --- and When It Silently Reverses. Preprint, 2026",
    "Repair Headroom, Not Dying Gates: When an Own-Segment Bias Helps Continual Learning --- and When It Silently Reverses",
    "Ranxi Zhou", "Preprint", "2026", "https://github.com/zhouranxi/delp-anchor-budget", "",
    claim="S4：S3 伴生论文，尚未发表"))

cits.append(mk(
    "Zhou, Ranxi. Where Do the Gains Come From? An Audit of Dendritic Context-Gated Protection for Continual Learning. Preprint, Zenodo, version 2, doi:10.5281/zenodo.23602281 (2026)",
    "Where Do the Gains Come From? An Audit of Dendritic Context-Gated Protection for Continual Learning",
    "Ranxi Zhou", "Zenodo", "2026", "https://doi.org/10.5281/zenodo.23602281",
    "10.5281/zenodo.23602281",
    claim="S4：前身审计论文（DeLP 方法的被审计对象）"))

for i, c in enumerate(cits, 1):
    c["id"] = i
    c["_paper"] = "S3" if i <= 7 else "S4"

with open(os.path.join(OUT, 'citations.json'), 'w', encoding='utf-8') as f:
    json.dump(cits, f, ensure_ascii=False, indent=1)

print(f'写入 {len(cits)} 条引用 -> {OUT}/citations.json')
for c in cits:
    print(f"  {c['id']:2d} [{c['_paper']}] {c['_seed']['doi'] or c['_seed']['url'] or '(无 DOI/URL)':<45} {c['_seed']['title'][:60]}")
