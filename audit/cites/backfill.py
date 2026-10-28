# -*- coding: utf-8 -*-
"""Stage 4：把已核实的组件标签回填进 citations.json。

核验依据（全部来自本轮实际查询）：
- DOI 10.3389/fnbot.2022.846219 → CrossRef exist（title/journal/vol/year/authors 全对）
- DOI 10.1073/pnas.1611835114 → CrossRef：Kirkpatrick et al., PNAS 114(13):3521-3526, 2017 ✓
- DOI 10.1073/pnas.1803839115 → CrossRef + EuropePMC：Masse et al., PNAS 115(44):E10467-E10475, 2018 ✓
- DOI 10.1038/s41586-022-05340-6 → CrossRef：Boahen, Nature 612(7938):43-50, 2022 ✓
- DOI 10.1038/s41583-020-0301-7 → CrossRef：Poirazi & Papoutsi, Nat Rev Neurosci 21(6):303-321, 2020 ✓
- DOI 10.1016/j.neunet.2019.01.012 → CrossRef：Parisi et al., Neural Networks 113:54-71, 2019 ✓
- Zenodo 10.5281/zenodo.23637973 → DataCite：Zhou, Ranxi；Preprint；2026；version 1 ✓
- Zenodo 10.5281/zenodo.23602281 → DataCite：Zhou, Ranxi；Preprint；2026 ✓
- S3 (zhou2026repair) 无 DOI/URL → 自引预印本，本地仓库可核
"""
import json, os

OUT = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit_cites'

with open(os.path.join(OUT, 'citations.verified.json'), encoding='utf-8') as f:
    cits = json.load(f)

# id -> (components, best_source, note, triage)
BACK = {
    1: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='match'),
        'CrossRef 10.3389/fnbot.2022.846219 — Frontiers in Neurorobotics 16 (2022)',
        'CrossRef 标题/期刊/卷/年/6 位作者全部逐字吻合； bib 已给 DOI，核验通过。',
        'verifiable'),
    2: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='unknown'),
        'CrossRef 10.1038/s41586-022-05340-6 — Nature 612(7938):43-50 (2022)',
        'CrossRef 反查标题逐字吻合，作者/期刊/卷/年/页码全对；bib 缺 DOI 与期号（可补，非实质错误）。',
        'verifiable'),
    3: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='unknown'),
        'CrossRef 10.1038/s41583-020-0301-7 — Nature Reviews Neuroscience 21(6):303-321 (2020)',
        'CrossRef 反查标题逐字吻合，作者/期刊/卷/年/页码全对；bib 缺 DOI 与期号（可补，非实质错误）。',
        'verifiable'),
    4: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='unknown'),
        'CrossRef 10.1073/pnas.1803839115 — PNAS 115(44):E10467-E10475 (2018)',
        'CrossRef 标题/作者/卷/期全对；页码经 Europe PMC 独立确认 E10467-E10475。',
        'verifiable'),
    5: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='unknown'),
        'CrossRef 10.1073/pnas.1611835114 — PNAS 114(13):3521-3526 (2017)',
        '按 DOI 直查 CrossRef：标题/期刊/卷(114)/期(13)/页(3521-3526)/年(2017)/首三位作者全对。'
        '标题反查未命中正主，故改用 DOI 直查确认。',
        'verifiable'),
    6: (dict(title='match', authors='match', publisher='match', date='match',
             url='unknown', doi='unknown'),
        'CrossRef 10.1016/j.neunet.2019.01.012 — Neural Networks 113:54-71 (2019)',
        'CrossRef 反查标题逐字吻合，5 位作者/期刊/卷/年/页码全对。',
        'verifiable'),
    7: (dict(title='match', authors='match', publisher='match', date='match',
             url='match', doi='match'),
        'Zenodo record 23637973 (DataCite) — "Segment Specialization in Active-Dendrite '
        'Continual Learning: An Audit, an Artifact, and a Causal Repair", Zhou Ranxi, Preprint, 2026',
        'DataCite API 确认存在且状态 findable；标题、作者、出版方、年份、DOI 全对。'
        '注意：DataCite 显示 version=1，而 S1 已在本次审计后修订——引用版本号需与实际发布版本同步。',
        'verifiable'),
    8: (dict(title='match', authors='match', publisher='partial', date='match',
             url='match', doi='unknown'),
        '本地仓库 github.com/zhouranxi/delp-anchor-budget — S3_修复边界/paper/main.tex'
        '（标题与作者逐字一致；尚未有 DOI/出版方）',
        '自引预印本：标题/作者与仓库中的 S3 论文逐字一致，仓库 URL 存活。'
        '出版方为「Preprint」尚无正式 DOI —— 属未定稿引用的正常状态，非编造。',
        'verifiable'),
    9: (dict(title='match', authors='match', publisher='match', date='match',
             url='match', doi='match'),
        'Zenodo record 23602281 (DataCite) — "Where Do the Gains Come From? An Audit of '
        'Dendritic Context-Gated Protection for Continual Learning", Zhou Ranxi, Preprint, 2026',
        'DataCite API 确认存在且状态 findable；标题、作者、出版方、年份、DOI 全对。'
        'DataCite 未给 version 号（ bib 写 version 2），需与 Zenodo 页面实际版本核对。',
        'verifiable'),
}

for c in cits:
    comps, best, note, tri = BACK[c['id']]
    c['components'] = comps
    c['best_source'] = best
    c['note'] = note
    c['triage'] = tri

with open(os.path.join(OUT, 'citations.backfilled.json'), 'w', encoding='utf-8') as f:
    json.dump(cits, f, ensure_ascii=False, indent=1)
print('回填完成 ->', os.path.join(OUT, 'citations.backfilled.json'))
for c in cits:
    comp = c['components']
    bad = [f'{k}={v}' for k, v in comp.items() if v in ('not', 'partial', 'weak')]
    print(f"  {c['id']} [{c['_paper']}] {c['_seed']['title'][:45]:<47} {bad if bad else 'OK'}")
