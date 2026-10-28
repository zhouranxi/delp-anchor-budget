# -*- coding: utf-8 -*-
"""Stage 3-4 的确定性部分：
1) Zenodo DOI → DataCite API 查存在性 + 抓真实元数据
2) 6 条无 DOI 的期刊文献 → CrossRef 按标题反查，核对标题/作者/期刊/年/卷/页
"""
import json, urllib.request, urllib.parse, time, sys, os

OUT = r'C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit_cites'
UA = 'WorkBuddy-cite-audit/1.0 (mailto:zhouranxi@foxmail.com)'


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8', 'replace')


def datacite(doi):
    url = f'https://api.datacite.org/dois/{urllib.parse.quote(doi)}'
    try:
        st, body = fetch(url)
        if st != 200:
            return {'found': False, 'status': st, 'reason': f'HTTP {st}'}
        d = json.loads(body)['data']['attributes']
        return {
            'found': True,
            'doi': d.get('doi'),
            'title': (d.get('titles') or [{}])[0].get('title'),
            'creators': [c.get('name') for c in (d.get('creators') or [])],
            'publisher': d.get('publisher'),
            'publication_year': d.get('publicationYear'),
            'types': (d.get('types') or {}).get('resourceTypeGeneral'),
            'url': d.get('url'),
            'version': d.get('version'),
            'state': d.get('state'),
            'related': [r.get('relationType') + '|' + r.get('relatedIdentifier')
                        for r in (d.get('relatedIdentifiers') or [])],
        }
    except Exception as e:
        return {'found': False, 'reason': f'{type(e).__name__}: {e}'}


def crossref_title(title):
    q = urllib.parse.urlencode({'query.bibliographic': title, 'rows': 3,
                                'select': 'DOI,title,author,container-title,issued,volume,issue,page,type,publisher'})
    url = f'https://api.crossref.org/works?{q}'
    try:
        st, body = fetch(url)
        items = json.loads(body)['message']['items']
        out = []
        for it in items:
            out.append({
                'doi': it.get('DOI'),
                'title': (it.get('title') or [''])[0],
                'authors': [' '.join(filter(None, [a.get('given'), a.get('family')]))
                            for a in (it.get('author') or [])],
                'container': (it.get('container-title') or [''])[0],
                'year': (it.get('issued', {}).get('date-parts') or [[None]])[0][0],
                'volume': it.get('volume'), 'issue': it.get('issue'),
                'page': it.get('page'), 'type': it.get('type'),
                'publisher': it.get('publisher'),
            })
        return out
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {e}'}]


print('=' * 80)
print('A. Zenodo DOI → DataCite 核验')
print('=' * 80)
zen = {}
for doi in ('10.5281/zenodo.23637973', '10.5281/zenodo.23602281', '10.5281/zenodo.23668523'):
    r = datacite(doi)
    zen[doi] = r
    print(f'\n--- {doi} ---')
    if not r['found']:
        print('  NOT FOUND:', r.get('reason'))
    else:
        print(f"  title     : {r['title']}")
        print(f"  creators  : {r['creators']}")
        print(f"  publisher : {r['publisher']}")
        print(f"  year      : {r['publication_year']}   version={r['version']}  state={r['state']}")
        print(f"  type      : {r['types']}")
        print(f"  url       : {r['url']}")
        print(f"  related   : {r['related']}")

print('\n' + '=' * 80)
print('B. 期刊文献 → CrossRef 按标题反查')
print('=' * 80)

BIB = [
    dict(key='iyer2022', doi='10.3389/fnbot.2022.846219',
         title='Avoiding catastrophe: Active dendrites enable multi-task learning in dynamic environments',
         authors='Iyer; Grewal; Velu; Souza; Forest; Ahmad',
         journal='Frontiers in Neurorobotics', year='2022', vol='16', pages=None),
    dict(key='boahen2022', doi=None,
         title='Dendrocentric learning for synthetic intelligence',
         authors='Boahen', journal='Nature', year='2022', vol='612', pages='43-50'),
    dict(key='poirazi2020', doi=None,
         title='Illuminating dendritic function with computational models',
         authors='Poirazi; Papoutsi', journal='Nature Reviews Neuroscience', year='2020', vol='21', pages='303-321'),
    dict(key='masse2018', doi=None,
         title='Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization',
         authors='Masse; Grant; Freedman', journal='Proceedings of the National Academy of Sciences',
         year='2018', vol='115', issue='44', pages='E10467-E10475'),
    dict(key='kirkpatrick2017', doi=None,
         title='Overcoming catastrophic forgetting in neural networks',
         authors='Kirkpatrick; Pascanu; Rabinowitz; et al.',
         journal='Proceedings of the National Academy of Sciences', year='2017', vol='114', issue='13', pages='3521-3526'),
    dict(key='parisi2019', doi=None,
         title='Continual lifelong learning with neural networks: A review',
         authors='Parisi; Kemker; Part; Kanan; Wermter',
         journal='Neural Networks', year='2019', vol='113', pages='54-71'),
]

results = {}
for b in BIB:
    print(f'\n=== {b["key"]} ===')
    print(f'  bib: {b["title"]}')
    hits = crossref_title(b['title'])
    if hits and 'error' in hits[0]:
        print('  ERR:', hits[0]['error']); results[b['key']] = hits; continue
    for h in hits[:2]:
        print(f"  hit DOI={h.get('doi')}")
        print(f"      title : {h.get('title')}")
        print(f"      jour  : {h.get('container')}")
        print(f"      year  : {h.get('year')}  vol={h.get('volume')} issue={h.get('issue')} page={h.get('page')}")
        print(f"      auth  : {h.get('authors')[:6]}")
    results[b['key']] = hits
    time.sleep(0.5)

with open(os.path.join(OUT, 'lookup_results.json'), 'w', encoding='utf-8') as f:
    json.dump({'datacite': zen, 'crossref': results}, f, ensure_ascii=False, indent=1)
print('\nsaved lookup_results.json')
