# -*- coding: utf-8 -*-
"""Refresh the Zenodo metadata for v8.

The metadata file checked into the repository was still the v1 one (old
title, version 1, no reference to the merged manuscript).  v8 carries the
current title, the merged-manuscript related identifiers, and a description
that names the three changes this version makes.
"""
import json
import os

PKG = os.environ.get(
    'ZENODO_PKG',
    r'C:\Users\Lance\WorkBuddy\2026-10-25-10-08-27\_zenodo_v8')
META = os.path.join(PKG, 'paper', 'zenodo_metadata.json')

TITLE = ("What an Active-Dendrite Continual Learner Actually Learns: "
         "A Dead-Neuron Artifact, a Granularity Result That Did Not Survive "
         "Its Own Controls, and a Repair That Is a Damping Factor")

DESCRIPTION = (
    "This record (version 8) deposits the combined manuscript of three "
    "previously separate studies of dendritic context-gated protection in "
    "continual learning, together with the controls that were run after the "
    "first review of this line of work. The three original studies were "
    "written independently and published separately (Zenodo records 23685865 "
    "for S1, 23685385 for S3, and 23686689 for S4); this record supersedes "
    "them as a single, self-consistent narrative, and the earlier records "
    "are retained unchanged so that their DOIs continue to resolve to the "
    "documents they were issued for. The merge was necessary because the "
    "three sets of results could not all be true at once: S1 reported the "
    "own-segment bias as a causal repair of the missing segment-task "
    "specialization, S3 showed that the repair does not transfer to a "
    "convolutional backbone, and S4 showed that the mechanism is a "
    "saturation of a gate that is never context-dependent. "
    "Version 2 added five controls that a review of version 1 identified as "
    "missing rather than as errors: a specificity control with four arms "
    "showing that the bias acts on whichever segment the network is using "
    "rather than specifically on the task's own segment; a positive control "
    "in which task-dominant gating is forced by construction, which reads "
    "1.000 where the base network reads 0.055; a construction-diversity "
    "control showing that the granularity null does not survive a "
    "non-self-limiting anchor, so the general claim that sub-neuron "
    "protection granularity is worthless is withdrawn; a comparison with "
    "contemporary replay methods (ER and DER, 40 runs), which shows the "
    "audited method is not competitive with them; and equivalence testing "
    "(TOST at a 0.5-point margin) together with a metric-sensitivity "
    "analysis, replacing the manuscript's earlier reliance on failure to "
    "reject. "
    "Version 8 adds three further changes, all of them consequences of "
    "re-running the key experiments on a second machine and of fixing a "
    "statistical artefact that the re-run exposed. "
    "(i) An independent replication on a second machine re-derived the "
    "mechanism-attribution numbers with five seeds per dataset rather than "
    "three (72 of 72 arm-level comparisons agree within floating-point "
    "tolerance), and the manuscript now quotes those five-seed values, "
    "annotated as independently re-run. The most consequential number "
    "changes very little: even with the per-task gradient mask switched off, "
    "the own-segment win rate stays at chance (0.101 on MNIST, 0.102 on "
    "Fashion-MNIST), so the central negative result survives a second "
    "machine and a larger seed count. "
    "(ii) The per-phase one-sample t values reported for the signed-drift "
    "experiment were found not to be reproducible at the original three "
    "seeds: the means agree across machines to one part in ten thousand, "
    "while the t values move by several units, because the sample standard "
    "deviation of three numbers amplifies floating-point noise. They are now "
    "reported from ten seeds, and the accompanying claim is stated as a "
    "comparison of effect magnitudes (the peak per-phase displacement is 6.5 "
    "times larger under the dose) rather than as a pattern of per-phase "
    "significance. "
    "(iii) A mortality column was added to the full width-by-normalization "
    "factorial scan, giving 160 additional runs across two datasets, two "
    "widths, two normalization settings, two doses and ten seeds. This "
    "tests the dying-gate hypothesis at the level of configurations rather "
    "than within a single configuration, and the two are uncorrelated "
    "(Pearson r = -0.11, p = 0.79; Spearman rho = +0.05, p = 0.91) with the "
    "ordering locally inverted: the cells in which mortality is essentially "
    "eliminated have the smallest repair gain, negative on both datasets, "
    "while the highest-mortality cells have the largest positive gain. The "
    "causal chain described in the manuscript is therefore a "
    "within-configuration one and does not extend between configurations. "
    "All statistics are recomputed from the raw per-seed records with Holm "
    "correction within each metric family, 95% confidence intervals, and "
    "paired standardized effect sizes; the loader that assembles them was "
    "made deterministic, because several cells had been collected twice and "
    "the earlier version let directory order decide which run survived. "
    "The package contains the LaTeX source, the compiled PDF in both the "
    "Elsevier (submission) and TMLR (double-blind, anonymized) formats, the "
    "supplementary material, all figures, the bibliography, the "
    "machine-readable recomputed statistics, the scripts that produced "
    "them, and the raw records of both the original and the "
    "second-machine runs. All code, per-run records, pre-registrations and "
    "dated amendments are archived in the project repository linked in the "
    "manuscript.")

KEYWORDS = [
    'continual learning', 'catastrophic forgetting', 'active dendrites',
    'dendritic segments', 'elastic weight consolidation',
    'segment-task specialization', 'measurement validity', 'dead neurons',
    'multiple comparison correction', 'reproducibility',
    'equivalence testing', 'preprint',
]

RELATED = [
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23730915',
     'relation': 'isversionof'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23685865',
     'relation': 'references'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23685385',
     'relation': 'references'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23686689',
     'relation': 'references'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23685906',
     'relation': 'references'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23685788',
     'relation': 'references'},
    {'scheme': 'doi', 'identifier': '10.5281/zenodo.23685822',
     'relation': 'references'},
    {'scheme': 'url',
     'identifier': 'https://github.com/zhouranxi/delp-anchor-budget',
     'relation': 'issupplementto'},
]

FILE_NOTE = (
    "paper/: the Elsevier-format manuscript (main.tex, main.pdf, main.bbl), "
    "the supplementary material (supplementary.tex, supplementary.pdf), the "
    "eight figures, the bibliography, and the machine-readable statistics "
    "(stats_recomputed.json/.md, s5_analysis.json/.md, "
    "scan_mortality.json/.md, metric_sensitivity.json, tost_results.json). "
    "tmlr/: the double-blind anonymized version with its source, style files "
    "and figures. "
    "data/: the raw per-seed records, including both the original runs and "
    "the second-machine re-runs (server_*). "
    "code/: the scripts that produced them, together with the "
    "protocol-consistency checker that runs as step 0 of the build and the "
    "cross-machine comparison tool used for the replication.")


def main():
    meta = {
        'title': TITLE,
        'upload_type': 'publication',
        'publication_type': 'Preprint (submitted to TMLR; not yet accepted)',
        'publication_date': '2026-10-28',
        'version': '8',
        'description': DESCRIPTION,
        'keywords': KEYWORDS,
        'creators': [{
            'given_name': 'Ranxi',
            'family_name': 'Zhou',
            'affiliation': 'Hunan University of Chinese Medicine, Changsha, China',
            'role': 'Researcher',
        }],
        'related_identifiers': RELATED,
        'access_right': 'open',
        'license': {'id': 'cc-by-4.0'},
        'communities': [],
        'resource_type': {'general': 'publication', 'kind': 'publication'},
        'file_note': FILE_NOTE,
    }
    with open(META, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'已更新 {META}')
    print(f'  title   : {TITLE[:60]}...')
    print(f'  version : {meta["version"]}')
    print(f'  desc    : {len(DESCRIPTION)} chars')
    print(f'  related : {len(RELATED)} 条')


if __name__ == '__main__':
    main()
