"""Build the Zenodo v3 upload package for the S2 manuscript.

v2 (online) still ships the 28-page manuscript and the nine original audit
scripts. v3 carries everything produced in this round:

  - the 34-page manuscript (non-parametric tests in the text, the Funding
    declaration, five new references, Limitations item 6, concept DOIs)
  - 11 new audit files (verify_all, check_latex, audit_language,
    audit_nonparametric, rewrite_abstract, make_declarations_docx,
    make_graphical_abstract, rename_figures, REPRODUCE.md, ...)
  - the CIFAR policy-grid record shards that were missing from results/
  - build.sh, requirements.txt, Dockerfile (the reproducibility container)

Usage:  python audit/make_zenodo_v3.py
Output: zenodo_v3/S2_zenodo_package.zip   (upload this file)
        zenodo_v3/zenodo_v3_description.md (paste into the Zenodo form)
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "zenodo_v3")
ZIP = os.path.join(OUTDIR, "S2_zenodo_package.zip")

FILES = [
    # manuscript
    "main.pdf", "main.tex", "main.bbl", "refs.bib", "elsarticle-num.bst",
    "appx_mat_t10.tex", "appx_mat_t20.tex",
    # submission package
    "highlights.txt", "cover_letter.md", "declarations.docx",
    "graphical_abstract.png", "graphical_abstract.tiff",
    "PR_UPLOAD_CHECKLIST.md", "README.md", "TARGET_JOURNAL.md",
    "SUBMISSION_STRATEGY.md", "requirements.txt", "Dockerfile", "build.sh",
]

DIRS = [
    ("figs/", "*.pdf"),
    ("figs/", "*.png"),
    ("code/", "*.py"),
    ("prereg/", "*"),
    ("audit/", "*"),
    ("results/", "*.json*"),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    n, missing = 0, []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            src = os.path.join(ROOT, f)
            if not os.path.exists(src):
                missing.append(f)
                continue
            z.write(src, f)
            n += 1
        import glob
        for d, pat in DIRS:
            for src in glob.glob(os.path.join(ROOT, d, pat)):
                arc = os.path.relpath(src, ROOT).replace("\\", "/")
                z.write(src, arc)
                n += 1
    print(f"packaged {n} files -> {ZIP}")
    print(f"size: {os.path.getsize(ZIP)/1024:.0f} KB")
    if missing:
        print(f"MISSING (not packaged): {missing}")
    # quick integrity check: verify the new scripts are inside
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        for probe in ["audit/audit_nonparametric.py",
                      "audit/verify_all.py",
                      "audit/REPRODUCE.md",
                      "Dockerfile",
                      "results/s2r6_b_t20.json.part0",
                      "main.pdf"]:
            print(f"  {'OK ' if probe in names else 'MISSING'} {probe}")


if __name__ == "__main__":
    main()
