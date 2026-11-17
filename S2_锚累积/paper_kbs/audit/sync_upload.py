"""Sync the upload/ package from the current build (Knowledge-Based Systems).

Run after every successful build so that what sits in upload/ is exactly what
the compiled manuscript contains:

    python audit/sync_upload.py

What it does
------------
1. Copies main.pdf and supplementary.pdf into upload/.
2. Renames the manuscript's figures to the Elsevier logical naming convention
   (Figure_1, Figure_2, ...) in the order they appear in main.tex, and copies
   the supplementary-only figures as Figure_S1 / Figure_S2.
3. Removes stale artefacts that belong to a different journal's requirements
   (for KBS there is no author-vitae/photo requirement, and no graphical
   abstract requirement, so neither is uploaded).
4. Prints a final manifest with sizes so the mismatch is visible.

KBS artwork rules this implements (official guide for authors):
    "Submit each image as a separate file using a logical naming convention
     for your files (for example, Figure_1, Figure_2 etc.)."
"""
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
UPLOAD = os.path.join(ROOT, "upload")

# Files that are simply copied across.
COPY_PLAIN = ["main.pdf", "supplementary.pdf", "cover_letter.docx",
              "cover_letter.md", "cover_letter.txt", "declarations.docx",
              "highlights.docx", "highlights.txt",
              # KBS has no graphical-abstract requirement, but uploading one is
              # harmless and may help discoverability -> keep it optional.
              "graphical_abstract.png"]

# Artefacts deliberately NOT uploaded for KBS.
#   vitae.docx : author vitae / author photo is a Pattern Recognition and
#   Information Sciences requirement; the KBS guide states neither.
NOT_UPLOADED = ["vitae.docx"]


def figure_order(tex_path):
    """Return the includegraphics stems in the order they appear in the .tex."""
    with open(tex_path, encoding="utf-8") as fh:
        text = fh.read()
    blocks = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", text, re.S)
    stems = []
    for block in blocks:
        hit = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", block)
        if hit:
            stems.append(os.path.splitext(os.path.basename(hit.group(1)))[0])
    return stems


def pdf_page_count(path):
    """Page count from the pdflatex log, falling back to the PDF itself."""
    log = os.path.splitext(path)[0] + ".log"
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="ignore") as fh:
            hit = re.search(r"Output written on .*?\(\s*(\d+)\s+pages?", fh.read())
            if hit:
                return int(hit.group(1))
    return None


def refresh_page_counts(pages):
    """Keep the 'N pages' statements in the checklist / EM notes truthful."""
    if not pages:
        return
    targets = [
        os.path.join(UPLOAD, "UPLOAD_CHECKLIST.md"),
        os.path.join(UPLOAD, "EM_STEP_BY_STEP.txt"),
        os.path.join(UPLOAD, "metadata_for_em.txt"),
    ]
    for path in targets:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new = re.sub(r"(双倍行距\s*)\d+(\s*页)", rf"\g<1>{pages}\g<2>", text)
        new = re.sub(r"(正文约\s*)\d+(\s*页)", rf"\g<1>{pages}\g<2>", new)
        new = re.sub(r"(Pages:\s*)\d+(\s*double-spaced)", rf"\g<1>{pages}\g<2>", new)
        if new != text:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
            print(f"  page count refreshed in {os.path.basename(path)}: {pages}")


def main():
    os.makedirs(UPLOAD, exist_ok=True)

    for name in COPY_PLAIN:
        src = os.path.join(ROOT, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(UPLOAD, name))

    for name in NOT_UPLOADED:
        path = os.path.join(UPLOAD, name)
        if os.path.exists(path):
            os.remove(path)

    # clear previously generated figure files
    for name in os.listdir(UPLOAD):
        if re.match(r"^Figure_.*\.pdf$", name):
            os.remove(os.path.join(UPLOAD, name))

    main_stems = figure_order(os.path.join(ROOT, "main.tex"))
    supp_stems = figure_order(os.path.join(ROOT, "supplementary.tex"))

    made = []
    for i, stem in enumerate(main_stems, 1):
        src = os.path.join(FIGS, stem + ".pdf")
        if os.path.exists(src):
            dst = os.path.join(UPLOAD, f"Figure_{i}.pdf")
            shutil.copyfile(src, dst)
            made.append((f"Figure_{i}.pdf", stem))

    for i, stem in enumerate(supp_stems, 1):
        src = os.path.join(FIGS, stem + ".pdf")
        if os.path.exists(src):
            dst = os.path.join(UPLOAD, f"Figure_S{i}.pdf")
            shutil.copyfile(src, dst)
            made.append((f"Figure_S{i}.pdf", stem))

    print("=" * 68)
    print("upload/ synced for Knowledge-Based Systems")
    print("=" * 68)
    pages = pdf_page_count(os.path.join(ROOT, "main.pdf"))
    if pages:
        print(f"manuscript pages   : {pages}")
        refresh_page_counts(pages)
    print(f"manuscript figures : {len(main_stems)} -> " +
          ", ".join(f"Figure_{i}" for i in range(1, len(main_stems) + 1)))
    print(f"supplementary figs  : {len(supp_stems)} -> " +
          ", ".join(f"Figure_S{i}" for i in range(1, len(supp_stems) + 1)))
    print()
    for dst, stem in made:
        print(f"  {dst:<16} <- figs/{stem}.pdf")
    print()
    print("manifest:")
    for name in sorted(os.listdir(UPLOAD)):
        size = os.path.getsize(os.path.join(UPLOAD, name))
        print(f"  {name:<24} {size:>9,} bytes")
    print()
    print("NOT uploaded (not required by KBS): " + ", ".join(NOT_UPLOADED))
    print("graphical_abstract.png is included but OPTIONAL: the KBS guide has "
          "no graphical-abstract section at all.")


if __name__ == "__main__":
    main()
