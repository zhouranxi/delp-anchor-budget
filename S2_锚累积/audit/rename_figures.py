"""Create PR-style figure files (Figure_1 ... Figure_5) alongside the originals.

Pattern Recognition asks for logically named, separately uploaded artwork
("Figure_1", "Figure_2", ...). This script copies (does not move) each figure
into that naming scheme, keeping the originals intact so main.tex needs no
change and the source tree stays self-consistent.

Run from the package root:  python audit/rename_figures.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")

# main.tex includegraphics order -> (stem, PR name)
MAPPING = [
    ("fig1_pareto", "Figure_1"),
    ("fig2_ageprofile", "Figure_2"),
    ("fig3_criteria", "Figure_3"),
    ("fig4_stacktrace", "Figure_4"),
    ("fig5_cifar_stacktrace", "Figure_5"),
]

if __name__ == "__main__":
    made = []
    for stem, pr_name in MAPPING:
        for ext in ("pdf", "png"):
            src = os.path.join(FIGS, f"{stem}.{ext}")
            dst = os.path.join(FIGS, f"{pr_name}.{ext}")
            if not os.path.exists(src):
                print(f"  [skip] missing {src}")
                continue
            shutil.copyfile(src, dst)
            made.append(os.path.basename(dst))

    print("=" * 64)
    print("PR-style figure copies created")
    print("=" * 64)
    for f in made:
        print(f"  {f}")
    print(f"\n{len(made)} files. Upload these (PDF preferred for vector art);")
    print("originals (fig*.pdf/png) kept in place so main.tex is unchanged.")
