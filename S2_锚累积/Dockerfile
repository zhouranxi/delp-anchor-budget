# Reproducibility container for the S2 manuscript.
#
# Build:  docker build -t delp-anchor-budget .
# Run:    docker run --rm delp-anchor-budget
#
# The container verifies that every number quoted in the manuscript is
# re-derivable from the raw per-seed records in results/.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="The Anchor Accumulation Problem"
LABEL org.opencontainers.image.description="Reproducibility package for the Pattern Recognition submission on bounded anchor sets in usage-localized elastic weight consolidation."
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

# System deps for matplotlib/Pillow; LaTeX is not needed to verify the
# numbers, only to typeset the PDF.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libfreetype6 \
        libpng16-16 \
        libjpeg62-turbo \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Source: manuscript, code, raw records and the audit/verification scripts.
COPY main.tex main.bbl refs.bib elsarticle-num.bst ./
COPY appx_mat_t10.tex appx_mat_t20.tex ./
COPY figs/ ./figs/
COPY code/ ./code/
COPY results/ ./results/
COPY prereg/ ./prereg/
COPY audit/ ./audit/
COPY highlights.txt cover_letter.md README.md ./

# Fail the build if any verification check fails.
CMD ["python", "audit/verify_all.py"]
