"""
Fetches the sample document this recipe indexes: NIST's "Artificial
Intelligence Risk Management Framework (AI RMF 1.0)" -- a real, free,
48-page regulatory document with genuine multi-level structure (Parts ->
sections -> subsections -> numbered categories).

Not the "Attention Is All You Need" PDF used elsewhere in this repo.
That paper has a shallow structure (~7 top-level sections, at most one
level of sub-numbering) -- it wouldn't demonstrate PageIndex's actual
advantage, which depends on a document deep and well-organized enough
that navigating to the right section beats keyword/semantic similarity.
"""

import os
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
DEFAULT_PDF_PATH = DATA_DIR / "nist_ai_rmf.pdf"


def download_sample_pdf(url: str = DEFAULT_PDF_URL, dest: Path = DEFAULT_PDF_PATH) -> Path:
    """Download the sample PDF if it isn't already cached locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached PDF: {dest}")
        return dest

    print(f"Downloading sample PDF from {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


if __name__ == "__main__":
    pdf_url = os.environ.get("SAMPLE_PDF_URL", DEFAULT_PDF_URL)
    download_sample_pdf(pdf_url)
