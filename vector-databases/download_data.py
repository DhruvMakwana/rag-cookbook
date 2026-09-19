"""
Fetches a small sample PDF to build a chunk corpus from.

Default: "Attention Is All You Need" (Vaswani et al., 2017) from arXiv --
the same sample document used across most recipes in this repo, so a
metadata field like "section" or "page" on each chunk means something
real and consistent across the vector database demos on this page.
"""

import os
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"


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
