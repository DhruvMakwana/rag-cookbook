"""
Fetches three real papers to build a genuinely multi-document corpus --
the point of this tutorial is comparing chunking/retrieval strategies
and evaluating a real vector database against more than one toy PDF.

- "Attention Is All You Need" (Vaswani et al., 2017) -- the Transformer
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) -- GPT-3

All three are freely downloadable from arXiv, and closely related enough
(pretraining/architecture choices in NLP) that real cross-document
questions make sense against this corpus.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"

PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "gpt3.pdf": "https://arxiv.org/pdf/2005.14165",
}


def download_papers() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, url in PAPERS.items():
        dest = DATA_DIR / filename
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"Downloading {filename} from {url} ...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
        paths[filename] = dest
    return paths


if __name__ == "__main__":
    paths = download_papers()
    for name, path in paths.items():
        print(f"{name}: {path.stat().st_size / 1024:.0f} KB")
