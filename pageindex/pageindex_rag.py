"""
PageIndex: vectorless, reasoning-based retrieval. No embeddings, no vector
database — a document's actual structure (headers, sections, table of
contents) becomes a tree, and an LLM navigates that tree to find the
right section, the way a person flips to the right chapter instead of
scanning for similar-looking text.

Uses the real `pageindex` package (VectifyAI), local mode: documents are
indexed on this machine, using your own LLM keys (Anthropic here, via
LiteLLM's `anthropic/` model prefix) rather than PageIndex's cloud.

Compared against this repo's usual naive text RAG baseline (chunk, embed,
cosine-similarity search) — but on a different sample document than every
other recipe here. "Attention Is All You Need" has a shallow structure
(~7 top-level sections, at most one level of sub-numbering) that wouldn't
exercise what this technique is actually built for. This recipe instead
uses NIST's "AI Risk Management Framework (AI RMF 1.0)" — a real,
48-page regulatory document with genuine multi-level structure (Parts ->
sections -> subsections -> numbered categories, 4 levels deep).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

import llm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "nist_ai_rmf.pdf"
DEFAULT_PDF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LiteLLM's `anthropic/` prefix, since local-mode `pageindex` routes model
# calls through LiteLLM rather than a bare model name. A cheap model is
# enough for tree construction (structure detection + node summaries);
# the chat/reasoning-over-the-tree step gets the more capable model.
INDEX_MODEL = "anthropic/claude-haiku-4-5"
CHAT_MODEL = "anthropic/claude-sonnet-5"

DOC_ID_CACHE = Path(__file__).parent / ".pageindex_doc_id"

# A small, custom eval set for this specific document -- not the 8-question
# set reused elsewhere in this repo, since that set is about the "Attention
# Is All You Need" paper. Each keyword was checked directly against the
# PDF's real page text before being used here.
NIST_EVAL_SET = [
    ("What does the AI RMF's discussion of Risk Tolerance say it's adapted from?", "ISO GUIDE 73"),
    ("How is the GOVERN function described relative to the other AI RMF functions?", "cross-cutting"),
    ("What does MEASURE 2.7 evaluate?", "security and resilience"),
    ("According to Appendix B, are AI risks comprehensively addressed by current risk frameworks and approaches?", "not comprehensively addressed"),
    ("Is it easy to tell whether an opaque AI system is accurate, private, secure, or fair?", "difficult to determine"),
]


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def download_sample_pdf(url: str = DEFAULT_PDF_URL, dest: Path = DEFAULT_PDF_PATH) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


# ======================================================================
# Naive text RAG baseline -- same pattern used across this repo
# ======================================================================


def load_document_text() -> str:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_document_text())


def build_text_index(chunks: list[str], text_model: SentenceTransformer) -> np.ndarray:
    return text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def text_search(query: str, chunks: list[str], chunk_embeddings: np.ndarray, text_model: SentenceTransformer, k: int = 3) -> list[tuple[str, float]]:
    query_embedding = text_model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(k, len(chunks)))
    return [(chunks[i], sims[i].item()) for i in top_k.indices.tolist()]


# ======================================================================
# PageIndex -- vectorless, reasoning-based retrieval
# ======================================================================


def get_pageindex_client():
    from pageindex import PageIndexClient

    return PageIndexClient(index_model=INDEX_MODEL, chat_model=CHAT_MODEL, storage_path="./.pageindex")


def index_document(client, pdf_path: Path = DEFAULT_PDF_PATH) -> str:
    """Build the document's tree once and cache the doc_id locally, so
    re-running this script doesn't re-index (and re-pay for) the same
    document every time."""
    if DOC_ID_CACHE.exists():
        return DOC_ID_CACHE.read_text().strip()

    result = client.submit_document(str(pdf_path))
    doc_id = result["doc_id"]
    DOC_ID_CACHE.write_text(doc_id)
    return doc_id


def pageindex_search(question: str, client, doc_id: str) -> tuple[str, list[int]]:
    """Retrieval and generation happen together here -- the LLM navigates
    the tree, pulls the section(s) it decides are relevant, and answers,
    all in one `chat()` call. `citations=True` returns which pages it
    actually used, parsed out of the answer text."""
    import re

    answer = client.chat(question, doc_id=doc_id, citations=True)
    pages = sorted({int(m) for m in re.findall(r'<cite doc="[^"]*" page="(\d+)"', answer)})
    clean_answer = re.sub(r'\s*<cite doc="[^"]*" page="\d+"/>', "", answer)
    return clean_answer, pages


# ======================================================================
# Comparison
# ======================================================================


def run_comparison() -> None:
    load_environment()

    print("=== Naive text RAG ===")
    chunks = chunk_sample_text()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    chunk_embeddings = build_text_index(chunks, text_model)
    naive_hits = 0
    for question, keyword in NIST_EVAL_SET:
        start = time.perf_counter()
        results = text_search(question, chunks, chunk_embeddings, text_model, k=3)
        elapsed = time.perf_counter() - start
        hit = any(keyword.lower() in chunk.lower() for chunk, _ in results)
        naive_hits += hit
        print(f"  {'PASS' if hit else 'FAIL'}  {elapsed:.3f}s  {question[:60]}")

    print("\n=== PageIndex ===")
    client = get_pageindex_client()
    doc_id = index_document(client)
    pageindex_hits = 0
    for question, keyword in NIST_EVAL_SET:
        start = time.perf_counter()
        answer, pages = pageindex_search(question, client, doc_id)
        elapsed = time.perf_counter() - start
        hit = keyword.lower() in answer.lower()
        pageindex_hits += hit
        print(f"  {'PASS' if hit else 'FAIL'}  {elapsed:5.1f}s  pages={pages}  {question[:50]}")

    n = len(NIST_EVAL_SET)
    print(f"\nNaive text RAG: {naive_hits}/{n} correct")
    print(f"PageIndex:      {pageindex_hits}/{n} correct")


def run_query(mode: str, question: str) -> None:
    load_environment()

    if mode == "pageindex":
        client = get_pageindex_client()
        doc_id = index_document(client)
        answer, pages = pageindex_search(question, client, doc_id)
        print(f"Question: {question}\n")
        print(f"Pages used: {pages}\n")
        print(f"Answer: {answer}")
    else:
        chunks = chunk_sample_text()
        text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        chunk_embeddings = build_text_index(chunks, text_model)
        results = text_search(question, chunks, chunk_embeddings, text_model, k=3)
        print(f"Question: {question}\n")
        for chunk, score in results:
            print(f"[TEXT] (score {score:.2f}) {chunk[:150]}")
        print()
        context = "\n---\n".join(chunk for chunk, _ in results)
        answer = llm.generate(f"Answer using the context below.\n\nContext:\n{context}\n\nQuestion: {question}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PageIndex demo.")
    parser.add_argument("--mode", choices=["pageindex", "naive"], default="pageindex")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="Naive text RAG vs. PageIndex, across a custom eval set for this document.")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    elif args.query:
        run_query(args.mode, args.query)
    else:
        parser.print_help()
