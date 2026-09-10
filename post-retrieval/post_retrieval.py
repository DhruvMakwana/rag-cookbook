"""
Three post-retrieval techniques, each demonstrated as a live before/after
against a scenario matched to that technique's actual "when to use" case:
cross-encoder reranking, contextual compression, and LLM-based listwise
reranking (RankGPT-style) — a genuinely modern alternative to classic
cross-encoders that trades cost/latency for occasionally higher accuracy.

Cross-encoder reranking needs no LLM key (fully local). Contextual
compression and listwise reranking both need one.

Run:
    python post_retrieval.py --technique cross_encoder
    python post_retrieval.py --technique compression
    python post_retrieval.py --technique listwise
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# One scenario per technique, each matched to that technique's own
# documented "when to use" case. Keywords verified present verbatim in
# the actual paper text.
SCENARIOS = {
    "cross_encoder": {
        "question": "What is the complexity per layer for a recurrent layer compared to self-attention?",
        "keyword": "O(n·d2)",
        "candidate_pool": 15,
        "top_n": 3,
    },
    "compression": {
        "question": "What optimizer did they use for training?",
        "keyword": "Adam",
        "k": 5,
    },
    "listwise": {
        "question": "What is the complexity per layer for a recurrent layer compared to self-attention?",
        "keyword": "O(n·d2)",
        "candidate_pool": 15,
        "top_n": 3,
    },
}


# ======================================================================
# Setup / data loading
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Loads the sample PDF's text, with the front-matter boilerplate
    (permission notice + the 7-author name/affiliation/email block)
    trimmed off — same fix applied in query-transformation/, for the
    same reason: that block is single-newline-separated with no real
    paragraph structure, and none of this page's questions are about
    authorship, so keeping it in only adds noisy, low-signal chunks."""
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    abstract_start = full_text.find("Abstract")
    return full_text[abstract_start:] if abstract_start != -1 else full_text


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def naive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 5) -> list[str]:
    """Bi-encoder retrieval: query and chunks embedded independently,
    compared with cosine similarity. Fast, scalable, pre-computable —
    but query and chunk never actually attend to each other."""
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [cosine_sim(query_vec, v) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


# ======================================================================
# 1. Cross-Encoder Reranking
# ======================================================================

# --8<-- [start:cross_encoder]
def cross_encoder_rerank(question: str, candidates: list[str], top_n: int = 3) -> list[str]:
    """Bi-encoder retrieval casts a wide net (candidates); the
    cross-encoder reads query+chunk TOGETHER in one forward pass per
    pair (bidirectional attention, no pre-computation possible) and
    outputs a real relevance score per candidate — then keeps only the
    top_n. Too expensive to run over a whole corpus, fine over a
    already-narrowed candidate set."""
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return ranked[:top_n]
# --8<-- [end:cross_encoder]


# ======================================================================
# 2. Contextual Compression
# ======================================================================

# --8<-- [start:compression]
COMPRESS_PROMPT = """From the passage below, extract ONLY the sentences \
that are relevant to answering the question. Return them verbatim, \
nothing added. If nothing in the passage is relevant, return "NONE".

Question: {query}

Passage:
{chunk}"""


def contextual_compress(question: str, chunks: list[str], provider: str | None = None) -> list[str]:
    """Runs one compression call per retrieved chunk (independent of
    each other — parallelizable in a real deployment) using a cheap
    LLM call, trimming each chunk down to just the sentences relevant
    to the question. Chunks that compress to "NONE" are dropped."""
    compressed = []
    for chunk in chunks:
        result = llm.generate(COMPRESS_PROMPT.format(query=question, chunk=chunk), provider=provider).strip()
        if result and result.upper() != "NONE":
            compressed.append(result)
    return compressed
# --8<-- [end:compression]


# ======================================================================
# 3. LLM-Based Listwise Reranking (RankGPT-style)
# ======================================================================

# --8<-- [start:listwise]
LISTWISE_PROMPT = """You are ranking passages by relevance to a question.

Question: {query}

Passages:
{numbered_passages}

Think through which passages are most relevant if you like. When you \
are done, output your final ranking as passage numbers from MOST to \
LEAST relevant, wrapped in <ranking></ranking> tags, with numbers \
separated by commas and every number included exactly once.

Example: <ranking>3,1,4,2</ranking>"""


def llm_listwise_rerank(question: str, candidates: list[str], top_n: int = 3, provider: str | None = None) -> list[str]:
    """RankGPT-style listwise reranking: instead of scoring each
    candidate independently (pointwise, like a cross-encoder), the LLM
    sees the WHOLE candidate list at once and outputs a full relevance
    ordering in one call — it can reason about candidates relative to
    each other, not just each one in isolation. Costs one LLM call for
    the whole list (small lists) instead of one lightweight forward
    pass per candidate, and typically has higher latency and $ cost
    than a cross-encoder for a comparable-size candidate set.

    An earlier version of this prompt instructed the model to "return
    ONLY the list, nothing else" — the model routinely ignored that and
    explained its reasoning first, and a parser trusting the whole
    response to be just numbers picked up stray digits from that prose
    ("Table 1", "[4]") and silently corrupted the ranking. Fighting a
    model's tendency to explain itself is a losing battle; instead this
    prompt lets it reason freely and asks it to wrap ONLY the final
    answer in an unambiguous <ranking> tag — trivial to extract
    correctly regardless of how much reasoning precedes it."""
    import re

    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(candidates))
    response = llm.generate(
        LISTWISE_PROMPT.format(query=question, numbered_passages=numbered), provider=provider
    ).strip()

    match = re.search(r"<ranking>(.*?)</ranking>", response, re.S)
    order = []
    if match:
        for tok in match.group(1).replace(" ", "").split(","):
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(candidates) and idx not in order:
                    order.append(idx)
    for i in range(len(candidates)):
        if i not in order:
            order.append(i)

    return [candidates[i] for i in order[:top_n]]
# --8<-- [end:listwise]


# ======================================================================
# CLI
# ======================================================================

TECHNIQUES = ["cross_encoder", "compression", "listwise"]


def run(technique: str, provider: str | None = None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    scenario = SCENARIOS[technique]
    question = scenario["question"]

    if technique == "compression":
        k = scenario["k"]
        retrieved = naive_retrieve(question, chunks, embed_model, k)
        compressed = contextual_compress(question, retrieved, provider=provider)

        raw_chars = sum(len(c) for c in retrieved)
        compressed_chars = sum(len(c) for c in compressed)

        print(f"Question: {question}\n")
        print(f"Retrieved {len(retrieved)} chunks, {raw_chars} chars total.")
        print(f"Compressed to {len(compressed)} non-empty extracts, {compressed_chars} chars total.")
        print(f"Reduction: {100 * (1 - compressed_chars / raw_chars):.0f}%\n")
        kw = scenario["keyword"]
        print(f'"{kw}" survives compression: {any(kw in c for c in compressed)}')
        return

    candidate_pool, top_n, kw = scenario["candidate_pool"], scenario["top_n"], scenario["keyword"]
    candidates = naive_retrieve(question, chunks, embed_model, candidate_pool)
    naive_top_n = candidates[:top_n]

    if technique == "cross_encoder":
        reranked = cross_encoder_rerank(question, candidates, top_n=top_n)
    else:
        reranked = llm_listwise_rerank(question, candidates, top_n=top_n, provider=provider)

    print(f"Question: {question}\n")
    print(f'"{kw}" found in naive top-{top_n}: {any(kw in c for c in naive_top_n)}')
    print(f'"{kw}" found in {technique} top-{top_n}: {any(kw in c for c in reranked)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-retrieval technique demos.")
    parser.add_argument("--technique", choices=TECHNIQUES, default="cross_encoder")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()
    run(args.technique, provider=args.provider)
