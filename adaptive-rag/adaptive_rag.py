"""
Adaptive RAG: an LLM router classifies each query before retrieval, and
the pipeline applies only the pre-retrieval technique that query's
category actually needs — instead of applying every technique to every
query (expensive) or none of them (misses genuine failure modes).

This ties together query-transformation/ and post-retrieval/ into one
end-to-end pipeline: router -> matching pre-retrieval fix (or none) ->
wide retrieval -> cross-encoder rerank -> generate. Contextual
compression is intentionally NOT wired into the default pipeline — see
the README for why; run with --compress to add it as a final step.

Run:
    python adaptive_rag.py --query "What optimizer did they use?"
    python adaptive_rag.py --compare   # runs the 5-query comparison table
"""

from __future__ import annotations

import argparse
import re
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
CANDIDATE_POOL = 15
FINAL_K = 3

# 5 demo queries, one per router category, each reusing a scenario
# already keyword-verified in query-transformation/ and this repo's
# other recipes — not re-guessed here.
COMPARISON_SET = [
    {"question": "What optimizer did they use for training?", "keyword": "Adam", "expected_category": "CLEAR"},
    {"question": "why dont they just use RNNs like everyone else did before", "keyword": "precludes parallelization", "expected_category": "VAGUE"},
    {"question": "What optimizer did they use, and how many attention heads did they use?", "keywords": ["Adam", "h = 8"], "expected_category": "COMPOUND"},
    {"question": "What lets this model connect words that are far apart in a sentence without losing track over long distances?", "keyword": "maximum path length", "expected_category": "BROAD"},
    {"question": "What value did they use for Pdrop during training?", "keyword": "Pdrop", "expected_category": "NEEDS_CONTEXT"},
]


# ======================================================================
# Setup / data loading
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Same front-matter trim applied in query-transformation/ and
    post-retrieval/, for the same reason: the author/affiliation block
    is noisy, low-signal, single-newline text with no real paragraph
    structure, and irrelevant to every question this pipeline is asked."""
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


def bi_encoder_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int) -> list[str]:
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [cosine_sim(query_vec, v) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


def naive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = FINAL_K) -> list[str]:
    """The baseline every comparison in this repo measures against:
    embed the raw query, retrieve top-k, nothing else."""
    return bi_encoder_retrieve(question, chunks, embed_model, k)


# ======================================================================
# 1. Router
# ======================================================================

# --8<-- [start:router]
ROUTER_PROMPT = """Classify the query below into EXACTLY ONE category, \
based on what problem (if any) it poses for retrieval against a \
technical document. Categories, each with an example:

CLEAR - a single specific fact that is a NAMED thing (an algorithm \
name, a hardware type, a dataset name) — not a numeric parameter value \
— already phrased using the kind of technical vocabulary the document \
itself would use.
  Example: "What optimizer did they use for training?"

VAGUE - a single specific fact whose complete answer is ONE short \
thing, but the question is phrased colloquially, avoiding the \
technical vocabulary the document actually uses.
  Example: "why dont they just use the old-school approach everyone \
used before"

COMPOUND - explicitly asks for two or more distinct, independently \
answerable facts, usually joined by "and".
  Example: "What optimizer did they use, and how many layers does it have?"

BROAD - one open-ended question whose COMPLETE answer requires citing \
several distinct reasons or mechanisms (not one short fact) — decide \
based on whether a full answer needs multiple points, regardless of \
whether the wording sounds casual or technical.
  Example: "What lets this model handle relationships between things \
that are positioned far apart, without losing track over long spans?"

NEEDS_CONTEXT - asks for one narrow NUMERIC or configuration parameter \
value (a rate, a dimension, a count) where knowing the broader \
principle/section it belongs to would help interpret or locate the \
answer.
  Example: "What value did they use for the regularization strength?"

Query: {query}

Respond with ONLY the category name, wrapped in <category></category> tags."""


def route_query(question: str, provider: str | None = None) -> str:
    """One LLM call classifies the query so the pipeline can apply only
    the pre-retrieval fix that category actually needs, instead of
    running every technique on every query (expensive, and — as
    measured on the pre-retrieval page — often actively counterproductive
    on queries that don't have the failure mode a technique targets)."""
    response = llm.generate(ROUTER_PROMPT.format(query=question), provider=provider)
    match = re.search(r"<category>(.*?)</category>", response, re.S)
    category = match.group(1).strip() if match else "CLEAR"
    return category if category in {"CLEAR", "VAGUE", "COMPOUND", "BROAD", "NEEDS_CONTEXT"} else "CLEAR"
# --8<-- [end:router]


# ======================================================================
# 2. Pre-retrieval technique implementations
# (same prompts as query-transformation/, copied in per this repo's
# self-contained-recipe convention)
# ======================================================================

REWRITE_PROMPT = """Rewrite this query to be clearer and more specific \
for searching a technical document. Return ONLY the rewritten query, \
nothing else.

Query: {query}"""

MULTI_QUERY_PROMPT = """Generate 3 different phrasings of the question \
below, each capturing a different angle on the same underlying \
information need. Return one per line, nothing else.

Question: {query}"""

STEP_BACK_PROMPT = """What is a more general question that would help \
answer this specific question? Return ONLY the general question.

Specific question: {query}"""

DECOMPOSE_PROMPT = """If the question below is asking multiple distinct \
things, break it into separate sub-questions, one per line. If it's \
already a single question, return it unchanged on one line.

Question: {query}"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def apply_pre_retrieval(
    category: str, question: str, chunks: list[str], embed_model: SentenceTransformer, provider: str | None = None
) -> tuple[list[str], str]:
    """Routes to the candidate-gathering step for whichever technique
    the router picked. Returns (CANDIDATE POOL, rerank_query) — the pool
    isn't the final top-k yet (reranking narrows it down after this),
    and rerank_query is the query the reranker should score against.

    That second part matters more than it looks: a cross-encoder is
    just another model reading text, so it suffers the SAME
    vocabulary-mismatch problem a bi-encoder does. Reranking a VAGUE
    query's candidates against the original colloquial wording throws
    away the exact fix rewriting just made — the reranker ends up
    scoring every candidate as a poor match, including the correct one,
    because it never sees the rewritten, technically-phrased query.
    So VAGUE reranks against the REWRITTEN query, not the original."""
    if category == "VAGUE":
        rewritten = llm.generate(REWRITE_PROMPT.format(query=question), provider=provider).strip()
        return bi_encoder_retrieve(rewritten, chunks, embed_model, CANDIDATE_POOL), rewritten

    if category == "COMPOUND":
        # Reranking this pool against the single merged question — instead
        # of per sub-question — collapses it right back to one topic,
        # undoing decomposition's whole guarantee that each sub-question
        # gets independent representation. Handled specially in
        # adaptive_retrieve: COMPOUND reranks (and allocates slots) per
        # sub-question, not as one merged pool.
        response = llm.generate(DECOMPOSE_PROMPT.format(query=question), provider=provider)
        sub_questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        per_question_pool = max(3, CANDIDATE_POOL // len(sub_questions))
        candidate_sets = {sub_q: bi_encoder_retrieve(sub_q, chunks, embed_model, per_question_pool) for sub_q in sub_questions}
        return candidate_sets, question

    if category == "BROAD":
        response = llm.generate(MULTI_QUERY_PROMPT.format(query=question), provider=provider)
        variations = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        variations.append(question)
        ranked_lists = [bi_encoder_retrieve(v, chunks, embed_model, CANDIDATE_POOL) for v in variations]
        return reciprocal_rank_fusion(ranked_lists)[:CANDIDATE_POOL], question

    if category == "NEEDS_CONTEXT":
        general_question = llm.generate(STEP_BACK_PROMPT.format(query=question), provider=provider).strip()
        specific_hits = bi_encoder_retrieve(question, chunks, embed_model, CANDIDATE_POOL)
        general_hits = bi_encoder_retrieve(general_question, chunks, embed_model, CANDIDATE_POOL)
        combined, seen = [], set()
        for chunk in specific_hits + general_hits:
            if chunk not in seen:
                combined.append(chunk)
                seen.add(chunk)
        return combined[:CANDIDATE_POOL], question

    # CLEAR: no transformation needed, straight bi-encoder retrieval
    return bi_encoder_retrieve(question, chunks, embed_model, CANDIDATE_POOL), question


# ======================================================================
# 3. Post-retrieval: cross-encoder rerank (always on), compression (opt-in)
# ======================================================================

_reranker: CrossEncoder | None = None


def cross_encoder_rerank(question: str, candidates: list[str], top_n: int = FINAL_K) -> list[str]:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    scores = _reranker.predict([(question, c) for c in candidates])
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return ranked[:top_n]


COMPRESS_PROMPT = """From the passage below, extract ONLY the sentences \
that are relevant to answering the question. Return them verbatim, \
nothing added. If nothing in the passage is relevant, return "NONE".

Question: {query}

Passage:
{chunk}"""


def contextual_compress(question: str, chunks: list[str], provider: str | None = None) -> list[str]:
    compressed = []
    for chunk in chunks:
        result = llm.generate(COMPRESS_PROMPT.format(query=question, chunk=chunk), provider=provider).strip()
        if result and result.upper() != "NONE":
            compressed.append(result)
    return compressed


# ======================================================================
# 4. The adaptive pipeline
# ======================================================================

# --8<-- [start:pipeline]
def adaptive_retrieve(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    compress: bool = False,
) -> dict:
    """Router -> matching pre-retrieval technique (or none) -> wide
    candidate pool -> cross-encoder rerank -> optional compression.
    Returns the category the router picked alongside the final chunks,
    so a caller can see WHY the pipeline did what it did."""
    category = route_query(question, provider=provider)
    candidates, rerank_query = apply_pre_retrieval(category, question, chunks, embed_model, provider=provider)
    pool_text = " ".join(c for pool in candidates.values() for c in pool) if category == "COMPOUND" else " ".join(candidates)

    if category == "COMPOUND":
        # candidates is {sub_question: pool} here — rerank EACH
        # sub-question's own pool against ITS OWN question, and give
        # each sub-question a guaranteed, equal slice of the final
        # budget, instead of one merged rerank that would silently
        # let one sub-topic crowd out the other.
        # A compound question genuinely needs more total slots than a
        # single-fact one — floor-dividing the single-fact FINAL_K
        # across sub-questions starves each one down to 1 slot, which
        # isn't enough margin for the reranker to reliably keep the
        # right chunk. Guarantee at least 2 slots per sub-question instead.
        sub_questions = list(candidates.keys())
        per_question_final = max(2, FINAL_K // len(sub_questions))
        reranked, seen = [], set()
        for sub_q in sub_questions:
            for chunk in cross_encoder_rerank(sub_q, candidates[sub_q], top_n=per_question_final):
                if chunk not in seen:
                    reranked.append(chunk)
                    seen.add(chunk)
    else:
        reranked = cross_encoder_rerank(rerank_query, candidates, top_n=FINAL_K)

    final_chunks = contextual_compress(question, reranked, provider=provider) if compress else reranked

    return {"category": category, "chunks": final_chunks, "pool_text": pool_text}
# --8<-- [end:pipeline]


# ======================================================================
# CLI
# ======================================================================


def run_single(question: str, provider: str | None, compress: bool) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    result = adaptive_retrieve(question, chunks, embed_model, provider=provider, compress=compress)
    print(f"Question: {question}")
    print(f"Router category: {result['category']}\n")
    for i, c in enumerate(result["chunks"]):
        print(f"--- chunk {i} ---")
        print(c[:300])
        print()


def run_comparison(provider: str | None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"{'Question':<55} {'Category':<14} {'Naive':<7} {'Pool':<7} {'Final':<7}")
    print("-" * 92)
    for item in COMPARISON_SET:
        question = item["question"]
        keywords = item.get("keywords") or [item["keyword"]]

        naive_hits = naive_retrieve(question, chunks, embed_model, k=FINAL_K)
        naive_text = " ".join(naive_hits)
        naive_ok = all(kw in naive_text for kw in keywords)

        result = adaptive_retrieve(question, chunks, embed_model, provider=provider)
        pool_ok = all(kw in result["pool_text"] for kw in keywords)
        final_text = " ".join(result["chunks"])
        final_ok = all(kw in final_text for kw in keywords)

        label = question if len(question) <= 53 else question[:50] + "..."
        print(f"{label:<55} {result['category']:<14} {str(naive_ok):<7} {str(pool_ok):<7} {str(final_ok):<7}")
    print()
    print("Naive = raw-query bi-encoder top-3. Pool = target fact present ANYWHERE")
    print("in the router-selected candidate pool, before reranking. Final = present")
    print("in what the pipeline actually returns, after cross-encoder reranking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG pipeline demo.")
    parser.add_argument("--query", type=str, default=None, help="Run the pipeline on a single query.")
    parser.add_argument("--compare", action="store_true", help="Run the 5-query naive-vs-adaptive comparison.")
    parser.add_argument("--compress", action="store_true", help="Add contextual compression as a final step.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.provider)
    else:
        query = args.query or "What optimizer did they use for training?"
        run_single(query, args.provider, args.compress)
