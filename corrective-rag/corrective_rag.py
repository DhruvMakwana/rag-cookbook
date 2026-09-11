"""
Corrective RAG (CRAG): a lightweight evaluator grades the whole
retrieved candidate set in one batched call, and the outcome decides
what happens next — not a single confidence gate, but three distinct
corrective actions:

  Correct   -> refine: split into fine-grained "knowledge strips",
               re-score each one, keep only the strips that hold up
  Incorrect -> discard the retrieved chunks entirely, rewrite the
               query into search-engine keywords, and fall back to a
               real web search (Tavily)
  Ambiguous -> do both: refined internal strips AND web search results,
               combined before generation

Run:
    python corrective_rag.py --query "What optimizer did they use?"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVE_K = 5

# Evaluator scores each retrieved chunk 1-5. These thresholds turn the
# k scores into one 3-way decision — same spirit as the CRAG paper's
# upper/lower confidence bounds, just on a 1-5 scale instead of a
# continuous confidence score.
UPPER_THRESHOLD = 4  # max score >= this -> Correct
LOWER_THRESHOLD = 2  # max score <= this -> Incorrect


# ======================================================================
# Setup / data loading
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Same front-matter trim applied in every other recipe in this
    repo — the author/affiliation block is noisy, single-newline text
    irrelevant to every question asked here."""
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


def naive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = RETRIEVE_K) -> list[str]:
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


# ======================================================================
# 1. Retrieval evaluator (one batched call, not k round-trips)
# ======================================================================

# --8<-- [start:evaluator]
EVALUATOR_PROMPT = """Score how well each passage below helps answer the \
question, on a scale of 1 (irrelevant) to 5 (directly and fully answers \
it). Score EVERY passage, even if most seem irrelevant.

Question: {question}

Passages:
{numbered_passages}

Respond with ONLY the scores, one per passage in order, comma-separated, \
wrapped in <scores></scores> tags. Example for 3 passages: <scores>2,5,1</scores>"""


def evaluate_retrieval(question: str, chunks: list[str], provider: str | None = None) -> list[int]:
    """One LLM call scores the WHOLE candidate set at once — same
    "batch it, don't loop it" principle as listwise reranking. Falls
    back to a score of 1 for any chunk the model doesn't return a
    parseable score for, so a malformed response degrades toward
    "treat as irrelevant" rather than crashing or silently misaligning
    scores to the wrong chunks."""
    numbered = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    response = llm.generate(EVALUATOR_PROMPT.format(question=question, numbered_passages=numbered), provider=provider)
    match = re.search(r"<scores>(.*?)</scores>", response, re.S)

    scores = [1] * len(chunks)
    if match:
        parts = match.group(1).replace(" ", "").split(",")
        for i, part in enumerate(parts[: len(chunks)]):
            if part.isdigit():
                scores[i] = max(1, min(5, int(part)))
    return scores


def classify(scores: list[int]) -> str:
    top = max(scores) if scores else 1
    if top >= UPPER_THRESHOLD:
        return "Correct"
    if top <= LOWER_THRESHOLD:
        return "Incorrect"
    return "Ambiguous"
# --8<-- [end:evaluator]


# ======================================================================
# 2. Knowledge refinement (the "Correct" branch)
# ======================================================================

# --8<-- [start:refine]
STRIP_RELEVANCE_PROMPT = """Score how well each short passage below \
helps answer the question, on a scale of 1 (irrelevant) to 5 (directly \
answers it).

Question: {question}

Passages:
{numbered_strips}

Respond with ONLY the scores, comma-separated, wrapped in <scores></scores> \
tags. Example for 3 passages: <scores>2,5,1</scores>"""


def refine_knowledge(question: str, chunks: list[str], provider: str | None = None, keep_threshold: int = 3) -> list[str]:
    """Even a chunk that scored well overall can have sentences that
    don't actually help — split into strips (sentences), re-score each
    one at this finer grain, and keep only the strips that individually
    clear the bar. Same idea as contextual compression, but gated
    specifically on the evaluator's "Correct" verdict rather than
    applied unconditionally to every retrieval."""
    strips = [s.strip() for c in chunks for s in re.split(r"(?<=[.!?])\s+", c) if len(s.strip()) > 20]
    if not strips:
        return chunks

    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(strips))
    response = llm.generate(STRIP_RELEVANCE_PROMPT.format(question=question, numbered_strips=numbered), provider=provider)
    match = re.search(r"<scores>(.*?)</scores>", response, re.S)

    if not match:
        return chunks

    parts = match.group(1).replace(" ", "").split(",")
    kept = [strips[i] for i, part in enumerate(parts[: len(strips)]) if part.isdigit() and int(part) >= keep_threshold]
    return kept if kept else chunks
# --8<-- [end:refine]


# ======================================================================
# 3. Web search fallback (the "Incorrect" branch)
# ======================================================================

# --8<-- [start:websearch]
SEARCH_REWRITE_PROMPT = """Rewrite the question below into a short, \
keyword-style web search query — the way you'd type it into a search \
engine, not a full sentence. Return ONLY the search query.

Question: {question}"""


def rewrite_for_search(question: str, provider: str | None = None) -> str:
    return llm.generate(SEARCH_REWRITE_PROMPT.format(question=question), provider=provider).strip()


def web_search(query: str, max_results: int = 3) -> list[str]:
    import os

    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Get a free key at https://tavily.com and add it to .env.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    return [f"{r['title']}: {r['content']}" for r in response.get("results", [])]
# --8<-- [end:websearch]


# ======================================================================
# 4. Generate
# ======================================================================

GENERATE_PROMPT = """Answer the question using the context below. If \
context from the web is included, you may cite it as such.

Context:
{context}

Question: {question}"""


def generate_answer(question: str, chunks: list[str], provider: str | None = None) -> str:
    context = "\n---\n".join(chunks) if chunks else "(no context available)"
    return llm.generate(GENERATE_PROMPT.format(context=context, question=question), provider=provider)


# ======================================================================
# 5. The full pipeline
# ======================================================================

# --8<-- [start:pipeline]
def corrective_rag(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    k: int = RETRIEVE_K,
) -> dict:
    """Retrieve -> evaluate the whole candidate set in one call ->
    branch on the verdict -> generate. Each branch is a genuinely
    different corrective ACTION, not just a different confidence
    label."""
    retrieved = naive_retrieve(question, chunks, embed_model, k)
    scores = evaluate_retrieval(question, retrieved, provider=provider)
    verdict = classify(scores)

    trace = {"retrieved": len(retrieved), "scores": scores, "verdict": verdict}

    if verdict == "Correct":
        context = refine_knowledge(question, retrieved, provider=provider)
        trace["source"] = "internal (refined)"

    elif verdict == "Incorrect":
        search_query = rewrite_for_search(question, provider=provider)
        context = web_search(search_query)
        trace["source"] = "web"
        trace["search_query"] = search_query

    else:  # Ambiguous
        refined = refine_knowledge(question, retrieved, provider=provider)
        search_query = rewrite_for_search(question, provider=provider)
        web_results = web_search(search_query)
        context = refined + web_results
        trace["source"] = "internal (refined) + web"
        trace["search_query"] = search_query

    trace["context"] = context
    trace["answer"] = generate_answer(question, context, provider=provider)
    return trace
# --8<-- [end:pipeline]


# ======================================================================
# CLI
# ======================================================================


def run(question: str, provider: str | None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    result = corrective_rag(question, chunks, embed_model, provider=provider)
    print(f"Question: {question}\n")
    print(f"Evaluator scores (1-5 per retrieved chunk): {result['scores']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Source used: {result['source']}")
    if "search_query" in result:
        print(f"Web search query: {result['search_query']}")
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrective RAG (CRAG) demo.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()
    run(args.query, args.provider)
