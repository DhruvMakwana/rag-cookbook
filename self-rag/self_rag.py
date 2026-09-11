"""
A prompted Self-RAG loop: retrieve, grade each chunk's relevance,
generate from only the relevant ones, grade the answer's groundedness
(is it actually supported by what was retrieved?) and usefulness (does
it actually address the question?), and either return the answer or
regenerate once before falling back to an honest "insufficient
support" response.

Relevance and usefulness grading are LLM calls (structured yes/no via
tags). Groundedness grading uses a local NLI (Natural Language
Inference) cross-encoder instead of an LLM call — premise = the
relevant retrieved context, hypothesis = the generated answer,
checking for entailment. This is the one piece not in the original
Self-RAG paper or most write-ups: an NLI model is meaningfully cheaper
than an LLM-judge call for this specific check (local, ~200-500ms vs.
a full LLM round-trip) and, per recent benchmarking, comparably
accurate for QA-over-retrieved-context faithfulness checks specifically
(it's a worse fit for open-ended generation tasks with no clear source
document).

Run:
    python self_rag.py --query "What optimizer did they use for training?"
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
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
RETRIEVE_K = 5
MAX_REGENERATIONS = 1


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
# 1. Grade relevance (LLM, per chunk)
# ======================================================================

# --8<-- [start:relevance]
RELEVANCE_PROMPT = """Is the passage below relevant to answering the \
question? A passage is relevant if it contains information that would \
help answer the question, even partially.

Question: {question}

Passage:
{chunk}

Respond with ONLY yes or no, wrapped in <relevant></relevant> tags."""


def grade_relevance(question: str, chunk: str, provider: str | None = None) -> bool:
    response = llm.generate(RELEVANCE_PROMPT.format(question=question, chunk=chunk), provider=provider)
    match = re.search(r"<relevant>(.*?)</relevant>", response, re.S)
    return bool(match) and match.group(1).strip().lower() == "yes"
# --8<-- [end:relevance]


# ======================================================================
# 2. Generate
# ======================================================================

# --8<-- [start:generate]
GENERATE_PROMPT = """Answer the question using ONLY the context below. \
If the context doesn't contain enough to answer, say so explicitly \
rather than guessing.

Context:
{context}

Question: {question}"""


def generate_answer(question: str, chunks: list[str], provider: str | None = None) -> str:
    context = "\n---\n".join(chunks) if chunks else "(no relevant context found)"
    return llm.generate(GENERATE_PROMPT.format(context=context, question=question), provider=provider)
# --8<-- [end:generate]


# ======================================================================
# 3. Grade groundedness (local NLI model, no LLM call)
# ======================================================================

_nli_model: CrossEncoder | None = None

# --8<-- [start:groundedness]
def grade_groundedness(answer: str, chunks: list[str]) -> dict:
    """premise = the relevant retrieved context (concatenated),
    hypothesis = the generated answer. "entailment" means the context
    actually supports the answer; "contradiction" or "neutral" both
    count as NOT grounded — either the context disagrees with the
    answer, or it simply doesn't say enough to support it.

    Known simplification: this checks the answer as ONE hypothesis
    against the whole context as one premise. A more precise version
    would decompose the answer into individual claims and check each
    separately — this single-pair version is enough to demonstrate the
    mechanism, but can miss a partially-ungrounded multi-claim answer
    where most of it is supported and only one clause isn't."""
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(NLI_MODEL_NAME)

    if not chunks:
        return {"grounded": False, "label": "no_context"}

    context = " ".join(chunks)
    scores = _nli_model.predict([(context, answer)])[0]
    label = _nli_model.config.id2label[int(np.argmax(scores))]
    return {"grounded": label == "entailment", "label": label}
# --8<-- [end:groundedness]


# ======================================================================
# 4. Grade usefulness (LLM)
# ======================================================================

# --8<-- [start:usefulness]
USEFULNESS_PROMPT = """Does the answer below actually address the \
question — regardless of whether it's factually correct? An answer \
that's grounded but off-topic, or that just says "I don't know" when \
a real answer was possible, is NOT useful.

Question: {question}

Answer: {answer}

Respond with ONLY yes or no, wrapped in <useful></useful> tags."""


def grade_usefulness(question: str, answer: str, provider: str | None = None) -> bool:
    response = llm.generate(USEFULNESS_PROMPT.format(question=question, answer=answer), provider=provider)
    match = re.search(r"<useful>(.*?)</useful>", response, re.S)
    return bool(match) and match.group(1).strip().lower() == "yes"
# --8<-- [end:usefulness]


# ======================================================================
# 5. The full loop
# ======================================================================

# --8<-- [start:pipeline]
def self_rag(
    question: str,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    k: int = RETRIEVE_K,
) -> dict:
    """Retrieve -> grade relevance -> generate -> grade groundedness ->
    grade usefulness -> return, regenerate once, or fall back to an
    honest "insufficient support" response instead of a second silent
    failure."""
    retrieved = naive_retrieve(question, chunks, embed_model, k)
    relevant = [c for c in retrieved if grade_relevance(question, c, provider=provider)]

    trace = {"retrieved": len(retrieved), "relevant": len(relevant), "attempts": []}

    for attempt in range(MAX_REGENERATIONS + 1):
        answer = generate_answer(question, relevant, provider=provider)
        groundedness = grade_groundedness(answer, relevant)
        useful = grade_usefulness(question, answer, provider=provider)
        trace["attempts"].append({"answer": answer, "groundedness": groundedness, "useful": useful})

        if groundedness["grounded"] and useful:
            trace["final_answer"] = answer
            trace["status"] = "ok"
            return trace

    trace["final_answer"] = (
        "I don't have enough reliably-supported information in the retrieved context to answer this confidently."
    )
    trace["status"] = "fell_back"
    return trace
# --8<-- [end:pipeline]


# ======================================================================
# CLI
# ======================================================================


def run(question: str, provider: str | None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    result = self_rag(question, chunks, embed_model, provider=provider)
    print(f"Question: {question}\n")
    print(f"Retrieved: {result['retrieved']}, graded relevant: {result['relevant']}\n")
    for i, attempt in enumerate(result["attempts"]):
        print(f"--- attempt {i + 1} ---")
        print(f"Answer: {attempt['answer']}")
        print(f"Groundedness: {attempt['groundedness']['label']} (grounded={attempt['groundedness']['grounded']})")
        print(f"Useful: {attempt['useful']}\n")
    print(f"Status: {result['status']}")
    print(f"Final answer: {result['final_answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompted Self-RAG loop demo.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()
    run(args.query, args.provider)
