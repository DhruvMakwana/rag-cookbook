"""
Six pre-retrieval query transformation techniques, each demonstrated as a
live before/after: naive retrieval on the raw query, vs. retrieval after
the technique is applied — against the same sample document, but with
its own scenario matched to that technique's actual "when to use" case
(see SCENARIOS below), not one shared generic question set. A vague,
colloquial query for query rewriting; a broad multi-faceted question for
multi-query/RAG-Fusion; a numeric-fact question for HyDE; and so on.

All 6 need an LLM key — query transformation is inherently LLM-driven.

Run:
    python query_transformation.py --technique rewrite
    python query_transformation.py --technique multi_query
    python query_transformation.py --technique hyde
    python query_transformation.py --technique step_back
    python query_transformation.py --technique decompose
    python query_transformation.py --technique ircot
"""

from __future__ import annotations

import argparse
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
RRF_K = 60  # standard damping constant for Reciprocal Rank Fusion

# Same 8 question/keyword pairs used in chunking-strategies/ and
# embedding-model-selection/ — keywords verified against the real
# "Attention Is All You Need" text there. Kept as a general-purpose
# baseline (e.g. to show the corpus-cleanup effect on Recall@k), but no
# longer used to demo any single technique — each technique below gets
# its own scenario matched to its actual "when to use" case instead.
EVAL_SET = [
    {"question": "How many attention heads did they use?", "keyword": "h = 8"},
    {"question": "What is the model's embedding dimension?", "keyword": "dmodel = 512"},
    {"question": "How many layers are in the encoder?", "keyword": "N = 6"},
    {"question": "What optimizer was used for training?", "keyword": "Adam"},
    {"question": "What BLEU score did they get on English-to-German translation?", "keyword": "28.4"},
    {"question": "What GPUs was the model trained on?", "keyword": "P100"},
    {"question": "How long did the base model train for?", "keyword": "12 hours"},
    {"question": "What dropout rate did they use?", "keyword": "Pdrop = 0.1"},
]

# One scenario per technique, each built to match that technique's own
# documented "when to use" case (not the generic EVAL_SET above) — see
# each technique's blog section for why this particular query is a good
# fit. Keywords verified present verbatim in the actual paper text.
SCENARIOS = {
    "rewrite": {
        "question": "why dont they just use RNNs like everyone else did before",
        "keyword": "precludes parallelization",
        "k": 3,
    },
    "multi_query": {
        "question": "What lets this model connect words that are far apart in a sentence without losing track over long distances?",
        "keyword": "maximum path length",
        "k": 5,
    },
    "hyde": {
        "question": "What BLEU score did they get on English-to-German translation?",
        "keyword": "28.4",
        "k": 1,
    },
    "step_back": {
        "question": "What value did they use for Pdrop during training?",
        "keyword": "Pdrop",
        "k": 3,
    },
    "decompose": {
        "question": "What optimizer did they use, and how many attention heads did they use?",
        "keywords": ["Adam", "h = 8"],
        "k": 6,
    },
}


# ======================================================================
# Setup / data loading / eval helpers
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Loads the sample PDF's text, with the front-matter boilerplate
    (permission notice + the 7-author name/affiliation/email block)
    trimmed off. That block is single-newline-separated with no real
    paragraph structure for recursive splitting to respect, and none of
    the questions on this page are about authorship — keeping it in
    only adds noisy, low-signal chunks competing for retrieval slots
    that are relevant to nothing being asked here."""
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


def top_k_indices(query_vec: np.ndarray, chunk_vecs: np.ndarray, k: int) -> list[int]:
    sims = [cosine_sim(query_vec, v) for v in chunk_vecs]
    return list(np.argsort(sims)[::-1][:k])


def recall_at_k(chunks: list[str], embed_model: SentenceTransformer, retrieve_fn, k: int = 3) -> float:
    """`retrieve_fn(question, chunks, embed_model, k)` must return the
    top-k retrieved chunk texts for one question. Checks whether each
    question's keyword shows up anywhere in its top-k."""
    hits = 0
    for item in EVAL_SET:
        retrieved = retrieve_fn(item["question"], chunks, embed_model, k)
        if item["keyword"] in " ".join(retrieved):
            hits += 1
    return hits / len(EVAL_SET)


def naive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3) -> list[str]:
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return [chunks[i] for i in top_k_indices(query_vec, chunk_vecs, k)]


# ======================================================================
# 1. Query Rewriting
# ======================================================================
# --8<-- [start:rewrite]
REWRITE_PROMPT = """Rewrite this query to be clearer and more specific \
for searching a technical document. Return ONLY the rewritten query, \
nothing else.

Query: {query}"""


def rewrite_query(query: str, provider: str | None = None) -> str:
    return llm.generate(REWRITE_PROMPT.format(query=query), provider=provider).strip()


def rewrite_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    rewritten = rewrite_query(question, provider=provider)
    return naive_retrieve(rewritten, chunks, embed_model, k)
# --8<-- [end:rewrite]


# ======================================================================
# 2. Multi-Query Retrieval & RAG-Fusion (RRF)
# ======================================================================
# --8<-- [start:multi_query]
MULTI_QUERY_PROMPT = """Generate 3 different phrasings of the question \
below, each capturing a different angle on the same underlying \
information need. Return one per line, nothing else.

Question: {query}"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Each chunk's fused score = sum of 1/(k + rank) across every list
    it appears in. A chunk ranked highly by multiple query variations
    outscores one that only one variation liked."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def multi_query_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    response = llm.generate(MULTI_QUERY_PROMPT.format(query=question), provider=provider)
    variations = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    variations.append(question)  # always include the original

    ranked_lists = []
    for variant in variations:
        query_vec = embed_model.encode(variant, convert_to_numpy=True)
        chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        ranked_indices = top_k_indices(query_vec, chunk_vecs, k=10)
        ranked_lists.append([chunks[i] for i in ranked_indices])

    return reciprocal_rank_fusion(ranked_lists)[:k]
# --8<-- [end:multi_query]


# ======================================================================
# 3. HyDE (Hypothetical Document Embeddings)
# ======================================================================
# --8<-- [start:hyde]
HYDE_PROMPT = "Write a short passage that answers this question: {query}"


def hyde_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    hypothetical_answer = llm.generate(HYDE_PROMPT.format(query=question), provider=provider)
    query_vec = embed_model.encode(hypothetical_answer, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return [chunks[i] for i in top_k_indices(query_vec, chunk_vecs, k)]
# --8<-- [end:hyde]


# ======================================================================
# 4. Step-Back Prompting
# ======================================================================
# --8<-- [start:step_back]
STEP_BACK_PROMPT = """What is a more general question that would help \
answer this specific question? Return ONLY the general question.

Specific question: {query}"""


def step_back_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    general_question = llm.generate(STEP_BACK_PROMPT.format(query=question), provider=provider).strip()

    specific_hits = naive_retrieve(question, chunks, embed_model, k=k)
    general_hits = naive_retrieve(general_question, chunks, embed_model, k=k)

    combined, seen = [], set()
    for chunk in specific_hits + general_hits:
        if chunk not in seen:
            combined.append(chunk)
            seen.add(chunk)
    return combined[:k]
# --8<-- [end:step_back]


# ======================================================================
# 5. Query Decomposition
# ======================================================================
# --8<-- [start:decompose]
DECOMPOSE_PROMPT = """If the question below is asking multiple distinct \
things, break it into separate sub-questions, one per line. If it's \
already a single question, return it unchanged on one line.

Question: {query}"""


def decompose_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None) -> list[str]:
    response = llm.generate(DECOMPOSE_PROMPT.format(query=question), provider=provider)
    sub_questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

    per_question_k = max(1, k // len(sub_questions))
    combined, seen = [], set()
    for sub_q in sub_questions:
        for chunk in naive_retrieve(sub_q, chunks, embed_model, k=per_question_k):
            if chunk not in seen:
                combined.append(chunk)
                seen.add(chunk)
    return combined[:k]
# --8<-- [end:decompose]


# ======================================================================
# 6. Multi-Hop Query Planning (IRCoT-style)
# ======================================================================
# --8<-- [start:ircot]
IRCOT_REASON_PROMPT = """You are answering a question using a document \
retrieval system — you do NOT have the document memorized, but each \
reasoning step you write will be used as a search query against it.

Question: {question}

Retrieved so far:
{context}

Write ONE short, specific, searchable sentence describing the next \
fact you need to look up to make progress (this sentence becomes the \
next search query, so keep it concrete and on-topic — do not ask what \
document this is, assume it's a technical ML paper). If the retrieved \
information above is already enough to answer, start your sentence \
with "ANSWER:" instead."""


FINAL_SYNTHESIS_PROMPT = """Question: {question}

Retrieved passages across {n_hops} rounds of search:
{context}

Using ONLY the passages above, give the final answer in 1-2 sentences. \
If they genuinely don't contain enough to answer, say so plainly."""


def ircot_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3, provider: str | None = None, max_hops: int = 3) -> dict:
    """Interleave reasoning and retrieval: generate one reasoning step,
    use THAT step as the next retrieval query (not the original question
    again), and repeat — each hop's retrieval is informed by what the
    model has actually seen retrieved so far, unlike decomposition's
    fixed upfront split. Critically, the model must be shown the real
    retrieved TEXT at each hop, not just its own prior reasoning
    sentences — otherwise it has no evidence to judge whether it knows
    enough yet."""
    context_chunks: list[str] = []
    reasoning_trace: list[str] = []

    for hop in range(max_hops):
        context_text = "\n---\n".join(context_chunks) if context_chunks else "(nothing retrieved yet)"
        step = llm.generate(IRCOT_REASON_PROMPT.format(question=question, context=context_text), provider=provider).strip()
        reasoning_trace.append(step)

        if step.upper().startswith("ANSWER:"):
            break

        hop_chunks = naive_retrieve(step, chunks, embed_model, k=2)
        for chunk in hop_chunks:
            if chunk not in context_chunks:
                context_chunks.append(chunk)

    final_context = "\n---\n".join(context_chunks) if context_chunks else "(nothing retrieved)"
    final_answer = llm.generate(
        FINAL_SYNTHESIS_PROMPT.format(question=question, n_hops=len(reasoning_trace), context=final_context),
        provider=provider,
    )

    return {"reasoning_trace": reasoning_trace, "retrieved_chunks": context_chunks[:k], "final_answer": final_answer}
# --8<-- [end:ircot]


# ======================================================================
# CLI
# ======================================================================

TECHNIQUES = ["rewrite", "multi_query", "hyde", "step_back", "decompose", "ircot"]


def run(technique: str, provider: str | None = None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if technique == "ircot":
        question = "What optimizer did they use, and how many attention heads did they use?"
        result = ircot_retrieve(question, chunks, embed_model, provider=provider)
        print(f"Question: {question}\n")
        for i, step in enumerate(result["reasoning_trace"]):
            print(f"Hop {i + 1}: {step}")
        print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks across all hops.")
        print(f"\nFinal answer: {result['final_answer']}")
        return

    retrieve_fns = {
        "rewrite": lambda q, c, m, k: rewrite_retrieve(q, c, m, k, provider=provider),
        "multi_query": lambda q, c, m, k: multi_query_retrieve(q, c, m, k, provider=provider),
        "hyde": lambda q, c, m, k: hyde_retrieve(q, c, m, k, provider=provider),
        "step_back": lambda q, c, m, k: step_back_retrieve(q, c, m, k, provider=provider),
        "decompose": lambda q, c, m, k: decompose_retrieve(q, c, m, k, provider=provider),
    }

    scenario = SCENARIOS[technique]
    question, k = scenario["question"], scenario["k"]
    keywords = scenario.get("keywords") or [scenario["keyword"]]

    naive_hits = naive_retrieve(question, chunks, embed_model, k)
    technique_hits = retrieve_fns[technique](question, chunks, embed_model, k)
    naive_text = " ".join(naive_hits)
    technique_text = " ".join(technique_hits)

    print(f"Question: {question}\n")
    for kw in keywords:
        found_naive = kw in naive_text
        found_technique = kw in technique_text
        print(f'"{kw}" found — naive: {found_naive}, {technique}: {found_technique}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query transformation demos.")
    parser.add_argument("--technique", choices=TECHNIQUES, default="rewrite")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()
    run(args.technique, provider=args.provider)
