"""
Six retrieval methods, run against the same sample document. Two run
fully locally (mmr, hierarchical); four need an LLM key (self_query,
hypothetical_questions, raptor, adaptive).

Run:
    python retrieval_methods.py --method mmr
    python retrieval_methods.py --method hierarchical
    python retrieval_methods.py --method self_query --provider anthropic
    python retrieval_methods.py --method hypothetical_questions
    python retrieval_methods.py --method raptor
    python retrieval_methods.py --method adaptive
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# A small metadata-tagged corpus for the self-query demo — real metadata,
# not extracted from the sample PDF (which has none), so the technique
# is demonstrable without needing a metadata-rich source document.
METADATA_CORPUS = [
    {"text": "Employees may work remotely up to 3 days per week with manager approval.", "section": "remote_work", "year": 2024},
    {"text": "International remote work requires HR approval and is limited to 30 days per year.", "section": "remote_work", "year": 2024},
    {"text": "All travel expenses must be submitted within 30 days with receipts attached.", "section": "expenses", "year": 2023},
    {"text": "Home office equipment up to $500/year is reimbursable with manager approval.", "section": "expenses", "year": 2024},
    {"text": "The company matches 401k contributions up to 4% of base salary.", "section": "benefits", "year": 2023},
    {"text": "New parents receive 16 weeks of paid parental leave.", "section": "benefits", "year": 2024},
]


# ======================================================================
# Setup / data loading helpers
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


# ======================================================================
# 1. Maximal Marginal Relevance (MMR)
# ======================================================================
# --8<-- [start:mmr]
def mmr_retrieve(query: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 5, lambda_mult: float = 0.5) -> list[str]:
    """Diversify top-k results: greedily pick the chunk that's most
    relevant to the query MINUS most similar to chunks already picked.
    `lambda_mult=1.0` is pure relevance (identical to plain top-k);
    `lambda_mult=0.0` is pure diversity, ignoring the query entirely."""
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    selected_indices = maximal_marginal_relevance(query_vec, list(chunk_vecs), lambda_mult=lambda_mult, k=k)
    return [chunks[i] for i in selected_indices]
# --8<-- [end:mmr]


# ======================================================================
# 2. Self-Query Retrieval
# ======================================================================
# --8<-- [start:self_query]
SELF_QUERY_PROMPT = """Given the natural-language query below, extract:
1. "semantic_query": the part that should be matched by meaning
2. "filters": any metadata constraints mentioned, as a JSON object with
   only the keys below — omit any key not clearly implied by the query.

Available metadata fields (this is the schema — without it, the model
has no way to know these are valid categorical values, not just query
content; this is exactly what a real self-query retriever's field-info
schema provides):
- "section": one of "remote_work", "expenses", "benefits"
- "year": an integer, e.g. 2023 or 2024

Return ONLY a JSON object like {{"semantic_query": "...", "filters": {{...}}}}.

Query: {query}"""


def self_query_retrieve(query: str, corpus: list[dict], provider: str | None = None, k: int = 3) -> list[dict]:
    """Ask an LLM to split a natural-language query into a semantic part
    and structured metadata filters, then apply the filters BEFORE
    ranking by semantic similarity — so "2024 remote work policy" only
    ever considers 2024 remote_work entries, rather than hoping semantic
    similarity alone surfaces the right year."""
    response = llm.generate(SELF_QUERY_PROMPT.format(query=query), provider=provider)
    parsed = json.loads(response.strip().strip("`").removeprefix("json").strip())
    semantic_query = parsed["semantic_query"]
    filters = parsed.get("filters", {})

    candidates = [doc for doc in corpus if all(doc.get(key) == value for key, value in filters.items())]
    if not candidates:
        return []

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vec = embed_model.encode(semantic_query, convert_to_numpy=True)
    doc_vecs = embed_model.encode([doc["text"] for doc in candidates], convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidates, doc_vecs), key=lambda pair: -cosine_sim(query_vec, pair[1]))
    return [doc for doc, _ in scored[:k]]
# --8<-- [end:self_query]


# ======================================================================
# 3. Hypothetical Questions
# ======================================================================
# --8<-- [start:hypothetical_questions]
HYPOTHETICAL_QUESTIONS_PROMPT = """Generate 3 short questions that the \
passage below could answer. Return one question per line, nothing else.

Passage: {passage}"""


def build_hypothetical_questions_index(chunks: list[str], provider: str | None = None) -> list[dict]:
    """At INDEX time (not query time), generate plausible questions each
    chunk could answer, and embed those questions instead of the chunk's
    own declarative text — question-to-question similarity at query time
    is often stronger than question-to-answer similarity."""
    index = []
    for chunk in chunks:
        response = llm.generate(HYPOTHETICAL_QUESTIONS_PROMPT.format(passage=chunk), provider=provider)
        questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
        for question in questions:
            index.append({"question": question, "chunk": chunk})
    return index


def retrieve_via_hypothetical_questions(query: str, question_index: list[dict], embed_model: SentenceTransformer, k: int = 3) -> list[str]:
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    question_vecs = embed_model.encode([item["question"] for item in question_index], convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(question_index, question_vecs), key=lambda pair: -cosine_sim(query_vec, pair[1]))
    seen_chunks, results = set(), []
    for item, _ in scored:
        if item["chunk"] not in seen_chunks:
            results.append(item["chunk"])
            seen_chunks.add(item["chunk"])
        if len(results) >= k:
            break
    return results
# --8<-- [end:hypothetical_questions]


# ======================================================================
# 4. Hierarchical Index Retrieval
# ======================================================================
# --8<-- [start:hierarchical]
def hierarchical_retrieve(query: str, chunks: list[str], embed_model: SentenceTransformer, section_size: int = 5, k: int = 3) -> list[str]:
    """Narrow the search space in two stages instead of searching every
    chunk flat: group chunks into sections, embed each section (here, by
    embedding its concatenated text — production systems often use an
    LLM summary instead for a sharper section-level representation),
    find the best-matching section first, then rank only its chunks."""
    sections = [chunks[i:i + section_size] for i in range(0, len(chunks), section_size)]
    section_texts = [" ".join(section) for section in sections]

    query_vec = embed_model.encode(query, convert_to_numpy=True)
    section_vecs = embed_model.encode(section_texts, convert_to_numpy=True, show_progress_bar=False)
    best_section_idx = int(np.argmax([cosine_sim(query_vec, vec) for vec in section_vecs]))

    candidate_chunks = sections[best_section_idx]
    chunk_vecs = embed_model.encode(candidate_chunks, convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidate_chunks, chunk_vecs), key=lambda pair: -cosine_sim(query_vec, pair[1]))
    return [chunk for chunk, _ in scored[:k]]
# --8<-- [end:hierarchical]


# ======================================================================
# 5. RAPTOR (scaled down)
# ======================================================================
# --8<-- [start:raptor]
RAPTOR_SUMMARY_PROMPT = """Summarize the following related passages in \
2-3 sentences, capturing the common theme:

{passages}"""


def build_raptor_tree(chunks: list[str], embed_model: SentenceTransformer, n_clusters: int = 3, provider: str | None = None) -> dict:
    """Recursively cluster chunks by embedding similarity and summarize
    each cluster with an LLM, building one extra tree layer. (The full
    RAPTOR paper recurses this multiple times up to a single root
    summary; one layer is enough to demonstrate the mechanism without
    the full offline cost.)"""
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    n_clusters = min(n_clusters, len(chunks))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(chunk_vecs)

    summary_nodes = []
    for cluster_id in range(n_clusters):
        cluster_chunks = [c for c, label in zip(chunks, labels) if label == cluster_id]
        if not cluster_chunks:
            continue
        summary = llm.generate(
            RAPTOR_SUMMARY_PROMPT.format(passages="\n\n".join(cluster_chunks)), provider=provider
        )
        summary_nodes.append({"summary": summary, "cluster_chunks": cluster_chunks})

    return {"leaves": chunks, "summaries": summary_nodes}


def raptor_retrieve(query: str, tree: dict, embed_model: SentenceTransformer, k: int = 2) -> list[str]:
    """Search leaf chunks AND summary nodes together — a narrow factual
    question tends to match best at leaf level; a broad synthesis
    question tends to match best at a summary node."""
    query_vec = embed_model.encode(query, convert_to_numpy=True)

    candidates = [(leaf, "leaf") for leaf in tree["leaves"]]
    candidates += [(node["summary"], "summary") for node in tree["summaries"]]

    texts = [c[0] for c in candidates]
    vecs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    scored = sorted(zip(candidates, vecs), key=lambda pair: -cosine_sim(query_vec, pair[1]))
    return [f"[{kind}] {text}" for (text, kind), _ in scored[:k]]
# --8<-- [end:raptor]


# ======================================================================
# 6. Adaptive retrieval (simplified FLARE via self-assessed confidence)
# ======================================================================
# --8<-- [start:adaptive]
DRAFT_PROMPT = "Answer this question in one sentence, from general knowledge: {question}"
CONFIDENCE_PROMPT = """Sentence: "{sentence}"

On a scale of 1 (pure guess) to 5 (certain, well-known fact), how \
confident are you this sentence is factually correct? Reply with ONLY \
the number."""


def adaptive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, confidence_threshold: int = 4, provider: str | None = None) -> dict:
    """A simplified stand-in for FLARE: instead of monitoring token-level
    log-probabilities during generation (which needs logprob access most
    hosted chat APIs don't expose the same way), ask the model to
    self-report confidence in its own draft answer, and only retrieve
    (and regenerate) if that self-reported confidence is low. Cruder
    than real FLARE, but the same "retrieve only when needed" shape,
    and fully implementable against any chat-completions API."""
    draft = llm.generate(DRAFT_PROMPT.format(question=question), provider=provider)
    confidence_response = llm.generate(CONFIDENCE_PROMPT.format(sentence=draft), provider=provider)
    confidence = int(re.search(r"\d", confidence_response).group())

    if confidence >= confidence_threshold:
        return {"answer": draft, "confidence": confidence, "retrieved": False}

    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    best_idx = int(np.argmax([cosine_sim(query_vec, vec) for vec in chunk_vecs]))
    context = chunks[best_idx]

    grounded_prompt = f"Using ONLY this context, answer in one sentence: {context}\n\nQuestion: {question}"
    grounded_answer = llm.generate(grounded_prompt, provider=provider)
    return {"answer": grounded_answer, "confidence": confidence, "retrieved": True, "context_used": context}
# --8<-- [end:adaptive]


# ======================================================================
# CLI
# ======================================================================

METHODS = ["mmr", "self_query", "hypothetical_questions", "hierarchical", "raptor", "adaptive"]


def run(method: str, provider: str | None = None, threshold: int = 4) -> None:
    load_environment()

    if method == "self_query":
        for query in ["What's the 2024 remote work policy?", "What benefits were introduced in 2023?"]:
            print(f"\nQuery: {query}")
            for doc in self_query_retrieve(query, METADATA_CORPUS, provider=provider):
                print(f"  -> [{doc['section']}, {doc['year']}] {doc['text']}")
        return

    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query = "What is multi-head attention and how many heads are used?"

    if method == "mmr":
        print("--- lambda_mult=1.0 (pure relevance, like plain top-k) ---")
        for c in mmr_retrieve(query, chunks, embed_model, k=4, lambda_mult=1.0):
            print(f"  {c[:100]}...")
        print("\n--- lambda_mult=0.3 (diversity-weighted) ---")
        for c in mmr_retrieve(query, chunks, embed_model, k=4, lambda_mult=0.3):
            print(f"  {c[:100]}...")

    elif method == "hierarchical":
        for c in hierarchical_retrieve(query, chunks, embed_model, k=3):
            print(f"  {c[:150]}...")

    elif method == "hypothetical_questions":
        sample_chunks = chunks[:8]  # keep the demo's LLM-call count small
        print(f"Building hypothetical-question index for {len(sample_chunks)} chunks...")
        index = build_hypothetical_questions_index(sample_chunks, provider=provider)
        print(f"Generated {len(index)} question->chunk pairs.\n")
        for c in retrieve_via_hypothetical_questions(query, index, embed_model, k=3):
            print(f"  {c[:150]}...")

    elif method == "raptor":
        sample_chunks = chunks[:12]  # keep clustering + summarization calls small
        print(f"Building RAPTOR tree for {len(sample_chunks)} chunks...")
        tree = build_raptor_tree(sample_chunks, embed_model, n_clusters=3, provider=provider)
        for node in tree["summaries"]:
            print(f"\n[cluster summary] {node['summary']}")
        print(f"\nRetrieving for a broad question: 'What is this paper about overall?'")
        for r in raptor_retrieve("What is this paper about overall?", tree, embed_model, k=2):
            print(f"  {r[:200]}...")

    elif method == "adaptive":
        for question in ["What is the capital of France?", "What is the exact FLOPS count for training the big Transformer model?"]:
            result = adaptive_retrieve(question, chunks, embed_model, confidence_threshold=threshold, provider=provider)
            print(f"\nQ: {question}")
            print(f"   confidence={result['confidence']}, retrieved={result['retrieved']}")
            print(f"   A: {result['answer']}")
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval methods demos.")
    parser.add_argument("--method", choices=METHODS, default="mmr")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    parser.add_argument("--threshold", type=int, default=4, help="Confidence threshold (1-5) for --method adaptive.")
    args = parser.parse_args()
    run(args.method, provider=args.provider, threshold=args.threshold)
